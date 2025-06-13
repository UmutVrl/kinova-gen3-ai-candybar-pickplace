#########################################################################################
#    Kinova Gen3 Robotic Arm                                                            #
#                                                                                       #
#    Get and transform Target coordinates                                               #
#                                                                                       #
#    written by: U. Vural                                                               #
#                                                                                       #
#                                                                                       #
#                                                                                       #
#    for KISS Project at Furtwangen University                                          #
#                                                                                       #
#########################################################################################
# common
import cv2
import numpy as np
import argparse
import pickle
import threading
import time
from scipy.spatial.transform import Rotation as R

# kinova
from kortex_api.autogen.messages import VisionConfig_pb2
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Base_pb2
from kortex_api.autogen.client_stubs.VisionConfigClientRpc import VisionConfigClient
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.client_stubs.DeviceConfigClientRpc import DeviceConfigClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient

# local
import utilities
import intrinsics
import extrinsics

#  KINOVA
TIMEOUT_DURATION = 10  # (seconds)
BASE01_POS_X2 = 0.10  # (meters)

# Object Position (pixel)
OBJECT_X = 640
OBJECT_Y = 360

# ARUCO
MARKER_ID = 10  # marker ID
MARKER_SIZE_CM = 4.4 # in cm

# Load camera matrix and distortion coefficients from pickle file
with open("calibration_data.pkl", "rb") as f:
    data = pickle.load(f)
    cameraMatrix = data['cameraMatrix']
    dist = data['dist']
print(cameraMatrix)
print(dist)


# Create closure to set an event after an END or an ABORT
def check_for_sequence_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications on a sequence

    Arguments:
    e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)
    """

    def check(notification, e=e):
        event_id = notification.event_identifier
        task_id = notification.task_index
        if event_id == Base_pb2.SEQUENCE_TASK_COMPLETED:
            print("Sequence task {} completed".format(task_id))
        elif event_id == Base_pb2.SEQUENCE_ABORTED:
            print("Sequence aborted with error {}:{}".format(notification.abort_details,
                                                             Base_pb2.SubErrorCodes.Name(notification.abort_details)))
            e.set()
        elif event_id == Base_pb2.SEQUENCE_COMPLETED:
            print("Sequence completed.")
            e.set()
    return check


# Create closure to set an event after an END or an ABORT
def check_for_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications

    Arguments:
    e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)
    """
    def check(notification, e=e):
        print("EVENT : " + Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event == Base_pb2.ACTION_END or notification.action_event == Base_pb2.ACTION_ABORT:
            e.set()
    return check


def get_camera_position_from_feedback(base_cyclic, extrinsics_read):
    """
    Computes the camera position in the robot base (world) frame using real-time feedback.

    Parameters:
        base_cyclic: Kinova BaseCyclicClient instance
        extrinsics_read: ExtrinsicParameters object (from vision_config.GetExtrinsicParameters)

    Returns:
        camera_position: (3,) numpy array, [x, y, z] of camera in world frame (meters)
    """
    feedback = base_cyclic.RefreshFeedback()
    # End effector position in world frame
    gripper_position = np.array([
        feedback.base.tool_pose_x,
        feedback.base.tool_pose_y,
        feedback.base.tool_pose_z
    ])  # meters

    # End effector orientation (Euler angles in degrees)
    rx = feedback.base.tool_pose_theta_x
    ry = feedback.base.tool_pose_theta_y
    rz = feedback.base.tool_pose_theta_z

    # Convert Euler angles to rotation matrix ('xyz' order, degrees)
    EE_rotation_matrix = R.from_euler('xyz', [rx, ry, rz], degrees=True).as_matrix()

    # Extract extrinsic translation vector (camera position relative to gripper, in gripper frame)
    extrinsic_translation = np.array([
        extrinsics_read.translation.t_x,
        extrinsics_read.translation.t_y,
        extrinsics_read.translation.t_z
    ])  # meters

    # Transform extrinsic translation from gripper frame to world frame
    camera_offset_world = EE_rotation_matrix @ extrinsic_translation
    camera_position = gripper_position + camera_offset_world

    camera_rotation = EE_rotation_matrix

    print("Camera position in world frame:", camera_position)
    return camera_position, camera_rotation


def go_to_retract(base, base_cyclic):
    # Make sure the arm is in Single Level Servoing mode
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)

    print("Going to default Retract Position ...")
    action_type = Base_pb2.RequestedActionType()
    action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
    action_list = base.ReadAllActions(action_type)
    action_handle = None

    for action in action_list.action_list:
        if action.name == "Retract":
            action_handle = action.handle

    if action_handle is None:
        print("Can't reach safe position. Exiting")
        return False

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(check_for_end_or_abort(e), Base_pb2.NotificationOptions())

    base.ExecuteActionFromReference(action_handle)
    finished = e.wait(TIMEOUT_DURATION)
    time.sleep(10)
    base.Unsubscribe(notification_handle)

    # Get robot pose
    feedback = base_cyclic.RefreshFeedback()
    current_pose = [
        feedback.base.tool_pose_x,
        feedback.base.tool_pose_y,
        feedback.base.tool_pose_z,
        feedback.base.tool_pose_theta_x,
        feedback.base.tool_pose_theta_y,
        feedback.base.tool_pose_theta_z
    ]

    if finished:
        print("Retract position reached\n")
        # Print the pose information
        print(f"  Position: X={current_pose[0]:.3f}, Y={current_pose[1]:.3f}, Z={current_pose[2]:.3f}")
        print(f"  Orientation: Rx={current_pose[3]:.3f}, Ry={current_pose[4]:.3f}, Rz={current_pose[5]:.3f}")
        print("--------------------")
    else:
        print("Timeout on action notification wait\n")
    return finished


def go_to_start_position(base, base_cyclic, object_in_robot):
    pass  # LOOK LATER


def go_to_target_position(base, base_cyclic, object_in_robot):
    pass  # LOOK LATER


def get_object_world_coordinates_from_frame(camera_pos, camera_rot, object_px, object_depth, cameraMatrix, dist):
    u, v = object_px
    fx = cameraMatrix[0, 0]
    fy = cameraMatrix[1, 1]
    cx = cameraMatrix[0, 2]
    cy = cameraMatrix[1, 2]
    x_cam = (u - cx) * object_depth / fx
    y_cam = (v - cy) * object_depth / fy
    z_cam = object_depth
    object_in_camera = np.array([x_cam, y_cam, z_cam])

    # Use camera_rot directly (3x3), camera_pos directly (3,)
    object_in_world = camera_rot @ object_in_camera + camera_pos
    print(f"Object coordinates in robot/world frame: {object_in_world}")
    return object_in_world


def process_frame(frame, cameraMatrix, dist):
    """
    Detects the specified ArUco marker in the frame and returns:
      - object_px: (u, v) pixel coordinates (marker center)
      - object_depth: z distance from camera to marker (in meters)
    Returns (object_px, object_depth) or (None, None) if not found.
    """
    marker_size = MARKER_SIZE_CM / 100

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_size, cameraMatrix, dist
        )
        try:
            idx = [i for i, id in enumerate(ids) if id[0] == MARKER_ID][0]
        except IndexError:
            print(f"Marker {MARKER_ID} not found in this frame.")
            return None, None

        marker_tvec = tvecs[idx][0]
        object_depth = marker_tvec[2]
        marker_corners = corners[idx][0]
        object_px = tuple(map(int, marker_corners.mean(axis=0)))
        cv2.circle(frame, object_px, 5, (0, 255, 0), -1)
        print(f"Marker {MARKER_ID} position in camera space: {marker_tvec}")
        print(f"Object pixel: {object_px}, Object depth: {object_depth:.2f} m")
        return object_px, object_depth
    else:
        print(f"No markers found in this frame.")
        return None, None


def take_screenshot(rtsp_url="rtsp://192.168.1.10/color", retries=3):
    for _ in range(retries):
        cap = cv2.VideoCapture(rtsp_url)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                return frame
        time.sleep(1)
    print("Failed to capture frame after retries.")
    return None


def main():
    parser = argparse.ArgumentParser()
    args = utilities.parseConnectionArguments(parser)

    with utilities.DeviceConnection.createTcpConnection(args) as router:
        device_manager = DeviceManagerClient(router)
        vision_config = VisionConfigClient(router)
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)

        vision_device_id = intrinsics.get_vision_device_id(device_manager)
        print(f"Vision Device ID: {vision_device_id}")

        if vision_device_id != 0:
            settings = VisionConfig_pb2.SensorSettings()
            settings.sensor = VisionConfig_pb2.SENSOR_COLOR
            settings.resolution = VisionConfig_pb2.RESOLUTION_1280x720
            settings.frame_rate = VisionConfig_pb2.FRAMERATE_15_FPS
            settings.bit_rate = VisionConfig_pb2.BITRATE_10_MBPS  # Try higher if error
            print("protobuf settings:", settings)

            # Set the sensor settings
            vision_config.SetSensorSettings(settings, vision_device_id)
            extrinsics.set_custom_extrinsics(vision_config, vision_device_id)
            extrinsics_read = vision_config.GetExtrinsicParameters(vision_device_id)

            # Autofocus disable
            focus_action_msg = VisionConfig_pb2.SensorFocusAction()
            focus_action_msg.sensor = VisionConfig_pb2.SENSOR_COLOR
            focus_action_msg.focus_action = VisionConfig_pb2.FOCUSACTION_DISABLE_FOCUS
            vision_config.DoSensorFocusAction(focus_action_msg, vision_device_id)

            bilgi = intrinsics.get_active_routed_vision_intrinsics(vision_config, vision_device_id)
            intrinsics.routed_vision_set_manual_focus_medium_distance(vision_config, vision_device_id)

            resolution_string = intrinsics.resolution_to_string(bilgi.resolution)
            frame_width, frame_height = map(int, resolution_string.split('x'))
            print(f"FRAME_WIDTH: {frame_width}")
            print(f"FRAME_HEIGHT: {frame_height}")

        try:
            while True:
                go_to_retract(base, base_cyclic)
                time.sleep(5)
                frame = take_screenshot()
                cv2.imwrite("Debug_Frame", frame)
                if frame is not None:
                    # Process frame and get marker depth and pixel location
                    object_px, object_depth = process_frame(frame, cameraMatrix, dist)
                    camera_pos, camera_rot = get_camera_position_from_feedback(base_cyclic, extrinsics_read)
                    if object_px is not None and object_depth is not None:
                        target_in_world = get_object_world_coordinates_from_frame(
                            camera_pos,
                            camera_rot,
                            object_px,
                            object_depth,
                            cameraMatrix,
                            dist
                        )
                        if target_in_world is not None:
                            go_to_target_position(base, base_cyclic, target_in_world)
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()