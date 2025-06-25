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

# KINOVA
TIMEOUT_DURATION = 10  # (seconds)
BASE01_POS_X2 = 0.10  # (meters)
# Object Position (pixel)
OBJECT_X = 640
OBJECT_Y = 360
# ARUCO
MARKER_ID = 10  # marker ID
MARKER_SIZE_CM = 2.05  # (centimeters)

# Load camera matrix and distortion coefficients from pickle file
with open("../01_calibration/calibration_data.pkl", "rb") as f:
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


def get_camera_pose_matrix(base, extrinsics_read):
    feed = base.GetMeasuredCartesianPose()

    end_effector_position = np.array([
        feed.x,
        feed.y,
        feed.z
    ])
    rx = feed.theta_x
    ry = feed.theta_y
    rz = feed.theta_z
    EE_rotation_matrix = R.from_euler('xyz', [rx, ry, rz], degrees=True).as_matrix()
    extrinsic_translation = np.array([
        extrinsics_read.translation.t_x,
        extrinsics_read.translation.t_y,
        extrinsics_read.translation.t_z
    ])
    camera_offset_world = EE_rotation_matrix @ extrinsic_translation
    camera_position = end_effector_position + camera_offset_world

    print("EE orientation (deg):", rx, ry, rz)
    print("Rotation matrix (ZYX):\n", R.from_euler('zyx', [rz, ry, rx], degrees=True).as_matrix())
    print("Rotation matrix (XYZ):\n", R.from_euler('xyz', [rx, ry, rz], degrees=True).as_matrix())

    # Build 4x4 pose
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = EE_rotation_matrix
    camera_pose[:3, 3] = camera_position
    print("Camera position in world frame:", camera_pose)
    return camera_pose


def get_marker_pose_in_world(frame, marker_id, marker_size, camera_matrix, dist_coeff, camera_pose):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_size, camera_matrix, dist_coeff
        )
        try:
            idx = [i for i, id in enumerate(ids) if id[0] == marker_id][0]
        except IndexError:
            print(f"Marker {marker_id} not found in this frame.")
            return None, None

        marker_tvec = tvecs[idx][0]  # (x, y, z) in camera frame
        marker_rvec = rvecs[idx][0]  # rotation vector
        # Convert rvec to rotation matrix
        marker_rmat, _ = cv2.Rodrigues(marker_rvec)
        # Build 4x4 marker pose in camera frame
        marker_in_camera = np.eye(4)
        marker_in_camera[:3, :3] = marker_rmat
        # Sign flip (OpenCV vs World):
        marker_in_camera[:3, 3] = [-marker_tvec[0], -marker_tvec[1], marker_tvec[2]]
        # Chain with camera pose in world frame
        marker_in_world = camera_pose @ marker_in_camera
        # Extract world coordinates for robot
        object_world_position = marker_in_world[:3, 3]
        print(f"Marker position in world frame: {object_world_position}")
        return object_world_position, marker_in_world
    else:
        print(f"No markers found in this frame.")
        return None, None


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


def print_all_poses(feed, camera_pose, marker_pos, marker_rot=None):
    # End-Effector
    print("End-Effector Position (m):")
    print(f"  X = {feed.x:.3f}")
    print(f"  Y = {feed.y:.3f}")
    print(f"  Z = {feed.z:.3f}")
    print("End-Effector Orientation (deg):")
    print(f"  Rx = {feed.theta_x:.3f}")
    print(f"  Ry = {feed.theta_y:.3f}")
    print(f"  Rz = {feed.theta_z:.3f}")
    print("-" * 40)

    # Camera
    print("Camera Position (m):")
    print(f"  X = {camera_pose[0,3]:.3f}")
    print(f"  Y = {camera_pose[1,3]:.3f}")
    print(f"  Z = {camera_pose[2,3]:.3f}")
    # Extract Euler angles from rotation matrix (assuming 'zyx' order, degrees)
    cam_rot = R.from_matrix(camera_pose[:3,:3]).as_euler('xyz', degrees=True)  # check
    print("Camera Orientation (deg):")
    print(f"  Rx = {cam_rot[2]:.3f}")
    print(f"  Ry = {cam_rot[1]:.3f}")
    print(f"  Rz = {cam_rot[0]:.3f}")
    print("-" * 40)

    # Marker
    print("Marker Position (m):")
    print(f"  X = {marker_pos[0]:.3f}")
    print(f"  Y = {marker_pos[1]:.3f}")
    print(f"  Z = {marker_pos[2]:.3f}")
    if marker_rot is not None:
        print("Marker Orientation (deg):")
        print(f"  Rx = {marker_rot[0]:.3f}")
        print(f"  Ry = {marker_rot[1]:.3f}")
        print(f"  Rz = {marker_rot[2]:.3f}")
    print("-" * 40)


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
        feed = base.GetMeasuredCartesianPose()

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
                time.sleep(2)
                frame = take_screenshot()
                if frame is not None:
                    camera_pose = get_camera_pose_matrix(base, extrinsics_read)
                    object_world_position, marker_in_world = get_marker_pose_in_world(
                        frame,
                        MARKER_ID,
                        MARKER_SIZE_CM / 100,
                        cameraMatrix,
                        dist,
                        camera_pose
                    )
                    print_all_poses(feed, camera_pose, object_world_position, marker_rot=None)
                    if object_world_position is not None:
                        go_to_target_position(base, base_cyclic, object_world_position)
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()