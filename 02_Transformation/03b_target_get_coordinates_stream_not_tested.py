#########################################################################################
#    Kinova Gen3 Robotic Arm                                                            #
#                                                                                       #
#    Pose Estimation with aruco markers                                      #
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
MARKER_ID_1 = 10  # first marker ID
MARKER_SIZE_CM = 4.4  # in cm

# Shared data dictionary
shared_data = {"object_in_camera": None}

# Load camera matrix and distortion coefficients from pickle file
with open("calibration_data.pkl", "rb") as f:
    data = pickle.load(f)
    cameraMatrix = data['cameraMatrix']
    dist = data['dist']
print(cameraMatrix)
print(dist)


def camera_stream(shared_data, extrinsics_log):
    # Initialize ArUco detector - openCV 4.11
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    source = cv2.VideoCapture("rtsp://192.168.1.10/color")

    win_name = 'Camera Preview'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    while cv2.waitKey(1) != 27:
        try:
            has_frame, frame = source.read()

            if not has_frame or frame is None or frame.size == 0:
                print("Invalid frame, skipping")
                continue

            # Detect ArUco markers
            corners, ids, rejected = detector.detectMarkers(frame)

            if ids is not None:
                # Draw detected markers
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                # Estimate pose of each marker03_target_pose_estimation.py
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE_CM, cameraMatrix, dist)
                for i in range(len(ids)):
                    tvec = tvecs[i][0]  # [x, y, z] in camera coordinates
                    print(f"Marker {ids[i][0]} position in camera space: {tvec}")

                # Find the specified markers
                corners = find_marker_by_id(corners, ids, MARKER_ID_1)

                if corners is not None:
                    # Get the corresponding tvecs
                    idx1 = [i for i, id in enumerate(ids) if id[0] == MARKER_ID_1][0]
                    marker_tvec = tvecs[idx1][0]  # [x, y, z] in camera coordinates

                    # tvecs[idx][0] is the translation vector for marker at index idx
                    distance_to_camera = np.linalg.norm(marker_tvec)
                    cv2.putText(frame,
                                f"Distance from camera to marker {MARKER_ID_1} : {distance_to_camera:.2f} cm",
                                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # Object pixel coordinates
                    object_px = (OBJECT_X, OBJECT_Y)

                    # Use marker's depth (Z-coordinate) for object
                    object_depth = marker_tvec[2]

                    # Convert pixel to camera coordinates
                    u, v = object_px
                    fx = cameraMatrix[0, 0]
                    fy = cameraMatrix[1, 1]
                    cx = cameraMatrix[0, 2]
                    cy = cameraMatrix[1, 2]
                    x_cam = (u - cx) * object_depth / fx
                    y_cam = (v - cy) * object_depth / fy
                    z_cam = object_depth

                    object_in_camera = np.array([x_cam, y_cam, z_cam])
                    print(f"Object coordinates in camera frame: {object_in_camera}")
                    # ... compute object_in_camera ...
                    shared_data["object_in_camera"] = object_in_camera
                    # Optionally: also compute and store object_in_robot here
                    shared_data["object_in_robot"] = get_object_position(object_in_camera, extrinsics_log)

                    # Draw circle at the center
                    center = tuple(map(int, corners.mean(axis=0)))
                    cv2.circle(frame, center, 5, (0, 255, 0), -1)

                else:
                    # Optionally: print a message or handle the absence of the marker
                    print(f"Marker {MARKER_ID_1} not found in this frame.")

            height, width = frame.shape[:2]
            center_x, center_y = int(width / 2), int(height / 2)
            cv2.circle(frame, (center_x, center_y), radius=3, color=(0, 0, 255), thickness=-1)
            cv2.circle(frame, (center_x, center_y), radius=12, color=(0, 0, 255), thickness=2)

            cv2.imshow(win_name, frame)

        except Exception as e:
            print(f"Error: {e}")

    source.release()
    cv2.destroyAllWindows()


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


def find_marker_by_id(corners, ids, target_id):
    if ids is not None:
        for i, id in enumerate(ids):
            if id[0] == target_id:
                return corners[i][0]
    return None


def get_object_position(object_in_camera, extrinsics):
    # Extract rotation matrix from extrinsics
    R = np.array([
        [extrinsics.rotation.row1.column1, extrinsics.rotation.row1.column2, extrinsics.rotation.row1.column3],
        [extrinsics.rotation.row2.column1, extrinsics.rotation.row2.column2, extrinsics.rotation.row2.column3],
        [extrinsics.rotation.row3.column1, extrinsics.rotation.row3.column2, extrinsics.rotation.row3.column3]
    ])
    # Extract translation vector
    t = np.array([
        extrinsics.translation.t_x,
        extrinsics.translation.t_y,
        extrinsics.translation.t_z
    ])

    # Transform object coordinates from camera to robot frame
    object_in_robot = R @ object_in_camera + t
    print(f"Object coordinates in robot frame: {object_in_robot}")
    return object_in_robot



def get_target_pose(base_cyclic, extrinsics_log):
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

    # End_Effector
    extrinsics.print_extrinsic_parameters(extrinsics_log)


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


def go_to_start_position(base, base_cyclic, pos_x, pos_z, ang_x):  # cartesian_action
    print("Starting Cartesian action movement to go to Pickup location ...")
    action = Base_pb2.Action()
    feedback = base_cyclic.RefreshFeedback()

    cartesian_pose = action.reach_pose.target_pose
    cartesian_pose.x = feedback.base.tool_pose_x + pos_x  # (meters)
    cartesian_pose.y = feedback.base.tool_pose_y   # (meters)
    cartesian_pose.z = feedback.base.tool_pose_z   # (meters)
    cartesian_pose.theta_x = feedback.base.tool_pose_theta_x  # (degrees)
    cartesian_pose.theta_y = feedback.base.tool_pose_theta_y  # (degrees)
    cartesian_pose.theta_z = feedback.base.tool_pose_theta_z  # (degrees)

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    print("Executing action")
    base.ExecuteAction(action)

    print("Waiting for movement to finish ...")
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
        print("Sequence Start location reached\n")
        # Print the pose information
        print(f"  Position: X={current_pose[0]:.3f}, Y={current_pose[1]:.3f}, Z={current_pose[2]:.3f}")
        print(f"  Orientation: Rx={current_pose[3]:.3f}, Ry={current_pose[4]:.3f}, Rz={current_pose[5]:.3f}")
        print("--------------------")
    else:
        print("Timeout on action notification wait\n")
    return finished


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
            extrinsics_log = vision_config.GetExtrinsicParameters(vision_device_id)

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

        shared_data = {"object_in_camera": None, "object_in_robot": None}
        camera_thread = threading.Thread(target=camera_stream, args=(shared_data, extrinsics_log))
        camera_thread.start()

        try:
            while True:
                if shared_data["object_in_robot"] is not None:
                    print("Object in robot frame:", shared_data["object_in_robot"])
                    # Use this position for robot actions
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            camera_thread.join()


if __name__ == "__main__":
    main()