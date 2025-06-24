##########################################################
#    Kinova Gen3 Robotic Arm Control                     #
#                                                        #
#    Integration Main  (Project Candydbar)               #
#                                                        #
#    written by: Umut Can Vural                          #
#                                                        #
#    for KISS Project @ Furtwangen University            #
#    (06.2025)                                           #
##########################################################
# Kortex API 2.7.0
# Python 3.11
# Google Protobuf 3.20
# Opencv 4.11
# Mediapipe 0.10.10

# common
import argparse
import cv2
import logging
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import multiprocessing
import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R
import threading
import time

# kinova
from kortex_api.autogen.messages import VisionConfig_pb2
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Base_pb2
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.client_stubs.VisionConfigClientRpc import VisionConfigClient

# local
import intrinsics
import extrinsics
import utilities

# GLOBAL VARIABLES
# KINOVA
TIMEOUT_DURATION = 10  # (seconds)
BASE01_POS_X = 0.20  # (meters)
BASE01_POS_Z = 0.10  # (meters) (dz= 45.89 cm appr.)
# TARGET DEPTH
TARGET_POS_Z = 0.285  # (meters)
# GRIPPER
GRIPPER_POS_01 = 0.00  # gripper full open
GRIPPER_POS_02 = 0.60  # gripper close (width = X.X cm appr.)
# TABLE
TABLE_HEIGHT = 0.0  # Table surface Z in robot base frame (meters)
Z_TOLERANCE = 0.02  # Acceptable deviation from table (meters, e.g., 2 cm)

# CALIBRATION
with open("pixel_to_cm_calibration.pkl", "rb") as f:
    calib = pickle.load(f)
    x_ratio = calib['x_ratio']
    y_ratio = calib['y_ratio']

# Load camera calibration
with open("../01_calibration/calibration_data.pkl", "rb") as f:
    calib = pickle.load(f)
    camera_matrix = calib['cameraMatrix']
    dist_coeff = calib['dist']

# LOGGING
logging.basicConfig(
    level=logging.INFO,
    format='[%(processName)s] %(levelname)s: %(message)s'
)


class RTSPCameraStream:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.cap = None
        self.frame = None
        self.stopped = True
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        if not self.stopped:
            return  # Already running
        self.stopped = False
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.frame = frame
                else:
                    # Try to reconnect if frame not received
                    self.cap.release()
                    time.sleep(1)
                    self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            else:
                time.sleep(1)
                self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if self.cap is not None:
            self.cap.release()

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        if self.thread is not None:
            self.thread.join()
        if self.cap is not None:
            self.cap.release()


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
    print("Rotation matrix (XYZ):\n", R.from_euler('xyz', [rx, ry, rz], degrees=True).as_matrix())

    # Build 4x4 pose
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = EE_rotation_matrix
    camera_pose[:3, 3] = camera_position
    print("Camera position in world frame:", camera_pose)
    return camera_pose


def get_object_offset_cm(frame, center_x, center_y, x_ratio, y_ratio):
    # IMPORTANT #
    # There is amn alighment issue taht we have to chect the signatures of x and y axis
    # (as we did before with aruco tvecs)
    frame_height, frame_width = frame.shape[:2]
    image_center = (frame_width // 2, frame_height // 2)

    # Calculate pixel offsets
    dx_pixels = center_x - image_center[0]
    dy_pixels = center_y - image_center[1]

    print(f"Raw pixel offsets: dx={dx_pixels}, dy={dy_pixels}")

    # Apply sign correction for robotics coordinates:
    # - Keep X sign (right = positive)
    # - Invert Y sign (up = positive)
    dx_cm = dx_pixels / x_ratio
    dy_cm = -dy_pixels / y_ratio  # Invert Y sign

    print(f"Corrected cm offsets: dx={dx_cm:.3f}, dy={dy_cm:.3f}")
    return dx_cm, dy_cm


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
    time.sleep(1)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Retract position reached\n")
    else:
        print("Timeout on action notification wait\n")
    return finished


def go_to_start(base, base_cyclic, pos_x, pos_z):  # cartesian_action
    print("Starting Cartesian action movement to go to Start location ...")
    action = Base_pb2.Action()
    feedback = base_cyclic.RefreshFeedback()

    cartesian_pose = action.reach_pose.target_pose
    cartesian_pose.x = feedback.base.tool_pose_x + pos_x  # (meters)
    cartesian_pose.y = feedback.base.tool_pose_y  # (meters)
    cartesian_pose.z = feedback.base.tool_pose_z + pos_z  # (meters)
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
    time.sleep(0)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Start location reached\n")
    else:
        print("Timeout on action notification wait\n")
    return finished


def gripper_control(base, target_position, step=0.1, delay=0.5):
    print("Starting Gripper control command ...")

    if not 0.0 <= target_position <= 1.0:
        print("Error: Position must be between 0.0 and 1.0")
        return

    # Query the actual current position
    gripper_request = Base_pb2.GripperRequest()
    gripper_request.mode = Base_pb2.GRIPPER_POSITION
    gripper_measure = base.GetMeasuredGripperMovement(gripper_request)
    if len(gripper_measure.finger):
        current_position = gripper_measure.finger[0].value
    else:
        current_position = 0.0  # fallback

    while abs(current_position - target_position) > 0.01:
        if current_position < target_position:
            current_position = min(current_position + step, target_position)
        else:
            current_position = max(current_position - step, target_position)

        gripper_command = Base_pb2.GripperCommand()
        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        finger = gripper_command.gripper.finger.add()
        finger.finger_identifier = 1
        finger.value = current_position

        print(f"Going to position {finger.value:.2f} ...")
        base.SendGripperCommand(gripper_command)
        time.sleep(delay)

    print("Gripper movement is finished\n")


def load_candybar_detector():
    # Load your TFLite model (using tflite_runtime or mediapipe)
    # Load the MediaPipe Object Detector
    base_options = python.BaseOptions(model_asset_path='../03_MediaPipe_AI_Framework/Model'
                                                       '/candybar_objectdetection_model.tflite')
    options = vision.ObjectDetectorOptions(base_options=base_options,
                                           score_threshold=0.7,
                                           max_results=1)  # Number of detected object, should be 1
    object_detector = vision.ObjectDetector.create_from_options(options)
    return object_detector


def move_ee_to_camera_target(base, base_cyclic, extrinsics_read, target_cam, TIMEOUT_DURATION):
    camera_pose = get_camera_pose_matrix(base, extrinsics_read)
    target_base = camera_pose @ target_cam
    EE_rotation_matrix = camera_pose[:3, :3]
    cam_to_ee = np.array([
        extrinsics_read.translation.t_x,
        extrinsics_read.translation.t_y,
        extrinsics_read.translation.t_z
    ])
    print(f"cam to EE: {cam_to_ee}")
    ee_target_position = target_base[:3] - EE_rotation_matrix @ cam_to_ee

    # Print security check before movement
    print_target_coordinates(base, camera_pose, target_base, ee_target_position)

    action = Base_pb2.Action()
    cartesian_pose = action.reach_pose.target_pose
    cartesian_pose.x = float(ee_target_position[0])
    cartesian_pose.y = -float(ee_target_position[1])  # IMPORTANT. Signature Change
    cartesian_pose.z = float(ee_target_position[2])
    feedback = base_cyclic.RefreshFeedback()
    cartesian_pose.theta_x = feedback.base.tool_pose_theta_x
    cartesian_pose.theta_y = feedback.base.tool_pose_theta_y
    cartesian_pose.theta_z = feedback.base.tool_pose_theta_z

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    print("Executing Cartesian action to go to target")
    base.ExecuteAction(action)
    print("Waiting for movement to finish...")
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Target position reached.\n")
    else:
        print("Timeout on action notification wait.\n")
    return finished


def print_target_coordinates(base, camera_pose, target_base, ee_target_position):
    feed = base.GetMeasuredCartesianPose()
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
    print(f"  X = {camera_pose[0, 3]:.3f}")
    print(f"  Y = {camera_pose[1, 3]:.3f}")
    print(f"  Z = {camera_pose[2, 3]:.3f}")
    cam_rot = R.from_matrix(camera_pose[:3, :3]).as_euler('xyz', degrees=True)  # check
    print("Camera Orientation (deg):")
    print(f"  Rx = {cam_rot[2]:.3f}")
    print(f"  Ry = {cam_rot[1]:.3f}")
    print(f"  Rz = {cam_rot[0]:.3f}")
    print("-" * 40)

    print("\n[SECURITY CHECK] Planned Movement:")
    print(
        f"  Target (object) position in robot base frame: X={target_base[0]:.3f}, Y={target_base[1]:.3f}, Z={target_base[2]:.3f}")
    print(
        f"  End-Effector destination: X={ee_target_position[0]:.3f}, Y={ee_target_position[1]:.3f}, Z={ee_target_position[2]:.3f}")
    print("-" * 40)


def process_detect_candybar(shared):
    logging.info(f"Starting object detection process...")
    detector = load_candybar_detector()
    while not shared.get('exit', False):
        frame = shared.get('frame', None)
        if frame is not None:
            # Convert the frame to RGB as MediaPipe expects RGB images
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detection_result = detector.detect(mp_image)
            found = False
            for detection in detection_result.detections:
                bbox = detection.bounding_box
                # Draw bounding box
                x1, y1 = int(bbox.origin_x), int(bbox.origin_y)
                x2, y2 = x1 + int(bbox.width), y1 + int(bbox.height)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Calculate center coordinates
                center_x = x1 + (x2 - x1) // 2
                center_y = y1 + (y2 - y1) // 2
                shared['detection_data'] = (center_x, center_y)
                found = True
            if found:
                shared['new_detection'] = True  # Signal new detection
            else:
                shared['new_detection'] = False  # No detection
            shared['detected_frame'] = frame
        time.sleep(0.01)


def process_display_detection(shared):
    logging.info(f"Starting display process...")
    # Create a white placeholder image (e.g., 720p)
    placeholder = np.ones((720, 1280, 3), dtype=np.uint8) * 255  # White image

    window_opened = False
    while True:
        frame = shared.get('detected_frame', None)
        if frame is not None:
            cv2.imshow("RTSP Stream", frame)
            window_opened = True
        else:
            # Only show the placeholder until the first real frame arrives
            if not window_opened:
                cv2.imshow("RTSP Stream", placeholder)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break
        time.sleep(0.03)
    cv2.destroyAllWindows()
    logging.info("Display process ended.")


def process_stream(rtsp_url, shared):
    logging.info(f"Starting stream process for {rtsp_url}...")
    while not shared.get('exit', False):
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            print("Failed to open stream, retrying...")
            time.sleep(2)
            continue
        while not shared.get('exit', False):
            ret, frame = cap.read()
            if not ret:
                print("Frame not received, reconnecting...")
                break  # Exit inner loop to reconnect
            shared['frame'] = frame
        cap.release()
        time.sleep(2)  # Wait before reconnecting
    logging.info("Stream process ended.")


def main():
    parser = argparse.ArgumentParser()
    args = utilities.parseConnectionArguments(parser)

    with utilities.DeviceConnection.createTcpConnection(args) as router:
        device_manager = DeviceManagerClient(router)
        vision_config = VisionConfigClient(router)
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)
        # feed = base.GetMeasuredCartesianPose()
        # detector = load_candybar_detector()

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

        manager = multiprocessing.Manager()
        shared = manager.dict()
        shared['frame'] = None  # RTSP frame
        shared['detected_frame'] = None  # Object Detection Frame
        shared['detection_data'] = None  # To store detection results (e.g., center coordinates)
        shared['new_detection'] = False  # Flag for new detection
        shared['exit'] = False

        stream_proc = multiprocessing.Process(target=process_stream, args=("rtsp://192.168.1.10/color", shared))
        display_proc = multiprocessing.Process(target=process_display_detection, args=(shared,))
        detect_proc = multiprocessing.Process(target=process_detect_candybar, args=(shared,))

        stream_proc.start()  # start RTSP stream
        time.sleep(0.5)
        display_proc.start()
        time.sleep(0.5)
        logging.info("Do not close the display until prompted to do so.")
        detect_proc.start()  # start object detection
        time.sleep(3)

        go_to_retract(base, base_cyclic)  # go to default base
        time.sleep(1)
        gripper_control(base, 0.3, 0.1, 0.1)
        time.sleep(0.5)

        go_to_start(base, base_cyclic, BASE01_POS_X, BASE01_POS_Z)  # go to starting base
        time.sleep(10)

        try:
            detection_timeout = 10  # seconds
            start_time = time.time()
            detected = False

            while time.time() - start_time < detection_timeout:
                if shared['new_detection']:
                    center_x, center_y = shared['detection_data']
                    print(f"center coordinates X: {center_x}, Y: {center_y}")
                    shared['new_detection'] = False
                    detected = True

                    # Convert pixel coordinates to real-world offsets
                    frame = shared['detected_frame']
                    dx_cm, dy_cm = get_object_offset_cm(frame, center_x, center_y, x_ratio, y_ratio)
                    # If ratios are in pixels/meter:
                    TARGET_POS_X = dx_cm / 100  # Convert cm to meters
                    TARGET_POS_Y = dy_cm / 100
                    # Build homogeneous target_cam vector
                    target_cam = np.array([TARGET_POS_X, TARGET_POS_Y, TARGET_POS_Z, 1])
                    print(f"target vector : {target_cam}")
                    break
                time.sleep(0.05)

            if detected and target_cam is not None:

                # Move robot to detected location
                print("OBJECT IS DETECTED")
                #gripper_control(base, 0.7, 0.1, 0.1)  # detection warning
                gripper_control(base, 0.3, 0.1, 0.1)
                time.sleep(5)
                move_ee_to_camera_target(base, base_cyclic, extrinsics_read, target_cam, TIMEOUT_DURATION)
                time.sleep(10)
                gripper_control(base, 0.7, 0.1, 0.1)
            else:
                print("No detection in 10 seconds, returning to retract position.")
                go_to_retract(base, base_cyclic)

        finally:
            # This ensures cleanup happens even if an exception occurs
            shared['exit'] = True
            detect_proc.join()
            go_to_retract(base, base_cyclic)
            time.sleep(1)
            gripper_control(base, 0.7, 0.1, 0.1)
            time.sleep(0.5)

            logging.info("Session ended. Press ESC to close Display")
            display_proc.join()  # Wait for display window to close (ESC)
            stream_proc.join()

            # Prompt the user to press 'q' to exit
            while True:
                user_input = input("Press 'q' and Enter to quit the program: ")
                if user_input.strip().lower() == 'q':
                    print("Program terminated.")
                    break


if __name__ == "__main__":
    main()
