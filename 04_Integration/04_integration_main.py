##########################################################
#    Kinova Gen3 Robotic Arm                             #
#                                                        #
#    Integration Main                                    #
#                                                        #
#                                                        #
#    written by: U. Vural                                #
#                                                        #
#                                                        #
#    for KISS Project at Furtwangen University           #
#    (06.2025)                                           #
##########################################################
# Kortex API 2.7.0
# Python 3.11
# Google Protobuf 3.20
# Opencv 4.11
# Mediapipe 0.10.10

# common
import cv2
import numpy as np
import argparse
import multiprocessing
import pickle
import threading
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
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

# GLOBAL VARIABLES
# KINOVA
TIMEOUT_DURATION = 20  # (seconds)
BASE01_POS_X = 0.20  # (meters)
BASE01_POS_Z = 0.10  # (meters) (dz= 45.89 cm appr.)
# TARGET
TARGET_POS_Z = 0.30  # (meters)
TARGET_POS_X = None
TARGET_POS_Y = None
# GRIPPER
GRIPPER_POS_01 = 0.00  # gripper full open
GRIPPER_POS_02 = 0.60  # gripper close (width = X.X cm appr.)
# TABLE
TABLE_HEIGHT = 0.0  # Table surface Z in robot base frame (meters)
Z_TOLERANCE = 0.02  # Acceptable deviation from table (meters, e.g., 2 cm)

with open("pixel_to_cm_calibration.pkl", "rb") as f:
    calib = pickle.load(f)
    x_ratio = calib['x_ratio']
    y_ratio = calib['y_ratio']

# Load camera calibration
with open("../01_calibration/calibration_data.pkl", "rb") as f:
    calib = pickle.load(f)
    camera_matrix = calib['cameraMatrix']
    dist_coeff = calib['dist']


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


def detect_candybar(frame, detector):
    # Convert the frame to RGB as MediaPipe expects RGB images
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    detection_result = detector.detect(mp_image)
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        # Draw bounding box
        x1, y1 = int(bbox.origin_x), int(bbox.origin_y)
        x2, y2 = x1 + int(bbox.width), y1 + int(bbox.height)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw label if available
        if detection.categories:
            label = detection.categories[0].category_name
            score = detection.categories[0].score
            cv2.putText(frame, f"{label} {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


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


def get_candybar_center(frame, detector):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    detection_result = detector.detect(mp_image)
    centers = []
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        x1, y1 = int(bbox.origin_x), int(bbox.origin_y)
        x2, y2 = x1 + int(bbox.width), y1 + int(bbox.height)
        # Calculate center
        cx = x1 + int(bbox.width // 2)
        cy = y1 + int(bbox.height // 2)
        centers.append((cx, cy))
    return centers  # List of centers for all detected objects


def get_object_offset_cm(frame, detector, x_ratio, y_ratio):
    centers = get_candybar_center(frame, detector)
    frame_height, frame_width = frame.shape[:2]
    image_center = (frame_width // 2, frame_height // 2)
    offsets_cm = []
    for (cx, cy) in centers:
        dx_pixels = cx - image_center[0]
        dy_pixels = cy - image_center[1]
        dx_cm = dx_pixels / x_ratio
        dy_cm = dy_pixels / y_ratio
        offsets_cm.append((dx_cm, dy_cm))
    return offsets_cm  # List of (dx_cm, dy_cm) for each detected object


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
        print("Pickup location reached\n")
    else:
        print("Timeout on action notification wait\n")
    return finished


def go_to_target(base, base_cyclic, pos_x, pos_y, pos_z):
    print("Starting Cartesian action movement to go to Pickup location ...")
    action = Base_pb2.Action()
    feedback = base_cyclic.RefreshFeedback()

    cartesian_pose = action.reach_pose.target_pose
    cartesian_pose.x = feedback.base.tool_pose_x + pos_x  # (meters)
    cartesian_pose.y = feedback.base.tool_pose_y + pos_y  # (meters)
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
        print("Pickup location reached\n")
    else:
        print("Timeout on action notification wait\n")
    return finished


def gripper_control(base, position):
    print("Starting Gripper control command ...")
    gripper_command = Base_pb2.GripperCommand()
    finger = gripper_command.gripper.finger.add()

    gripper_command.mode = Base_pb2.GRIPPER_POSITION
    position = position
    finger.finger_identifier = 1
    while position < 1.0:
        finger.value = position
        print("Going to position {:0.2f} ...".format(finger.value))
        base.SendGripperCommand(gripper_command)
        time.sleep(1)
        break
    print("Gripper movement is finished\n")


def load_candybar_detector():
    # Load your TFLite model (using tflite_runtime or mediapipe)
    # Load the MediaPipe Object Detector
    base_options = python.BaseOptions(model_asset_path='../03_MediaPipe_AI_Framework/Model'
                                                       '/candybar_objectdetection_model.tflite')
    options = vision.ObjectDetectorOptions(base_options=base_options,
                                           score_threshold=0.8,
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
    ee_target_position = target_base[:3] - EE_rotation_matrix @ cam_to_ee

    # Print security check before movement
    print_target_coordinates(target_base, ee_target_position)

    # action = Base_pb2.Action()
    cartesian_pose = action.reach_pose.target_pose
    cartesian_pose.x = float(ee_target_position[0])
    cartesian_pose.y = float(ee_target_position[1])
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


def print_target_coordinates(target_base, ee_target_position):
    print("\n[SECURITY CHECK] Planned Movement:")
    print(
        f"  Target (object) position in robot base frame: X={target_base[0]:.3f}, Y={target_base[1]:.3f}, Z={target_base[2]:.3f}")
    print(
        f"  End-Effector destination: X={ee_target_position[0]:.3f}, Y={ee_target_position[1]:.3f}, Z={ee_target_position[2]:.3f}")
    print("-" * 40)


def process_display(shared):
    # Create a white placeholder image (e.g., 720p)
    placeholder = np.ones((720, 1280, 3), dtype=np.uint8) * 255  # White image

    window_opened = False
    while True:
        frame = shared.get('frame')
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


def process_stream(rtsp_url, shared):
    stream = RTSPCameraStream(rtsp_url)
    stream.start()
    try:
        while True:
            frame = stream.read()
            if frame is not None:
                shared['frame'] = frame
    finally:
        stream.stop()


def main():
    parser = argparse.ArgumentParser()
    args = utilities.parseConnectionArguments(parser)

    with utilities.DeviceConnection.createTcpConnection(args) as router:
        device_manager = DeviceManagerClient(router)
        vision_config = VisionConfigClient(router)
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)
        feed = base.GetMeasuredCartesianPose()
        detector = load_candybar_detector()

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
        shared['frame'] = None

        stream_proc = multiprocessing.Process(target=process_stream, args=("rtsp://192.168.1.10/color", shared))
        display_proc = multiprocessing.Process(target=process_display, args=(shared,))
        robot_retract = threading.Thread(target=go_to_retract, args=(base, base_cyclic))
        robot_retract_end =  threading.Thread(target=go_to_retract, args=(base, base_cyclic))
        robot_start_base = threading.Thread(target=go_to_start,
                                                    args=(base, base_cyclic, BASE01_POS_X, BASE01_POS_Z))

        stream_proc.start()
        time.sleep(10)
        display_proc.start()
        time.sleep(10)

        # ... do other things, start/stop other processes ...
        robot_retract.start()
        # Other code can run here in parallel
        robot_retract.join()  # Wait for the robot to finish (optional)
        time.sleep(0.5)
        robot_start_base.start()
        robot_start_base.join()
        time.sleep(0.5)
        robot_retract_end.start()
        robot_retract_end.join()

        display_proc.join()  # Wait for display window to close (ESC)
        stream_proc.terminate()


if __name__ == "__main__":
    main()
