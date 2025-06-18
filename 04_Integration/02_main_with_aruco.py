##########################################################
#    Kinova Gen3 Robotic Arm                             #
#                                                        #
#    Taking timed screenshots from the camera            #
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
TIMEOUT_DURATION = 10  # (seconds)
BASE01_POS_X2 = 0.10  # (meters)
# Object Position (pixel)
OBJECT_X = 640
OBJECT_Y = 360
# ARUCO
MARKER_ID = 10  # marker ID
MARKER_SIZE_CM = 2.05  # (centimeters)

# Fixed pixel and depth
u, v = 640, 360
z = 0.35  # 35 cm


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


def get_candybar_pose_in_world(frame, centers, marker_id, marker_size, camera_matrix, dist_coeff, camera_pose):
    # get BB center
    x_center = centers[0]
    y_center = centers[1]

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = aruco_detector.detectMarkers(frame)
    if ids is None or marker_id not in ids:
        print(f"Marker {marker_id} not found in this frame.")
        return None, None

    idx = [i for i, id in enumerate(ids) if id[0] == marker_id][0]
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        [corners[idx]], marker_size, camera_matrix, dist_coeff
    )
    marker_tvec = tvecs[0][0]  # (x, y, z) in camera frame

    # 3. Apply sign flip (OpenCV to world alignment)
    z = marker_tvec[2]

    # 4. Back-project (x_center, y_center, z) to camera coordinates
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    x_cam = (x_center - cx) * z / fx
    y_cam = (y_center - cy) * z / fy

    # Apply sign flip to x, y to match your world convention
    x_cam_flipped = -x_cam
    y_cam_flipped = -y_cam

    # Compose 3D point in camera frame (homogeneous)
    candybar_pos_camera = np.array([x_cam_flipped, y_cam_flipped, z, 1.0])

    # 5. Transform to world coordinates using camera_pose
    candybar_pos_world = camera_pose @ candybar_pos_camera
    print(f"Candy bar position in world frame: {candybar_pos_world[:3]}")

    return candybar_pos_world[:3], candybar_pos_camera[:3]


def detect_candybar(frame, detector):
    """
    Detects candy bars in the frame, draws bounding boxes, and returns a list of center points.
    Each center point is a tuple: (x_center, y_center)
    """
    centers = []
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    detection_result = detector.detect(mp_image)
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        x1, y1 = int(bbox.origin_x), int(bbox.origin_y)
        x2, y2 = x1 + int(bbox.width), y1 + int(bbox.height)
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw label if available
        if detection.categories:
            label = detection.categories[0].category_name
            score = detection.categories[0].score
            cv2.putText(frame, f"{label} {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # Calculate center point
        x_center = x1 + (x2 - x1) // 2
        y_center = y1 + (y2 - y1) // 2
        centers.append((x_center, y_center))
    return centers


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


def get_pixel_cm_ratio(frame, marker_id, marker_size_cm):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is not None:
        for i, id in enumerate(ids):
            if id[0] == marker_id:
                pts = corners[i][0]
                # Compute average side length in pixels
                side_lengths = [
                    np.linalg.norm(pts[0] - pts[1]),
                    np.linalg.norm(pts[1] - pts[2]),
                    np.linalg.norm(pts[2] - pts[3]),
                    np.linalg.norm(pts[3] - pts[0])
                ]
                avg_side_length_px = np.mean(side_lengths)
                pixel_cm_ratio = avg_side_length_px / marker_size_cm
                return pixel_cm_ratio
    return None


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
    # feedback = base_cyclic.RefreshFeedback()
    # current_pose = [
    #    feedback.base.tool_pose_x,
    #    feedback.base.tool_pose_y,
    #    feedback.base.tool_pose_z,
    #    feedback.base.tool_pose_theta_x,
    #    feedback.base.tool_pose_theta_y,
    #    feedback.base.tool_pose_theta_z
    # ]

    if finished:
        print("Retract position reached\n")
        # Print the pose information
        # print(f"  Position: X={current_pose[0]:.3f}, Y={current_pose[1]:.3f}, Z={current_pose[2]:.3f}")
        # print(f"  Orientation: Rx={current_pose[3]:.3f}, Ry={current_pose[4]:.3f}, Rz={current_pose[5]:.3f}")
        # print("--------------------")
    else:
        print("Timeout on action notification wait\n")
    return finished


def go_to_start(base, base_cyclic):
    pass


def go_to_pick(base, base_cyclic, world_pos):
    pass


def go_to_drop(base, base_cyclic):
    pass


# Load detection model
def load_candybar_detector():
    # Load your TFLite model (using tflite_runtime or mediapipe)
    # Load the MediaPipe Object Detector
    base_options = python.BaseOptions(model_asset_path='../03_MediaPipe_AI_Framework/Model'
                                                       '/candybar_objectdetection_model.tflite')
    options = vision.ObjectDetectorOptions(base_options=base_options,
                                           score_threshold=0.6,
                                           max_results=1)
    object_detector = vision.ObjectDetector.create_from_options(options)
    return object_detector


def pixel_to_world(u, v, z, camera_matrix, camera_pose):
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    point_cam = np.array([x_cam, y_cam, z, 1.0])
    point_world = camera_pose @ point_cam
    return point_world[:3]


def pixel_to_cm_xy(p1, p2, pixel_cm_ratio):
    # Calculate x and y distances in pixels
    x_dist_px = abs(p1[0] - p2[0])
    y_dist_px = abs(p1[1] - p2[1])
    # Convert to centimeters
    x_dist_cm = x_dist_px / pixel_cm_ratio
    y_dist_cm = y_dist_px / pixel_cm_ratio
    return x_dist_cm, y_dist_cm


def print_all_poses(feed, camera_pose, target_pos, target_rot=None):
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
    # Extract Euler angles from rotation matrix (assuming 'zyx' order, degrees)
    cam_rot = R.from_matrix(camera_pose[:3, :3]).as_euler('zyx', degrees=True)  # check
    print("Camera Orientation (deg):")
    print(f"  Rx = {cam_rot[2]:.3f}")
    print(f"  Ry = {cam_rot[1]:.3f}")
    print(f"  Rz = {cam_rot[0]:.3f}")
    print("-" * 40)

    # Target
    print("Target Position (m):")
    print(f"  X = {target_pos[0]:.3f}")
    print(f"  Y = {target_pos[1]:.3f}")
    print(f"  Z = {target_pos[2]:.3f}")
    if target_rot is not None:
        print("Target Orientation (deg):")
        print(f"  Rx = {target_rot[0]:.3f}")
        print(f"  Ry = {target_rot[1]:.3f}")
        print(f"  Rz = {target_rot[2]:.3f}")
    print("-" * 40)


def robust_detect_candybar(frame, detector, retries=5, delay=0.1):
    for attempt in range(retries):
        if frame is None:
            time.sleep(delay)
            continue
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detection_result = detector.detect(mp_image)
        if detection_result.detections:
            return detection_result.detections[0]
        time.sleep(delay)
    print("Candy bar not detected after retries on this frame.")
    return None


# Load camera calibration
with open("../01_calibration/calibration_data.pkl", "rb") as f:
    calib = pickle.load(f)
    camera_matrix = calib['cameraMatrix']
    dist_coeff = calib['dist']


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

        stream = RTSPCameraStream("rtsp://192.168.1.10/color")
        stream.start()

        try:
            while True:
                go_to_retract(base, base_cyclic)
                time.sleep(2)
                go_to_start(base, base_cyclic)
                time.sleep(2)

                frame = stream.read()
                if frame is None:
                    print("No frame received from camera.")
                    time.sleep(1)  # Avoid tight loop
                    continue

                centers = detect_candybar(frame, detector)

                #
                cv2.imshow("RTSP Stream", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                    break
                aruco_world_pos, aruco_pose_matrix = get_marker_pose_in_world(
                    frame, MARKER_ID, MARKER_SIZE_CM, camera_matrix, dist_coeff, camera_pose
                )
                print(f"ArUco marker world position: {aruco_world_pos}")

                ptw_world = pixel_to_world(u, v, z, camera_matrix, camera_pose)
                print(f"Pixel-to-world (center, z=35cm): {ptw_world}")

                # --- Compare results ---
                if aruco_world_pos is not None:
                    diff = np.linalg.norm(ptw_world - aruco_world_pos)
                    print(f"Difference between pixel-to-world and ArUco: {diff:.4f} meters")


                # If you want to use detection results for robot actions:
                world_pos = None
                camera_pose = None
                if centers:
                    x_center, y_center = centers[0]  # Use first detection
                    camera_pose = get_camera_pose_matrix(base, extrinsics_read)
                    world_pos, cam_pos = get_candybar_pose_in_world(
                        frame, x_center, y_center, MARKER_ID, MARKER_SIZE_CM, camera_matrix, dist_coeff, camera_pose
                    )
                    # Now world_pos contains the 3D world coordinates for the detected candy bar
                else:
                    print("No candy bar detected in this frame.")

                if camera_pose is not None and world_pos is not None:
                    print_all_poses(feed, camera_pose, world_pos, target_rot=None)
                else:
                    print("Skipping pose print due to missing data.")

                if world_pos is not None:
                    # Stop stream before robot movement if needed for safety/resource reasons
                    stream.stop()
                    print("RTSP stream stopped for pick-and-place operation.")
                    time.sleep(2)

                    go_to_pick(base, base_cyclic, world_pos)
                    time.sleep(2)
                    go_to_drop(base, base_cyclic)
                    time.sleep(10)

                    # Restart stream after movement if needed
                    stream.start()
                    print("RTSP stream restarted for next cycle.")

        except KeyboardInterrupt:
            pass
        finally:
            stream.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
