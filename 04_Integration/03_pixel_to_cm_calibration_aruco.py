##########################################################
#    Kinova Gen3 Robotic Arm                             #
#                                                        #
#    Pixel_to_cm Calibration                             #
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
BASE01_POS_X = 0.20  # (meters)
BASE01_POS_Z = -0.10  # (meters)

# Object Position (pixel)
OBJECT_X = 640
OBJECT_Y = 360
# ARUCO
MARKER_ID = 10  # marker ID
MARKER_SIZE_CM = 4.45  # (centimeters)

# Load camera matrix and distortion coefficients from pickle file
with open("../01_calibration/calibration_data.pkl", "rb") as f:
    data = pickle.load(f)
    camera_matrix = data['cameraMatrix']
    dist_coeff = data['dist']


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


def get_xy_pixel_cm_ratio(frame, marker_id, marker_size, camera_matrix, dist_coeff,
                          save_path="pixel_to_cm_calibration.pkl"):
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

        # Use the corners of the detected marker
        marker_corners = corners[idx][0]  # (4,2) array

        # Calculate side lengths in pixels
        top = np.linalg.norm(marker_corners[0] - marker_corners[1])
        right = np.linalg.norm(marker_corners[1] - marker_corners[2])
        bottom = np.linalg.norm(marker_corners[2] - marker_corners[3])
        left = np.linalg.norm(marker_corners[3] - marker_corners[0])

        # Average x (horizontal) and y (vertical) side lengths
        x_px = (top + bottom) / 2
        y_px = (left + right) / 2
        x_ratio = x_px / marker_size
        y_ratio = y_px / marker_size

        # Save to pickle file
        calib_data = {'x_ratio': x_ratio, 'y_ratio': y_ratio}
        with open(save_path, "wb") as f:
            pickle.dump(calib_data, f)
        print(f"Saved calibration: x_ratio={x_ratio:.4f}, y_ratio={y_ratio:.4f} to {save_path}")

        return x_ratio, y_ratio

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


def go_to_start(base, base_cyclic, pos_x, pos_z):  # cartesian_action
    print("Starting Cartesian action movement to go to Pickup location ...")
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

        stream = RTSPCameraStream("rtsp://192.168.1.10/color")
        stream.start()
        try:
            display_enabled = True
            while True:
                go_to_retract(base, base_cyclic)
                time.sleep(3)
                frame = stream.read()
                if frame is not None:

                    if display_enabled:
                        cv2.imshow("RTSP Stream", frame)
                        time.sleep(3)
                        go_to_start(base, base_cyclic, BASE01_POS_X, BASE01_POS_Z)  # dz = 45.89 cm
                        time.sleep(3)
                        fresh_frame = stream.read()
                        get_xy_pixel_cm_ratio(fresh_frame, MARKER_ID, MARKER_SIZE_CM, camera_matrix, dist_coeff,
                                              save_path="pixel_to_cm_calibration.pkl")

                        if cv2.waitKey(1) & 0xFF == 27:
                            display_enabled = False
                            cv2.destroyWindow("RTSP Stream")
        except KeyboardInterrupt:
            pass
        finally:
            stream.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
