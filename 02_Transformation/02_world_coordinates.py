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
from scipy.spatial.transform import Rotation as R

# kinova
from kortex_api.autogen.messages import VisionConfig_pb2
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Base_pb2
from kortex_api.autogen.client_stubs.VisionConfigClientRpc import VisionConfigClient
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient

# local
import utilities
import intrinsics
import extrinsics

TIMEOUT_DURATION = 10  # (seconds)
BASE01_POS_X2 = 0.10  # (meters)

# Load camera matrix and distortion coefficients from pickle file
with open("../01_calibration/calibration_data.pkl", "rb") as f:
    data = pickle.load(f)
    cameraMatrix = data['cameraMatrix']
    dist = data['dist']
print(f"cameraMatrix is: {cameraMatrix}")
print(f"dist Coeefs are: {dist}")


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
    rotation = R.from_euler('xyz', [rx, ry, rz], degrees=True)
    gripper_rotation_matrix = rotation.as_matrix()

    # Extract extrinsic translation vector (camera position relative to gripper, in gripper frame)
    extrinsic_translation = np.array([
        extrinsics_read.translation.t_x,
        extrinsics_read.translation.t_y,
        extrinsics_read.translation.t_z
    ])  # meters

    # Transform extrinsic translation from gripper frame to world frame
    camera_offset_world = gripper_rotation_matrix @ extrinsic_translation
    camera_position = gripper_position + camera_offset_world

    print("Camera position in world frame:", camera_position)
    return camera_position


def go_to_retract(base, base_cyclic, extrinsics_read):
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

    # Get gripper pose
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
        extrinsics.print_extrinsic_parameters(extrinsics_read)
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
            extrinsics_read = extrinsics.set_custom_extrinsics(vision_config, vision_device_id)

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

        # Robotic arm control - added try catch
        try:
            go_to_retract(base, base_cyclic, extrinsics_read)
            time.sleep(1)  # Wait for 1 seconds
            get_camera_position_from_feedback(base_cyclic, extrinsics_read)
            time.sleep(1)  # Wait for 1 seconds

            # go_to_start_position(base, base_cyclic, BASE01_POS_X2, 0, 0)
            # time.sleep(1)

        except Exception as e:
            print(f"An error occurred during robot movement: {e}")

        finally:
            pass


if __name__ == "__main__":
    main()
