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
#

import cv2
import os
import time

# Import the utilities helper module of Kinova
import argparse
import utilities
import intrinsics
from kortex_api.autogen.client_stubs.VisionConfigClientRpc import VisionConfigClient
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.client_stubs.DeviceConfigClientRpc import DeviceConfigClient
from kortex_api.autogen.messages import VisionConfig_pb2


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    args = utilities.parseConnectionArguments(parser)

    # Create connection to the device and get the router
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        device_manager = DeviceManagerClient(router)
        device_config = DeviceConfigClient(router)
        vision_config = VisionConfigClient(router)
        vision_device_id = intrinsics.get_vision_device_id(device_manager)

        # Set the camera to stream at 1280x720 (add this after getting vision_device_id)
        #print(dir(VisionConfig_pb2))
        #print(dir(vision_config))

        if vision_device_id != 0:
            print(f"Vision Device ID: {vision_device_id}")
            #print(type(intrinsics))
            #print(dir(intrinsics))
            #print("Sensor: {0}".format(intrinsics.get_sensor_name(1)))
            #print("Sensor: {0}".format(intrinsics.get_sensor_supported_options(1)))

            # sensor settings
            settings = VisionConfig_pb2.SensorSettings()
            settings.sensor = VisionConfig_pb2.SENSOR_COLOR
            settings.resolution = VisionConfig_pb2.RESOLUTION_1280x720
            settings.frame_rate = VisionConfig_pb2.FRAMERATE_15_FPS
            settings.bit_rate = VisionConfig_pb2.BITRATE_10_MBPS  # Try higher if error
            print("protobuf settings:", settings)

            # Set the sensor settings
            vision_config.SetSensorSettings(settings, vision_device_id)

            # Autofocus disable
            focus_action_msg = VisionConfig_pb2.SensorFocusAction()
            focus_action_msg.sensor = VisionConfig_pb2.SENSOR_COLOR
            focus_action_msg.focus_action = VisionConfig_pb2.FOCUSACTION_DISABLE_FOCUS
            vision_config.DoSensorFocusAction(focus_action_msg, vision_device_id)

            # CHECK instrinsics.py. Use default values if necessary from Kinova Github.
            intrinsics.routed_vision_set_intrinsics(vision_config, vision_device_id)
            #intrinsics.disable_autofocus(vision_config, vision_device_id)  # disable autofocus
            bilgi = intrinsics.get_active_routed_vision_intrinsics(vision_config, vision_device_id)  # Get Intrinsics
            #intrinsics.routed_vision_set_manual_focus_medium_distance(vision_config, vision_device_id) # 35 cm

            resolution_string = intrinsics.resolution_to_string(bilgi.resolution)
            frame_width, frame_height = map(int, resolution_string.split('x'))
            print(f"FRAME_WIDTH: {frame_width}")
            print(f"FRAME_HEIGHT: {frame_height}")
            # intrinsics.routed_vision_set_intrinsics(vision_config, vision_device_id)
            # intrinsics.routed_vision_confirm_saved_sensor_options_values(vision_config, device_config,
            # vision_device_id)

        # Camera Streaming via Ethernet
        source = cv2.VideoCapture("rtsp://192.168.1.10/color")

        count = 0
        waiting_duration_secs = 60
        write_path = os.getcwd() + "/resources/calibration_screenshot"

        # Initialize the time tracker
        last_capture_time = time.time()

        while cv2.waitKey(1) != 27:  # press ESC to exit
            has_frame, frame = source.read()
            frame = cv2.resize(frame, (frame_width, frame_height), 3)
            # Note: Be careful with frame dimensions. These have to match with the main pick&place program.

            if has_frame:
                #print("Frame Shape", frame.shape)
                #frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                cv2.imshow("Output Video", frame)
                # cv2.imshow("Grayscale Video", frame_gray)
                if time.time() - last_capture_time > 5:  # More than 5 seconds have passed
                    cv2.imwrite(write_path + str(count) + ".jpg", frame)
                    cv2.rectangle(frame, (0, frame_width//3), (frame_width, frame_height//2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, "Scan Saved", (frame_width//2-120, frame_height//2+60),
                                cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 3)
                    cv2.imshow("Output Video", frame)
                    print("Scan Saved. Frame Shape:", frame.shape)
                    cv2.waitKey(waiting_duration_secs)  # Waiting duration between each screenshot.
                    # Take many screenshots with different position & angle combinations for better precision (100+)
                    count += 1
                    last_capture_time = time.time()  # Reset the last capture time
            else:
                print("No frame")
                break

        source.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
