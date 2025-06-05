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
#                                                        #
##########################################################
#    specs:                                              #
#    Python 3.9                                          #
#    Kinova Kortex 2.6.0                                 #
#    Gen3 firmware Bundle 2.5.2-r.2                      #
##########################################################

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

        if vision_device_id != 0:
            print(f"Vision Device ID: {vision_device_id}")
            # print(type(intrinsics))
            # print(dir(intrinsics))
            # print("Sensor: {0}".format(intrinsics.get_sensor_name(1)))
            # print("Sensor: {0}".format(intrinsics.get_sensor_supported_options(1)))

            intrinsics.disable_autofocus(vision_config, vision_device_id)  # disable autofocus
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
        write_path = os.getcwd() + "/resources/calibration_screen"

        # Initialize the time tracker
        last_capture_time = time.time()

        while cv2.waitKey(1) != 27:  # press ESC to exit
            has_frame, frame = source.read()
            frame = cv2.resize(frame, (frame_width, frame_height), 3)
            # Note: Be careful with frame dimensions. These have to match with the main pick&place program.

            if has_frame:
                # print("Frame Shape", frame.shape)
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                cv2.imshow("Output Video", frame)
                # cv2.imshow("Grayscale Video", frame_gray)
                if time.time() - last_capture_time > 3:  # More than 5 seconds have passed
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
