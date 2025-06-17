#########################################################################################
#    Kinova Gen3 Robotic Arm                                                            #
#                                                                                       #
#    Pose Estimation with aruco markers                                                 #
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
# kinova
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.client_stubs.DeviceConfigClientRpc import DeviceConfigClient
from kortex_api.autogen.client_stubs.VisionConfigClientRpc import VisionConfigClient
from kortex_api.autogen.messages import VisionConfig_pb2
# local
import intrinsics
import utilities


# Load camera matrix and distortion coefficients from pickle file
with open("calibration_data.pkl", "rb") as f:
    data = pickle.load(f)
    cameraMatrix = data['cameraMatrix']
    dist = data['dist']
print(cameraMatrix)
print(dist)


def find_marker_by_id(corners, ids, target_id):
    if ids is not None:
        for i, id in enumerate(ids):
            if id[0] == target_id:
                return corners[i][0]
    return None


def main():
    # Specify the IDs of the two markers
    MARKER_ID_1 = 10  # first marker ID
    MARKER_SIZE_CM = 2.05  # in cm

    parser = argparse.ArgumentParser()
    args = utilities.parseConnectionArguments(parser)

    with utilities.DeviceConnection.createTcpConnection(args) as router:
        device_manager = DeviceManagerClient(router)
        vision_config = VisionConfigClient(router)
        vision_device_id = intrinsics.get_vision_device_id(device_manager)

        if vision_device_id != 0:
            print(f"Vision Device ID: {vision_device_id}")

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

            bilgi = intrinsics.get_active_routed_vision_intrinsics(vision_config, vision_device_id)
            intrinsics.routed_vision_set_manual_focus_medium_distance(vision_config, vision_device_id)

            resolution_string = intrinsics.resolution_to_string(bilgi.resolution)
            frame_width, frame_height = map(int, resolution_string.split('x'))
            print(f"FRAME_WIDTH: {frame_width}")
            print(f"FRAME_HEIGHT: {frame_height}")

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

                    # Estimate pose of each marker
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
                        object_px = (640, 360)

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

                        # Draw circle at the center
                        center = tuple(map(int, corners.mean(axis=0)))
                        cv2.circle(frame, center, 5, (0, 255, 0), -1)

                height, width = frame.shape[:2]
                center_x, center_y = int(width / 2), int(height / 2)
                cv2.circle(frame, (center_x, center_y), radius=3, color=(0, 0, 255), thickness=-1)
                cv2.circle(frame, (center_x, center_y), radius=12, color=(0, 0, 255), thickness=2)

                cv2.imshow(win_name, frame)

            except Exception as e:
                print(f"Error: {e}")

        source.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
