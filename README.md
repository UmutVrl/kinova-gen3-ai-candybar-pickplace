
# Project Candybar

This project is part of the KISS (Künstliche Intelligenz Services & Systeme) initiative, a collaborative effort led by Hochschule Furtwangen and Hochschule für Musik Trossingen, and funded by the German Federal Ministry of Education and Research. KISS aims to advance the application of artificial intelligence (AI) across diverse domains—including cognitive robotics, smart production, autonomous systems, health technologies, and music—by developing both foundational modules and practical demonstrators.
Project Candybar showcases the dynamic integration of AI-based object detection with robotic manipulation. Using the Kinova Gen3 robotic arm equipped with a vision module, this project demonstrates a complete pipeline:
  Camera calibration for precise vision,
  Real-time object detection using a custom-trained MediaPipe model,
  3D pose estimation and coordinate transformation,
  Closed-loop pick-and-place of a candy bar.
Unlike traditional robotics workflows that rely on static, pre-programmed positions, this project leverages AI to enable dynamic, vision-guided manipulation. The object detection model was trained with custom-labeled data (using tools like Label Studio) and deployed in a way that allows the robot to adapt to new object locations and orientations in real time. All code, calibration data, and models are organized to facilitate reproducibility and adaptation for other AI-driven robotics tasks.

[VIDEO HERE]


## 01_Calibration
### Overview
This folder contains all scripts and resources required for calibrating the Kinova Gen3 robotic arm’s vision system. Accurate camera calibration is essential for reliable object detection, pose estimation, and precise robotic manipulation. The calibration process ensures that the camera’s intrinsic parameters (such as focal length, principal point, and distortion coefficients) are accurately determined. This is critical for tasks like object detection, 3D localization, and robot-to-camera transformations. Workflow includes:
Taking timed screenshots from the robot’s camera for calibration data collection.
Performing chessboard-based camera calibration.
Evaluating calibration quality and undistorting images.
Testing calibration using ArUco markers for spatial validation.
### Folder Structure & Scripts
01_screenshot_taker_with_timer.py	Captures timed screenshots from the Kinova camera stream to collect calibration images.
02_camera_calibration.py	Performs camera calibration using chessboard images and saves intrinsic parameters.
02b_camera_calibration.py	Evaluates calibration quality: computes reprojection error and undistorts sample images.
03_aruco_tester.py	Tests calibration by detecting ArUco markers, estimating their pose, and measuring distances.
resources/	Contains calibration images and chessboard/ArUco marker files.
### Calibration Workflow
1. Image Collection
Use 01_screenshot_taker_with_timer.py to capture images from the Kinova camera at regular intervals. Images are saved in the resources/calibration_screenshot directory. Collect images from different angles and positions for robust calibration.
2. Camera Calibration
Run 02_camera_calibration.py to process the captured chessboard images. The script detects chessboard corners and computes the camera matrix and distortion coefficients. Calibration results are saved to calibration_data.pkl.
3. Calibration Evaluation
Use 02b_camera_calibration.py to evaluate calibration quality. The script computes the reprojection error and demonstrates image undistortion. Lower reprojection error indicates better calibration accuracy.
4. ArUco Marker Validation
Run 03_aruco_tester.py to validate calibration using ArUco markers. The script detects specified ArUco markers, estimates their pose, and measures distances between them and to the camera. Visual feedback is provided by drawing markers and lines on the camera stream.

[SCREENSHOT HERE]

## 02_Transformation
This folder contains scripts for transforming object positions detected by the Kinova Gen3 robotic arm’s camera into real-world and robot base coordinates. The main goal is to accurately estimate the 3D pose of objects (such as ArUco markers or target items) in the robot’s workspace, enabling precise pick-and-place operations.
### Overview
Pose estimation and coordinate transformation are critical for robotic manipulation tasks. By detecting ArUco markers in the camera image, the scripts compute the marker’s 3D position relative to the camera, then transform this position into the robot’s world (base) coordinate system. This enables the robot to interact with objects based on vision feedback, rather than relying on hardcoded positions.
### Folder Structure & Scripts
01_pose_estimation.py	Detects ArUco markers in the camera feed and estimates their 3D pose relative to the camera.
02_world_coordinates.py	Transforms detected marker/object positions from camera coordinates to robot/world frame, using robot kinematics and extrinsic parameters.
03_target_get_coordinates_screenshot.py	Captures screenshots and computes the world coordinates of detected targets for pick-and-place.
### Transformation Workflow
1. Pose Estimation with ArUco Markers
01_pose_estimation.py detects ArUco markers in the camera stream and estimates their position and orientation (pose) in the camera coordinate system using OpenCV’s ArUco module. The script uses previously saved camera calibration parameters for accurate 3D localization.
2. Coordinate Transformation
02_world_coordinates.py transforms the detected marker/object pose from the camera frame to the robot’s world (base) frame.This involves:
Reading the robot’s end-effector pose using Kinova’s API.
Using extrinsic calibration (camera-to-gripper transformation).
Applying kinematic transformations to compute the object’s position in the world frame.
4. Target Coordinate Capture
03_target_get_coordinates_screenshot.py automates the process of capturing the camera feed, detecting the target (e.g., a candy bar or marker), and saving its computed world coordinates for later use in robotic manipulation.


[SCREENSHOT HERE]

## 03_Mediapipe_AI_Framework
### Overview
This section focuses on training a custom object detection model using MediaPipe Model Maker on Google Colab. The trained model enables the Kinova robot to detect the candy bar in real-time, which is essential for the pick-and-place task. Due to dependencies and environment requirements, the training code is designed to run in Google Colab rather than local IDEs like PyCharm.
### Contents
Model Training Notebook (Google Colab)
A Python notebook/script that:
  Prepares training and validation datasets labeled in COCO format.
  Loads datasets using MediaPipe’s object_detector.Dataset.
  Trains a MobileNet-based object detection model.
  Evaluates model performance.
  Exports the trained TensorFlow Lite model (.tflite).
Model Folder
  Contains the exported model files:
    candybar_object_detectionmodel.tflite — the trained TensorFlow Lite model.
    label.txt — label file mapping class IDs to names.
Training and Validation Data Folders
  Sample images used for training and validation, labeled with tools such as Label Studio.
### Usage
Training:
Upload the training scripts and dataset folders to Google Colab and run the notebook to train the model.
The notebook installs required dependencies (tensorflow, mediapipe-model-maker) and uses MediaPipe’s API for training.
Deployment:
Use the exported .tflite model and label file in the robot’s AI framework for real-time candy bar detection.
### Important Notes
The training environment must be Google Colab due to dependency and compatibility constraints. Running locally may cause errors. The dataset should be well-labeled and organized in COCO format for best results. The model uses a MobileNet backbone optimized for edge devices like the Kinova Gen3’s onboard computer.


[SCREENSHOT HERE]

## 04_Integration
### Overview
The Integration stage is where all the core components—camera calibration, coordinate transformation, and object detection—come together to enable the Kinova Gen3 robot to autonomously detect, localize, and manipulate a candy bar using a custom-trained MediaPipe model. This folder contains scripts for testing the object detector, calibrating pixel-to-centimeter ratios, running the main pick-and-place logic, and a process skeleton for workflow reference.
### Folder Structure & Scripts
01_object_detector_test.py	Tests the MediaPipe object detection model on the camera stream and visualizes detections.
02_skeleton.py	Outlines the high-level workflow for the pick-and-place process, from initialization to shutdown.
03_pixel_to_cm_calibration.py	Calibrates the pixel-to-centimeter ratio using ArUco markers, crucial for accurate physical movement.
04_integration_main.py	The main script: integrates detection, coordinate transformation, and robot control for autonomous pick-and-place.
### Integration Workflow
Object Detection Test
  Use 01_object_detector_test.py to verify that the MediaPipe model correctly detects the candy bar in the camera feed. This step ensures your AI model is working before integrating with the robot.
Process Skeleton
  02_skeleton.py provides a visual and logical outline of the entire pick-and-place workflow, including initialization, movement, detection, and cleanup. Use this as a reference for understanding or modifying the integration logic.
Pixel-to-Centimeter Calibration
  Run 03_pixel_to_cm_calibration.py to calibrate the pixel-to-real-world (cm) ratio using ArUco markers. This step is essential for translating detected object positions (in pixels) to actual robot movement commands.
  Note: Manual adjustment of the vertical offset (dz) is often required to fine-tune the robot’s end-effector height, compensating for small calibration errors or setup differences. Precise calibration here directly impacts the accuracy and reliability of the pick-and-place operation.
Finally, Main Integration Script
  04_integration_main.py brings together all previous modules:
    Loads calibration parameters from 01_Calibration (camera intrinsics, distortion coefficients).
    Uses transformation logic from 02_Transformation to convert detected object positions to robot coordinates.
    Applies the custom MediaPipe model from 03_Mediapipe_AI_Framework for object detection.
    Controls the Kinova Gen3 arm to pick up and drop the detected candy bar.

### Important Notes
Calibration: The entire integration depends on precise camera calibration. Even small errors in camera intrinsics or pixel-to-cm ratio can cause the robot to miss or improperly grasp the object.
Manual Adjustment: The vertical offset (dz) often needs to be manually fine-tuned for your specific setup. This compensates for differences in camera mounting, table height, or object thickness, and is a normal part of deploying vision-guided robotics46.
Reusability: Always use the calibration files generated in 01_Calibration and 03_pixel_to_cm_calibration.py. These files ensure consistency and reproducibility across sessions and between different machines or environments6.


[SCREENSHOT HERE]

##Requirements
Kinova Gen3 robotic arm with Vision module
Calibrated camera parameters (calibration_data.pkl)
Pixel-to-cm calibration file (pixel_to_cm_calibration.pkl)
Trained MediaPipe object detection model (candybar_objectdetection_model.tflite)
Python 3.11, OpenCV 4.11, MediaPipe 0.10.10, Kinova Kortex API 2.7.0
See envirnemnet.yml (conda) & requirements.txt 

## References & Further Learning

Kinova Gen3: https://www.kinovarobotics.com/product/gen3-robots
Opencv_calibration: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
Opencv_aruco: https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
Google_Colab: https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/tutorials/object_detection/Object_Detection_for_3_dogs.ipynb
Google_Mediapipe: https://ai.google.dev/edge/mediapipe/solutions/guide
Label Studio: https://labelstud.io/
Training AI object_detection model: https://www.youtube.com/watch?v=vODSFXEP-XY


## License & Acknowledgement

This repository and all its code, models, and documentation were developed by Umut Can Vural as part of the KISS Project at Furtwangen University.
KISS is funded by the German Federal Ministry of Education and Research and brings together expertise from computer science, music, and engineering to foster interdisciplinary AI innovation.
For more about KISS, visit: https://www.projekt-kiss.net/

License:
This repository is released under the GPL-3.0 license.
See LICENSE for details.


