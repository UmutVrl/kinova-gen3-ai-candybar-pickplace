
# Project Candybar
Below is a draft README section for your 01_Calibration folder, summarizing its purpose, workflow, and the scripts it contains. This will help users understand the calibration process for the Kinova Gen3 robotic arm’s camera and how each script fits into the workflow.

## 01_Calibration
This folder contains all scripts and resources required for calibrating the Kinova Gen3 robotic arm’s vision system. Accurate camera calibration is essential for reliable object detection, pose estimation, and precise robotic manipulation.

### Overview
The calibration process ensures that the camera’s intrinsic parameters (such as focal length, principal point, and distortion coefficients) are accurately determined. This is critical for tasks like object detection, 3D localization, and robot-to-camera transformations. Workflow includes:
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

### Further Learning

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
02_world_coordinates.py transforms the detected marker/object pose from the camera frame to the robot’s world (base) frame.
  This involves:
   Reading the robot’s end-effector pose using Kinova’s API.
   Using extrinsic calibration (camera-to-gripper transformation).
   Applying kinematic transformations to compute the object’s position in the world frame.
4. Target Coordinate Capture
03_target_get_coordinates_screenshot.py automates the process of capturing the camera feed, detecting the target (e.g., a candy bar or marker), and saving its computed world coordinates for later use in robotic manipulation.

### Further Learning

## 03_Mediapipe_AI_Framework

#Overview
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
The training environment must be Google Colab due to dependency and compatibility constraints. Running locally may cause errors. The dataset should be well-labeled and organized in COCO format for best results.

The model uses a MobileNet backbone optimized for edge devices like the Kinova Gen3’s onboard computer.

References & Further Learning
https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/tutorials/object_detection/Object_Detection_for_3_dogs.ipynb
https://ai.google.dev/edge/mediapipe/solutions/guide
https://www.youtube.com/watch?v=vODSFXEP-XY




