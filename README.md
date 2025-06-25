
## Project Candybar
Below is a draft README section for your 01_Calibration folder, summarizing its purpose, workflow, and the scripts it contains. This will help users understand the calibration process for the Kinova Gen3 robotic arm’s camera and how each script fits into the workflow.

# 01_Calibration
This folder contains all scripts and resources required for calibrating the Kinova Gen3 robotic arm’s vision system. Accurate camera calibration is essential for reliable object detection, pose estimation, and precise robotic manipulation.
#Overview
The calibration process ensures that the camera’s intrinsic parameters (such as focal length, principal point, and distortion coefficients) are accurately determined. This is critical for tasks like object detection, 3D localization, and robot-to-camera transformations.

# The workflow includes:
Taking timed screenshots from the robot’s camera for calibration data collection.
Performing chessboard-based camera calibration.
Evaluating calibration quality and undistorting images.
Testing calibration using ArUco markers for spatial validation.

# Folder Structure & Scripts
# Script/File	Purpose
01_screenshot_taker_with_timer.py	Captures timed screenshots from the Kinova camera stream to collect calibration images.
02_camera_calibration.py	Performs camera calibration using chessboard images and saves intrinsic parameters.
02b_camera_calibration.py	Evaluates calibration quality: computes reprojection error and undistorts sample images.
03_aruco_tester.py	Tests calibration by detecting ArUco markers, estimating their pose, and measuring distances.
resources/	Contains calibration images and chessboard/ArUco marker files.

# Calibration Workflow
1. Image Collection
Use 01_screenshot_taker_with_timer.py to capture images from the Kinova camera at regular intervals. Images are saved in the resources/calibration_screenshot directory. Collect images from different angles and positions for robust calibration.
2. Camera Calibration
Run 02_camera_calibration.py to process the captured chessboard images. The script detects chessboard corners and computes the camera matrix and distortion coefficients. Calibration results are saved to calibration_data.pkl.
3. Calibration Evaluation
Use 02b_camera_calibration.py to evaluate calibration quality. The script computes the reprojection error and demonstrates image undistortion. Lower reprojection error indicates better calibration accuracy.
4. ArUco Marker Validation
Run 03_aruco_tester.py to validate calibration using ArUco markers. The script detects specified ArUco markers, estimates their pose, and measures distances between them and to the camera. Visual feedback is provided by drawing markers and lines on the camera stream.
