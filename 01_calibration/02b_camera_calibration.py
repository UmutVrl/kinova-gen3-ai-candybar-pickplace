#########################################################################################
#    Kinova Gen3 Robotic Arm                                                            #
#    Evaluating the quality of a previously performed camera calibration                #
#    (Reprojection Error and Undistortion)                                              #
#    written by: U. Vural                                                               #
#    based on:                                                                          #
#    https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html                    #
#    see chessboard.png under resources folder                                          #
#                                                                                       #
#                                                                                       #
#    for KISS Project at Furtwangen University                                          #
#                                                                                       #
#########################################################################################

import cv2
import pickle
import time

# Start timing
first_capture_time = time.time()

# Load all calibration parameters and points
with open('calibration_data.pkl', 'rb') as f:
    data = pickle.load(f)
cameraMatrix = data['cameraMatrix']
dist = data['dist']
objpoints = data['objpoints']
imgpoints = data['imgpoints']
rvecs = data['rvecs']
tvecs = data['tvecs']

# Load a sample calibration image
img = cv2.imread(r'resources\calibration_screenshot1.jpg')
h, w = img.shape[:2]

# Compute new optimal camera matrix
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))

# Undistort the image using the optimal camera matrix
undistorted = cv2.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

# Crop the undistorted image to the valid ROI
x, y, w, h = roi
undistorted_cropped = undistorted[y:y + h, x:x + w]

# (Optional) Save or display the undistorted image
# cv2.imwrite('undistorted_result.png', undistorted_cropped)
# cv2.imshow('Undistorted', undistorted_cropped)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Calculate reprojection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

mean_error /= len(objpoints)
print("Camera matrix:", cameraMatrix)
print("Dist coeffs.:", dist)
print("Total reprojection error: {:.4f} pixels".format(mean_error))

# Timing and summary
last_capture_time = time.time()
duration = last_capture_time - first_capture_time
hours = int(duration // 3600)
minutes = int(duration // 60 % 60)
seconds = int(duration % 60)
print(f"Program Run Duration: {hours} hours:{minutes} mins:{seconds} secs")
print(f"{len(objpoints)} object pictures and {len(imgpoints)} image points used.")