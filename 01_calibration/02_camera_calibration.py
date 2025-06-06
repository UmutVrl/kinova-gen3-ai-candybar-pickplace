#########################################################################################
#    Kinova Gen3 Robotic Arm                                                            #
#    Camera Calibration (Part 2 - addition of undistortion)                             #
#    Taking a screenshot from the camera                                                #
#    (Chessboard Detection and Calibration)                                                                                   #
#    written by: U. Vural                                                               #
#    based on:                                                                          #
#    https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html                    #
#    see chessboard.png under resources folder                                          #
#                                                                                       #
#                                                                                       #
#    for KISS Project at Furtwangen University                                          #
#                                                                                       #
#########################################################################################

import numpy as np
import cv2
import glob
import pickle
import time

# Start timer
start_time = time.time()

# Chessboard settings
chessboard_size = (12, 11)  # (columns, rows) of inner corners
square_size = 10  # mm

# Prepare object points (0,0,0), (1,0,0), ... scaled by square size
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays for points
objpoints = []
imgpoints = []

# Get images
images = glob.glob(r'resources\*.jpg')

# Criteria for cornerSubPix
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(500)
cv2.destroyAllWindows()

# Calibration
if objpoints and imgpoints:
    img_shape = gray.shape[::-1]
    ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)
    # Save everything needed for later evaluation
    with open("calibration_data.pkl", "wb") as f:
        pickle.dump({
            'cameraMatrix': cameraMatrix,
            'dist': dist,
            'objpoints': objpoints,
            'imgpoints': imgpoints,
            'rvecs': rvecs,
            'tvecs': tvecs
        }, f)
    print("Calibration successful. All parameters saved in calibration_data.pkl.")
else:
    print("No chessboard corners were found. Calibration failed.")

# Timing
elapsed = time.time() - start_time
print(f"Processed {len(objpoints)} valid images in {elapsed:.1f} seconds.")
