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


with open("pixel_to_cm_calibration.pkl", "rb") as f:
    calib = pickle.load(f)
    x_ratio = calib['x_ratio']
    y_ratio = calib['y_ratio']

# Load camera calibration
with open("../01_calibration/calibration_data.pkl", "rb") as f:
    calib = pickle.load(f)
    camera_matrix = calib['cameraMatrix']
    dist_coeff = calib['dist']

