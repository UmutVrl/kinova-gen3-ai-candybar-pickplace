#########################################################################################
#    Kinova Gen3 Robotic Arm                                                            #
#                                                                                       #
#    MediaPipe Model Training using Google Colab                                        #
#                                                                                       #
#    written by: U. Vural                                                               #
#                                                                                       #
#                                                                                       #
#                                                                                       #
#    for KISS Project at Furtwangen University                                          #
#                                                                                       #
#########################################################################################

# IMPORTANT NOTE:
# This code is designed to be run in Google Colab, not in Anaconda or a local Jupyter environment.
# Google Colab provides the required dependencies and a compatible environment for MediaPipe Model Maker and TensorFlow.
# Running this code locally may cause installation or compatibility issues.
# Please upload and execute this notebook in Google Colab for best results.


## SETUP ##
#!pip install --upgrade pip
#!pip install --upgrade tensorflow
#!pip install 'keras<3.0.0' mediapipe-model-maker
#!pip install mediapipe-model-maker

#######################
## LIBRARIES ##
from google.colab import files
import os
import json
import tensorflow as tf
assert tf.__version__.startswith('2')
from mediapipe_model_maker import object_detector

########################################################

import sys
print(sys.path)
#!python --version
#!python -m pip freeze

#########################################################

#!rm -rf /content/blocks/train
os.getcwd()

#########################################################

## PREPARE DATA ##
train_dataset_path = "candybar_blocks/train"
validation_dataset_path = "candybar_blocks/validation"

#########################################################

## REVIEW DATASET ##
with open(os.path.join(train_dataset_path, "labels.json"), "r") as f:
  labels_json = json.load(f)
for category_item in labels_json["categories"]:
  print(f"{category_item['id']}: {category_item['name']}")

###########################################################

## CREATE DATASET ##
train_data = object_detector.Dataset.from_coco_folder(train_dataset_path, cache_dir="/tmp/od_data/train")
validation_data = object_detector.Dataset.from_coco_folder(validation_dataset_path, cache_dir="/tmp/od_data/validation")
print("train_data size: ", train_data.size)
print("validation_data size: ", validation_data.size)

###########################################################

## RETRAIN MODEL ##
# https://developers.google.com/mediapipe/solutions/customization/object_detector#hyperparameters
spec = object_detector.SupportedModels.MOBILENET_MULTI_AVG
hparams = object_detector.HParams(export_dir='exported_model')
options = object_detector.ObjectDetectorOptions(
    supported_model=spec,
    hparams=hparams
)

###########################################################
model = object_detector.ObjectDetector.create(
    train_data=train_data,
    validation_data=validation_data,
    options=options)

###########################################################

## EVALUATE THE MODEL PERFORMANCE ##
loss, coco_metrics = model.evaluate(validation_data, batch_size=4)
print(f"Validation loss: {loss}")
print(f"Validation coco metrics: {coco_metrics}")

###########################################################

## EXPORT MODEL ##
model.export_model()
#!ls exported_model
files.download('exported_model/model.tflite')

###################################