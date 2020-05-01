from functions import *
from digits_predictor import *
from image_processing import *

import os
import joblib

## LOAD IMAGES FOR TESTING ########################################
path_img = './img/'

working_ims = []
for root, dirs, files in os.walk(path_img):
    for filename in files:
        working_ims.append(resizeImage(loadImage(path_img + filename), scale=20))
###################################################################

## GET AND SAVE THE CLASSIFIER ####################################
"""
digits_pred = get_digits_predictor()
joblib.dump(digits_pred, './classifiers/digits_pred.pkl', compress=3)
"""
###################################################################

## IMAGE PROCESSING ###############################################
img1 = working_ims[4]
showImage(img1)
processed_img = process_image(img1)
showImage(processed_img)
###################################################################
