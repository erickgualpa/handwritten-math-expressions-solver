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
digits_pred = get_digits_predictor()
joblib.dump(digits_pred, 'digits_pred.pkl', compress=3)
###################################################################
