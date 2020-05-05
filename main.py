from utils import *
from digits_predictor_training import *
from image_processing import *
from text_detection import *

import os
import joblib

## LOAD IMAGES FOR TESTING ########################################
path_img = './img/'

working_ims = []
for root, dirs, files in os.walk(path_img):
    for filename in files:
        working_ims.append(loadImage(path_img + filename))
###################################################################

## GET AND SAVE THE CLASSIFIER ####################################
"""
digits_pred = get_digits_predictor()
joblib.dump(digits_pred, './classifiers/digits_pred.pkl', compress=3)
"""
###################################################################

## IMAGE PROCESSING ###############################################

img1 = working_ims[3]
showImage(resizeImage(img1, 30))
# processed_img = process_image(img1)
# showImage(processed_img)


"""
vid = cv2.VideoCapture(0)
while(True):
    ret, frame = vid.read()
    processed_image = process_image(frame)
    cv2.imshow('frame', processed_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()
"""
###################################################################

## TEXT DETECTION #################################################
detection = detect_text_on_image(img1, 0.5)
showImage(resizeImage(detection, 30))
###################################################################
