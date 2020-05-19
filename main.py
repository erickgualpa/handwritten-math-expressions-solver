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
working_im = working_ims[1]
showImage(resizeImage(working_im, 30))
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
boxes, detection = detect_text_on_image(working_im, 0.2)    # (image, min_confidence)
showImage(resizeImage(detection, 30))
detected_texts = get_boxes_as_images(boxes, working_im)    # Extract de bounding rectangles of detected text as images
###################################################################

for im in detected_texts:
    showImage(resizeImage(im, 30))