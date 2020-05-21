from utils import *
from digits_predictor_training import *
from digits_symbols_predictor_training import *
from image_processing import *
from text_detection import *

import os
import joblib
import time

## LOAD IMAGES FOR TESTING ########################################
path_img = './img/'

working_ims = []
for root, dirs, files in os.walk(path_img):
    for filename in files:
        working_ims.append(loadImage(path_img + filename))
###################################################################

## GET AND SAVE THE CLASSIFIER ####################################
"""
start_time = time.time()
digits_symbols_pred = get_digits_symbols_predictor()
joblib.dump(digits_symbols_pred, './classifiers/digits_symbols_pred.pkl', compress=3)
print("--- Elapsed time: %s seconds ---" % (time.time() - start_time))
"""
###################################################################

## LOAD CLASSIFIER ################################################
clf = joblib.load('./classifiers/digits_symbols_pred.pkl')
###################################################################

## SET WORKING IMAGE ##############################################
working_im = working_ims[1]
showImage(resizeImage(working_im, 30))

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
boxes, im_detection = detect_text_on_image(working_im, 0.2)    # (image, min_confidence)
# showImage(resizeImage(im_detection, 30))
detected_texts = get_boxes_as_images(boxes, working_im)    # Extract de bounding rectangles of detected text as images
###################################################################

## DIGITS AND SYMBOLS EXTRACTION AS IMAGES ########################
expression_list = []
expression = []
for im_text in detected_texts:
    detections, _ = process_image(im_text.copy())
    detected_digits_symbols = get_boxes_as_images(detections, im_text)
    # showImage(resizeImage(im_contours, 30))
    expression = []
    for im_digit_symbol in detected_digits_symbols:
        # Convert 'im_digit_symbol' to grayscale
        im_gray = cv2.cvtColor(im_digit_symbol, cv2.COLOR_BGR2GRAY)
        # showImage(resizeImage(im_gray, 30))

        # TODO: Probar procesado de imagen o HOG
        im_gray = pre_svm_image_processing(im_gray)
        showImage(resizeImage(im_gray, 30))

        # Reshape image to (8,8)
        im_gray = cv2.resize(im_gray, (8, 8), interpolation=cv2.INTER_AREA)

        # Flatten image
        im_gray = np.array(im_gray).flatten()

        res = clf.predict([im_gray])
        expression.append(res)
###################################################################

for item in expression:
    print(item)

