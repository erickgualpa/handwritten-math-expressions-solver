from image_processing import *
from text_detection import detect_text_on_image, get_boxes_as_images
from solving_expression_module import solve_expression
from keras.models import load_model

import os
import json

DIGITS_SYMBOLS_MAPPING = "digits-symbols-mapping.json"
# Load digits-symbols mapping from categorical to numerical
with open(DIGITS_SYMBOLS_MAPPING, "r") as dig_sym_mapping_file:
    labels_mapping = json.load(dig_sym_mapping_file)["INV_DIGITS_SYMBOLS_MAPPING"]

## LOAD IMAGES FOR TESTING ########################################
path_img = './img/'

working_ims = []
for root, dirs, files in os.walk(path_img):
    for filename in files:
        working_ims.append(loadImage(path_img + filename))
###################################################################

## LOAD CLASSIFIER ################################################
clf = load_model('./classifiers/digits_symbols_cnn_classif.h5')
###################################################################

## SET WORKING IMAGE ##############################################
working_im = working_ims[1]
showImage(resizeImage(working_im, 0.3))

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
boxes, im_detection = detect_text_on_image(working_im, 0.8)    # (image, min_confidence)
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
    math_exp = []    # math_exp -> Symbol list
    for im_digit_symbol in detected_digits_symbols:
        # Convert 'im_digit_symbol' to grayscale
        im_gray = cv2.cvtColor(im_digit_symbol, cv2.COLOR_BGR2GRAY)
        # showImage(resizeImage(im_gray, 30))

        im_gray = pre_classification_image_processing(im_gray)
        showImage(resizeImage(im_gray, 0.3))

        # Resize image to (28,28)
        im_gray = cv2.resize(im_gray, (28, 28), interpolation=cv2.INTER_AREA)

        # Reshape Image to (28,28,1)
        im_gray = im_gray.reshape(28, 28, 1)

        # Predict expression symbol
        symbol = clf.predict_classes(np.array([im_gray]))

        # Append symbol on math expression
        math_exp.append(labels_mapping[str(symbol[0])])
    expression_list.append(np.array(math_exp))
expression_list = np.array(expression_list)
###################################################################

## EXPRESSION SOLVING ############################################
# TODO: Colocar el resultado debajo del texto detectado
im_result = resizeImage(working_im, 0.3)
for exp in expression_list:
    str_exp, result = solve_expression(exp)
    print(str_exp, '=', result)
    # Show expression result on image
    im_result = write_message_on_img(im_result, str(str_exp) + '=' + str(result))

showImage(im_result)
##################################################################


