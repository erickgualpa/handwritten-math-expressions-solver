from image_processing import *
from text_detection import detect_text_on_image, get_boxes_as_images
from solving_expression_module import solve_expression
from keras.models import load_model


import os
import json

WORKING_IM_WIDTH = 1000
WORKING_IM_HEIGHT = 1000
DIGITS_SYMBOLS_MAPPING = "digits-symbols-mapping.json"

# Load digits-symbols mapping from categorical to numerical
with open(DIGITS_SYMBOLS_MAPPING, "r") as dig_sym_mapping_file:
    labels_mapping = json.load(dig_sym_mapping_file)["INV_DIGITS_SYMBOLS_MAPPING"]

## LOAD IMAGES FOR TESTING ########################################
path_img = './testing_imgs/'
path_results_img = './testing_imgs_results/'

working_ims = []
for root, dirs, files in os.walk(path_img):
    for filename in files:
        working_ims.append(loadImage(path_img + filename))
###################################################################

try:
    ## LOAD CLASSIFIER ################################################
    clf = load_model('./classifiers/digits_symbols_cnn_classif_128_4.h5')
    ###################################################################
    try:
        ## SET WORKING IMAGE ##############################################
        im_index = 4
        working_im = working_ims[im_index]
        working_im = resize_image_by_dim(working_im, WORKING_IM_WIDTH, WORKING_IM_HEIGHT)
        showImage(working_im)
        ###################################################################

        ## TEXT DETECTION #################################################
        boxes, im_detection = detect_text_on_image(working_im, 0.8)    # (image, min_confidence)
        showImage(im_detection)
        detected_texts = get_boxes_as_images(boxes, working_im)    # Extract de bounding rectangles of detected text as images
        ###################################################################

        ## DIGITS AND SYMBOLS EXTRACTION AS IMAGES ########################
        expression_list = []
        expression = []
        for im_text in detected_texts:
            detections, _ = process_image(im_text.copy())
            detected_digits_symbols = get_boxes_as_images(detections, im_text)
            # showImage(im_contours)
            math_exp = []    # math_exp -> Symbol list
            for im_digit_symbol in detected_digits_symbols:
                # Convert 'im_digit_symbol' to grayscale
                im_gray = cv2.cvtColor(im_digit_symbol, cv2.COLOR_BGR2GRAY)

                im_gray = pre_classification_image_processing(im_gray)

                # Put image over a square canvas
                im_gray = add_square_canvas_to_image(im_gray)
                im_gray = add_borders_to_image(im_gray, border_scale=0.2, color=(0, 0, 0))
                showImage(im_gray)

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
        im_result = working_im.copy()
        message_height_offset = 50

        for (exp, box) in zip(expression_list, boxes):
            str_exp, result = solve_expression(exp)
            if len(result) > 0:
                print(str_exp, '=', result)
                # Show expression result on image
                x_pos = box[0]
                y_pos = (box[3] + message_height_offset) if (box[3] + message_height_offset) < im_result.shape[0] else im_result.shape[0]
                im_result = write_message_on_img(im_result, str(str_exp) + '=' + str(result), position=(x_pos, y_pos))
        showImage(im_result)
        ##################################################################

        ## SAVE IMAGE RESULT #############################################
        im_res_filename = path_results_img + str(im_index) + '_result.jpg'
        cv2.imwrite(im_res_filename, im_result)
        ##################################################################

    except Exception as e:
        print('[ERROR]:', e)

except OSError as ose:
    print('[ERROR]: Classifier filename not found')
    print(ose)
except Exception as e:
    print('[ERROR]:', e)

print("[INFO]: Finishing program...")


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


