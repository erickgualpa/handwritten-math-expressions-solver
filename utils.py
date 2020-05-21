import cv2
import numpy as np

def loadImage(img_name):
    img = cv2.imread(img_name, cv2.IMREAD_COLOR)
    return img

def showImage(img):
    if type(img) == str:
        img = cv2.imread(img, cv2.IMREAD_COLOR)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resizeImage(img, scale):
    scale_percent = scale
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized

def resizeImageByDim(img, width, height):
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

def getOffsetForAllDirections(scale_x, scale_y, width, height):
    offset_x = int(width * scale_x) // 2
    offset_y = int(height * scale_y) // 2
    return offset_x, offset_y

def get_boxes_as_images(boxes, img): # 'boxes' item:(start_x, start_y, end_x, end_y)
    detected_texts = []

    # Sort 'boxes' by x-axis
    boxes = np.array(boxes)
    boxes = boxes[boxes[:, 0].argsort()]

    for box in boxes:   # box:(start_x, start_y, end_x, end_y)
        start_x = box[0]
        start_y = box[1]
        end_x = box[2]
        end_y = box[3]

        detected_texts.append(img[start_y:end_y, start_x:end_x, :])

    return np.array(detected_texts)
