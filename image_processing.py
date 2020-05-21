from PIL import Image, ImageEnhance
from utils import *
from skimage.filters import threshold_sauvola

import cv2
import numpy as np

## FUNCTIONS #################################################################
def dilate(img, kernel_size):
    kernel = np.ones(kernel_size, np.uint8)
    im_dilation = cv2.dilate(img, kernel, iterations=3)

    return im_dilation

def erode(img, kernel_size):
    kernel = np.ones(kernel_size, np.uint8)
    im_erosion = cv2.erode(img, kernel, iterations=1)

    return im_erosion

def closing(img, kernel_size):
    kernel = np.ones(kernel_size, np.uint8)
    im_closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    return im_closed

def opening(img, kernel_size):
    kernel = np.ones(kernel_size, np.uint8)
    im_opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    return im_opened

def max_contrast(img, scale):
    img = Image.fromarray(img)
    en = ImageEnhance.Contrast(img)
    img = en.enhance(scale)
    contrast = np.array(img)

    return contrast

def spot_numbers(img):
    # Smooth the image with contrast increasing
    im_contrast = max_contrast(img, 2)
    #showImage(im_contrast)

    # Spot edges
    im_canny = cv2.Canny(im_contrast, 30, 200)

    # Close the found edges
    im_closed = closing(im_canny, (9, 9))

    return im_closed

def find_and_draw_contours(original, img):
    height, width = original.shape[:2]

    contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in contours]

    detections = []
    for rect in rects:
        # Calculate contour areas for extracting it from the image later
        start_x = rect[0]
        start_y = rect[1]
        w = rect[2]
        h = rect[3]
        end_x = start_x + w
        end_y = start_y + h

        rect_offset_x, rect_offset_y = getOffsetForAllDirections(scale_x=0.3, scale_y=0.3, width=w, height=h)

        start_x = (start_x - rect_offset_x) if (start_x - rect_offset_x) > 0 else 0
        start_y = (start_y - rect_offset_y) if (start_y - rect_offset_y) > 0 else 0
        end_x = (end_x + rect_offset_x) if (end_x + rect_offset_x) < width else width
        end_y = (end_y + rect_offset_y) if (end_y + rect_offset_y) < height else height

        cv2.rectangle(original, (start_x, start_y), (end_x, end_y), (0, 255, 0), 3)
        detections.append((start_x, start_y, end_x, end_y))

    return np.array(detections), original
##############################################################################

def process_image(img):
    # STEP 1 - Convert input image to grayscale
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # STEP 2 - Apply Bilateral filter for noise removing
    im_bilateral = cv2.bilateralFilter(im_gray, 11, 17, 17)

    # STEP 3 - Spot the digits
    im_digits = spot_numbers(im_bilateral)
    # showImage(im_digits)

    # STEP 4 - Find and draw the contours
    detections, im_contours = find_and_draw_contours(img, im_digits)

    return detections, im_contours

def pre_svm_image_processing(img):
    # Smooth the image with contrast increasing
    im_contrast = max_contrast(img, 2)
    # showImage(im_contrast)

    # Sauvola thresholding
    window_size = 25
    tresh_sauvola = threshold_sauvola(im_contrast, window_size=window_size)
    im_contrast[im_contrast > tresh_sauvola] = 255
    im_contrast[im_contrast <= tresh_sauvola] = 0

    return im_contrast