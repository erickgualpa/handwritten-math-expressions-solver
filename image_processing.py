from PIL import Image, ImageEnhance
import cv2
import numpy as np
from utils import *

## FUNCTIONS #################################################################
def dilate(img, kernel_size):
    kernel = np.ones(kernel_size, np.uint8)
    dilation = cv2.dilate(img, kernel, iterations=3)

    return dilation

def erode(img, kernel_size):
    kernel = np.ones(kernel_size, np.uint8)
    erosion = cv2.erode(img, kernel, iterations=1)

    return erosion

def closing(img, kernel_size):
    kernel = np.ones(kernel_size, np.uint8)
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    return closed

def opening(img, kernel_size):
    kernel = np.ones(kernel_size, np.uint8)
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    return opened

def max_contrast(img, scale):
    img = Image.fromarray(img)
    en = ImageEnhance.Contrast(img)
    img = en.enhance(scale)
    contrast = np.array(img)

    return contrast

def spot_numbers(img):
    # Smooth the image with contrast increasing
    contrast = max_contrast(img, 2)
    #showImage(contrast)

    # Spot edges
    canny = cv2.Canny(contrast, 30, 200)

    # Clean the image
    #clos = closing(canny, kernel_size=(5,5))

    return canny

def find_and_draw_contours(original, img):
    #showImage(img)
    contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in contours]

    # TODO: Remove rects intersection

    for rect in rects:
        cv2.rectangle(original, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)

    return original
##############################################################################

def process_image(img):
    # STEP 1 - Convert input image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # STEP 2 - Apply Bilateral filter for noise removing
    bilateral = cv2.bilateralFilter(gray, 11, 17, 17)

    # STEP 3 - Spot the digits
    digits = spot_numbers(bilateral)

    # STEP 4 - Find and draw the contours
    contours = find_and_draw_contours(img, digits)

    return contours
