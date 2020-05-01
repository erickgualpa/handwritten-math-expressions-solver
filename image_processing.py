from PIL import Image, ImageEnhance
import cv2
import numpy as np
from functions import *

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
    """
    img = Image.fromarray(img)
    en = ImageEnhance.Contrast(img)
    img = en.enhance(2.0)
    contrast = np.array(img)
    """
    contrast = img * scale

    return contrast

def spot_numbers(img):
    # Smooth the image with contrast increasing
    contrast = max_contrast(img, 2)
    showImage(contrast)

    # Spot edges
    canny = cv2.Canny(contrast, 30, 200)

    # Clean the image
    clos = closing(canny, kernel_size=(5,5))

    return clos
##############################################################################

def process_image(img):
    # STEP 1 - Convert input image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # STEP 2 - Apply Bilateral filter for noise removing
    bilateral = cv2.bilateralFilter(gray, 11, 17, 17)

    # STEP 3 - Spot the digits
    digits = spot_numbers(bilateral)

    # STEP 4 - Find the countourns
    cnts = cv2.findContours(digits.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    

    return digits