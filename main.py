import cv2
import os
from functions import *

## Classifier modules ######################
import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import numpy as np
############################################

## LOAD IMAGES FOR TESTING #################
path_img = "./img/"

working_ims = []
for root, dirs, files in os.walk(path_img):
    for filename in files:
        working_ims.append(resizeImage(loadImage(path_img + filename), scale=20))
############################################

## TRAIN THE CLASSIFIER ####################
digits = datasets.load_digits()
print(digits.data.shape)
"""
plt.gray()
plt.matshow(digits.images[10])
plt.show()
"""
############################################
