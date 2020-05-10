from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from os import listdir

DATASET_PATH = "../extracted_images_Kaggle/"

class DigitsSymbolsDataset:
    def __init__(self):
        self.__targets = []
        self.__samples = []
        self.__size = 0

    def add_target(self, target):
        self.__targets.append(target)

    def get_targets(self):
        return self.__targets

    def add_sample(self, sample):
        self.__samples.append(sample)

    def get_samples(self):
        return self.__samples

    def __len__(self):
        return self.__samples.__len__()

def get_digits_predictor():
    dataset = DigitsSymbolsDataset()

    labels = listdir(DATASET_PATH)
    for label in labels:
        data = os.listdir(DATASET_PATH + label)
        for sample in data:
            # print("- ", label, " --> ", sample)
            # TODO: read image -> resize image to 8x8 -> save in dataset
            dataset.add_target(label)
            dataset.add_sample(DATASET_PATH + sample)

    images_and_labels = list(zip(dataset.get_samples(), dataset.get_targets()))
    n_samples = len(images_and_labels)
    print(n_samples)

    # TODO: Flatten image

if __name__ == '__main__':
    get_digits_predictor()