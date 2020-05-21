from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from os import listdir
from utils import *
import numpy as np

DATASET_PATH = "../Kaggle_reducted_dataset/"
WIDTH_DS_ITEM = 8   # Dataset sample width
HEIGHT_DS_ITEM = 8  # Dataset sample height

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

def get_digits_symbols_predictor():
    # Prepare Kaggle Digits and Symbols object
    dataset = DigitsSymbolsDataset()

    # Plot setting for labels and predictions display
    _, axes = plt.subplots(2, 4)

    # Load Kaggle Digits and Symbols
    labels = np.array(listdir(DATASET_PATH))
    for label in labels[8:12]:
        data = np.array(os.listdir(DATASET_PATH + label))
        for sample in data:
            # print("- ", label, " --> ", sample)
            dataset.add_target(label)
            filename = DATASET_PATH + label + "/" + sample
            im_sample = resize_image_by_dim(loadImage(filename), WIDTH_DS_ITEM, HEIGHT_DS_ITEM)
            im_sample = cv2.cvtColor(im_sample, cv2.COLOR_BGR2GRAY)
            dataset.add_sample(im_sample)

    # Join images and targets in a single list
    images_and_labels = list(zip(dataset.get_samples(), dataset.get_targets()))

    # Add the previous combination to the plot first axis
    for ax, (image, label) in zip(axes[0, :], images_and_labels[:4]):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_title('Training: ' + label)

    # showImage(resizeImage(dataset.get_samples()[0], 1000))

    # Flatten the images set
    n_samples = len(images_and_labels)
    data = np.array(dataset.get_samples()).reshape((n_samples, -1))

    # Create the classifier
    classifier = svm.SVC(gamma=0.001)

    # Split data into train and tests subsets
    x_train, x_test, y_train, y_test = train_test_split(data, np.array(dataset.get_targets()), test_size=0.45, shuffle=True)

    # Learn the digits on the first half of the digits
    classifier.fit(x_train, y_train)

    # Predict the value of the digit on the second half
    predicted = classifier.predict(x_test)

    # Join the first half of the images and the predicted labels in a single list
    images_and_predictions = list(zip(dataset.get_samples()[n_samples // 2:], predicted))

    # Add the previous combination to the plot second axis
    for ax, (image, prediction) in zip(axes[1, :], images_and_predictions[4:]):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_title('Prediction: ' + prediction)

    # Plot the achieved metrics
    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(y_test, predicted)))

    disp = metrics.plot_confusion_matrix(classifier, x_test, y_test)
    disp.figure_.suptitle("Confusion Matrix")
    print("Confusion matrix:\n%s" % disp.confusion_matrix)

    plt.show()
    return classifier


"""
if __name__ == '__main__':
    get_digits_symbols_predictor()
"""
