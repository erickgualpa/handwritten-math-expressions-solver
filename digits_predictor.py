from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def get_digits_predictor():
    # Load digits dataset
    digits = datasets.load_digits()

    # Plot setting for labels and predictions display
    _, axes = plt.subplots(2, 4)

    # Join images and targets in a single list
    images_and_labels = list(zip(digits.images, digits.target))

    # Add the previous combination to the plot first axis
    for ax, (image, label) in zip(axes[0, :], images_and_labels[:4]):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_title('Training: %i' % label)

    # Flatten the images set
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    # Create the classifier
    classifier = svm.SVC(gamma=0.001)

    # Split data into train and tests subsets
    x_train, x_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False)

    # Learn the digits on the first half of the digits
    classifier.fit(x_train, y_train)

    # Predict the value of the digit on the second half
    predicted = classifier.predict(x_test)

    # Join the first half of the images and the predicted labels in a single list
    images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))

    # Add the previous combination to the plot second axis
    for ax, (image, prediction) in zip(axes[1, :], images_and_predictions[4:]):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_title('Prediction: %i' % prediction)

    # Plot the achieved metrics
    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(y_test, predicted)))

    disp = metrics.plot_confusion_matrix(classifier, x_test, y_test)
    disp.figure_.suptitle("Confusion Matrix")
    print("Confusion matrix:\n%s" % disp.confusion_matrix)

    plt.show()
    return classifier

