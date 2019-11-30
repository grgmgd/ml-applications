import numpy as np
import matplotlib.pyplot as plt
from utils import load_train, load_test, plot_cm

TRAINING_PATH = "Train"
TESTING_PATH = "Test"


def means(set):
    return set.reshape(10, 240, 784).mean(axis=1)


def variances(means, set):
    output = set.reshape(10, 240, 784).var(axis=1)
    output[output < 0.01] = 0.01
    return output


def normal_dist(x, means, variances):
    first_factor = np.power(np.sqrt(2 * np.pi * variances), -1)
    second_factor = np.exp(-np.square(x - means) / (2 * variances))
    normal_dist_values = np.multiply(first_factor, second_factor)
    output = normal_dist_values.prod(axis=1)

    return np.argmax(output)


def main():
    train = np.divide(load_train(TRAINING_PATH, (2400, 784)), 255)
    test = np.divide(load_test(TESTING_PATH, (200, 784)), 255)

    mean_values = means(train)
    variance_values = variances(mean_values, train)

    output = np.zeros(200)
    for i in range(len(test)):
        output[i] = normal_dist(test[i], mean_values, variance_values)

    reshaped = output.reshape(10, 20)
    confusion_matrix = np.zeros((100))
    for i in range(10):
        unique, counts = np.unique(reshaped[i], return_counts=True)
        np.put(confusion_matrix, unique.astype(int) + i*10, counts)

    confusion_matrix = confusion_matrix.reshape(10, 10)

    plot_cm(confusion_matrix, "confusion/naive_bayes/Confusion.jpg")


main()
