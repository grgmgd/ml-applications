import numpy as np
import matplotlib.pyplot as plt
from utils import load_train, load_test, plot_cm

TRAINING_PATH = "Train"
TESTING_PATH = "Test"


def fit(classifier, LSTerm):
    size = 240
    T = np.empty(2400)
    T.fill(-1)
    T[size*classifier:size*(classifier + 1):1] = 1
    weights = np.matmul(LSTerm, T)
    return weights


def pred(weights, set):
    weights = np.transpose(weights)
    pred = np.dot(set, weights)
    normalizedPred = np.zeros_like(pred)
    normalizedPred[np.arange(len(pred)), np.argpartition(
        pred, -1, axis=1)[:, -1]] = 1
    normalizedPred = normalizedPred.reshape(10, 20, 10).sum(axis=1)
    return normalizedPred


def init_least_squares(X):
    XT = np.transpose(X)
    temp1 = np.matmul(XT, X)
    temp2 = np.linalg.pinv(temp1)
    LSTerm = np.matmul(temp2, XT)
    return LSTerm


def runs():
    confusion_matrix = np.empty((10, 10))
    training = np.append(load_train(
        TRAINING_PATH, (2400, 784)), np.ones((2400, 1)), axis=1)
    testing = np.append(load_test(TESTING_PATH, (200, 784)),
                        np.ones((200, 1)), axis=1)
    LSTerm = init_least_squares(training)
    weights = np.empty((10, 785))

    for classifier in range(10):
        weights[classifier] = fit(classifier, LSTerm)

    y = pred(weights, testing)
    return y


def main():
    confusion_matrix = runs()
    plot_cm(confusion_matrix, "confusion/least_squares/Confusion.jpg")


main()
