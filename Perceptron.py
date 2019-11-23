import numpy as np
import matplotlib.pyplot as plt
from utils import plot_cm

TRAINING_PATH = "Train"
TESTING_PATH = "Test"
LEARNING_RATE = 1
YESPRINT = True


def load_train():
    trainSet = np.empty((2400, 785))
    for file in range(1, 2401):
        path = TRAINING_PATH + "/" + str(file) + ".jpg"
        image = plt.imread(path).flatten()
        image = np.append(image, 1)
        trainSet[file - 1] = image
    return trainSet


def load_test():
    testSet = np.empty((200, 785))
    for file in range(1, 201):
        path = TESTING_PATH + "/" + str(file) + ".jpg"
        image = plt.imread(path).flatten()
        image = np.append(image, 1)
        testSet[file - 1] = image
    return testSet


def perceptron(classifier, training):
    print(LEARNING_RATE)
    T = np.empty(2400)
    T.fill(-1)
    size = 240
    T[size*classifier:size*(classifier + 1):1] = 1
    weight = np.zeros(785)
    weight[0] = 1
    for limit in range(500):
        for i in range(2400):
            weightTrans = np.transpose(weight)
            pred = np.dot(training[i], weightTrans)
            if (((T[i] == 1) and (pred <= 0)) or ((T[i] == -1) and (pred > 0))):
                weight += training[i]*LEARNING_RATE*T[i]
    return weight


def pred(weights, set):
    weights = np.transpose(weights)
    pred = np.dot(set, weights)
    normalizedPred = np.zeros_like(pred)
    normalizedPred[np.arange(len(pred)), np.argpartition(
        pred, -1, axis=1)[:, -1]] = 1
    normalizedPred = normalizedPred.reshape(10, 20, 10).sum(axis=1)
    return normalizedPred


def runs():
    training = load_train()
    testing = load_test()
    weights = np.zeros((10, 785))
    for i in weights:
        i[0] = 1
    for classifier in range(10):
        weights[classifier] = perceptron(classifier, training)

    y = pred(weights, testing)
    return y


def main():
    learningRate = [1, 10**-1, 10**-2, 10**-3, 10**-
                    4, 10**-5, 10**-6, 10**-7, 10**-8, 10**-9.]

    global LEARNING_RATE
    for n in learningRate:
        LEARNING_RATE = n
        confusion_matrix = runs()
        plot_cm(confusion_matrix, "perceptron/Confusion" + str(n) + ".jpg")


main()
