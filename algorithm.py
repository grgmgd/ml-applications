import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

TRAINING_PATH = "Train"
TESTING_PATH = "Test"


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


def fit(classifier, LSTerm):
    size = 240
    T = np.empty(2400)
    T.fill(-1)
    T[size*classifier:size*(classifier + 1):1] = 1
    weights = np.matmul(LSTerm, T)
    return weights


def pred(weights):
    X = load_test()
    pred = np.dot(X, weights)
    normalizedPred = np.select(
        [pred < 0, pred >= 0], [np.zeros_like(pred), np.ones_like(pred)])
    groupedPred = np.reshape(normalizedPred, (10, 20))
    return groupedPred.sum(axis=1)


def init_least_squares(X):
    XT = np.transpose(X)
    temp1 = np.matmul(XT, X)
    temp2 = np.linalg.pinv(temp1)
    LSTerm = np.matmul(temp2, XT)
    return LSTerm


def runs():
    confusion_matrix = np.empty((10, 10))
    training = load_train()
    LSTerm = init_least_squares(training)

    for classifier in range(10):
        weights = fit(classifier, LSTerm)
        y = pred(weights)
        confusion_matrix[classifier] = y

    print(confusion_matrix)
    return confusion_matrix


def main():
    confusion_matrix = runs()
    sns.heatmap(confusion_matrix, annot=True, linewidths=.5)
    plt.show()


main()
