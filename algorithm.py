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
    training = load_train()
    testing = load_test()
    LSTerm = init_least_squares(training)
    weights = np.empty((10, 785))

    for classifier in range(10):
        weights[classifier] = fit(classifier, LSTerm)

    y = pred(weights, testing)
    return y


def plot_cm(data):
    rows, cols = data.shape
    fig, ax = plt.subplots()
    ax.imshow(data, cmap="bone_r", interpolation='nearest')
    ax.set_xticks(np.arange(len(data)))
    ax.set_yticks(np.arange(len(data)))
    [ax.text(j, i, data[i, j],
             ha="center", va="center", color="w") for i in range(rows) for j in range(cols)]

    fig.tight_layout()
    plt.savefig("Confusion.jpg")
    plt.show()


def main():
    confusion_matrix = runs()
    plot_cm(confusion_matrix)


main()
