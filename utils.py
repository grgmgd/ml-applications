import numpy as np
import matplotlib.pyplot as plt


def plot_cm(data, filepath):
    rows, cols = data.shape
    fig, ax = plt.subplots()
    ax.imshow(data, cmap="bone_r", interpolation='nearest')
    ax.set_xticks(np.arange(len(data)))
    ax.set_yticks(np.arange(len(data)))
    [ax.text(j, i, data[i, j],
             ha="center", va="center", color="w") for i in range(rows) for j in range(cols)]

    fig.tight_layout()
    plt.savefig(filepath)


def load_train(TRAINING_PATH, size):
    trainSet = np.empty(size)
    for file in range(1, size[0] + 1):
        path = TRAINING_PATH + "/" + str(file) + ".jpg"
        image = plt.imread(path).flatten()
        trainSet[file - 1] = image
    return trainSet


def load_test(TESTING_PATH, size):
    testSet = np.empty(size)
    for file in range(1, size[0] + 1):
        path = TESTING_PATH + "/" + str(file) + ".jpg"
        image = plt.imread(path).flatten()
        testSet[file - 1] = image
    return testSet
