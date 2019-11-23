import numpy as np
import matplotlib.pyplot as plt

TRAINING_PATH = "Train"
TESTING_PATH = "Test"


def load_train():
    trainSet = np.empty((2400, 784))
    for file in range(1, 2401):
        path = TRAINING_PATH + "/" + str(file) + ".jpg"
        image = plt.imread(path).flatten()
        trainSet[file - 1] = np.divide(image, 255)
    return trainSet


def load_test():
    testSet = np.empty((200, 784))
    for file in range(1, 201):
        path = TESTING_PATH + "/" + str(file) + ".jpg"
        image = plt.imread(path).flatten()
        testSet[file - 1] = np.divide(image, 255)
    return testSet


def means(set):
    return set.reshape(10, 240, 784).mean(axis=1)


def variances(means, set):
    output = np.square(set.reshape(10, 240, 784).std(axis=1))
    output[output < 0.01] = 0.01
    return output


def normal_dist(x, means, variances):
    first_factor = np.power(np.sqrt(2 * np.pi * variances), -1)
    second_factor = np.exp(-np.square(x - means) / (2 * variances))
    normal_dist_values = np.multiply(first_factor, second_factor)
    output = normal_dist_values.prod(axis=1)

    return np.argmax(output)


def plot_cm(data):
    rows, cols = data.shape
    fig, ax = plt.subplots()
    ax.imshow(data, cmap="bone_r", interpolation='nearest')
    ax.set_xticks(np.arange(len(data)))
    ax.set_yticks(np.arange(len(data)))
    [ax.text(j, i, data[i, j],
             ha="center", va="center", color="w") for i in range(rows) for j in range(cols)]

    fig.tight_layout()
    plt.savefig("NaiveBayes_Confusion.jpg")
    plt.show()


def main():
    train = load_train()
    mean_values = means(train)
    variance_values = variances(mean_values, train)
    test = load_test()

    output = np.zeros(200)
    for i in range(len(test)):
        output[i] = normal_dist(test[i], mean_values, variance_values)

    reshaped = output.reshape(10, 20)
    confusion_matrix = np.zeros((100))
    for i in range(10):
        unique, counts = np.unique(reshaped[i], return_counts=True)
        np.put(confusion_matrix, unique.astype(int) + i*10, counts)

    confusion_matrix = confusion_matrix.reshape(10, 10)

    plot_cm(confusion_matrix)


main()
