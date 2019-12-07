import numpy
import numpy as np
import matplotlib.pyplot as plt
from utils import load_train, load_test, plot_cm, most_frequet

TRAINING_PATH = "Train"
TESTING_PATH = "Test"

K = 10


def init_clusters(set):
    original_set = set.copy()
    clusters = np.empty((K))
    index = np.random.choice(set.shape[0], 1)[0]

    for i in range(K):
        point = set[index]
        set = np.delete(set, index, axis=0)
        index = np.argmax(np.linalg.norm(point - set, axis=1))
        clusters[i] = index

    return np.take(original_set, clusters.astype(int), axis=0)


def k_means(set, means):
    while(True):
        new_means = np.zeros(means.shape)
        difference = set[:, np.newaxis] - means
        norms = np.linalg.norm(difference, axis=2)
        belongs = np.argmin(norms, axis=1)

        for i in range(K):
            slice = set[belongs == i]
            new_means[i] = (set[belongs == i]).mean(
                axis=0) if slice.size > 0 else 0
        if(np.array_equal(new_means, means)):
            return new_means, belongs
        means = new_means.copy()


def cluster(x, means):
    difference = x[:, np.newaxis] - means
    norms = np.linalg.norm(difference, axis=2)
    belongs = np.argmin(norms, axis=1)
    return belongs


def main():
    train = np.where(load_train(TRAINING_PATH, (2400, 784)) > 140, 1, 0)
    test = np.where(load_test(TESTING_PATH, (200, 784)) > 140, 1, 0)
    for _ in range(2):
        clusters = init_clusters(train)
        means, belongs = k_means(train, clusters)
        belongs = belongs.reshape(10, 240)
        cuts = most_frequet(belongs)
        if(cuts.size < K):
            continue
        sorted_means = means[cuts.argsort()]
        predicted = cluster(test, sorted_means)
        actual = np.repeat(np.arange(10), 20)
        print(predicted, cuts, cuts.argsort(), np.linalg.norm(
            means, axis=1), np.linalg.norm(sorted_means, axis=1))


main()
