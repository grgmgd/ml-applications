import numpy
import numpy as np
import matplotlib.pyplot as plt
from utils import load_train, plot_cm, most_frequet

TRAINING_PATH = "Train"

K = 10
EPOCHS = 30


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


def cluster(set, means):
    difference = set[:, np.newaxis] - means
    norms = np.linalg.norm(difference, axis=2)
    belongs = np.argmin(norms, axis=1)
    return belongs


def k_means(set, means):
    while(True):
        new_means = np.zeros(means.shape)
        belongs = cluster(set, means)

        for i in range(K):
            slice = set[belongs == i]
            new_means[i] = slice.mean(
                axis=0) if slice.size > 0 else 0
        if(np.array_equal(new_means, means)):
            return new_means, belongs
        means = new_means.copy()


def DBI(set, belongs, means):
    clusters = [set[belongs == C] for C in range(K)]
    variances = np.empty(10)
    for i in range(K):
        variances[i] = np.mean(np.linalg.norm(clusters[i] - means[i]))

    indicies = [((variances[i] + variances[j]) / np.linalg.norm(means[i] - means[j]))
                for i in range(K) for j in range(K) if i != j]
    return np.max(indicies)/K


def main():
    train = np.where(load_train(TRAINING_PATH, (2400, 784)) > 140, 1, 0)
    best_dbi = float('inf')
    best_clustering = None
    best_cuts = None

    epoch = 0
    while(epoch != EPOCHS):
        clusters = init_clusters(train)
        means, belongs = k_means(train, clusters)
        belongs_reshaped = belongs.reshape(10, 240)
        cuts = most_frequet(belongs_reshaped)
        if(cuts.size < K):
            continue
        epoch += 1
        dbi = DBI(train, belongs, means)
        print("Epoch: ", epoch, " With a dbi score = ", dbi)
        if(dbi < best_dbi):
            best_dbi = dbi
            best_clustering = belongs_reshaped
            best_cuts = cuts
    plot(best_clustering.reshape((10, 240)), best_cuts)


def plot(clusters, cuts):
    counts = [((clusters[i] == cuts[i]).sum()) for i in range(cuts.size)]
    fig, ax = plt.subplots()
    ax.plot(np.arange(0, K), counts)
    fig.tight_layout()
    plt.savefig("confusion/k_means/counts.jpg")
    plt.show()


main()
