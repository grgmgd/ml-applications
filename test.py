import numpy as np


def most_frequet(set):
    cuts = np.array([])
    for i in range(set.shape[0]):
        slice = set[i]
        for value in cuts:
            slice = slice[slice != value]
        value = np.bincount(slice).argmax()
        cuts = np.append(cuts, value)


set = np.array([[1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]])

most_frequet(set)
