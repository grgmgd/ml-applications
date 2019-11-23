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
