import seaborn as sns
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

training = "Train"
testing = "Test"

X = np.empty((2400, 785))

for file in range(1, 2401):
    path = training + "/" + str(file) + ".jpg"
    image = np.asarray(Image.open(open(path, 'rb'))).flatten()
    image = np.append(image, 1)
    X[file - 1] = image

XT = np.transpose(X)
ft = np.matmul(XT, X)
inv = np.linalg.pinv(ft)
st = np.matmul(inv, XT)


def train(epoch):
    size = 240
    T = np.empty(2400)
    T.fill(-1)
    T[size*epoch:size*(epoch + 1):1] = 1
    weights = np.matmul(st, T)
    return weights


def test(weights, epoch):
    count = 0
    index = 0
    for file in range(1, 201):
        path = testing + "/" + str(file) + ".jpg"
        image = np.asarray(Image.open(open(path, 'rb'))).flatten()
        image = np.append(image, 1)
        value = np.matmul(weights, image)
        count += 1 if value >= 0 else 0
        if(file % 20 == 0):
            testing_values[index][epoch] = count
            count = 0
            index += 1


def runs():
    for epoch in range(10):
        weights = train(epoch)
        test(weights, epoch)


testing_values = np.empty((10, 10))
runs()
sns.heatmap(testing_values, annot=True, linewidths=.5)
plt.show()
