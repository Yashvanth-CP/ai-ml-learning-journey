import numpy as np
import matplotlib.pyplot as plt


dlc = {
    "dlblue": "blue",
    "dlmagenta": "magenta"
}


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def plot_data(X, y, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    X = np.array(X)
    y = np.array(y)

    for i in range(len(y)):
        if y[i] == 1:
            ax.scatter(X[i, 0], X[i, 1], marker='x', s=100)
        else:
            ax.scatter(X[i, 0], X[i, 1], marker='o', s=100)

    return ax