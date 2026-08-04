import numpy as np
import matplotlib.pyplot as plt


def plt_one_addpt_onclick(x, y, w, b, logistic=False):
    fig, ax = plt.subplots()

    # Plot training data
    pos = y == 1
    neg = y == 0

    ax.scatter(x[pos], y[pos], marker='x', s=80, label='y=1')
    ax.scatter(x[neg], y[neg], marker='o', s=80, label='y=0')

    # Plot prediction curve
    x_min = np.min(x) - 1
    x_max = np.max(x) + 1

    x_plot = np.linspace(x_min, x_max, 100)

    if logistic:
        z = w * x_plot + b
        y_plot = 1 / (1 + np.exp(-z))
    else:
        y_plot = w * x_plot + b

    ax.plot(x_plot, y_plot, label='Model')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Logistic Regression')
    ax.legend()
    ax.grid(True)

    return fig