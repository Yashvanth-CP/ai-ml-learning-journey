import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_cost_logistic(x, y, w, b):
    m = len(x)

    z = w * x + b
    f_wb = sigmoid(z)

    epsilon = 1e-15
    f_wb = np.clip(f_wb, epsilon, 1 - epsilon)

    loss = -y * np.log(f_wb) - (1 - y) * np.log(1 - f_wb)

    return np.sum(loss) / m


def compute_logistic_loss(x, y, w, b):
    z = w * x + b
    f_wb = sigmoid(z)

    epsilon = 1e-15
    f_wb = np.clip(f_wb, epsilon, 1 - epsilon)

    loss = -y * np.log(f_wb) - (1 - y) * np.log(1 - f_wb)

    return loss


def soup_bowl():
    w = np.linspace(-10, 10, 100)
    b = np.linspace(-10, 10, 100)

    W, B = np.meshgrid(w, b)

    # Example data
    x = np.array([0., 1., 2., 3., 4., 5.])
    y = np.array([0., 0., 0., 1., 1., 1.])

    Z = np.zeros_like(W)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Z[i, j] = compute_cost_logistic(
                x, y, W[i, j], B[i, j]
            )

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(W, B, Z, cmap='viridis')

    ax.set_xlabel('w')
    ax.set_ylabel('b')
    ax.set_zlabel('Cost')
    ax.set_title('Logistic Regression Cost Function')

    plt.show()


def plt_simple_example(x_train, y_train):
    plt.figure(figsize=(8, 5))

    plt.scatter(
        x_train,
        y_train,
        marker='x',
        s=80
    )

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Training Data')

    plt.yticks([0, 1])

    plt.grid(True)
    plt.show()


def plt_logistic_squared_error(x_train, y_train):
    w = 1
    b = -3

    x_values = np.linspace(-10, 10, 200)

    f_values = sigmoid(w * x_values + b)

    plt.figure(figsize=(8, 5))

    plt.plot(
        x_values,
        f_values,
        label='Sigmoid prediction'
    )

    plt.scatter(
        x_train,
        y_train,
        marker='x',
        s=80,
        label='Training data'
    )

    plt.xlabel('x')
    plt.ylabel('Prediction')
    plt.title('Logistic Regression')

    plt.yticks([0, 1])

    plt.grid(True)
    plt.legend()

    plt.show()


def plt_two_logistic_loss_curves():
    x = np.linspace(0.01, 0.99, 200)

    loss_y1 = -np.log(x)
    loss_y0 = -np.log(1 - x)

    plt.figure(figsize=(8, 5))

    plt.plot(
        x,
        loss_y1,
        label='y = 1'
    )

    plt.plot(
        x,
        loss_y0,
        label='y = 0'
    )

    plt.xlabel('Prediction f(x)')
    plt.ylabel('Logistic Loss')

    plt.title('Logistic Loss Curves')

    plt.grid(True)
    plt.legend()

    plt.show()


def plt_logistic_cost(x_train, y_train):
    w_values = np.linspace(-10, 10, 200)

    costs = []

    b = -3

    for w in w_values:
        cost = compute_cost_logistic(
            x_train,
            y_train,
            w,
            b
        )

        costs.append(cost)

    plt.figure(figsize=(8, 5))

    plt.plot(
        w_values,
        costs
    )

    plt.xlabel('w')
    plt.ylabel('Cost')

    plt.title('Logistic Regression Cost')

    plt.grid(True)

    plt.show()

    return costs