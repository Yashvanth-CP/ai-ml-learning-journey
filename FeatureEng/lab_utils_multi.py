import numpy as np

def zscore_normalize_features(X):
    """
    Normalize the features using Z-score normalization.
    Returns:
        X_norm : normalized features
        mu     : mean of each feature
        sigma  : standard deviation of each feature
    """
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)

    # Avoid division by zero
    sigma[sigma == 0] = 1

    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma


def compute_cost(X, y, w, b):
    m = len(y)
    predictions = X @ w + b
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost


def compute_gradient(X, y, w, b):
    m = len(y)

    predictions = X @ w + b
    error = predictions - y

    dj_dw = (1 / m) * (X.T @ error)
    dj_db = (1 / m) * np.sum(error)

    return dj_dw, dj_db


def run_gradient_descent_feng(X, y, iterations=1000, alpha=0.01):
    """
    Performs gradient descent for linear regression.

    Returns:
        w : learned weights
        b : learned bias
    """
    m, n = X.shape

    w = np.zeros(n)
    b = 0.0

    for i in range(iterations):

        dj_dw, dj_db = compute_gradient(X, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i % max(1, iterations // 10) == 0:
            cost = compute_cost(X, y, w, b)
            print(f"Iteration {i:6d}: Cost = {cost:.6f}")

    return w, b