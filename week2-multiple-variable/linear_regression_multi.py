import numpy as np

def predict(X, w, b):
    """
    X: matrix (m, n)
    w: vector (n,)
    b: scalar
    """
    return np.dot(X, w) + b


def compute_cost(X, y, w, b):
    m = X.shape[0]
    predictions = predict(X, w, b)
    cost = np.sum((predictions - y) ** 2) / (2 * m)
    return cost


def gradient_descent(X, y, w, b, alpha, iterations):
    m = X.shape[0]

    for _ in range(iterations):
        predictions = predict(X, w, b)

        dj_dw = np.dot(X.T, (predictions - y)) / m
        dj_db = np.sum(predictions - y) / m

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

    return w, b
