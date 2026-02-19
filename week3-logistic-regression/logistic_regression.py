import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict(X, w, b):
    z = np.dot(X, w) + b
    return sigmoid(z)


def compute_cost(X, y, w, b):
    m = X.shape[0]
    predictions = predict(X, w, b)

    cost = - (1/m) * np.sum(
        y * np.log(predictions) + 
        (1 - y) * np.log(1 - predictions)
    )

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
