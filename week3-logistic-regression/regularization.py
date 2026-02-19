import numpy as np
from logistic_regression import sigmoid

def compute_cost_reg(X, y, w, b, lambda_):
    m = X.shape[0]
    predictions = sigmoid(np.dot(X, w) + b)

    cost = - (1/m) * np.sum(
        y * np.log(predictions) + 
        (1 - y) * np.log(1 - predictions)
    )

    reg_cost = (lambda_ / (2 * m)) * np.sum(w ** 2)

    return cost + reg_cost
