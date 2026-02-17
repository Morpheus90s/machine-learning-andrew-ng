import numpy as np

def compute_cost(x, y, w, b):
    """
    Compute cost using Mean Squared Error.
    """
    m = len(x)
    total_cost = 0

    for i in range(m):
        prediction = w * x[i] + b
        total_cost += (prediction - y[i]) ** 2

    return total_cost / (2 * m)
