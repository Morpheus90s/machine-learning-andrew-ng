import numpy as np

def add_polynomial_features(X, degree):
    """
    Create polynomial features up to given degree.
    """
    X_poly = X.copy()

    for d in range(2, degree + 1):
        X_poly = np.column_stack((X_poly, X ** d))

    return X_poly
