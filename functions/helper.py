import numpy as np

def r_squared(y, y_hat):
    SS_res = np.sum((y - y_hat) ** 2, axis=0)
    SS_tot = np.sum((y - np.mean(y)) ** 2, axis=0)
    return 1 - SS_res / SS_tot


def adjusted_r_squared(y, y_hat, X_shape):
    assert len(X_shape) == 2
    n = X_shape[0]
    k = X_shape[1]
    return 1 - (1 - r_squared(y, y_hat)) * (n - 1) / (n - k - 1)
