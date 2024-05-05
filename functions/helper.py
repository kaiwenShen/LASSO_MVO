import numpy as np
import pandas as pd


def _geo_mean(iterable):
    a = np.array(iterable)
    return a.prod() ** (1.0 / len(a))


def geometric_return(returns):
    if isinstance(returns, pd.DataFrame):
        return returns.apply(lambda x: _geo_mean(x + 1) - 1, axis=0)
    elif isinstance(returns, np.ndarray):
        if returns.ndim == 1:
            return _geo_mean(returns + 1) - 1
        else:
            return np.apply_along_axis(_geo_mean, 0, returns + 1) - 1
    else:
        raise ValueError("Input must be a DataFrame or numpy array")


def r_squared(y, y_hat):
    SS_res = np.sum((y - y_hat) ** 2, axis=0)
    SS_tot = np.sum((y - np.mean(y)) ** 2, axis=0)
    return 1 - SS_res / SS_tot


def adjusted_r_squared(y, y_hat, X_shape):
    assert len(X_shape) == 2
    n = X_shape[0]
    k = X_shape[1]
    return 1 - (1 - r_squared(y, y_hat)) * (n - 1) / (n - k - 1)
