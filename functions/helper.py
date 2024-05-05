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


def cal_ols_beta(returns, factRet):
    xtx = np.dot(factRet.T, factRet)
    xty = np.dot(factRet.T, returns)
    try:
        beta = np.dot(np.linalg.inv(xtx), xty)
    except np.linalg.LinAlgError:
        # then we try sudo inverse
        beta = np.dot(np.linalg.pinv(xtx), xty)
    residuals = returns - np.dot(factRet, beta)
    return beta, residuals


def cal_Q(beta, factRet, residuals):
    """compute the vcov matrix with formula Q = B'FB + delta"""
    F = np.cov(factRet, rowvar=False)
    delta = np.diag(np.var(residuals, axis=0))
    Q = np.dot(np.dot(beta.T, F), beta) + delta
    return Q
