import numpy as np


def r_squared(y, y_hat):
    SS_res = np.sum((y - y_hat) ** 2, axis=0)
    SS_tot = np.sum((y - np.mean(y)) ** 2, axis=0)
    return 1 - SS_res / SS_tot


def adjusted_r_squared(y, y_hat, X_shape):
    assert len(X_shape) == 2
    n = X_shape[0]  # number of observations
    k = X_shape[1]  # number of factors
    return 1 - (1 - r_squared(y, y_hat)) * (n-1) / (n - k - 1)


def adjusted_r_squared_w_0(y, y_hat, beta, X_shape):
    """
    this is the function that iteratively calculate the adjusted r2 for each stock in y
    since the given beta may contain 0, we should not penalize those in adjusted r_squared
    """
    tol = 1e-6
    res_adj_r2 = []
    for i in range(y.shape[1]):
        y_i = y[:, i]
        y_hat_i = y_hat[:, i]
        num_of_factors = np.sum(np.abs(beta[:, i]) > tol)
        n = X_shape[0]
        res = adjusted_r_squared(y_i, y_hat_i, (n, num_of_factors))
        res_adj_r2.append(res)
    return np.array(res_adj_r2)


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
    tol = 1e-6
    dof_penalty = np.apply_along_axis(lambda x: np.sum(np.abs(x) > tol), axis=0, arr=beta)
    dof_penalty = 1 / (48 - dof_penalty - 1)
    delta = np.diag(dof_penalty*np.var(residuals, axis=0))
    Q = np.dot(np.dot(beta.T, F), beta) + delta
    return Q


def cal_vcov(returns):
    return np.cov(returns, rowvar=False)
