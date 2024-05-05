# import cvxpy as cp
import numpy as np
from functions.helper import r_squared, adjusted_r_squared
from scipy.stats import gmean


def OLS(returns, factRet, lambda_, K):
    """
    % Use this function to perform an OLS regression. Note that you will
    % not use lambda or K in this model (lambda is for LASSO, and K is for
    % BSS).
    """

    # *************** WRITE YOUR CODE HERE ***************
    # ----------------------------------------------------------------------

    # mu =          % n x 1 vector of asset exp. returns
    # Q  =          % n x n asset covariance matrix
    # ----------------------------------------------------------------------
    # deduct the risk free rate
    returns = returns.sub(factRet['Mkt_RF'], axis=0)
    # add intercept
    factRet = np.hstack([np.ones((len(factRet), 1)), factRet])
    xtx = np.dot(factRet.T, factRet)
    xty = np.dot(factRet.T, returns)
    try:
        beta = np.dot(np.linalg.inv(xtx), xty)
    except np.linalg.LinAlgError:
        # then we try sudo inverse
        beta = np.dot(np.linalg.pinv(xtx), xty)
    factor_mu = gmean(factRet + 1) - 1
    mu = np.dot(factor_mu, beta)
    print(f'OLS insample R2: \n{r_squared(returns, np.dot(factRet, beta))}')
    print(f'OLS insample adj R2: \n{adjusted_r_squared(returns, np.dot(factRet, beta), factRet.shape)}')
    # compute the vcov matrix with formula Q = B'FB + delta
    residuals = returns - np.dot(factRet, beta)
    delta = np.diag(np.var(residuals, axis=0))
    F = np.cov(factRet, rowvar=False)
    Q = np.dot(np.dot(beta.T, F), beta) + delta
    return mu, Q
