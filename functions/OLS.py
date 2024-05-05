import cvxpy as cp
import numpy as np


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

    # add intercept
    factRet = np.hstack([np.ones((len(factRet), 1)), factRet])
    xtx = np.dot(factRet.T, factRet)
    xty = np.dot(factRet.T, returns)
    beta = np.dot(np.linalg.inv(xtx), xty)

    # return mu, Q
