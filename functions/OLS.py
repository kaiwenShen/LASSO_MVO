# import cvxpy as cp
import numpy as np
from functions.helper import adjusted_r_squared, cal_ols_beta, cal_Q
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
    # add intercept
    factRet = np.hstack([np.ones((len(factRet), 1)), factRet])
    # OLS calculation
    beta, residuals = cal_ols_beta(returns, factRet)
    factor_mu = gmean(factRet + 1) - 1
    mu = np.dot(factor_mu, beta)
    # Q calculation
    Q = cal_Q(beta, factRet, residuals)
    return mu, Q, adjusted_r_squared(returns, np.dot(factRet, beta), factRet.shape)
