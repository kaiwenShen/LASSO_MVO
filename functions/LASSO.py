import cvxpy as cp
import numpy as np
from scipy.stats import gmean

from functions.helper import cal_Q


def LASSO(returns, factRet, lambda_, K):
    """
    % Use this function for the LASSO model. Note that you will not use K
    % in this model (K is for BSS).
    %
    % You should use an optimizer to solve this problem. Be sure to comment
    % on your code to (briefly) explain your procedure.


    """

    # *************** WRITE YOUR CODE HERE ***************
    # ----------------------------------------------------------------------

    # mu =          % n x 1 vector of asset exp. returns
    # Q  =          % n x n asset covariance matrix
    # ----------------------------------------------------------------------
    factRet = np.hstack([np.ones((len(factRet), 1)), factRet])
    beta = cp.Variable((factRet.shape[1], returns.shape[1]))
    residual = returns.values - factRet @ beta
    loss = cp.sum_squares(residual) + lambda_ * cp.norm(beta, 1)
    problem = cp.Problem(cp.Minimize(loss))
    problem.solve()
    factor_mu = gmean(factRet + 1) - 1
    mu = np.dot(factor_mu, beta.value)
    Q = cal_Q(beta.value, factRet, returns - factRet @ beta.value)
    return mu, Q
