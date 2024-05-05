import cvxpy as cp
import numpy as np
import gurobipy as gp
from scipy.stats import gmean

from functions.helper import cal_Q, adjusted_r_squared


def BSS(returns, factRet, lambda_, K):
    """
    % Use this function for the BSS model. Note that you will not use
    % lambda in this model (lambda is for LASSO).
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
    loss = cp.sum_squares(residual)
    # Define the constraint (sparsity constraint on beta)
    # Since cvxpy does not directly support l0-norm, we use a trick with auxiliary boolean variables
    # Constraint to make sure that the number of true entries in aux_bool does not exceed K
    sparsity_constraints = [cp.norm1(beta[:, j]) <= K for j in range(returns.shape[1])]
    problem = cp.Problem(cp.Minimize(loss), constraints=sparsity_constraints)
    problem.solve()
    factor_mu = gmean(factRet + 1) - 1
    mu = np.dot(factor_mu, beta.value)
    Q = cal_Q(beta.value, factRet, returns - factRet @ beta.value)
    adj_r2 = adjusted_r_squared(returns, np.dot(factRet, beta.value), factRet.shape)
    tol = 1e-6
    print(f'BSS avg numbers of beta >0: {np.sum(beta.value > tol) / 20}')
    return mu, Q, adj_r2
