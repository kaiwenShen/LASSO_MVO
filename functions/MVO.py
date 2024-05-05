import cvxpy as cp
import numpy as np


def MVO(mu, Q, targetRet):
    """
    % Use this function to construct your MVO portfolio subject to the
    % target return, with short sales disallowed.
    %
    % You may use quadprog, Gurobi, cvxpy or any other optimizer you are familiar
    % with. Just be sure to comment on your code to (briefly) explain your
    % procedure.


    """

    # Find the total number of assets
    n = len(mu)

    # *************** WRITE YOUR CODE HERE ***************
    # -----------------------------------------------------------------------

    # x =           % Optimal asset weights
    # ----------------------------------------------------------------------
    x = cp.Variable(n)
    # Objective function
    obj = cp.Minimize(cp.quad_form(x, Q))
    # Constraints
    constraints = [-mu @ x <= -targetRet, x >= 0, cp.sum(x) == 1]
    # Solve the problem
    prob = cp.Problem(obj, constraints)
    prob.solve()
    x = x.value
    return x
