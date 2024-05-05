import numpy as np
from functions.OLS import OLS


def FF(returns, factRet, lambda_, K):
    """
    % Use this function to calibrate the Fama-French 3-factor model. Note
    % that you will not use lambda or K in this model (lambda is for LASSO,
    % and K is for BSS).
    """

    # *************** WRITE YOUR CODE HERE ***************
    # ----------------------------------------------------------------------

    # mu =          % n x 1 vector of asset exp. returns
    # Q  =          % n x n asset covariance matrix
    # ----------------------------------------------------------------------
    # market excess return
    mu, Q, adj_r2 = OLS(returns, factRet.loc[:, ['Mkt_RF', 'SMB', 'HML']], lambda_, K)
    return mu, Q, adj_r2
