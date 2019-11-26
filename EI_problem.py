import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm
#from unitFromGPR import one_iter_from_gpr

# this will be the evaluation function that is called each time
from pymop.factory import get_problem_from_func


def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    '''
    Computes the EI at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.

    Args:
        X: Points at which EI shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.

    Returns:
        Expected improvements at points X.
    '''
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)

    mu = mu.reshape(-1, 1)
    sigma = sigma.reshape(-1, 1)

    # for minization purpose, chose best point as the np.min(Y_sample)
    mu_sample_opt = np.min(Y_sample)

    with np.errstate(divide='warn'):
        imp = mu_sample_opt - mu
        # print(imp.shape)
        # print(sigma.shape)
        Z = imp / sigma
        ei1 = imp * norm.cdf(Z)
        ei1[sigma == 0.0] = 0.0
        ei2 = sigma * norm.pdf(Z)
        ei = (ei1 + ei2)


    return ei



def acqusition_function(x, out, X_sample, Y_sample, gpr, xi=0.01):


    dim = X_sample.shape[1]
    x = np.atleast_2d(x).reshape(-1, dim)

    # wrap EI method, use minus to minimize
    out["F"] = -expected_improvement(x, X_sample, Y_sample, gpr, xi=0.01)


