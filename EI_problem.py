import numpy as np
from scipy.stats import norm
from sklearn.utils.validation import check_array


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

    n_val = X.shape[1]

    # why previous testing on multi-variable problems are not reporting problems?
    # mu = mu.reshape(-1, n_val)
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


def Branin_g(x):
    x = check_array(x)
    x1 = np.atleast_2d(x[:, 0]).reshape(-1, 1)
    x2 = np.atleast_2d(x[:, 1]).reshape(-1, 1)

    f = -(x1 - 10.) ** 2 - (x2 - 15.) ** 2
    a = 1.0
    b = 5.1 / (4 * np.pi ** 2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)

    part1 = a * (x2 - b * x1 ** 2 + c * x1 - 6.0) ** 2.0
    part2 = s * (1 - t) * np.cos(x1)
    part3 = s

    g = part1 + part2 + part3 -5
    return g

def Branin_5_f(x):
    x = check_array(x)
    x1 = np.atleast_2d(x[:, 0]).reshape(-1, 1)
    x2 = np.atleast_2d(x[:, 1]).reshape(-1, 1)

    # minimization
    f = -(x1 - 10.0)**2 - (x2 -15.)**2

def acqusition_function(x, out, X_sample, Y_sample, gpr, xi=0.01):


    dim = X_sample.shape[1]
    x = np.atleast_2d(x).reshape(-1, dim)

    # wrap EI method, use minus to minimize
    out["F"] = -expected_improvement(x, X_sample, Y_sample, gpr, xi=0.01)
    out["G"] = Branin_g(x)


