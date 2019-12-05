import numpy as np
from scipy.stats import norm
from sklearn.utils.validation import check_array


# this will be the evaluation function that is called each time
from pymop.factory import get_problem_from_func


def expected_improvement(X, X_sample, Y_sample, gpr, gpr_g=None, xi=0.01):

    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)

    # why previous testing on multi-variable problems are not reporting problems?
    # mu = mu.reshape(-1, n_val)
    sigma = sigma.reshape(-1, 1)

    if gpr_g != None:
        mu_gx, sigma_gx = gpr_g.predict(X, return_std=True)

        with np.errstate(divide='warn'):
            pf = norm.cdf((0 - mu_gx) / sigma_gx)

        # check whether there is feasible solutions
        sample_n = X_sample.shape[0]
        a = np.linspace(0, sample_n - 1, sample_n, dtype=int)
        mu_g, sigma_g = gpr_g.predict(X_sample)
        mu_g[mu_g <= 0] = 0
        mu_cv = mu_g.sum(axis=1)
        infeasible = np.nonzero(mu_cv)
        feasible = np.setdiff1d(a, infeasible)

        if len(feasible) != 0:

            # if there is feasible solutions
            feas_f = Y_sample[feasible, :]
            mu_sample_opt = np.min(feas_f)
        else:
            return pf

    else:
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
    # input should be in the right range of defined problem
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

    g = part1 + part2 + part3 - 5
    return x, g

def Branin_5_prange_setting(x):

    x = check_array(x)

    # assume that values in x is in range [0, 1]
    if np.any(x > 1) or np.any(x < 0):
        raise Exception('Input range error, this Branin input should be in range [0, 1]')
        exit(1)

    x[:, 0] = -5 + (10 - (-5)) * x[:, 0]
    x[:, 1] = 0 + (15 - 0) * x[:, 1]
    return x


def Branin_5_f(x):
    x = check_array(x)
    x1 = np.atleast_2d(x[:, 0]).reshape(-1, 1)
    x2 = np.atleast_2d(x[:, 1]).reshape(-1, 1)

    # minimization
    f = -(x1 - 10.0)**2 - (x2 - 15.)**2
    return x, f


# this acqusition function on G should be refactored
def acqusition_function(x, out, X_sample, Y_sample, gpr, gpr_g, X_mean, X_std, xi=0.01):


    dim = X_sample.shape[1]
    x = np.atleast_2d(x).reshape(-1, dim)

    # wrap EI method, use minus to minimize
    out["F"] = -expected_improvement(x, X_sample, Y_sample, gpr, gpr_g, xi=0.01)



