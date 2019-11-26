from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

np.random.seed(1)


def f(x):
    y = (6 * x - 2) ** 2 * np.sin(12 * x - 4)
    return y.ravel()


def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    '''
    Compute EI for paramenter X given samples
    :param X: points at which EI shall be calculated
    :param X_sample: Sample locations (? dimension)
    :param Y_sample: Sample values (? dimension)
    :param gpr: a Gaussian process regressor fitted to samples
    :param xi: exploration and exploitation trade off parameter
    :return: Expected improvement at points X
    '''
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)

    sigma = sigma.reshape(-1, 1)

    mu_sample_opt = np.max(mu_sample)

    with np.errstate(divide = 'warn'):
        imp = mu - mu_sample_opt -xi
        Z = imp/sigma

        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei


def propose_location(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=25):
    '''
    propose the next sampling position by optimizing the acquisition functions
    :param acquisition: acquisition function/objective function
    :param X_sample: sample location
    :param Y_sample: sample value
    :param gpr: regressor
    :param bounds:
    :param n_restarts:
    :return: Location of the maximium value of the acquisition function
    '''
    dim = X_sample.shape[1]
    min_value = 0
    min_x = None

    def min_obj(X):
        return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr)

    for x0 in np.random.uniform([0.], [0.5], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds = bounds, method='L-BFGS-B')
        if res.fun < min_value:
            min_value = res.fun
            min_x = res.x

    return min_x.reshape(-1, 1)



if __name__ == '__main__':
    train_X = np.atleast_2d([0, 0.2, 0.4, 0.6, 0.7, 1]).T
    print(train_X)
    train_y = f(train_X)
    print(train_y)

    test_X = np.atleast_2d(np.linspace(0, 1, 100)).T
    test_Y_real = f(test_X)

    # fit GPR
    kernel = ConstantKernel(1.0) * RBF(1)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

    gpr.fit(train_X, train_y)
    mu, cov = gpr.predict(test_X, return_cov=True)
    test_y = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))

    # plotting
    plt.figure()
    plt.title("l=%.1f sigma_f=%.1f" % (gpr.kernel_.k2.length_scale, gpr.kernel_.k1.constant_value))
    plt.fill_between(test_X.ravel(), test_y + uncertainty, test_y - uncertainty, alpha=0.1)
    plt.plot(test_X, test_y, label="predict")
    plt.plot(test_X, test_Y_real, label='real_value')
    plt.scatter(train_X, train_y, label="train", c="red", marker="x")
    plt.legend()
    plt.show()
