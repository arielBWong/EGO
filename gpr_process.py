from __future__ import print_function

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from krige_dace import krige_dace
import time

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



    import pyKriging
    from pyKriging.krige import kriging
    from pyKriging.samplingplan import samplingplan
    from pyKriging.CrossValidation import Cross_Validation
    from pyKriging.utilities import saveModel

    from numpy import genfromtxt


    # x = genfromtxt('x.csv', delimiter=',')
    # y = genfromtxt('y.csv', delimiter=',')
    # x = np.atleast_2d(x)
    # y = np.atleast_2d(y).reshape(-1, 1)



    # train_X = np.atleast_2d([0, 0.2, 0.56, 0.23, 0.14, 0.3, 0.4, 0.5, 1.2, 0.8, 0.6, 0.7, 1]).T
    #print(train_X)
    #train_y = f(train_X)
    #train_y = np.atleast_2d(train_y).reshape(-1, 1)
    #print(train_y)

    start = time.time()
    x = np.linspace(0, 1, 500)
    # ytrg = ((6 * xtrg - 2). ^ 2). * sin(12 * xtrg - 4);
    y = (6 * x - 2)**2 * np.sin(12 * x - 4)
    x = np.atleast_2d(x).reshape(-1, 1)
    y = np.atleast_2d(y).reshape(-1, 1)


    mykriging = krige_dace(x, y)
    mykriging.train()
    # pred_y, _ = mykriging.predict(x)
    # print(pred_y)
    end = time.time()
    print('%0.4f' % (end-start))



    '''
    train_X = np.atleast_2d([0, 0.2, 0.3, 0.4, 0.5, 1.2, 0.8, 0.6, 0.7, 1]).T
    print(train_X)
    train_y = f(train_X)
    train_y = np.atleast_2d(train_y).reshape(-1, 1)
    print(train_y)

    test_X = np.atleast_2d([0, 0.2, 0.3, 0.4, 0.5, 1.2, 0.8, 0.6, 0.7, 1]).T
    # test_X = np.atleast_2d(np.linspace(0, 1, 100)).T
    test_Y_real = f(test_X)

    sm = KRG(theta0=[1e-2], print_global=False)
    sm.set_training_values(train_X, train_y)
    sm.train()

    test_y = sm.predict_values(test_X)
    uncertainty = sm.predict_variances(test_X)

 


    # plotting
    plt.figure()
    # plt.title("l=%.1f sigma_f=%.1f" % (gpr.kernel_.k2.length_scale, gpr.kernel_.k1.constant_value))
    # plt.fill_between(test_X.ravel(), test_y.ravel() + uncertainty.ravel(), test_y.ravel() - uncertainty.ravel(), alpha=0.8)

    plt.scatter(train_y.ravel(), pred_y.ravel(), label="pred", c="blue", marker="x")
    plt.legend()
    plt.show()
    '''


    
