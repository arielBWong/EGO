from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from random import Random
from time import time
from math import cos
from math import pi
from inspyred import ec
from inspyred.ec import terminators
from bayesian_optimization_util import plot_approximation, plot_acquisition
from matplotlib.pyplot import figure



np.random.seed(20)


def f(x):
    y = (6 * x - 2) ** 2 * np.sin(12 * x - 4)
    return y.ravel()


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

    # Needed for noise-based model,
    # otherwise use np.max(Y_sample).
    # See also section 2.4 in [...]
    mu_sample_opt = np.max(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        print(imp.shape)
        print(sigma.shape)
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei

def generate_EI(random, args):
    size = args.get('num_inputs', 1)
    return [random.uniform(0, 1.0) for i in range(size)]


def EA_location_selection(acquisition, X_sample, Y_sample, gpr, bounds, n_gens=10):
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

    # maxmize EI is to minimize negative EI
    def min_obj(X, args):
        X = np.atleast_2d(X)
        fitness = []
        for x in X:
            fitness.append(-acquisition(x.reshape(-1, dim), X_sample, Y_sample, gpr))
        return fitness

    # initialize population
    rand = Random()
    rand.seed(1)
    es = ec.ES(rand)
    es.terminator = terminators.evaluation_termination
    final_pop = es.evolve(generator=generate_EI,
                          evaluator=min_obj,
                          pop_size=10,
                          maximize=False,
                          bounder=ec.Bounder(0, 1),
                          max_evaluations=n_gens,
                          mutation_rate=0.25,
                          num_inputs=1)
    final_pop.sort()
    print(final_pop)
    print (final_pop[0])

    return final_pop[0]


if __name__ == '__main__':

    n_iter = 1
    train_X = np.atleast_2d([0, 0.5, 1]).T
    train_y = f(train_X)

    std_train_x = np.std(train_X)
    mean_train_x = np.mean(train_X)

    train_X = (train_X - mean_train_x) / std_train_x
    print(train_X)
    print(f(train_X))

    std_train_y = np.std(train_y)
    mean_train_y = np.mean(train_y)

    train_y = (train_y - mean_train_y) / std_train_y
    print(train_y)

    test_X = np.atleast_2d(np.linspace(0, 1, 100)).T
    test_Y_real = f(test_X)

    test_X = (test_X - mean_train_x) / std_train_x

    # fit GPR
    a = np.std(train_X)
    kernel = RBF(a, (np.exp(-1), np.exp(1)))
    # kernel = RBF(a, (a, a))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=30, alpha=0)

    gpr.fit(train_X, train_y)
    mu, cov = gpr.predict(test_X, return_cov=True)
    test_y = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))

    # Obtain next sampling point from the acquisition function (expected_improvement)
    EI = expected_improvement(test_X, train_X, train_y, gpr)

    #Xnext = EA_location_selection(expected_improvement, train_X, train_y, gpr, bounds, n_gens=10)

    # end of next location proposing method
    test_X = test_X * std_train_x + mean_train_x
    test_y = test_y * std_train_y + mean_train_y

    train_X = train_X * std_train_x + mean_train_x
    train_y = train_y * std_train_y + mean_train_y

    # plotting
    plt.figure(figsize=(10, 5))

    plt.subplot(n_iter, 2, 1)
    plt.title("l=%.1f" % (gpr.kernel_.length_scale))
    plt.fill_between(test_X.ravel(), test_y + uncertainty, test_y - uncertainty, alpha=0.5)
    plt.plot(test_X, test_y, label="predict")
    plt.plot(test_X, test_Y_real, label='real_value')
    plt.scatter(train_X, train_y, label="train", c="red", marker="x")
    plt.legend()

    plt.subplot(n_iter, 2, 2)
    EI = expected_improvement(test_X.reshape(-1, 1), train_X, train_y, gpr)
    plot_acquisition(test_X, EI, 0, show_legend=0 == 0)
    plt.show()

