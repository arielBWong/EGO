import numpy as np
from bayesian_optimization_util import plot_approximation, plot_acquisition
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt

noise = 0.1
bounds = np.array([[0, 0.5]])


def f(X, noise=noise):
    return (6 * X - 2) ** 2 * np.sin(12 * X - 4) #+ noise * np.random.randn(*X.shape)


X_init = np.array([[0], [0.1], [0.2], [0.3], [0.4], [0.5]])
Y_init = f(X_init)


X = np.array([x for x in np.linspace(0, 0.5, 300)]).reshape(-1, 1)
Y = f(X, 0)

plt.plot(X, Y, 'r--', lw=2, label='Noise free objective')
plt.plot(X, f(X), 'bx', lw=1, alpha = 0.1, label='Noisy samples')
plt.plot(X_init, Y_init, 'kx', mew=3, label='Initial Sample')
plt.legend()
plt.show()



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


m52 = ConstantKernel(1.0) * Matern(length_scale=0.01, nu=0.15)
gpr = GaussianProcessRegressor(kernel=m52, n_restarts_optimizer=5)

#kernel = ConstantKernel(constant_value = 1, constant_value_bounds=(1e-4, 1e4)) * RBF(length_scale=0.05, length_scale_bounds=(1e-4, 1e4))
#gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2)



X_sample = X_init
Y_sample = Y_init

n_iter = 10

plt.figure(figsize=(12, n_iter * 3))
plt.subplots_adjust(hspace=0.4)

for i in range(n_iter):
    gpr.fit(X_sample, Y_sample)
    X_next = propose_location(expected_improvement, X_sample, Y_sample, gpr, bounds)

    Y_next = f(X_next, noise)

    # Plot samples, surrogate function, noise-free objective and next sampling location
    plt.subplot(n_iter, 2, 2 * i + 1)
    plot_approximation(gpr, X, Y, X_sample, Y_sample, X_next, show_legend=i == 0)
    plt.title(f'Iteration {i + 1}')

    plt.subplot(n_iter, 2, 2 * i + 2)
    plot_acquisition(X, expected_improvement(X, X_sample, Y_sample, gpr), X_next, show_legend=i == 0)


    # Add sample to previous samples
    X_sample = np.vstack((X_sample, X_next))
    Y_sample = np.vstack((Y_sample, Y_next))

plt.show()






