from sklearn.gaussian_process.kernels import RBF
import numpy as np
from sklearn.utils.validation import check_X_y, check_array
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.linalg import cholesky, cho_solve, solve_triangular
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class GPM:

    def __init__(self, optimize=True):
        self.train_x, self.train_y = None, None
        self.optimize = optimize
        self.is_fit = False
        self.R = None
        self.R_inv = None
        self.params = {"theta": 10}
        self.bounds = [(1e-4, 1e4)]

    # this kernel is only for 1-D
    def kernel(self, kernel_x, kernel_y):
        dist = np.sum(kernel_x ** 2, 1).reshape(-1, 1) + np.sum(kernel_y ** 2, 1) - 2 * np.dot(kernel_x, kernel_y.T)
        R = np.exp(- self.params['theta'] * dist)
        return R

    def fit(self, x, y):
        # store train data
        x, y = check_X_y(x, y, multi_output=True, y_numeric=True)
        self.train_x = x
        self.train_y = y
        self.is_fit = True


        # hyper parameters optimization
        def negative_log_likelihood_loss(params):
            self.params["theta"] = params[0]
            n_samples = self.train_x.shape[0]

            self.R = self.kernel(self.train_x, self.train_x)
            self.R_inv = np.linalg.inv(self.R + 1e-8 * np.eye(self.train_x.shape[0]))

            # calculate equation (6)
            mu_ones = np.atleast_2d(np.ones(self.train_y.shape[0])).reshape(-1, 1)
            mu_first = (mu_ones.T.dot(self.R_inv)).dot(mu_ones)
            mu_first_inv = 1 / mu_first
            mu_hat = mu_first_inv * mu_ones.T.dot(self.R_inv).dot(self.train_y)  # eq(6) this should be a scalar
            self.mu_hat = mu_hat

            # calculate equation (7)
            n_samples = train_x.shape[0]
            phi_2 = 1 / n_samples * (self.train_y - mu_hat).T.dot(self.R_inv).dot(self.train_y - mu_hat)
            self.phi_2 = phi_2

            # negative loglikelihood
            lml = 1/2 * (n_samples * np.log(self.phi_2)) - np.linalg.slogdet(self.R)[1]
            print('print objective function value: %.2f ' % lml)
            return lml

        negative_log_likelihood_loss([self.params['theta']])
        '''
        # conduct optimization
        if self.optimize:
            res = minimize(negative_log_likelihood_loss, np.array(self.params['theta']),
                           bounds=self.bounds,
                           method='L-BFGS-B')
            self.params["theta"] = res.x[0]
        '''

    def prediction(self, X):
        if not self.is_fit:
            print("GPM Model not fit yet, means it has no training data.")
            return

        X = check_array(X)
        # Predict based on GP posterior

        # pre-calculate some parameters
        self.R = self.kernel(self.train_x, self.train_x)
        self.R_inv = np.linalg.inv(self.R + 1e-8 * np.eye(self.train_x.shape[0]))

        # calculate equation (6)
        mu_ones = np.atleast_2d(np.ones(self.train_y.shape[0])).reshape(-1, 1)
        mu_first = (mu_ones.T.dot(self.R_inv)).dot(mu_ones)
        mu_first_inv = 1 / mu_first
        mu_hat = mu_first_inv * mu_ones.T.dot(self.R_inv).dot(self.train_y)  # eq(6) this should be a scalar
        self.mu_hat = mu_hat

        # calculate equation (7)
        n_samples = train_x.shape[0]
        phi_2 = 1/n_samples * (self.train_y - mu_hat).T.dot(self.R_inv).dot(self.train_y - mu_hat)
        self.phi_2 = phi_2


        r_T = self.kernel(self.train_x, X).T  # equation (8)
        y_hat = self.mu_hat + r_T.dot(self.R_inv).dot(self.train_y - self.mu_hat)

        # calculate equation (4)
        ones = np.atleast_2d(np.ones(self.train_x.shape[0])).reshape(-1, 1)
        phi_2_new = self.phi_2 * (1 - r_T.dot(self.R_inv).dot(r_T.T) +
                                  (1 - ones.T.dot(self.R_inv).dot(r_T.T))**2 / (ones.T.dot(self.R_inv).dot(ones)))


        return y_hat, phi_2_new

def f(x):
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)


if __name__ == '__main__':
    np.random.seed(20)

    train_x = np.atleast_2d([0, 0.5, 1]).T
    train_y = f(train_x)

    test_x = np.atleast_2d(np.linspace(0, 1, 101)).T
    real_y = f(test_x)
    # test_x = np.atleast_2d([0.2]).T
    gpm = GPM()

    gpm.fit(train_x, train_y)
    mu, cov = gpm.prediction(test_x)
    test_y = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))

    plt.figure()
    plt.title("theta=%.2f" % (gpm.params["theta"]))
    plt.fill_between(test_x.ravel(), test_y + uncertainty, test_y - uncertainty, alpha=0.1)
    plt.plot(test_x, test_y, label="predict")
    plt.plot(test_x, real_y, label='ground truth')
    plt.scatter(train_x, train_y, label="train", c="red", marker="x")
    plt.legend()
    plt.show()
