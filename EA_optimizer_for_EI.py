import numpy as np
import matplotlib.pyplot as plt
import optimizer_para_EI
from pymop.factory import get_problem_from_func
from EI_problem import acqusition_function
from unitFromGPR import f, mean_std_save, reverse_zscore
from scipy.stats import norm, zscore
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from EI_problem import expected_improvement
from sklearn.utils.validation import check_array
import pyDOE
import multiprocessing



def function_m(x):
    x = check_array(x)
    if x.shape[1] > 0:
        x1 = x[:, 0]
        x2 = x[:, 1:]
    else:
        x1 = x
        x2 = np.zeros((x1.shape[0], 1))

    f1 = f(x1) + 20
    f2 = 1 + np.sum((x2 - 0.5) ** 2, axis=1)
    y = np.atleast_2d(f1 + f2).T
    return y


def train_data_norm(train_x, train_y):
    mean_train_x, std_train_x = mean_std_save(train_x)
    mean_train_y, std_train_y = mean_std_save(train_y)
    #
    norm_train_x = zscore(train_x, axis=0)
    norm_train_y = zscore(train_y, axis=0)

    return mean_train_x, mean_train_y, std_train_x, std_train_y, norm_train_x, norm_train_y


def test_data_1d(x_min, x_max):
    test_x = np.atleast_2d(np.linspace(x_min, x_max, 101)).T
    test_y = function_m(test_x)
    return test_x, test_y


def data_denorm(data_x, data_y, x_mean, x_std, y_mean, y_std):
    data_x = reverse_zscore(data_x, x_mean, x_std)
    data_y = reverse_zscore(data_y, y_mean, y_std)
    return data_x, data_y


def plot_for_1d_1(x_min,
                  x_max,
                  gpr,
                  mean_train_x,
                  std_train_x,
                  train_x,
                  train_y):
    test_x, test_y = test_data_1d(x_min, x_max)
    test_x_norm = (test_x - mean_train_x) / std_train_x

    plt.figure(figsize=(12, 5))
    # (1) plot initial gpr
    test_y_norm_predict, cov = gpr.predict(test_x_norm, return_cov=True)
    test_y_predict = reverse_zscore(test_y_norm_predict, mean_train_y, std_train_y)
    uncertainty = 1.96 * np.sqrt(np.diag(cov))

    plt.subplot(1, 3, 1)
    plt.title("l=%.2f" % gpr.kernel_.length_scale)
    plt.plot(test_x, test_y, label="real_value")
    plt.plot(test_x, test_y_predict, label='prediction')
    plt.fill_between(test_x.ravel(), test_y_predict.ravel() + uncertainty, test_y_predict.ravel() - uncertainty,
                     alpha=0.5)
    plt.scatter(train_x, train_y, label="train", c="red", marker="x")
    plt.legend()
    return plt


def plot_for_1d_2(plt, gpr, x_min, x_max, mean_train_x, std_train_x):
    test_x, test_y = test_data_1d(x_min, x_max)
    test_x_norm = (test_x - mean_train_x) / std_train_x

    # calculate EI landscape
    EI_landscape = expected_improvement(test_x_norm.reshape(-1, 1), norm_train_x, norm_train_y, gpr)

    plt.subplot(1, 3, 2)
    plt.plot(test_x, EI_landscape, 'r-', lw=1, label='expected_improvement')
    plt.axvline(x=next_x, ls='--', c='k', lw=1, label='Next sampling location')
    return plt


def plot_for_1d_3(plt, gpr, x_min, x_max, train_x, train_y, next_x, mean_train_x, std_train_x):
    test_x, test_y = test_data_1d(x_min, x_max)
    test_x_norm = (test_x - mean_train_x) / std_train_x

    test_y_norm_predict, cov = gpr.predict(test_x_norm, return_cov=True)
    test_y_predict = reverse_zscore(test_y_norm_predict, mean_train_y, std_train_y)
    uncertainty = 1.96 * np.sqrt(np.diag(cov))

    plt.subplot(1, 3, 3)
    plt.title("l=%.2f" % gpr.kernel_.length_scale)
    plt.plot(test_x, test_y, label="real_value")
    plt.plot(test_x, test_y_predict, label='prediction')
    plt.fill_between(test_x.ravel(), test_y_predict.ravel() + uncertainty, test_y_predict.ravel() - uncertainty,
                     alpha=0.5)
    plt.scatter(train_x, train_y, label="train", c="red", marker="x")
    plt.axvline(x=next_x, ls='--', c='k', lw=1, label='sampling location')
    plt.legend()
    plt.show()
    return None


if __name__ == "__main__":

    # this following one line is for work around 1d plot in multiple-processing settings
    multiprocessing.freeze_support()

    np.random.seed(20)
    n_iter = 10
    func_val = {'next_x': 0}

    # === preprocess data change in each iteration of EI ===
    # run gpr once for initialize gpr
    x_min = 0
    x_max = 1
    n_vals = 1
    number_of_initial_samples = 2*n_vals+1

    # initial samples with hyper cube sampling
    train_x = pyDOE.lhs(n_vals, number_of_initial_samples)

    # calculate initial train output
    train_y = function_m(train_x)
    # keep the mean and std of training data
    mean_train_x, mean_train_y, std_train_x, std_train_y, norm_train_x, norm_train_y = \
        train_data_norm(train_x, train_y)

    # === end of preprocess data ====
    # initialize gpr for EI problem definition
    kernel = RBF(1, (np.exp(-1), np.exp(3)))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=0)
    gpr.fit(norm_train_x, norm_train_y)

    if n_vals == 1:
        plot_for_1d_1(x_min, x_max, gpr, mean_train_x, std_train_x, train_x, train_y)

    # create EI problem
    n_variables = train_x.shape[1]
    evalparas = {'X_sample': norm_train_x, 'Y_sample': norm_train_y, 'gpr': gpr}
    upper_bound = np.ones(n_variables)
    lower_bound = np.ones(n_variables) * -1
    problem = get_problem_from_func(acqusition_function, lower_bound, upper_bound, n_var=n_variables
                                    , func_args=evalparas)

    nobj = problem.n_obj
    ncon = problem.n_constr
    nvar = problem.n_var

    bounds = np.zeros((nvar, 2))
    for i in range(nvar):
        bounds[i][1] = problem.xu[i]
        bounds[i][0] = problem.xl[i]
    bounds = bounds.tolist()

    # start the searching process
    for iteration in range(n_iter):

        print('iteration is %d' % iteration)

        # Running the EI optimizer
        '''
        pop_x, pop_f, pop_g, archive_x, archive_f, archive_g = optimizer(problem,
                                                                         nobj,
                                                                         ncon,
                                                                         bounds,
                                                                         mut=0.8,
                                                                         crossp=0.7,
                                                                         popsize=100,
                                                                         its=100,
                                                                         **evalparas)
        '''

        # use parallelised EI evolution
        pop_x, pop_f, pop_g, archive_x, archive_f, archive_g = optimizer_para_EI.optimizer(problem,
                                                                                           nobj,
                                                                                           ncon,
                                                                                           bounds,
                                                                                           mut=0.8,
                                                                                           crossp=0.7,
                                                                                           popsize=10,
                                                                                           its=10,
                                                                                           **evalparas)

        # propose next_x location
        next_x_norm = pop_x[0, :]
        next_x_norm = np.atleast_2d(next_x_norm).reshape(-1, nvar)

        # convert for plotting and additional data collection
        next_x = reverse_zscore(next_x_norm, mean_train_x, std_train_x)
        next_y = function_m(next_x)
        next_y_norm = (next_y - mean_train_y) / std_train_y

        if n_vals == 1:
            plot_for_1d_2(plt, gpr, x_min, x_max, mean_train_x, std_train_x)

        print('next location denormalized: ')
        print(next_x)
        print('real function value at proposed location is %.f' % next_y)

        # when adding next proposed data, first convert it to initial data range (denormalize)
        train_x = reverse_zscore(norm_train_x, mean_train_x, std_train_x)
        train_y = reverse_zscore(norm_train_y, mean_train_y, std_train_y)

        # add new proposed data
        train_x = np.vstack((train_x, next_x))
        train_y = np.vstack((train_y, next_y))

        # re-normalize after new collection
        mean_train_x, mean_train_y, std_train_x, std_train_y, norm_train_x, norm_train_y = \
            train_data_norm(train_x, train_y)

        # re-train gpr
        gpr.fit(norm_train_x, norm_train_y)

        # update problem.evaluation parameter kwargs for EI calculation
        evalparas['X_sample'] = norm_train_x
        evalparas['Y_sample'] = norm_train_y
        evalparas['gpr'] = gpr

        if n_vals == 1:
            plot_for_1d_3(plt, gpr, x_min, x_max, train_x, train_y, next_x, mean_train_x, std_train_x)
