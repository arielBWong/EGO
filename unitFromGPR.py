from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, zscore
from EI_problem import expected_improvement
from pymop.factory import get_problem_from_func
from optimizer import optimizer
from sklearn.metrics import mean_squared_error


def f(x):
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)
    #return y.ravel()

def mean_std_save(data):
    m = np.mean(data, axis=0)
    s = np.std(data, axis=0)
    return m, s

def reverse_zscore(data, m, s):
    return data * s + m


def cross_val(val_x, val_y, gpr):
    val_x = val_x.reshape(-1, 1)
    pred_y = gpr.predict(val_x)
    mse = mean_squared_error(val_y, pred_y)
    return mse

def one_iter_from_gpr(train_X, **kwargs):

    # calculate train output
    train_y = f(train_X)


    # keep the mean and std of training data
    a_y, b_y = mean_std_save(train_y)
    a_x, b_x = mean_std_save(train_X)

    # zscore calculation
    train_X = zscore(train_X)
    train_y = zscore(train_y)

    # generate test data
    test_X = np.atleast_2d(np.linspace(0, 1, 101)).T
    test_Y_real = f(test_X)

    # test_X is zscored with train mean and std
    test_X = (test_X - a_x) / b_x

    # split training data with leave one out
    n_samples = train_X.shape[0]
    val_results = []
    val_theta = []
    for val_iter in range(n_samples):

        v_test_x = train_X[val_iter, :]
        v_test_y = train_y[val_iter, :]

        tmp_x = train_X
        tmp_y = train_y

        v_train_x = np.delete(tmp_x, val_iter, axis=0)
        v_train_y = np.delete(tmp_y, val_iter, axis=0)




        # fit GPR
        # b is np.std(train_X)
        kernel = RBF(1, (np.exp(-1), np.exp(3)))
        # kernel = RBF(b_x, (b_x, b_x))
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=0)

        # test kernel with a single test point
        # test_X = (np.atleast_2d(np.array([0.2])).T - a_x) / b_x

        # gpr.fit(train_X, train_y)
        # use split data as training data
        gpr.fit(v_train_x, v_train_y)

        # external_test_of_loglikelihood(gpr)

        # single test on loglikehood value
        # value = np.array([0.8])
        # print(external_test_of_loglikelihood_one_value(gpr, value))



        # optimization on hyper-parameters
        def obj_func(x, out, gpr, xi=0.01):
            # x = [x]
            lml = gpr.log_marginal_likelihood(x)
            out["F"] = np.array([-lml])


        evalparas = {}
        evalparas['gpr'] = gpr


        bounds_prob = (np.array([-2]), np.array([3]))
        # length scale range is (0.13,20)
        problem = get_problem_from_func(obj_func, bounds_prob[0], bounds_prob[1], n_var=1, func_args=evalparas)

        nobj = problem.n_obj
        ncon = problem.n_constr
        nvar = problem.n_var

        bounds = np.zeros((nvar, 2))
        for i in range(nvar):
            bounds[i][1] = problem.xu[i]
            bounds[i][0] = problem.xl[i]
        bounds = bounds.tolist()

        val_data = (v_test_x, v_test_y)

        pop_x, pop_f, pop_g, archive_x, archive_f, archive_g = optimizer(problem,
                                                                         nobj,
                                                                         ncon,
                                                                         bounds,
                                                                         val_data,
                                                                         mut=0.8,
                                                                         crossp=0.7,
                                                                         popsize=10,
                                                                         its=10,
                                                                         **evalparas)
        best_theta = pop_x[0]
        val_theta.append(best_theta)
        gpr.kernel.theta = np.array([best_theta])
        gpr.kernel_.theta = np.array([best_theta])
        # if internal optimization is commented out, re-fit is needed for recalculating K
        gpr.fit(v_train_x, v_train_y)
        val_results.append(cross_val(v_test_x, v_test_y, gpr))


    print(val_results)

    # this works on list type
    min_val_results = val_results.index(min(val_results))
    min_val_theta = val_theta[min_val_results]

    # re-evaluate
    tmp_x = train_X
    tmp_y = train_y

    v_train_x = np.delete(tmp_x, min_val_results, axis=0)
    v_train_y = np.delete(tmp_y, min_val_results, axis=0)

    kernel = RBF(1, (np.exp(-1), np.exp(3)))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=0)
    gpr.kernel.theta = np.array([min_val_theta])
    # gpr.kernel_.theta = np.array([min_val_theta])
    gpr.fit(v_train_x, v_train_y)







    mu, std = gpr.predict(test_X, return_std=True)
    test_y = mu.ravel()

    # check




    # uncertainty = 1.96 * np.sqrt(np.diag(cov))


    # test_X = (np.atleast_2d(np.array([0, 0.2, 0.5, 0.7, 1])).T - a_x) / b_x
    # EI = expected_improvement(test_X.reshape(-1, 1), train_X, train_y, gpr)

    # for plotting, convert data back
    test_X = reverse_zscore(test_X, a_x, b_x)
    test_y = reverse_zscore(test_y, a_y, b_y)
    # test_y_real was not zscore processed

    train_X = reverse_zscore(train_X, a_x, b_x)
    train_y = reverse_zscore(train_y, a_y, b_y)


    # plot prediction
    plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    plt.title("l=%.1f" % (gpr.kernel_.length_scale))
    plt.fill_between(test_X.ravel(), test_y + 2*std, test_y - 2*std, alpha=0.5)
    plt.plot(test_X, test_y, label="predict")
    plt.plot(test_X, test_Y_real, label='real_value')
    plt.scatter(train_X, train_y, label="train", c="red", marker="x")
    plt.legend()
    plt.show()


    # external_test_of_loglikelihood(gpr)

    '''
    # plot acquisition
    next_x = kwargs['next_x']
    # plt.figure()
    plt.subplot(1, 2, 2)
    # EI = expected_improvement(test_X.reshape(-1, 1), train_X, train_y, gpr)
    plt.plot(test_X, EI, 'r-', lw=1, label='Acquisition function')
    plt.axvline(x=next_x, ls='--', c='k', lw=1, label='Next sampling location')
    plt.show()
    
    '''


    return gpr, train_X, train_y


def external_test_of_loglikelihood(gpr):
    # use gpr to call the loglikehood function

    print('\n\n now it is plotting loglikelihood process---')
    point_size = 1000
    lower_bound_length_scale = np.exp(-2)
    upper_bound_length_scale = np.exp(3)
    theta_list = np.atleast_2d(np.linspace(np.log(lower_bound_length_scale), np.log(upper_bound_length_scale), point_size)).reshape(-1, 1)
    log_likelihood_test = np.zeros((point_size, 1))
    # assign value to gpr.theta
    for counter, theta_item in enumerate(theta_list):
        gpr.kernel_.theta = theta_item
        log_likelihood_test[counter] = -(gpr.log_marginal_likelihood(gpr.kernel_.theta))

    # def log_marginal_likelihood(self, theta=None, eval_gradient=False)
    plt.figure()
    plt.plot(np.exp(theta_list), log_likelihood_test)
    plt.show()
    return None

def external_test_of_loglikelihood_one_value(gpr,theta_item):
    # use gpr to call the loglikehood function


    # assign value to gpr.theta

    gpr.kernel_.theta = theta_item
    log_likelihood_test = -(gpr.log_marginal_likelihood(gpr.kernel_.theta))
    return -log_likelihood_test


'''
def external_optimizer(gpr, bounds):
    population_size = 10
    initial_population = tf.random.uniform([population_size], minval= bounds[0], maxval =bounds[1])
    print(initial_population)
    tfp.optimizer.differential_evolution_one_step(-gpr.log_marginal_likelihood(),initial_population,differential_weight=0.5, crossover_prob=0.7, seed =1)
    return None
'''

if __name__ == '__main__':
    np.random.seed(20)
    train_x = np.atleast_2d([0, 0.5, 1, 0.2]).T


    paras = {'next_x': 0.3}


    one_iter_from_gpr(train_x, **paras)
