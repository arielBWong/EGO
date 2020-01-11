import numpy as np
# import matplotlib.pyplot as plt
import optimizer_EI
from pymop.factory import get_problem_from_func
from EI_problem import acqusition_function
from unitFromGPR import f, mean_std_save, reverse_zscore
from scipy.stats import norm, zscore
from sklearn.utils.validation import check_array
import pyDOE
import multiprocessing
from cross_val_hyperp import cross_val_gpr
from joblib import dump, load
import time
from surrogate_problems import branin
import os
import multiprocessing as mp
from pymop.problems.zdt import ZDT1
import shutil



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
    return x, y

def function_call(func, x):
    x, y = func(x)
    return x, y


def train_data_norm(train_x, train_y):
    mean_train_x, std_train_x = mean_std_save(train_x)
    mean_train_y, std_train_y = mean_std_save(train_y)
    #
    norm_train_x = zscore(train_x, axis=0)
    norm_train_y = zscore(train_y, axis=0)

    return mean_train_x, mean_train_y, std_train_x, std_train_y, norm_train_x, norm_train_y


def norm_data(x):
    mean_x, std_x = mean_std_save(x)
    norm_x = zscore(x, axis=0)

    return mean_x, std_x, norm_x

def test_data_1d(x_min, x_max):
    test_x = np.atleast_2d(np.linspace(x_min, x_max, 101)).T
    test_y = function_m(test_x)
    return test_x, test_y


def data_denorm(data_x, data_y, x_mean, x_std, y_mean, y_std):
    data_x = reverse_zscore(data_x, x_mean, x_std)
    data_y = reverse_zscore(data_y, y_mean, y_std)
    return data_x, data_y

'''
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
    plt.show()
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
'''


def hyper_cube_sampling_convert(xu, xl, n_var, x):
    x = check_array(x)

    if x.shape[1] != n_var:
        print('sample data given do not fit the problem number of variables')
        exit(1)

    # assume that values in x is in range [0, 1]
    if np.any(x > 1) or np.any(x < 0):
        raise Exception('Input range error, this Branin input should be in range [0, 1]')
        exit(1)

    x_first = np.atleast_2d(x[:, 0]).reshape(-1, 1)
    x_first = xl[0] + x_first * (xu[0] - xl[0])
    for i in np.arange(1, n_var):
        x_next = np.atleast_2d(x[:, 1]).reshape(-1, 1)
        # convert to defined range
        x_next = xl[i] + x_next * (xu[i] - xl[i])
        x_first = np.hstack((x_first, x_next))

    return x_first




def main(seed_index):
    # this following one line is for work around 1d plot in multiple-processing settings
    multiprocessing.freeze_support()

    np.random.seed(seed_index)
    n_iter = 2
    func_val = {'next_x': 0}

    # === preprocess data change in each iteration of EI ===
    # run gpr once for initialize gpr
    x_min = 0
    x_max = 1

    # configeration of the EGO for
    # number of variables
    # optimization target function
    # parameter range convertion
    n_vals = 2
    number_of_initial_samples = 10 * n_vals

    target_problem = branin.new_branin_5()
    # target_problem = ZDT1()
    # target_problem.n_var = 2
    # target_problem.xl = np.array([0, 0])
    # target_problem.xu = np.array([1, 1])
    print(target_problem.name())

    # collect problem parameters: number of objs, number of constraints
    # n_sur_objs = len(target_problem)
    n_sur_objs = target_problem.n_obj
    # n_sur_cons = len(target_constraints)
    n_sur_cons = target_problem.n_constr

    # initial samples with hyper cube sampling
    # pitfall is that this function generates values between 0 to 1
    # when create train_y, these train_x needs to be converted to the range
    # of their problem defined range
    train_x = pyDOE.lhs(n_vals, number_of_initial_samples)

    # For every problem definition, there should be a parameter
    # range converting method to transfer input
    # into the right range of the target problem
    train_x = hyper_cube_sampling_convert(target_problem.xu, target_problem.xl, target_problem.n_var,  train_x)
    archive_x_sur = train_x

    out = {}
    target_problem._evaluate(train_x, out)
    train_y = out['F']

    if 'G' in out.keys():
        cons_y = out['G']
    else:
        cons_y = None

    archive_y_sur = train_y
    archive_g_sur = cons_y

    # keep the mean and std of training data
    if n_sur_cons > 0:
        mean_cons_y, std_cons_y, norm_cons_y = norm_data(cons_y)
    else:
        norm_cons_y = None
        mean_cons_y = None
        std_cons_y = None

    mean_train_y, std_train_y, norm_train_y = norm_data(train_y)
    mean_train_x, std_train_x, norm_train_x = norm_data(train_x)



    # use cross validation for hyper-parameter
    gpr, gpr_g = cross_val_gpr(norm_train_x, norm_train_y, norm_cons_y)

    # krg, krg_g = cross_val_krg(norm_train_x, norm_train_y, norm_cons_y)

    # if n_vals == 1:
        # plot_for_1d_1(x_min, x_max, gpr, mean_train_x, std_train_x, train_x, train_y)

    # create EI problem
    n_variables = train_x.shape[1]
    evalparas = {'X_sample':  train_x,  # norm_train_x,
                 'Y_sample': train_y,  # norm_train_y,
                 'y_mean': mean_train_y,
                 'y_std': std_train_y,
                 'cons_g_mean': mean_cons_y,
                 'cons_g_std': std_cons_y,
                 'gpr': gpr,
                 'gpr_g': gpr_g,
                 'feasible': np.array([])}

    # For this upper and lower bound for EI sampling
    # should check whether it is reasonable?
    upper_bound = np.ones(n_variables)
    lower_bound = np.ones(n_variables)

    for i in range(n_variables):
        upper_bound[i] = (target_problem.xu[i] - mean_train_x[i]) / std_train_x[i]
        lower_bound[i] = (target_problem.xl[i] - mean_train_x[i]) / std_train_x[i]



    ei_problem = get_problem_from_func(acqusition_function,
                                       lower_bound,
                                       upper_bound,
                                       n_var=n_variables,
                                       func_args=evalparas)

    nobj = ei_problem.n_obj
    ncon = ei_problem.n_constr
    nvar = ei_problem.n_var

    # bounds settings for optimizer in main loop
    # each row refers to a variable, then [lower bound, upper bound]
    bounds = np.zeros((nvar, 2))
    for i in range(nvar):
        bounds[i][1] = ei_problem.xu[i]
        bounds[i][0] = ei_problem.xl[i]
    bounds = bounds.tolist()

    # start the searching process
    for iteration in range(n_iter):

        print('\n iteration is %d' % iteration)
        start = time.time()


        # check feasibility in main loop
        sample_n = train_x.shape[0]
        a = np.linspace(0, sample_n - 1, sample_n, dtype=int)
        target_problem._evaluate(train_x, out)

        if 'G' in out.keys():
            mu_g = out['G']

            mu_g[mu_g <= 0] = 0
            mu_cv = mu_g.sum(axis=1)
            infeasible = np.nonzero(mu_cv)
            feasible = np.setdiff1d(a, infeasible)
            feasible_y = evalparas['Y_sample'][feasible, :]
            evalparas['feasible'] = feasible_y

            if feasible.size > 0:
                print('feasible solutions: ')
                print(train_y[feasible, :])
                if n_sur_objs > 1:
                    target_problem.pareto_front(feasible_y)
                    nadir_p = target_problem.nadir_point()
            else:
                print('No feasible solutions in this iteration %d' % iteration)
        else:
            evalparas['feasible'] = -1



        # determine nadir for the target problem with current samples


        # use parallelised EI evolution
        # pop_x, pop_f, pop_g, archive_x, archive_f, archive_g = optimizer_para_EI.optimizer(ei_problem,
        #                                                                                   nobj,
        #                                                                                   ncon,
        #                                                                                   bounds,
        #                                                                                   mut=0.8,
        #                                                                                   crossp=0.7,
        #                                                                                   popsize=20,
        #                                                                                   its=20,
        #                                                                                   **evalparas)



        print('before ei opitimization')
        print(bounds)
        #
        pop_x, pop_f, pop_g, archive_x, archive_f, archive_g = optimizer_EI.optimizer(ei_problem,
                                                                                      nobj,
                                                                                      ncon,
                                                                                      bounds,
                                                                                      mut=0.8,
                                                                                      crossp=0.7,
                                                                                      popsize=20,
                                                                                      its=20,
                                                                                      **evalparas)


        # propose next_x location
        next_x_norm = pop_x[0, :]
        print('next_x_norm')
        print(next_x_norm)

        # dimension re-check
        next_x_norm = np.atleast_2d(next_x_norm).reshape(-1, nvar)

        # convert for plotting and additional data collection
        next_x = reverse_zscore(next_x_norm, mean_train_x, std_train_x)

        # generate corresponding f and g
        target_problem._evaluate(next_x, out)
        next_y = out['F']

        if 'G' in out.keys():
            next_cons_y = out['G']
        else:
            next_cons_y = None


        if next_x_norm[0, 0] < bounds[0][0] or next_x_norm[0, 0] > bounds[0][1] or next_x_norm[0, 1] < bounds[1][0] or next_x_norm[0, 1] > bounds[1][1]:
            print('out of range')

        # if n_vals == 1:
            # plot_for_1d_2(plt, gpr, x_min, x_max, mean_train_x, std_train_x)

        print('next location denormalized: ')
        print(next_x)
        print('real function value at proposed location is')
        print(next_y)
        print('constraint performance on this proposed location is')
        print(next_cons_y)
        print('\n')

        # when adding next proposed data, first convert it to initial data range (denormalize)
        train_x = reverse_zscore(norm_train_x, mean_train_x, std_train_x)
        train_y = reverse_zscore(norm_train_y, mean_train_y, std_train_y)

        if n_sur_cons > 0:
            cons_y = reverse_zscore(norm_cons_y, mean_cons_y, std_cons_y)
        else:
            cons_y = None

        # add new proposed data
        train_x = np.vstack((train_x, next_x))
        train_y = np.vstack((train_y, next_y))

        if n_sur_cons > 0:
            cons_y = np.vstack((cons_y, next_cons_y))

        archive_x_sur = np.vstack((archive_x_sur, next_x))
        archive_y_sur = np.vstack((archive_y_sur, next_y))

        if n_sur_cons > 0:
            archive_g_sur = np.vstack((archive_g_sur, next_cons_y))

        # re-normalize after new collection
        mean_train_x, std_train_x, norm_train_x = norm_data(train_x)
        mean_train_y, std_train_y, norm_train_y = norm_data(train_y)

        if n_sur_cons > 0:
            mean_cons_y, std_cons_y, norm_cons_y = norm_data(cons_y)
        else:
            mean_cons_y = None
            std_cons_y = None
            norm_cons_y = None

        upper_bound = (target_problem.xu - mean_train_x) / std_train_x
        lower_bound = (target_problem.xl - mean_train_x) / std_train_x

        # reset main loop bounds
        ei_problem.xu = upper_bound
        ei_problem.lu = lower_bound

        # reset optimization bounds
        for i in range(nvar):
            bounds[i][1] = ei_problem.xu[i]
            bounds[i][0] = ei_problem.xl[i]





        # use cross validation for hyper-parameter
        # the following is re-train from start
        # but gpr.fit is like re-train on previous parameters
        gpr, gpr_g = cross_val_gpr(norm_train_x, norm_train_y, norm_cons_y)



        # update problem.evaluation parameter kwargs for EI calculation
        evalparas['X_sample'] = train_x
        evalparas['Y_sample'] = train_y
        evalparas['y_mean'] = mean_train_y
        evalparas['y_std'] = std_train_y
        evalparas['cons_g_mean'] = mean_cons_y
        evalparas['cons_g_std'] = std_cons_y
        evalparas['gpr'] = gpr
        evalparas['gpr_g'] = gpr_g

        end = time.time()
        lasts = (end - start) / 60.
        print('main loop iteration %d uses %.2f' % (iteration, lasts))




        # if n_vals == 1:
            # plot_for_1d_3(plt, gpr, x_min, x_max, train_x, train_y, next_x, mean_train_x, std_train_x)


    # output best archive solutions
    sample_n = norm_train_x.shape[0]
    a = np.linspace(0, sample_n - 1, sample_n, dtype=int)
    target_problem._evaluate(train_x, out)
    if 'G' in out.keys():
        mu_g = out['G']

        mu_g[mu_g <= 0] = 0
        mu_cv = mu_g.sum(axis=1)
        infeasible = np.nonzero(mu_cv)
        feasible = np.setdiff1d(a, infeasible)

        feasible_solutions = archive_x_sur[feasible, :]
        feasible_f = archive_y_sur[feasible, :]

        n = len(feasible_f)
        print('number of feasible solutions in total %d solutions is %d ' % (sample_n, n))

        if n > 0:
            best_f = np.argmin(feasible_f, axis=0)
            print('Best solutions encountered so far')
            print(feasible_f[best_f, :])
            print(feasible_solutions[best_f, :])
        else:
            print('No best solutions encountered so far')
    else:
        best_f = np.argmin(train_y, axis=0)


    working_folder = os.getcwd()
    result_folder = working_folder + '\\outputs' + '\\' + target_problem.name()
    if os.path.isdir(result_folder):
        shutil.rmtree(result_folder)
        os.mkdir(result_folder)
    else:
        os.mkdir(result_folder)

    saveName_x = result_folder + '\\r_sample_x_seed_' + str(seed_index) + '.joblib'
    saveName_y = result_folder + '\\r_bset_f_seed_' + str(seed_index) + '.joblib'
    saveName_g = result_folder + '\\r_bset_x_seed_' + str(seed_index) + '.joblib'

    dump(train_x, saveName_x)

    if n_sur_cons > 0:
        dump(feasible_f[best_f, :], saveName_y)
        dump(feasible_solutions[best_f, :], saveName_g)
    else:
        dump(train_y[best_f, :], saveName_y)
        dump(train_x[best_f, :], saveName_g)



'''
    # save the gpr model for plotting
    dump(gpr, 'Branin.joblib')

    para_save = {}
    para_save['mean_x'] = mean_train_x
    para_save['mean_y'] = mean_train_y
    para_save['std_x'] = std_train_x
    para_save['std_y'] = std_train_y

    dump(para_save,  'normal_p.joblib')
'''






if __name__ == "__main__":
    # for i in np.arange(2, 2):
    # main(100)
    # target_problem = ZDT1()
    # print(target_problem.n_obj)

    # let's try multiple now

    # seeds = range(0, 10, 1)
    # seeds = tuple(seeds)
    # num_workers = 4

    # pool = mp.Pool(processes=num_workers)
    # pool.map(main, seeds)
    from smt.surrogate_models import KRG
    import pykring


