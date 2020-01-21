import numpy as np
import matplotlib.pyplot as plt
import optimizer_EI
from pymop.factory import get_problem_from_func
from pymop import ZDT1, ZDT2, ZDT3, ZDT4, DTLZ1, G1, DTLZ2, BNH, Carside, Kursawe, OSY, Truss2D, WeldedBeam, TNK
from EI_krg import acqusition_function
from unitFromGPR import f, mean_std_save, reverse_zscore
from scipy.stats import norm, zscore
from sklearn.utils.validation import check_array
import pyDOE
import multiprocessing
from cross_val_hyperp import cross_val_krg
from joblib import dump, load
import time
from surrogate_problems import branin, GPc, Gomez3, Mystery, Reverse_Mystery, SHCBc, HS100, Haupt_schewefel, MO_linearTest
import os
import copy
import multiprocessing as mp
import pygmo as pg



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

def saveNameConstr(problem_name, seed_index):

    working_folder = os.getcwd()
    result_folder = working_folder + '\\outputs' + '\\' + problem_name
    if not os.path.isdir(result_folder):
        # shutil.rmtree(result_folder)
        # os.mkdir(result_folder)
        os.mkdir(result_folder)
    # else:
    # os.mkdir(result_folder)
    savename_x = result_folder + '\\best_x_seed_' + str(seed_index) + '.joblib'
    savename_y = result_folder + '\\best_f_seed_' + str(seed_index) + '.joblib'
    savename_FEs = result_folder + '\\FEs_seed_' + str(seed_index) + '.joblib'
    return savename_x, savename_y, savename_FEs


def main(seed_index, target_problem):

    # this following one line is for work around 1d plot in multiple-processing settings
    multiprocessing.freeze_support()

    np.random.seed(seed_index)

    print('Problem')
    print(target_problem.name())
    print(seed_index)
    print('\n')

    # collect problem parameters: number of objs, number of constraints
    n_sur_objs = target_problem.n_obj
    n_sur_cons = target_problem.n_constr
    n_vals = target_problem.n_var

    # setting
    n_iter = 100 * n_vals
    number_of_initial_samples = 11 * n_vals - 1
    if n_vals > 3:
        number_of_initial_samples = 50
    n_iter = 400

    # initial samples with hyper cube sampling
    train_x = pyDOE.lhs(n_vals, number_of_initial_samples)

    # print(train_x)
    # transfer input into the right range of the target problem
    train_x = hyper_cube_sampling_convert(target_problem.xu, target_problem.xl, target_problem.n_var,  train_x)

    # train_x = np.atleast_2d([[0, 0], [2, 2], [1, 1]]) # this is only for test
    # print(train_x)
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


    krg, krg_g = cross_val_krg(train_x, train_y, cons_y)

    # create EI problem
    n_variables = train_x.shape[1]
    evalparas = {'train_x':  train_x,
                 'train_y': train_y,
                 'krg': krg,
                 'krg_g': krg_g,
                 'feasible': np.array([])}

    # For this upper and lower bound for EI sampling
    # should check whether it is reasonable?
    upper_bound = np.ones(n_variables)
    lower_bound = np.ones(n_variables)

    for i in range(n_variables):
        upper_bound[i] = target_problem.xu[i]
        lower_bound[i] = target_problem.xl[i]



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

        # print('\n iteration is %d' % iteration)
        start = time.time()

        # check feasibility in main loop
        sample_n = train_x.shape[0]
        a = np.linspace(0, sample_n - 1, sample_n, dtype=int)
        target_problem._evaluate(train_x, out)

        if 'G' in out.keys():
            mu_g = out['G']
            temp_mug = copy.deepcopy(out['G'])

            mu_g[mu_g <= 0] = 0
            mu_cv = mu_g.sum(axis=1)
            infeasible = np.nonzero(mu_cv)
            feasible = np.setdiff1d(a, infeasible)
            feasible_y = evalparas['train_y'][feasible, :]
            evalparas['feasible'] = feasible_y

            if feasible.size > 0:
                print('feasible solutions: ')
                # print(train_y[feasible, :])
                # print('feasible on constraints performance')
                # print(temp_mug[feasible, :])
                if n_sur_objs > 1:
                    target_problem.pareto_front(feasible_y)
                    nadir_p = target_problem.nadir_point()
            else:
                print('No feasible solutions in this iteration %d' % iteration)
        else:
            evalparas['feasible'] = -1


        # print('check bounds being same')
        # print(bounds)

        # main loop for finding next x
        pop_x, pop_f, pop_g, archive_x, archive_f, archive_g = optimizer_EI.optimizer(ei_problem,
                                                                                      nobj,
                                                                                      ncon,
                                                                                      bounds,
                                                                                      mut=0.1,
                                                                                      crossp=0.9,
                                                                                      popsize=20,
                                                                                      its=20,
                                                                                      **evalparas)

        # propose next_x location
        next_x = pop_x[0, :]
        # print(next_x)
        # dimension re-check
        next_x = np.atleast_2d(next_x).reshape(-1, nvar)

        # generate corresponding f and g
        target_problem._evaluate(next_x, out)
        next_y = out['F']
        # print(next_y)


        if 'G' in out.keys():
            next_cons_y = out['G']
        else:
            next_cons_y = None

        if next_x[0, 0] < bounds[0][0] or next_x[0, 0] > bounds[0][1] or next_x[0, 1] < bounds[1][0] or next_x[0, 1] > bounds[1][1]:
            print('out of range')

        # add new proposed data
        train_x = np.vstack((train_x, next_x))
        train_y = np.vstack((train_y, next_y))
        print('train x  size %d', train_x.shape[0])

        if n_sur_cons > 0:
            cons_y = np.vstack((cons_y, next_cons_y))

        archive_x_sur = np.vstack((archive_x_sur, next_x))
        archive_y_sur = np.vstack((archive_y_sur, next_y))

        if n_sur_cons > 0:
            archive_g_sur = np.vstack((archive_g_sur, next_cons_y))

        # use extended data to train krging model
        krg, krg_g = cross_val_krg(train_x, train_y, cons_y)

        # update problem.evaluation parameter kwargs for EI calculation
        evalparas['train_x'] = train_x
        evalparas['train_y'] = train_y
        evalparas['krg'] = krg
        evalparas['krg_g'] = krg_g

        end = time.time()  # on seconds
        lasts = (end - start) / 60.
        print('one loop iteration %d uses %.2f min' % (iteration, lasts))

        # check for termination
        # if target_problem.stop_criteria(next_x):
            # break


    # output best archive solutions
    sample_n = train_x.shape[0]
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
        # print('number of feasible solutions in total %d solutions is %d ' % (sample_n, n))

        if n > 0:
            best_f = np.argmin(feasible_f, axis=0)
            print('Best solutions encountered so far')
            print(feasible_f[best_f, :])
            best_f_out = feasible_f[best_f, :]
            best_x_out = feasible_solutions[best_f, :]
            print(feasible_solutions[best_f, :])
        else:
            best_f_out = None
            best_x_out = None
            print('No best solutions encountered so far')
    elif n_sur_objs == 1:
        best_f = np.argmin(train_y, axis=0)
        best_f_out = train_y[best_f, :]
        best_x_out = train_x[best_f, :]
    else:
        print('MO save pareto front from all y')
        ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(archive_y_sur)
        ndf = list(ndf)
        f_pareto = archive_y_sur[ndf[0], :]
        best_f_out = f_pareto
        best_x_out = archive_x_sur[ndf[0], :]

    savename_x, savename_f, savename_FEs = saveNameConstr(target_problem.name(), seed_index)

    dump(best_x_out, savename_x)
    dump(best_f_out, savename_f)
    dump(iteration, savename_FEs)


if __name__ == "__main__":

    # target_problem = ZDT1(3)
    # main(100, target_problem)

    # point_list = [[0, 0], [2, 2]]
    # point_reference = [2.2, 2.2]



    # hv = pg.hypervolume(point_list)
    # hv_value = hv.compute(point_reference)
    # print(hv_val e)

    # x = np.atleast_2d([5.,  5., 1.06457815,  5.,-0.71062037,  0.86922459, 1.01407611])
    # out = {}
    # f, g = target_problem._evaluate(x, out)
    # print(f)


    target_problems = [branin.new_branin_5(),
                       Gomez3.Gomez3(),
                       Mystery.Mystery(),
                       Reverse_Mystery.ReverseMystery(),
                       SHCBc.SHCBc(),
                       Haupt_schewefel.Haupt_schewefel(),
                       HS100.HS100(),
                       GPc.GPc()]



    MO_target_problems = [ZDT1(n_var=3),
                          ZDT2(n_var=3),
                          ZDT3(n_var=3),
                          ZDT4(n_var=3),
                          Kursawe(),
                          Truss2D(),
                          BNH(),
                          TNK(),
                          WeldedBeam(),
                          OSY()]
    args = []
    for p in MO_target_problems:
        args.append((100, p))

    num_workers = 6
    pool = mp.Pool(processes=num_workers)
    pool.starmap(main, ([arg for arg in args]))





                          #
    # for i in range(1, 2):
        # for j in np.arange(20):
            # main(j, target_problems[i])
    #
    # target_problem = ZDT1()
    # print(target_problem.n_obj)

    # let's try multiple now


    '''
    args = []
    problem = target_problems[5]
    seeds = range(0, 20, 1)
    for s in seeds:
        args.append((s, problem))
    num_workers = 7
    pool = mp.Pool(processes=num_workers)
    pool.starmap(main, ([arg for arg in args]))
    
    '''








