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
from surrogate_problems import branin, GPc, Gomez3, Mystery, Reverse_Mystery, SHCBc, HS100, Haupt_schewefel, MO_linearTest, single_krg_optim
import os
import copy
import multiprocessing as mp
import pygmo as pg
from optimizer import optimizer



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


def check_krg_ideal_points(krg, n_var, n_constr, n_obj, low, up):
    x_krg = []
    f_krg = []

    n_krg = len(krg)

    # identify ideal x and f for each objective
    for k in krg:
        problem = single_krg_optim.single_krg_optim(k, n_var, n_constr, 1, low, up)

        single_bounds = np.zeros((n_var, 2))
        for v in range(n_var):
            single_bounds[v, 0] = low[v]
            single_bounds[v, 1] = up[v]
        single_bounds = single_bounds.tolist()

        pop_x, pop_f, pop_g, archive_x, archive_f, archive_g = optimizer_EI.optimizer(problem,
                                                                                      nobj=1,
                                                                                      ncon=0,
                                                                                      bounds=single_bounds,
                                                                                      mut=0.1,
                                                                                      crossp=0.9,
                                                                                      popsize=100,
                                                                                      its=100)
        x_out = pop_x[0, :]
        f_out = pop_f[0, :]
        x_krg.append(x_out)
        f_krg.append(f_out)

    # ideal for one objective is nadir of the other objective
    adj_mat = np.zeros((n_krg, n_krg))
    for i, x in enumerate(x_krg):
        x = np.atleast_2d(x).reshape(-1, n_var)
        for j, k in enumerate(krg):
            adj_mat[i, j], _ = k.predict(x)


    # get ideal and nadir points
    nadir = np.amax(adj_mat, axis=0)
    ideal = np.amin(adj_mat, axis=0)

    return nadir, ideal, x_krg

def init_xy(number_of_initial_samples, target_problem):

    n_vals = target_problem.n_var
    n_sur_cons = target_problem.n_constr

    # initial samples with hyper cube sampling
    train_x = pyDOE.lhs(n_vals, number_of_initial_samples)

    # transfer input into the right range of the target problem
    train_x = hyper_cube_sampling_convert(target_problem.xu, target_problem.xl, target_problem.n_var, train_x)

    out = {}
    target_problem._evaluate(train_x, out)
    train_y = out['F']

    if 'G' in out.keys():
        cons_y = out['G']
        cons_y = np.atleast_2d(cons_y).reshape(-1, n_sur_cons)
    else:
        cons_y = None

    return train_x, train_y, cons_y

def feasible_check(train_x, target_problem, evalparas):

    out = {}
    sample_n = train_x.shape[0]
    n_sur_cons = target_problem.n_constr
    a = np.linspace(0, sample_n - 1, sample_n, dtype=int)
    target_problem._evaluate(train_x, out)

    if 'G' in out.keys():
        mu_g = out['G']
        mu_g = np.atleast_2d(mu_g).reshape(-1, n_sur_cons)
        temp_mug = copy.deepcopy(out['G'])

        mu_g[mu_g <= 0] = 0
        mu_cv = mu_g.sum(axis=1)
        infeasible = np.nonzero(mu_cv)
        feasible = np.setdiff1d(a, infeasible)
        feasible_y = evalparas['train_y'][feasible, :]
        evalparas['feasible'] = feasible_y

        if feasible.size > 0:
            print('feasible solutions: ')
        else:
            print('No feasible solutions in this iteration %d' % iteration)
    else:
        evalparas['feasible'] = -1

    return evalparas

def post_process(train_x, train_y, cons_y, target_probelm, seed_index):

    n_sur_objs = target_probelm.n_obj
    n_sur_cons = target_problem.n_constr
    # output best archive solutions
    sample_n = train_x.shape[0]
    a = np.linspace(0, sample_n - 1, sample_n, dtype=int)
    out = {}
    target_problem._evaluate(train_x, out)
    if 'G' in out.keys():
        mu_g = out['G']
        mu_g = np.atleast_2d(mu_g).reshape(-1, n_sur_cons)

        mu_g[mu_g <= 0] = 0
        mu_cv = mu_g.sum(axis=1)
        infeasible = np.nonzero(mu_cv)
        feasible = np.setdiff1d(a, infeasible)

        feasible_solutions = train_x[feasible, :]
        feasible_f = train_y[feasible, :]

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
        ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(train_y)
        ndf = list(ndf)
        f_pareto = train_y[ndf[0], :]
        best_f_out = f_pareto
        best_x_out = train_x[ndf[0], :]

    savename_x, savename_f, savename_FEs = saveNameConstr(target_problem.name(), seed_index)

    dump(best_x_out, savename_x)
    dump(best_f_out, savename_f)



def main(seed_index, target_problem, enable_crossvalidation):

    # this following one line is for work around 1d plot in multiple-processing settings
    multiprocessing.freeze_support()
    np.random.seed(seed_index)

    print('Problem')
    print(target_problem.name())
    print('seed %d' % seed_index)

    # collect problem parameters: number of objs, number of constraints
    n_sur_objs = target_problem.n_obj
    n_sur_cons = target_problem.n_constr
    n_vals = target_problem.n_var

    number_of_initial_samples = 11 * n_vals - 1
    n_iter = 300  # stopping criterion set
    train_x, train_y, cons_y = init_xy(number_of_initial_samples, target_problem)

    # create kriging
    krg, krg_g = cross_val_krg(train_x, train_y, cons_y, enable_crossvalidation)
    nadir_krg, ideal_krg, x_out = check_krg_ideal_points(krg, n_vals, n_sur_cons, n_sur_objs, target_problem.xl, target_problem.xu)

    # create EI problem
    evalparas = {'train_x':  train_x,
                 'train_y': train_y,
                 'krg': krg,
                 'krg_g': krg_g,
                 'nadir': nadir_krg,
                 'ideal': ideal_krg,
                 'feasible': np.array([])}

    ei_problem = get_problem_from_func(acqusition_function,
                                       target_problem.xl,
                                       target_problem.xu,
                                       n_var=n_vals,
                                       func_args=evalparas)

    x_bounds = np.vstack((target_problem.xl, target_problem.xu)).T.tolist()

    start_all = time.time()
    # start the searching process
    for iteration in range(n_iter):

        # check feasibility in main loop
        evalparas = feasible_check(train_x, target_problem, evalparas)

        start = time.time()
        # main loop for finding next x
        pop_x, pop_f, pop_g, archive_x, archive_f, archive_g = optimizer_EI.optimizer(ei_problem,
                                                                                      ei_problem.n_obj,
                                                                                      ei_problem.n_constr,
                                                                                      x_bounds,
                                                                                      mut=0.1,
                                                                                      crossp=0.9,
                                                                                      popsize=10,
                                                                                      its=10,
                                                                                      **evalparas)

        end = time.time()
        lasts = (end - start)
        # print('propose to next x in iteration %d uses %.2f sec' % (iteration, lasts))
        # propose next_x location
        next_x = pop_x[0, :]
        # print(next_x)
        # dimension re-check
        next_x = np.atleast_2d(next_x).reshape(-1, n_vals)

        # generate corresponding f and g
        out = {}
        target_problem._evaluate(next_x, out)
        next_y = out['F']
        # print(next_y)

        if 'G' in out.keys():
            next_cons_y = out['G']
            next_cons_y = np.atleast_2d(next_cons_y)
        else:
            next_cons_y = None


        # add new proposed data
        train_x = np.vstack((train_x, next_x))
        train_y = np.vstack((train_y, next_y))
        print('train x  size %d' % train_x.shape[0])

        if n_sur_cons > 0:
            cons_y = np.vstack((cons_y, next_cons_y))


        start = time.time()
        # use extended data to train krging model
        krg, krg_g = cross_val_krg(train_x, train_y, cons_y, enable_crossvalidation)
        end = time.time()  # on seconds

        # check whether there is any f that is even better/smaller than ideal
        ideal_true_samples = np.atleast_2d(np.amin(train_y, axis=0))
        compare = np.any(ideal_true_samples < ideal_krg, axis=1)
        print(ideal_true_samples)
        print(ideal_krg)
        print(compare)

        if sum(compare) > 0:
            print('New evaluation')
            # add true evaluation
            for x in x_out:
                x = np.atleast_2d(x).reshape(-1, n_vals)
                out = {}
                target_problem._evaluate(x, out)
                y = out['F']

                train_x = np.vstack((train_x, x))
                train_y = np.vstack((train_y, y))
                if 'G' in out:
                    g = np.atleast_2d(out['G']).reshape(-1, n_sur_cons)
                    cons_y = np.vstack((cons_y, g))
            # re-conduct krg training
            krg, krg_g = cross_val_krg(train_x, train_y, cons_y, enable_crossvalidation)



        lasts = (end - start)
        print('cross-validation %d uses %.2f sec' % (iteration, lasts))

        # update problem.evaluation parameter kwargs for EI calculation

        nadir_krg, ideal_krg, x_out = check_krg_ideal_points(krg, n_vals, n_sur_cons, n_sur_objs, target_problem.xl, target_problem.xu)
        evalparas['train_x'] = train_x
        evalparas['train_y'] = train_y
        evalparas['krg'] = krg
        evalparas['krg_g'] = krg_g
        evalparas['nadir'] = nadir_krg
        evalparas['ideal'] = ideal_krg

        # stopping criteria
        sample_n = train_x.shape[0]
        if sample_n == 200:
            break

    end_all = time.time()
    print('overall time %.4f ' % (end_all - start_all))


if __name__ == "__main__":
    MO_target_problems = [ZDT3(n_var=3)
                          #DTLZ2(n_obj=2)
                          # Kursawe(),
                          # Truss2D(),
                          # TNK()]
                          # BNH(),
                          # WeldedBeam()
                          ]

    target_problem = MO_target_problems[0]
    main(100, target_problem, False)

    # point_list = [[0, 0], [2, 2]]
    # point_reference = [2.2, 2.2]



    # hv = pg.hypervolume(point_list)
    # hv_value = hv.compute(point_reference)
    # print(hv_val e)

    # x = np.atleast_2d([5.,  5., 1.06457815,  5.,-0.71062037,  0.86922459, 1.01407611])
    # out = {}
    # f, g = target_problem._evaluate(x, out)
    # print(f)

    '''
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
                          OSY(),
                          Kursawe(),
                          Truss2D(),
                          BNH(),
                          TNK(),
                          WeldedBeam()]
    

   
    args = []
    for p in MO_target_problems:
        args.append((100, p))

    num_workers = 4
    pool = mp.Pool(processes=num_workers)
    pool.starmap(main, ([arg for arg in args]))
    '''




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








