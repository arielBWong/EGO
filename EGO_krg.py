import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import optimizer_EI
from pymop.factory import get_problem_from_func
from pymop import ZDT1, ZDT2, ZDT3, ZDT4, DTLZ1, G1, DTLZ2, DTLZ4, BNH, Carside, Kursawe, OSY, Truss2D, WeldedBeam, TNK
from EI_krg import acqusition_function, close_adjustment
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
import utilities



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


def saveNameConstr(problem_name, seed_index, method, run_signature):

    working_folder = os.getcwd()
    result_folder = working_folder + '\\outputs' + '\\' + problem_name + '_' + run_signature
    if not os.path.isdir(result_folder):
        # shutil.rmtree(result_folder)
        # os.mkdir(result_folder)
        os.mkdir(result_folder)
    # else:
    # os.mkdir(result_folder)
    savename_x = result_folder + '\\best_x_seed_' + str(seed_index) + '_' + method + '.joblib'
    savename_y = result_folder + '\\best_f_seed_' + str(seed_index) + '_' + method +'.joblib'
    savename_FEs = result_folder + '\\FEs_seed_' + str(seed_index) + '_' + method +'.joblib'
    return savename_x, savename_y, savename_FEs


def lexsort_with_certain_row(f_matrix, target_row_index):

    # f_matrix should have the size of n_obj * popsize
    # determine min
    target_row = f_matrix[target_row_index, :].copy()
    f_matrix = np.delete(f_matrix, target_row_index, axis=0)  # delete axis is opposite to normal


    f_min = np.min(f_matrix, axis=1)
    f_min = np.atleast_2d(f_min).reshape(-1, 1)
    # according to np.lexsort, put row with largest min values last row
    f_min_count = np.count_nonzero(f_matrix == f_min, axis=1)
    f_min_accending_index = np.argsort(f_min_count)
    # adjust last_f_pop
    last_f_pop = f_matrix[f_min_accending_index, :]

    # add saved target
    last_f_pop = np.vstack((last_f_pop, target_row))

    # apply np.lexsort (works row direction)
    lexsort_index = np.lexsort(last_f_pop)
    # print(last_f_pop[:, lexsort_index])
    selected_x_index = lexsort_index[0]

    return selected_x_index

def check_krg_ideal_points(krg, n_var, n_constr, n_obj, low, up):
    x_krg = []
    f_krg = []

    last_x_pop = []
    last_f_pop = []

    n_krg = len(krg)
    x_pop_size = 50
    x_pop_gen = 50

    # identify ideal x and f for each objective
    for k in krg:
        problem = single_krg_optim.single_krg_optim(k, n_var, n_constr, 1, low, up)
        single_bounds = np.vstack((low, up)).T.tolist()

        pop_x, pop_f, pop_g, archive_x, archive_f, archive_g, record = optimizer_EI.optimizer(problem,
                                                                                              nobj=1,
                                                                                              ncon=0,
                                                                                              bounds=single_bounds,
                                                                                              recordFlag=False,
                                                                                              pop_test=None,
                                                                                              mut=0.1,
                                                                                              crossp=0.9,
                                                                                              popsize=x_pop_size,
                                                                                              its=x_pop_gen)
        # save the last population
        last_x_pop = np.append(last_x_pop, pop_x)
        last_f_pop = np.append(last_f_pop, pop_f) # for test

    # long x
    last_x_pop = np.atleast_2d(last_x_pop).reshape(n_obj, -1)

    x_estimate = []
    for i in range(n_obj):
        x_pop = last_x_pop[i, :]
        x_pop = x_pop.reshape(x_pop_size, -1)
        all_f = []

        for k in krg:
            f_k, _ = k.predict(x_pop)
            all_f = np.append(all_f, f_k)
        # reorganise all f in obj * popsize shape
        all_f = np.atleast_2d(all_f).reshape(n_obj, -1)
        x_index = lexsort_with_certain_row(all_f, i)

        x_estimate = np.append(x_estimate, x_pop[x_index, :])

    x_estimate = np.atleast_2d(x_estimate).reshape(n_obj, -1)

    return x_estimate


def update_nadir(train_x,
                 train_y,
                 cons_y,
                 next_y,
                 problem,
                 x_krg,
                 krg,
                 krg_g,
                 nadir,
                 ideal,
                 enable_crossvalidation):

    '''
    # plot train_y
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(train_y[:, 0], train_y[:, 1], marker='x', c='blue')
    f1 = [nadir[0], nadir[0], ideal[0], ideal[0], nadir[0]]
    f2 = [nadir[1], ideal[1], ideal[1], nadir[1], nadir[1]]
    line = Line2D(f1, f2, c='green')
    ax.add_line(line)
    '''

    # check with new nadir and ideal point
    # update them if they do not meet ideal/nadir requirement
    n_var = problem.n_var
    n_obj = problem.n_obj
    x1 = np.atleast_2d(x_krg[0]).reshape(-1, n_var)
    x2 = np.atleast_2d(x_krg[1]).reshape(-1, n_var)

    # add new evaluation when next_y is better in any direction compared with
    # current ideal
    if next_y is not None:
        if np.any(next_y < ideal, axis=1):
            print('new next_y better than ideal')
            print(next_y)
            print(ideal)

            out = {}
            problem._evaluate(x1, out)
            y1 = out['F']

            if 'G' in out.keys():
                g1 = out['G']

            problem._evaluate(x2, out)
            y2 = out['F']

            if 'G' in out.keys():
                g2 = out['G']

            # whether there is smaller point than nadir
            train_x = np.vstack((train_x, x1, x2))
            train_y = np.vstack((train_y, y1, y2))
            if 'G' in out.keys():
                cons_y = np.vstack((cons_y, g1, g2))

            # solve the too small distance problem
            train_y = close_adjustment(train_y)
            nd_front = utilities.return_nd_front(train_y)

            nadir = np.amax(nd_front, axis=0)
            ideal = np.amin(nd_front, axis=0)

            print('ideal update')
            print(ideal)
            krg, krg_g = cross_val_krg(train_x, train_y, cons_y, enable_crossvalidation)


    '''
      
    ax.scatter(y1[:, 0], y1[:, 1], marker='o', c='m')
    ax.scatter(y2[:, 0], y2[:, 1], marker='o', c='red')
   
    # add new line
    f1 = [nadir_new[0], nadir_new[0], ideal_new[0], ideal_new[0], nadir_new[0]]
    f2 = [nadir_new[1], ideal_new[1], ideal_new[1], nadir_new[1], nadir_new[1]]

    line = Line2D(f1, f2, c='red')
    ax.add_line(line)

    ax.scatter(nd_front[:, 0], nd_front[:, 1], c='yellow')
    # ax.scatter(train_y[-1, 0], train_y[-1, 1], marker='D', c='g')
    ax.scatter(y1[:, 0], y1[:, 1], s=200, marker='_', c='m')
    ax.scatter(y2[:, 0], y2[:, 1], s=200, marker='_', c='red')

    up_lim = np.max(np.amax(train_y, axis=0))
    low_lim = np.min(np.amin(train_y, axis=0))
    ax.set(xlim=(low_lim-1, up_lim+1), ylim=(low_lim-1, up_lim+1))
    plt.show()
    '''



    return train_x, train_y, cons_y, krg, krg_g, nadir, ideal



def initNormalization(train_y):

    nd_front = utilities.return_nd_front(train_y)
    nadir = np.amax(nd_front, axis=0)
    ideal = np.amin(nd_front, axis=0)

    return nadir, ideal



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
            print('No feasible solutions in this iteration %d')
    else:
        evalparas['feasible'] = -1

    return evalparas


def post_process(train_x, train_y, cons_y, target_problem, seed_index, method_selection, run_signature):

    n_sur_objs = target_problem.n_obj
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

    savename_x, savename_f, savename_FEs = saveNameConstr(target_problem.name(), seed_index, method_selection, run_signature)

    dump(best_x_out, savename_x)
    dump(best_f_out, savename_f)


def referece_point_check(train_x, train_y, cons_y,  ideal_krg, x_out, target_problem, krg, krg_g, enable_crossvalidation):

    # check whether there is any f that is even better/smaller than ideal
    n_vals = target_problem.n_var
    n_sur_cons = target_problem.n_constr

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
    return krg, krg_g


def main(seed_index, target_problem, enable_crossvalidation, method_selection, run_signature):

    # this following one line is for work around 1d plot in multiple-processing settings
    multiprocessing.freeze_support()
    np.random.seed(seed_index)
    recordFlag = False

    #print(pyDOE.lhs(2, 10))


    print('Problem')
    print(target_problem.name())
    print('seed %d' % seed_index)

    # collect problem parameters: number of objs, number of constraints
    n_sur_objs = target_problem.n_obj
    n_sur_cons = target_problem.n_constr
    n_vals = target_problem.n_var
    if n_sur_objs > 2:
        stop = 200
    else:
        stop = 100

    number_of_initial_samples = 11 * n_vals - 1
    n_iter = 300  # stopping criterion set
    train_x, train_y, cons_y = init_xy(number_of_initial_samples, target_problem)

    # for evalparas compatibility
    nadir, ideal = initNormalization(train_y)

    # kriging initialization
    krg, krg_g = cross_val_krg(train_x, train_y, cons_y, enable_crossvalidation)

    # estimate nadir and ideal
    if method_selection == 'hvr' or method_selection == 'eim_r':
        x_out = check_krg_ideal_points(krg, n_vals, n_sur_cons, n_sur_objs, target_problem.xl, target_problem.xu)
        train_x, train_y, cons_y, krg, krg_g, nadir, ideal = update_nadir(train_x,
                                                                          train_y,
                                                                          cons_y,
                                                                          None,
                                                                          target_problem,
                                                                          x_out,
                                                                          krg,
                                                                          krg_g,
                                                                          nadir,
                                                                          ideal,
                                                                          enable_crossvalidation)

    # create EI problem
    evalparas = {'train_x':  train_x,
                 'train_y': train_y,
                 'krg': krg,
                 'krg_g': krg_g,
                 'nadir': nadir,
                 'ideal': ideal,
                 'feasible': np.array([]),
                 'ei_method': method_selection}

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

        if train_x.shape[0] % 5 == 0:
            saveName  = 'intermediate\\' + target_problem.name() + '_' + method_selection + '_seed_' + str(seed_index) + 'krg_iteration_' + str(iteration) + '.joblib'
            dump(krg, saveName)

            saveName = 'intermediate\\' + target_problem.name() + '_' + method_selection +  '_seed_' + str(seed_index) + 'nd_iteration_' + str(iteration) + '.joblib'
            utilities.save_pareto_front(train_y, saveName)

            saveName = 'intermediate\\' + target_problem.name() + '_' + method_selection + '_seed_' + str(
                seed_index) + '_nadir_iteration_' + str(iteration) + '.joblib'
            dump(nadir, saveName)

            saveName = 'intermediate\\' + target_problem.name() + '_' + method_selection + '_seed_' + str(
                seed_index) + '_ideal_iteration_' + str(iteration) + '.joblib'
            dump(ideal, saveName)

            recordFlag = True

        start = time.time()
        # main loop for finding next x
        candidate_x = np.zeros((1, n_vals))
        candidate_y = []
        for restart in range(4):

            '''
            # use best estimated location for
            ndfront = utilities.return_nd_front(train_y)
            _, _, _, _, _, pop_test,_ = utilities.samplex2f(ndfront,
                                                          n_sur_objs,
                                                          n_vals,
                                                          krg,
                                                          iteration,
                                                          method_selection,
                                                          nadir,
                                                          ideal)
            pop_test_bounds = np.atleast_2d(x_bounds).T
            pop_test = (pop_test - pop_test_bounds[0, :])/(pop_test_bounds[1, :] - pop_test_bounds[0, :])

            '''
            pop_x, pop_f, pop_g, archive_x, archive_f, archive_g, record = optimizer_EI.optimizer(ei_problem,
                                                                                                  ei_problem.n_obj,
                                                                                                  ei_problem.n_constr,
                                                                                                  x_bounds,
                                                                                                  recordFlag,
                                                                                                  # pop_test=pop_test,
                                                                                                  pop_test=None,
                                                                                                  mut=0.1,
                                                                                                  crossp=0.9,
                                                                                                  popsize=100,
                                                                                                  its=100,
                                                                                                  **evalparas)
            candidate_x = np.vstack((candidate_x, pop_x[0, :]))
            candidate_y = np.append(candidate_y, pop_f[0, :])

            if recordFlag:
                saveName = 'intermediate\\' + target_problem.name() + '_' + method_selection+ '_seed_' + str(seed_index) + 'search_record_iteration_' + str(iteration) + '_restart_' + str(restart) + '.joblib'
                dump(record, saveName)

        end = time.time()
        lasts = (end - start)

        # print('propose to next x in iteration %d uses %.2f sec' % (iteration, lasts))
        # propose next_x location
        w = np.argwhere(candidate_y == np.min(candidate_y))
        next_x = candidate_x[w[0]+1, :]

        # dimension re-check
        next_x = np.atleast_2d(next_x).reshape(-1, n_vals)

        # generate corresponding f and g
        out = {}
        target_problem._evaluate(next_x, out)
        next_y = out['F']

        if train_x.shape[0] % 5 == 0:
            saveName  = 'intermediate\\' + target_problem.name() + '_' + method_selection + '_seed_' + str(seed_index) + 'nextF_iteration_' + str(iteration) + '.joblib'
            dump(next_y, saveName)

        recordFlag = False
        if 'G' in out.keys():
            next_cons_y = out['G']
            next_cons_y = np.atleast_2d(next_cons_y)
        else:
            next_cons_y = None

        # add new proposed data
        train_x = np.vstack((train_x, next_x))
        train_y = np.vstack((train_y, next_y))
        # print('train x  size %d' % train_x.shape[0])

        if n_sur_cons > 0:
            cons_y = np.vstack((cons_y, next_cons_y))

        start = time.time()
        # use extended data to train krging model
        krg, krg_g = cross_val_krg(train_x, train_y, cons_y, enable_crossvalidation)
        end = time.time()  # on seconds

        # new evaluation added depending on condition
        if method_selection == 'hvr' or method_selection == 'eim_r':
            x_out = check_krg_ideal_points(krg, n_vals, n_sur_cons, n_sur_objs, target_problem.xl, target_problem.xu)
            train_x, train_y, cons_y, krg, krg_g, nadir, ideal = update_nadir(train_x,
                                                                              train_y,
                                                                              cons_y,
                                                                              next_y,
                                                                              target_problem,
                                                                              x_out,
                                                                              krg,
                                                                              krg_g,
                                                                              nadir,
                                                                              ideal,
                                                                              enable_crossvalidation)


        lasts = (end - start)
        print('cross-validation %d uses %.2f sec' % (iteration, lasts))


        evalparas['train_x'] = train_x
        evalparas['train_y'] = train_y
        evalparas['krg'] = krg
        evalparas['krg_g'] = krg_g
        evalparas['nadir'] = nadir
        evalparas['ideal'] = ideal

        # stopping criteria
        sample_n = train_x.shape[0]
        if sample_n >= stop:
            break

    end_all = time.time()
    print('overall time %.4f ' % (end_all - start_all))

    post_process(train_x, train_y, cons_y, target_problem, seed_index, method_selection, run_signature)


if __name__ == "__main__":


    MO_target_problems = [# ZDT3(n_var=6),
                          #ZDT1(n_var=6),
                          #ZDT2(n_var=6),
                          DTLZ2(n_var=8, n_obj=3),
                          # DTLZ4(n_var=8, n_obj=3),
                          # DTLZ1(n_var=6, n_obj=2),
                          # Kursawe(),
                          # Truss2D(),
                          # TNK()]
                          # BNH(),
                          # WeldedBeam()
                          ]

    # target_problem = MO_target_problems[1]
    # for seed in range(0, 10):
        # main(seed, target_problem, False, 'eim')


    args = []
    run_sig = ['hvr', 'eim_nobeyondfix_r']
    methods_ops = ['hvr', 'eim_r']

    for seed in range(0, 10):
        for target_problem in MO_target_problems:
            args.append((seed, target_problem, True, methods_ops[0], run_sig[0]))

    # main(8, MO_target_problems[0], False, 'hvr', 'hvr')


    num_workers = 6
    pool = mp.Pool(processes=num_workers)
    pool.starmap(main, ([arg for arg in args]))


    
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

    
    
    MO_target_problems = [ZDT3(n_var=6),
                          ZDT1(n_var=6),
                          ZDT2(n_var=6),
                          DTLZ2(n_var=8, n_obj=3),
                          DTLZ4(n_var=8, n_obj=3),
                          DTLZ1(n_var=6, n_obj=2),
                          # Kursawe(),
                          # Truss2D(),
                          # TNK()]
                          # BNH(),
                          # WeldedBeam()
                          ]
    
    methods_ops = ['hvr']#, 'hv', 'eim']
   
    args = []
    for seed in range(2, 5):
        for p in MO_target_problems:
            for m in methods_ops:
                args.append((seed, p, False, m))

    num_workers = 1
    pool = mp.Pool(processes=num_workers)
    pool.starmap(main, ([arg for arg in args]))
    
    '''









