import numpy as np
#from test_function import fobj
from create_child import create_child, create_child_c
from sort_population import sort_population
from sklearn.metrics import mean_squared_error
import time

def cross_val(val_x, val_y, **kwargs):
    gpr = kwargs['gpr']
    val_x = val_x.reshape(-1, 1)
    pred_y = gpr.predict(val_x)
    mse = mean_squared_error(val_y, pred_y)

    val_x = val_x[0, 0]
    val_y = val_y[0]
    pred_y = pred_y[0, 0]

    print('cross validation on x %.2f, real_y is %0.2f, predicted_y is %0.2f, mse is %0.2f' % (val_x, val_y, pred_y, mse))
    return mse

def optimizer(problem, nobj, ncon, bounds, mut, crossp, popsize, its,  **kwargs):
    '''

    :param problem:
    :param nobj:
    :param ncon:
    :param bounds: upper and lower bounds of problem variables
    :param mut: mutation rate
    :param crossp:  crossover rate
    :param popsize:  population size
    :param its:
    :param val_data:
    :return:
    '''
    dimensions = len(bounds)
    pop_g = []
    archive_g = []
    all_cv = []
    pop_cv = []
    a = np.linspace(0, 2*popsize-1, 2*popsize, dtype=int)
   
    all_cv = np.zeros((2*popsize, 1))
    all_g = np.zeros((2*popsize, ncon))
    pop_g = np.zeros((popsize, ncon))
    pop_cv = np.zeros((2*popsize, 1))
    child_g = np.zeros((popsize, ncon))
    archive_g = pop_g
    all_x = np.zeros((2*popsize, dimensions))
    all_f = np.zeros((2*popsize, nobj))
    pop_f = np.zeros((popsize, nobj))
    child_f = np.zeros((popsize, nobj))
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_x = min_b + pop * diff
    archive_x = pop
    archive_f = pop_f
    for ind in range(popsize):
        if ncon != 0:
            pop_f[ind, :], pop_g[ind, :] = problem.evaluate(pop_x[ind, :], return_values_of=["F", "G"], **kwargs)
            tmp = pop_g
            tmp[tmp <= 0] = 0
            pop_cv = tmp.sum(axis=1)

        if ncon == 0:
            # print('initialization loglikelihood check send in %d th theta: %0.4f ' % (ind, pop_x[ind, :]))
            pop_f[ind, :] = problem.evaluate(pop_x[ind, :], return_values_of=["F"], **kwargs)
       
    # Over the generations
    for i in range(its):
        # child_x = create_child(dimensions, bounds, popsize, crossp, mut, pop)
        start = time.time()
        child_x = create_child_c(dimensions, bounds, popsize, crossp, mut, pop, pop_f, 20, 30)
        end = time.time()
        # print('create child time used %.4f' % (end-start))

        start = time.time()
        # Evaluating the offspring
        for ind in range(popsize):
            trial_denorm = min_b + child_x[ind, :] * diff
            if ncon != 0:
                child_f[ind, :], child_g[ind, :] = problem.evaluate(trial_denorm, return_values_of=["F", "G"], **kwargs)
            if ncon == 0:
                # print('over generation %d send in %d th theta: ' % (i, ind))
                child_f[ind, :] = problem.evaluate(trial_denorm, return_values_of=["F"], **kwargs)
        end = time.time()
        # print('population evaluation time used %.4f' % (end - start))


        start = time.time()
        # Parents and offspring
        all_x = np.append(pop, child_x, axis=0)
        all_f = np.append(pop_f, child_f, axis=0)
        if ncon != 0:
            all_g = np.append(pop_g, child_g, axis=0)
            all_g[all_g <= 0] = 0
            all_cv = all_g.sum(axis=1)
            infeasible = np.nonzero(all_cv)
            feasible = np.setdiff1d(a, infeasible)
        if ncon == 0:
            feasible = a
            infeasible = []
     
        feasible = np.asarray(feasible)
        feasible = feasible.flatten()
        # Selecting the parents for the next generation
        selected = sort_population(popsize, nobj, ncon, infeasible, feasible, all_cv, all_f)
        end = time.time()
        # print('sorting  time used %.4f' % (end - start))
        
        pop = all_x[selected, :]
        pop_f = all_f[selected, :]

        # insert a crossvalidation
        if ncon != 0:
            pop_g = all_g[selected, :]
            
        # Storing all solutions in tha archive
        archive_x = np.append(archive_x, child_x, axis=0)
        archive_f = np.append(archive_f, child_f)
        if ncon != 0:
            archive_g = np.append(archive_g, child_g)

    # Getting the variables in appropriate bounds    
    pop_x = min_b + pop * diff
    archive_x = min_b + archive_x * diff
    return pop_x, pop_f, pop_g, archive_x, archive_f, archive_g