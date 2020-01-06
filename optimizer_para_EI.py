import numpy as np
from create_child import create_child
from sort_population import sort_population
import multiprocessing as mp
import time


def distribute_x(pop_x_bunch, problem, ncon, param):
    # retrieve key value parameters
    kwargs = param[0]

    pop_x_2d = np.atleast_2d(pop_x_bunch)
    n_indv = pop_x_2d.shape[0]

    # evaluation results
    out_f_bunch = []
    # print(os.getpid())

    # as a matter of fact, bunch is 1 in this function
    for each_x in range(n_indv):
        # so far this evaluation is only for non-constrained problems

        if ncon != 0:
            out_f, out_g = problem.evaluate(pop_x_2d[each_x, :], return_values_of=["F", "G"], **kwargs)
            out_f_bunch.append([out_f, out_g])
        else:
            # print('EI check')
            out_f = problem.evaluate(pop_x_2d[each_x, :], return_values_of=["F"], **kwargs)
            out_f_bunch.append(out_f)

    return out_f_bunch


def para_population_val(popsize, pop_x, problem, ncon, **kwargs):

    # mp.freeze_support()
    # assign number of cpus to use
    num_workers = 1
    pool = mp.Pool(processes=num_workers)


    # separate population
    n = pop_x.shape[0]

    # work around the pool.apply by add [] to key-value parameters
    para = [kwargs]
    # print(pop_x[1])

    # when using following line starmap, it returns a 'too many indices for array' error, not debugged
    results = pool.starmap(distribute_x, [(pop_x_indiv, problem, ncon, para) for pop_x_indiv in pop_x[:]])

    '''
    results = []
    for i in range(popsize):
        pop_x_bunch = pop_x[i, :]
        results.append(pool.apply_async(distribute_x, (pop_x_bunch, problem, para)))
    '''
    pool.close()
    pool.join()
    f = np.array(results).ravel()
    # print(results)
    # print(f)

    '''
    # the following pair with apply_async
    f = []
    for r in results:
        f.append(r.get())
    '''
    # ! the reshape below does not process multi-objective problems
    if ncon == 0:
        f_pop = np.atleast_2d(f).reshape(-1, 1)
        return f_pop
    else:
        f = np.atleast_2d(f).reshape(-1, 2)
        fitness = np.atleast_2d(f[:, 0]).reshape(-1, 1)
        constraint = np.atleast_2d(f[:, 1]).reshape(-1, 1)
        return fitness, constraint




def optimizer(problem, nobj, ncon, bounds, mut, crossp, popsize, its, **kwargs):

    dimensions = len(bounds)
    pop_g = []
    archive_g = []
    all_cv = []
    pop_cv = []
    a = np.linspace(0, 2 * popsize - 1, 2 * popsize, dtype=int)

    all_cv = np.zeros((2 * popsize, 1))
    all_g = np.zeros((2 * popsize, ncon))
    pop_g = np.zeros((popsize, ncon))
    pop_cv = np.zeros((2 * popsize, 1))
    child_g = np.zeros((popsize, ncon))
    archive_g = pop_g
    all_x = np.zeros((2 * popsize, dimensions))
    all_f = np.zeros((2 * popsize, nobj))
    pop_f = np.zeros((popsize, nobj))
    child_f = np.zeros((popsize, nobj))
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_x = min_b + pop * diff
    archive_x = pop
    archive_f = pop_f

    # for each population evaluation, parallel can be conducted
    if ncon == 0:
        pop_f = para_population_val(popsize, pop_x, problem, ncon, **kwargs)

    '''
    if ncon != 0:
        for ind in range(popsize):
            pop_f[ind, :], pop_g[ind, :] = problem.evaluate(pop_x[ind, :], return_values_of=["F", "G"], **kwargs)
            tmp = pop_g
            tmp[tmp <= 0] = 0
            pop_cv = tmp.sum(axis=1)
    '''

    # Over the generations
    for i in range(its):
        child_x = create_child(dimensions, bounds, popsize, crossp, mut, pop)

        # Evaluating the offspring
        trial_denorm = min_b + child_x * diff
        if ncon == 0:
            child_f_fit = para_population_val(popsize, trial_denorm, problem, ncon, **kwargs)
            child_f = child_f_fit

        # start = time.time()
        # if ncon != 0:
            # child_f, child_g = para_population_val(popsize, trial_denorm, problem, ncon, **kwargs)
        # end = time.time()
        # lasts = (end - start) / 60.
        # print('population evaluation in generation %d, uses %.2f minutes' % (i, lasts))

        '''
        # The following code commented is functioning 
        # and in sequential execution way
        for ind in range(popsize):
            trial_denorm = min_b + child_x[ind, :] * diff
            if ncon != 0:
                child_f[ind, :], child_g[ind, :] = problem.evaluate(trial_denorm, return_values_of=["F", "G"], **kwargs)
            
            if ncon == 0:
                # population evaluation that can be parallelized
                child_f[ind, :] = problem.evaluate(trial_denorm, return_values_of=["F"], **kwargs)
        '''
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
        #start = time.time()
        selected = sort_population(popsize, nobj, ncon, infeasible, feasible, all_cv, all_f)
        #end = time.time()
        #lasts = (start - end) / 60.
        #print('sorting in generation %d, uses %.2f minutes' % (i, lasts))

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