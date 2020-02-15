import numpy as np
from create_child import create_child, create_child_c
from sort_population import sort_population
import time
from scipy.optimize import differential_evolution


def optimizer(problem, nobj, ncon, bounds, recordFlag, pop_test, mut, crossp, popsize, its,  **kwargs):

    record_f = list()
    record_x = list()

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

    # print(pop)

    if pop_test is not None:
        pop = pop_test
        pop_x = min_b + pop * diff


    if ncon != 0:
        pop_f, pop_g = problem.evaluate(pop_x, return_values_of=["F", "G"], **kwargs)
        tmp = pop_g
        tmp[tmp <= 0] = 0
        pop_cv = tmp.sum(axis=1)

    if ncon == 0:
        # np.savetxt('test_x.csv', pop_x, delimiter=',')
        pop_f = problem.evaluate(pop_x, return_values_of=["F"], **kwargs)


    # Over the generations
    for i in range(its):

        start = time.time()
        child_x = create_child_c(dimensions, bounds, popsize, crossp, mut, pop, pop_f, 20, 30)
        end = time.time()
        # print('create child time used %.4f' % (end - start))

        start = time.time()
        trial_denorm = min_b + child_x * diff
        if ncon != 0:
            child_f, child_g = problem.evaluate(trial_denorm, return_values_of=["F", "G"], **kwargs)
        if ncon == 0:
            child_f = problem.evaluate(trial_denorm, return_values_of=["F"], **kwargs)
        end = time.time()
        # print(' evaluation time used %.4f' % (end - start))

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

        start = time.time()

        # Selecting the parents for the next generation
        selected = sort_population(popsize, nobj, ncon, infeasible, feasible, all_cv, all_f)
        end = time.time()
        # print('sort time used %.4f' % (end - start))

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

        if recordFlag:
            # record all best_individual
            record_f = np.append(record_f, pop_f[0, :])
            record_x = np.append(record_x, min_b + diff * pop[0, :])

    # Getting the variables in appropriate bounds
    pop_x = min_b + pop * diff
    archive_x = min_b + archive_x * diff

    return pop_x, pop_f, pop_g, archive_x, archive_f, archive_g, (record_f, record_x)


def optimizer_DE(problem, nobj, ncon, bounds, recordFlag, pop_test, F, CR, NP, itermax, **kwargs):
    #  NP: number of population members/popsize
    #  itermax: number of generation

    dimensions = len(bounds)


    # Check input variables
    VTR = -np.inf
    refresh = 0
    F = 0.8
    CR = 0.8
    strategy = 6
    use_vectorize = 1

    if NP < 5:
        NP = 5
        print('pop size is increased to minimize size 5')

    if CR < 0 or CR > 1:
        CR = 0.5
        print('CR should be from interval [0,1]; set to default value 0.5')

    if itermax <= 0:
        itermax = 200
        print('generation size is set to default 200')

    # Initialize population and some arrays
    # if pop is a matrix of size NPxD. It will be initialized with random
    # values between the min and max values of the parameters

    min_b, max_b = np.asarray(bounds).T
    pop = np.random.rand(NP, dimensions)
    pop_x = min_b + pop * (max_b - min_b)
    # for test
    # pop_x = np.loadtxt('pop.csv', delimiter=',')

    XVmin = np.repeat(np.atleast_2d(min_b), NP, axis=0)
    XVmax = np.repeat(np.atleast_2d(max_b), NP, axis=0)



    if ncon != 0:
        pop_f, pop_g = problem.evaluate(pop_x, return_values_of=["F", "G"], **kwargs)
        tmp = pop_g.copy()
        tmp[tmp <= 0] = 0
        pop_cv = tmp.sum(axis=1)

    if ncon == 0:
        # np.savetxt('test_x.csv', pop_x, delimiter=',')
        pop_f = problem.evaluate(pop_x, return_values_of=["F"], **kwargs)



    # best member of current iteration
    bestval = np.min(pop_f)  # single objective only
    ibest = np.where(pop_f == bestval)  # what if multiple best values?
    bestmemit = pop_x[ibest[0][0]]  # np.where return tuple of (row_list, col_list)

    # save best_x ever
    bestmem = bestmemit

    # DE-Minimization
    # popold is the population which has to compete. It is static through one
    # iteration. pop is the newly emerging population
    # initialize bestmember  matrix
    bm = np.zeros((NP, dimensions))

    # intermediate population of perturbed vectors
    ui = np.zeros((NP, dimensions))

    # rotating index array (size NP)
    rot = np.arange(0, NP)

    # rotating index array (size D)
    rotd = np.arange(0, dimensions)  # (0:1:D-1);

    iter = 1
    while iter < itermax and bestval > VTR:
        # save the old population
        # print('iteration: %d' % iter)
        oldpop_x = pop_x.copy()

        # index pointer array
        ind = np.random.permutation(4) + 1

        # shuffle locations of vectors
        a1 = np.random.permutation(NP)
        # for test
        # a1 = np.loadtxt('a1.csv', delimiter=',')
        # a1 = np.array(list(map(int, a1)))-1

        # rotate indices by ind(1) positions
        rt = np.remainder(rot + ind[0], NP)
        # rotate vector locations
        a2 = a1[rt]
        # for test
        # a2 = np.loadtxt('a2.csv', delimiter=',')
        # a2 = np.array(list(map(int, a2)))-1

        rt = np.remainder(rot + ind[1], NP)
        a3 = a2[rt]
        # for test
        # a3 = np.loadtxt('a3.csv', delimiter=',')
        # a3 = np.array(list(map(int, a3)))-1

        rt = np.remainder(rot + ind[2], NP)
        a4 = a3[rt]
        # for test
        # a4 = np.loadtxt('a4.csv', delimiter=',')
        # a4 = np.array(list(map(int, a4)))-1

        rt = np.remainder(rot + ind[3], NP)
        a5 = a4[rt]
        # for test
        # a5 = np.loadtxt('a5.csv', delimiter=',')
        # a5 = np.array(list(map(int, a5)))-1

        # shuffled population 1
        pm1 = oldpop_x[a1, :]
        pm2 = oldpop_x[a2, :]
        pm3 = oldpop_x[a3, :]
        pm4 = oldpop_x[a4, :]
        pm5 = oldpop_x[a5, :]

        # population filled with the best member of the last iteration
        # print(bestmemit)
        for i in range(NP):
            bm[i, :] = bestmemit

        mui = np.random.rand(NP, dimensions) < CR
        # mui = np.loadtxt('mui.csv', delimiter=',')

        if strategy > 5:
            st = strategy - 5
        else:
            # exponential crossover
            st = strategy
            # transpose, collect 1's in each column
            # did not implement following strategy process

        # inverse mask to mui
        # mpo = ~mui
        mpo = mui < 0.5


        if st == 1:  # DE/best/1
            # differential variation
            ui = bm + F * (pm1 - pm2)
            # crossover
            ui = oldpop_x * mpo + ui * mui

        if st == 2:  # DE/rand/1
            # differential variation
            ui = pm3 + F * (pm1 - pm2)
            # crossover
            ui = oldpop_x * mpo + ui * mui

        if st == 3:  # DE/rand-to-best/1
            ui = oldpop_x + F * (bm - oldpop_x) + F * (pm1 - pm2)
            ui = oldpop_x * mpo + ui * mui

        if st == 4:  # DE/best/2
            ui = bm + F * (pm1 - pm2 + pm3 - pm4)
            ui = oldpop_x * mpo + ui * mui

        if st == 5:  #DE/rand/2
            ui = pm5 + F * (pm1 - pm2 + pm3 - pm4)
            ui = oldpop_x * mpo + ui * mui


        # correcting violations on the lower bounds of the variables
        # validate components
        maskLB = ui > XVmin
        maskUB = ui < XVmax

        # part one: valid points are saved, part two/three beyond bounds are set as bounds
        ui = ui * maskLB * maskUB + XVmin * (~maskLB) + XVmax * (~maskUB)

        # Select which vectors are allowed to enter the new population
        if use_vectorize == 1:

            if ncon != 0:
                pop_f_temp, pop_g_temp = problem.evaluate(ui, return_values_of=["F", "G"], **kwargs)
                tmp = pop_g_temp.copy()
                tmp[tmp <= 0] = 0
                pop_cv_temp = tmp.sum(axis=1)

            if ncon == 0:
                # np.savetxt('test_x.csv', pop_x, delimiter=',')
                pop_f_temp = problem.evaluate(ui, return_values_of=["F"], **kwargs)

            # if competitor is better than value in "cost array"
            indx = pop_f_temp <= pop_f
            # replace old vector with new one (for new iteration)
            change = np.where(indx)
            pop_x[change[0], :] = ui[change[0], :]
            pop_f[change[0], :] = pop_f_temp[change[0], :]

            # we update bestval only in case of success to save time
            indx = pop_f_temp < bestval
            if np.sum(indx) != 0:
                # best member of current iteration
                bestval = np.min(pop_f_temp)  # single objective only
                ibest = np.where(pop_f_temp == bestval)  # what if multiple best values?
                if len(ibest[0]) > 1:
                    print(
                        "multiple best values, selected first"
                    )
                bestmem = ui[ibest[0][0], :]





            # freeze the best member of this iteration for the coming
            # iteration. This is needed for some of the strategies.
            bestmemit = bestmem.copy()

        if refresh == 1:
            print('Iteration: %d,  Best: %.4f,  F: %.4f,  CR: %.4f,  NP: %d' % (iter, bestval, F, CR, NP))

        iter = iter + 1

        del oldpop_x

    return np.atleast_2d(bestmem), np.atleast_2d(bestval)






