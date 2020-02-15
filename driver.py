def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]

if __name__ == "__main__":
    clear_all()
    # Driver for optimization
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import pygmo as pg
    from optimizer import optimizer
    from process import process
    
    from pymop.problem import Problem
    from pymop.problems.dtlz import DTLZ1, DTLZ2, DTLZ3, DTLZ4
    from pymop.problems.tnk import TNK
    from pymop.problems.osy import OSY
    from pymop.problems.ctp import CTP1
    from pymop.problems.ctp import CTP2
    from pymop.problems.ctp import CTP3
    from pymop.problems.ctp import CTP4
    from pymop.problems.ctp import CTP5
    from pymop.problems.ctp import CTP6
    from pymop.problems.ctp import CTP7
    from pymop.problems.ctp import CTP8
    from pymop.problems.dtlz import DTLZ1
    from pymop.problems.dtlz import DTLZ2
    from pymop.problems.dtlz import DTLZ3
    from pymop.problems.dtlz import DTLZ4
    from pymop.problems.dtlz import DTLZ5
    from pymop.problems.dtlz import DTLZ6
    from pymop.problems.dtlz import DTLZ7
    from pymop.problems.cdtlz import C1DTLZ1
    from pymop.problems.cdtlz import C1DTLZ3
    from pymop.problems.cdtlz import C2DTLZ2
    from pymop.problems.cdtlz import C3DTLZ4
    from pymop.problems.cantilevered_beam import CantileveredBeam
    from pymop.problems.pressure_vessel import PressureVessel
    from pymop.problems.welded_beam import WeldedBeam
    from pymop.problems.zdt import ZDT1
    from pymop.problems.zdt import ZDT2
    from pymop.problems.zdt import ZDT3
    from pymop.problems.zdt import ZDT4
    from pymop.problems.zdt import ZDT6
    from pymop.problems.g import G1
    from pymop.problems.g import G2
    from pymop.problems.g import G3
    from pymop.problems.g import G4
    from pymop.problems.g import G5
    from pymop.problems.g import G6
    from pymop.problems.g import G7
    from pymop.problems.g import G8
    from pymop.problems.g import G9
    from pymop.problems.g import G10
    from surrogate_problems import MO_linearTest
    from pymop.problems import Ackley
    import os
    import time
    from joblib import dump, load
    from optimizer_EI import optimizer_DE
    
    # Problem to run
    problem = G1()
    problem = Ackley(n_var=2)
    # problem = MO_linearTest.MO_test()
    # problem = DTLZ2(n_var=3, n_obj=2)

    np.random.seed(100)
    
    nobj = problem.n_obj
    ncon = problem.n_constr
    nvar = problem.n_var
    bounds = np.zeros((nvar, 2))
    for i in range(nvar):
        bounds[i][1] = problem.xu[i]
        bounds[i][0] = problem.xl[i]
    bounds = bounds.tolist()

    start = time.time()
    ## Running the optimizer
    evalparas = {}
    bestx, bestf = optimizer_DE(problem,
                                nobj,
                                ncon,
                                bounds,
                                recordFlag=False,
                                pop_test=None,
                                F=0.1,
                                CR=0.9,
                                NP=100,
                                itermax=100,
                                **evalparas)
    end = time.time()
    print('de optimizer evaluation: %0.4f' % (end-start))
    print(bestx)
    print(bestf)


    '''
    # Analyzing the results
    final_x, final_f, final_g, final_cv, feas_x, feas_f, final_nd_x, final_nd_f = process(nobj, ncon, result)
    
    if nobj == 2:
    # Plotting the final population and the nondominated front
        plt.plot(final_f[:, 0], final_f[:, 1], 'r.')
        plt.plot(final_nd_f[:, 0], final_nd_f[:, 1], 'bo')
        plt.show()
    
    if nobj == 3:
    # Plotting the final population and the nondominated front
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(final_f[:, 0], final_f[:, 1], final_f[:, 2], c='r', marker='.')
        ax.scatter(final_nd_f[:, 0], final_nd_f[:, 1], final_nd_f[:, 2], c='b', marker='o')
        plt.show()

        print(final_nd_f)
        a = np.sum(final_nd_f, axis=1)
        print(len(final_nd_f))
    
    if nobj == 1:
        print(final_nd_x)
        print(final_nd_f)

    # save pareto_front
    problem_name = problem.name()
    working_folder = os.getcwd()
    result_folder = working_folder + '\\NSGA2' + '\\' + problem_name
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    saveName = result_folder + '\\' + 'pareto_f.joblib'
    dump(final_nd_f, saveName)
    '''
