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
    
    # Problem to run
    problem = G1()
    
    nobj=problem.n_obj
    ncon=problem.n_constr
    nvar=problem.n_var
    bounds = np.zeros((nvar,2))
    for i in range(nvar):
        bounds[i][1]=problem.xu[i]
        bounds[i][0]=problem.xl[i]
    bounds=bounds.tolist()
    
    ## Running the optimizer
    result = optimizer(problem,nobj,ncon,bounds,mut=0.8,crossp=0.7,popsize=100,its=100)
    #
    # Analyzing the results
    final_x,final_f,final_g,final_cv,feas_x,feas_f,final_nd_x,final_nd_f = process(nobj,ncon,result)
    
    if nobj==2:
    # Plotting the final population and the nondominated front
        plt.plot(final_f[:,0],final_f[:,1],'r.')
        plt.plot(final_nd_f[:,0],final_nd_f[:,1],'bo')
    
    if nobj==3:
    # Plotting the final population and the nondominated front
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.scatter(final_f[:,0], final_f[:,1], final_f[:,2], c='r', marker='.')
        ax.scatter(final_nd_f[:,0], final_nd_f[:,1], final_nd_f[:,2], c='b', marker='o')
        plt.show()
    
    if nobj==1:
        print(final_nd_x)
        print(final_nd_f)
        
        
