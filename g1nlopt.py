from pymop.problems.g import G1
import numpy as np
if __name__ == "__main__":
# Problem to run
    problem = G1()
    nobj = problem.n_obj
    print('nobj', nobj)
    ncon = problem.n_constr
    print('ncon', ncon)
    nvar = problem.n_var
    print('nvar', nvar)
    bounds = np.zeros((nvar, 2))

    for i in range(nvar):
        bounds[i][1] = problem.xu[i]
        bounds[i][0] = problem.xl[i]
    bounds = bounds.tolist()
    print('bounds', bounds)

