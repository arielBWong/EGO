import pygmo as pg
import numpy as np


def process(nobj, ncon, result):
    final_x = result[0]
    final_f = result[1]
    final_g = result[2]


#    popsize, nobj=final_f.shape()
    final_cv = []
    feas_x = []
    feas_f = []
    final_nd_x = []
    final_nd_f = []
    popsize = len(final_f)
    a = np.linspace(0, popsize-1, popsize, dtype=int)
    feasible = a

    if ncon != 0:
        final_g[final_g <= 0] = 0
        final_cv = final_g.sum(axis=1)
        infeasible = np.nonzero(final_cv)
        feasible = np.setdiff1d(a, infeasible)
        feas_f = final_f[feasible, :]
        feas_x = final_x[feasible, :]

    if ncon == 0:
        feas_f = final_f
        feas_x = final_x

    if nobj > 1 and len(feasible) > 1:
        ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(feas_f)
        r1 = ndf[0]
        final_nd_x = feas_x[r1, :]
        final_nd_f = feas_f[r1, :]

    if nobj == 1 and len(feasible) > 1:
        final_nd_x = feas_x[0, :]
        final_nd_f = feas_f[0, :]

    return final_x, final_f, final_g, final_cv, feas_x, feas_f, final_nd_x, final_nd_f
