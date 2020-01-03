import numpy as np
import pygmo as pg
def sort_population(popsize,nobj,ncon,infeasible,feasible,all_cv,all_f):
    l2=[]
    l1=[]
    sl=[]
    ff=[]
    if ncon!=0:
        infeasible=np.asarray(infeasible)
        infeasible=infeasible.flatten()
        index1 = all_cv[infeasible].argsort()
        index1=index1.tolist()
        l2=infeasible[index1]
    if len(feasible)>=1:
        ff = all_f[feasible,:]
        if nobj==1:
            ff=ff.flatten()
            index1 = ff.argsort()
            index1=index1.tolist()
            l1=feasible[index1]
        if nobj>1:
            sl = pg.sort_population_mo(ff)
            l1 = feasible[sl]
    order=np.append(l1, l2, axis=0)
    order=order.flatten()
    selected=order[0:popsize]
    selected=selected.flatten()
    selected=selected.astype(int)
    return selected