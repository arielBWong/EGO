import autograd.numpy as anp
from surrogate_problems.sur_problem_base import Problem
from sklearn.utils.validation import check_array
import numpy as np
from scipy.special import comb
from itertools import combinations
import pygmo as pg
import multiprocessing as mp

from optproblems.wfg import WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9

def vec_nchoosek(v, k):
    """v is a vector, k is the number of elements to be chosen"""
    k_obj = combinations(v, k)
    out = []
    for ki in k_obj:
        out = np.append(out, ki)

    out = np.atleast_2d(out).reshape(-1, k)
    return out

def uniform_points(n_samples, n_obj):
    samples = None
    h1 = 1
    while comb(h1 + n_obj, n_obj - 1) <= n_samples:
        h1 = h1 + 1

    compo1 = vec_nchoosek(np.arange(1, h1 + n_obj), n_obj - 1)
    t = np.atleast_2d(np.arange(0, n_obj - 1))
    compo2 = np.repeat(t, comb(h1 + n_obj - 1, n_obj - 1), axis=0)
    W = compo1 - compo2 - 1

    t = np.atleast_2d(np.zeros((W.shape[0], 1))) + h1
    w1 = np.hstack((W, t))
    t2 = np.atleast_2d(np.zeros((W.shape[0], 1)))
    w2 = np.hstack((t2, W))
    W = (w1 - w2) / h1

    if h1 < n_obj:
        h2 = 0
        while comb(h1 + n_obj - 1, n_obj - 1) + comb(h2 + n_obj, n_obj - 1) <= n_samples:
            h2 = h2 + 1

        if h2 > 0:
            t1 = vec_nchoosek(np.arange(1, h2 + n_obj), n_obj - 1)
            t2 = np.atleast_2d(np.arange(0, n_obj - 1))
            t2 = np.repeat(t2, comb(h2 + n_obj - 1, n_obj - 1), axis=0)
            W2 = t1 - t2 - 1

            t1 = np.atleast_2d(np.zeros((W2.shape[0], 1))) + h2
            t1 = np.hstack((W2, t1))
            t2 = np.atleast_2d(np.zeros((W2.shape[0], 1)))
            t2 = np.hstack((t2, W2))
            W2 = (t1 - t2) / h2

            t3 = W2 / 2 + 1 / (2 * n_obj)
            W = np.vstack((W, t3))

    # fix 0
    W = np.maximum(W, 1e-6)
    new_sample_size = W.shape[0]
    return new_sample_size, W

def pareto_front(n_obj, n_pareto_points=100):
    new_n, pf = uniform_points(n_pareto_points, n_obj)
    t1 = np.sqrt(np.sum(pf ** 2, axis=1))
    t1 = np.repeat(np.atleast_2d(t1).reshape(-1, 1), n_obj, axis=1)
    pf = pf/t1
    t2 = np.atleast_2d(np.arange(2, 2 * n_obj + 1, 2))
    t2 = np.repeat(t2, new_n, axis=0)
    pf = t2 * pf

    return pf

def convex(x):
    n = x.shape[0]
    m = x.shape[1]
    temp1 = np.ones((n, 1))
    temp2 = 1 - np.cos(x[:, :-1] * np.pi/2.0)
    temp3 = 1 - np.sin(x[:, -2:-m-1:-1] * np.pi/2.0)

    tmp1 = np.hstack((temp1, temp2))
    tmp1 = np.fliplr(np.cumprod(tmp1, axis=1))

    tmp2 = np.hstack((temp1, temp3))
    output = tmp1 * tmp2
    return output

def linear(x):
    n = x.shape[0]
    m = x.shape[1]
    temp1 = np.ones((n, m))
    temp1[:, 1:] = x[:, :-1]
    temp1 = np.fliplr(np.cumprod(temp1, axis=1))

    temp2 = np.ones((n, m))
    temp2[:, 1:] = 1- x[:, -2:-m-1:-1]

    out = temp1 * temp2
    return out





def disc(x):
    output = 1 - x[:, 0] * (np.cos(5 * np.pi * x[:, 0])) ** 2
    return output

def mixed(x):
    output = 1 - x[:, 0] - np.cos(10 * np.pi * x[:, 0] + np.pi/2)/10/np.pi
    return output

class WFG_1(Problem):
    def __init__(self, n_var=6, n_obj=2, K=4):
        self.n_var = n_var
        self.K = K
        self.L = self.n_var - self.K
        self.n_constr = 0
        self.n_obj = n_obj
        self.xl = anp.array([0] * self.n_var)
        self.xu = anp.array(np.arange(2, 2*self.n_var+1, 2))

        self.WFG1 = WFG1(num_objectives=n_obj, num_variables=n_var, k=K)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        # is x right object to send to objective_function?
        f_objs = []
        for x_indiv in x:
            f_obj = self.WFG1.objective_function(x_indiv)
            f_objs.append(f_obj)

        out['F'] = np.atleast_2d(f_objs).reshape(-1, self.n_obj)
        # print(f_objs)

    def _calc_pareto_front(self, n_pareto_points=100):
        pf = None
        p_size, p = uniform_points(n_pareto_points, self.n_obj)
        c = np.ones((p_size, self.n_obj))
        for i in range(p_size):
            for j in range(1, self.n_obj):
                tmp = p[i, j]/p[i, 0] * np.prod(1 - c[i, self.n_obj - 1 - j + 1:self.n_obj-1])
                c[i, self.n_obj-j-1] = (tmp**2 - tmp + np.sqrt(2*tmp)) / (tmp**2 + 1)

        x = np.arccos(c) * 2 / np.pi
        tmp = (1 - np.sin(np.pi/2 * x[:, 1])) * p[:, self.n_obj-1] / p[:, self.n_obj-2]
        tmp = np.atleast_2d(tmp).reshape(-1, 1)
        a = np.arange(0, 1.00001, 0.0001)
        a = np.atleast_2d(a).reshape(1, -1)
        len_x = x.shape[0]
        E = np.abs(tmp.dot(1 - np.cos(np.pi/2*a)) - 1 +
                   np.repeat(a + np.cos(10 * np.pi * a + np.pi/2)/10/np.pi, len_x, axis=0))

        rank = np.argsort(E, axis=1)
        for i in range(len_x):
            x[i, 0] = a[0, min(rank[i, 1:10])]

        p = convex(x)
        p[:, self.n_obj-1] = mixed(x)
        pf = np.repeat(np.atleast_2d(np.arange(2, 2*self.n_obj+1, 2)), len(p), axis=0) * p

        return pf

class WFG_2(Problem):
    def __init__(self, n_var=6, n_obj=2, K=4):
        self.n_var = n_var
        self.K = K
        self.L = self.n_var - self.K
        self.n_constr = 0
        self.n_obj = n_obj
        self.xl = anp.array([0] * self.n_var)
        self.xu = anp.array(np.arange(2, 2*self.n_var+1, 2))

        self.WFG2 = WFG2(num_objectives=n_obj, num_variables=n_var, k=K)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        # is x right object to send to objective_function?
        f_objs = []
        for x_indiv in x:
            f_obj = self.WFG2.objective_function(x_indiv)
            f_objs.append(f_obj)

        out['F'] = np.atleast_2d(f_objs).reshape(-1, self.n_obj)
        # print(f_objs)

    def _calc_pareto_front(self, n_pareto_points=100):
        pf = None
        p_size, p = uniform_points(n_pareto_points, self.n_obj)
        c = np.ones((p_size, self.n_obj))
        for i in range(p_size):
            for j in range(1, self.n_obj):
                tmp = p[i, j]/p[i, 0] * np.prod(1 - c[i, self.n_obj - 1 - j + 1:self.n_obj-1])
                c[i, self.n_obj-j-1] = (tmp**2 - tmp + np.sqrt(2*tmp)) / (tmp**2 + 1)

        x = np.arccos(c) * 2 / np.pi
        tmp = (1 - np.sin(np.pi/2 * x[:, 1])) * p[:, self.n_obj-1] / p[:, self.n_obj-2]
        tmp = np.atleast_2d(tmp).reshape(-1, 1)
        a = np.arange(0, 1.00001, 0.0001)
        a = np.atleast_2d(a).reshape(1, -1)
        len_x = x.shape[0]
        E = np.abs(tmp.dot(1 - np.cos(np.pi/2 * a)) - 1 +
                   np.repeat(a * np.cos(5 * np.pi * a) ** 2, len_x, axis=0))

        rank = np.argsort(E, axis=1)
        for i in range(len_x):
            x[i, 0] = a[0, min(rank[i, 1:10])]

        p = convex(x)
        p[:, self.n_obj-1] = disc(x)

        ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(p)
        ndf = list(ndf)
        # tmp = np.atleast_2d([np.inf] * len(p))
        # tmp[:, ndf[0]] = 1

        p = p[ndf[0], :]
        len_p = len(p)
        p1 = np.repeat(np.atleast_2d(np.arange(2, 2 * self.n_obj+1, 2)), len_p, axis=0)
        pf = p1 * p

        return pf

class WFG_3(Problem):
    def __init__(self, n_var=6, n_obj=2, K=4):
        self.n_var = n_var
        self.K = K
        self.L = self.n_var - self.K
        self.n_constr = 0
        self.n_obj = n_obj
        self.xl = anp.array([0] * self.n_var)
        self.xu = anp.array(np.arange(2, 2*self.n_var+1, 2))

        self.WFG3 = WFG3(num_objectives=n_obj, num_variables=n_var, k=K)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        # is x right object to send to objective_function?
        f_objs = []
        for x_indiv in x:
            f_obj = self.WFG3.objective_function(x_indiv)
            f_objs.append(f_obj)

        out['F'] = np.atleast_2d(f_objs).reshape(-1, self.n_obj)
        # print(f_objs)

    def _calc_pareto_front(self, n_pareto_points=100):
        pf = None
        x0 = np.atleast_2d(np.arange(0, 1.0 + 1/n_pareto_points, 1/(n_pareto_points-1))).reshape(-1, 1)
        x2 = np.atleast_2d(np.zeros((n_pareto_points, 1))).reshape(-1, 1)

        if self.n_obj > 2:
            x1 = np.atleast_2d(np.zeros((n_pareto_points, self.n_obj - 2)) + 0.5).reshape(-1, self.n_obj - 2)
            X = np.hstack((x0, x1, x2))
        else:
            X = np.hstack((x0, x2))

        p = linear(X)
        pf = np.repeat(np.atleast_2d(np.arange(2, 2*self.n_obj+1, 2)), len(p), axis=0)
        pf = pf * p
        return pf


class WFG_4(Problem):
    def __init__(self, n_var=6, n_obj=2, K=4):
        self.n_var = n_var
        self.K = K
        self.L = self.n_var - self.K
        self.n_constr = 0
        self.n_obj = n_obj
        self.xl = anp.array([0] * self.n_var)
        self.xu = anp.array(np.arange(2, 2*self.n_var+1, 2))

        self.WFG4 = WFG4(num_objectives=n_obj, num_variables=n_var, k=K)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        # is x right object to send to objective_function?
        f_objs = []
        for x_indiv in x:
            f_obj = self.WFG4.objective_function(x_indiv)
            f_objs.append(f_obj)

        out['F'] = np.atleast_2d(f_objs).reshape(-1, self.n_obj)
        # print(f_objs)

    def _calc_pareto_front(self, n_pareto_points=100):
        pf = pareto_front(self.n_obj, n_pareto_points)
        return pf

class WFG_5(Problem):
    def __init__(self, n_var=6, n_obj=2, K=4):
        self.n_var = n_var
        self.K = K
        self.L = self.n_var - self.K
        self.n_constr = 0
        self.n_obj = n_obj
        self.xl = anp.array([0] * self.n_var)
        self.xu = anp.array(np.arange(2, 2*self.n_var+1, 2))

        self.WFG5 = WFG5(num_objectives=n_obj, num_variables=n_var, k=K)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        # is x right object to send to objective_function?
        f_objs = []
        for x_indiv in x:
            f_obj = self.WFG5.objective_function(x_indiv)
            f_objs.append(f_obj)

        out['F'] = np.atleast_2d(f_objs).reshape(-1, self.n_obj)
        # print(f_objs)

    def _calc_pareto_front(self, n_pareto_points=100):
        pf = pareto_front(self.n_obj, n_pareto_points)
        return pf

class WFG_6(Problem):

    def __init__(self, n_var=6, n_obj=2, K=4):
        self.n_var = n_var
        self.K = K
        self.L = self.n_var - self.K
        self.n_constr = 0
        self.n_obj = n_obj
        self.xl = anp.array([0] * self.n_var)
        self.xu = anp.array(np.arange(2, 2*self.n_var+1, 2))

        self.WFG6 = WFG6(num_objectives=n_obj, num_variables=n_var, k=K)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        # is x right object to send to objective_function?
        f_objs = []
        for x_indiv in x:
            f_obj = self.WFG6.objective_function(x_indiv)
            f_objs.append(f_obj)

        out['F'] = np.atleast_2d(f_objs).reshape(-1, self.n_obj)
        # print(f_objs)

    def _calc_pareto_front(self, n_pareto_points=100):
        pf = pareto_front(self.n_obj, n_pareto_points)
        return pf

class WFG_7(Problem):
    def __init__(self, n_var=6, n_obj=2, K=4):
        self.n_var = n_var
        self.K = K
        self.L = self.n_var - self.K
        self.n_constr = 0
        self.n_obj = n_obj
        self.xl = anp.array([0] * self.n_var)
        self.xu = anp.array(np.arange(2, 2*self.n_var+1, 2))

        self.WFG7 = WFG7(num_objectives=n_obj, num_variables=n_var, k=K)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        # is x right object to send to objective_function?
        f_objs = []
        for x_indiv in x:
            f_obj = self.WFG7.objective_function(x_indiv)
            f_objs.append(f_obj)

        out['F'] = np.atleast_2d(f_objs).reshape(-1, self.n_obj)
        # print(f_objs)

    def _calc_pareto_front(self, n_pareto_points=100):
        pf = pareto_front(self.n_obj, n_pareto_points)
        return pf

class WFG_8(Problem):
    def __init__(self, n_var=6, n_obj=2, K=4):
        self.n_var = n_var
        self.K = K
        self.L = self.n_var - self.K
        self.n_constr = 0
        self.n_obj = n_obj
        self.xl = anp.array([0] * self.n_var)
        self.xu = anp.array(np.arange(2, 2*self.n_var+1, 2))

        self.WFG8 = WFG8(num_objectives=n_obj, num_variables=n_var, k=K)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        # is x right object to send to objective_function?
        f_objs = []
        for x_indiv in x:
            f_obj = self.WFG8.objective_function(x_indiv)
            f_objs.append(f_obj)

        out['F'] = np.atleast_2d(f_objs).reshape(-1, self.n_obj)
        # print(f_objs)

    def _calc_pareto_front(self, n_pareto_points=100):
        pf = pareto_front(self.n_obj, n_pareto_points)
        return pf

class WFG_9(Problem):
    def __init__(self, n_var=6, n_obj=2, K=4):
        self.n_var = n_var
        self.K = K
        self.L = self.n_var - self.K
        self.n_constr = 0
        self.n_obj = n_obj
        self.xl = anp.array([0] * self.n_var)
        self.xu = anp.array(np.arange(2, 2*self.n_var+1, 2))

        self.WFG9 = WFG9(num_objectives=n_obj, num_variables=n_var, k=K)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        # is x right object to send to objective_function?
        f_objs = []
        for x_indiv in x:
            f_obj = self.WFG9.objective_function(x_indiv)
            f_objs.append(f_obj)

        out['F'] = np.atleast_2d(f_objs).reshape(-1, self.n_obj)
        # print(f_objs)

    def _calc_pareto_front(self, n_pareto_points=100):
        pf = pareto_front(self.n_obj, n_pareto_points)
        return pf


def test_method(x):

    # [1, 2, 3, 4, 5, 6],
    # x = np.atleast_2d([[1.5, 2.5, 3.5, 4.5, 5.5, 6.5]])
    pro = WFG_1(n_var=6, n_obj=3, K=4)
    print(pro.name())
    y = pro.pareto_front(n_pareto_points=20)
    print(y)

    obj = pro.evaluate(x)
    print(obj)


    # import matplotlib.pyplot as plt
    # plt.scatter(y[:, 0], y[:, 1])
    # plt.show()





if __name__ == "__main__":

    # test_method()


    args = []
    for _ in range(1, 1000):
         x = np.atleast_2d([0.1 * 10 + 0.5] * 6)
         args.append(x)

    num_workers = 2
    pool = mp.Pool(processes=num_workers)
    pool.map(test_method, ([arg for arg in args]))