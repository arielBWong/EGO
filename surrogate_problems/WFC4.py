import autograd.numpy as anp
from surrogate_problems.sur_problem_base import Problem
from sklearn.utils.validation import check_array
import numpy as np
import pygmo as pg
from scipy.special import comb
from functools import reduce
from math import fabs, ceil, floor, sin, cos, pi
from operator import mul
from copy import deepcopy
from itertools import combinations


class WFC4(Problem):

    def __init__(self, n_var=6, n_obj=2, K=2):
        self.n_var = n_var
        self.K = K
        self.L = self.n_var - self.K
        self.n_constr = 0
        self.n_obj = n_obj
        self.xl = anp.array([0] * self.n_var)
        self.xu = anp.array(np.arange(2, 2*self.n_var+1, 2))
        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):

        x = check_array(x)
        n = len(x)  # number rows

        z01 = self.destep(x)
        t1 = self._s_multi(z01, 30.0, 10.0, 0.35)  # 5.0, 10.0, 0.35
        t2 = np.zeros((n, self.n_obj))

        # calculate t2 (1 ~ n_obj-1)column -> index is 0 ~ n_obj-2
        for i in np.arange(1, self.n_obj):
            start_index = int((i-1) * self.K/(self.n_obj - 1) + 1)
            end_index = int((i * self.K)/(self.n_obj - 1))
            w_range = int(self.K/(self.n_obj - 1))

            y = t1[:, start_index-1: end_index]  # open bracket
            w = np.ones((1, w_range))

            t2[:, i-1] = self._r_sum(y, w)

        # calculate t2 last column
        t2[:, -1] = self._r_sum(t1[:, -self.L:], np.ones((1, self.L)))

        # construct concave_x
        concave_x = np.zeros((n, self.n_obj))
        for i in range(self.n_obj-1):  # calculating column 0 ~ self.n_obj-2
            comp1 = np.atleast_2d(t2[:, -1]).reshape(-1, 1)
            comp2 = np.ones((n, 1))
            tmp = np.max(np.hstack((comp1, comp2)), axis=1)
            concave_x[:, i] = tmp * (t2[:, i] - 0.5) + 0.5

        concave_x[:, -1] = t2[:, -1]

        h = self._concave(concave_x)
        S = np.arange(2, 2*self.n_obj + 1, 2)
        D = 1
        eva1 = np.repeat(D * np.atleast_2d(concave_x[:, self.n_obj-1]).reshape(-1, 1), self.n_obj, axis=1)
        eva2 = np.repeat(np.atleast_2d(S), n, axis=0)
        eva = eva1 + eva2*h
        out['F'] = eva

    def _calc_pareto_front(self, n_pareto_points=100):
        new_n, pf = self._uniform_points(n_pareto_points, self.n_obj)
        t1 = np.sqrt(np.sum(pf ** 2, axis=1))
        t1 = np.repeat(np.atleast_2d(t1).reshape(-1, 1), self.n_obj, axis=1)
        pf = pf/t1
        t2 = np.atleast_2d(np.arange(2, 2 * self.n_obj + 1, 2))
        t2 = np.repeat(t2, new_n, axis=0)
        pf = t2 * pf

        return pf


    def destep(self, vec):
        """Removes the [2, 4, 6,...] steps."""
        lvec = vec.shape[1]
        dev = np.arange(2, 2*(lvec+1), 2)
        return vec/dev

    def _s_multi(self, y, A, B, C):

        """Shift: Parameter Multi-Modal Transformation."""
        tmp1 = np.abs(y - C) / (2.0 * (np.floor(C - y) + C))
        tmp2 = (4.0 * A + 2.0) * np.pi * (0.5 - tmp1)
        return (1.0 + np.cos(tmp2) + 4.0 * B * tmp1 ** 2) / (B + 2.0)



    def _r_sum(self, y, w):

        """Weighted sum reduction transformation.'"""
        return  np.sum(y * w, axis=1)/np.sum(w)

    def _shape_concave(self, x, m):

        """Concave Pareto optimal shape function."""
        if m == 1:
            result = reduce(mul, (sin(0.5 * xi * pi) for xi in x[:len(x)]), 1.0)
        elif 1 < m <= len(x):
            result = reduce(mul, (sin(0.5 * xi * pi) for xi in x[:len(x) - m + 1]), 1.0)
            result *= cos(0.5 * x[len(x) - m + 1] * pi)
        else:
            result = cos(0.5 * x[0] * pi)
        return result

    def _concave(self, x):
        n = x.shape[0]
        m = x.shape[1]
        temp1 = np.ones((n, m))
        temp1[:, 1:] = np.sin(x[:, :-1] * np.pi / 2.0)

        temp2 = np.ones((n, m))
        temp2[:, 1:] = np.cos(x[:, -2:-m-1:-1] * np.pi / 2.0)

        temp1 = np.fliplr(np.cumprod(temp1, axis=1))
        return temp1 * temp2




    def _uniform_points(self, n_samples, n_obj):
        samples = None
        h1 = 1
        while comb(h1 + n_obj, n_obj - 1) <= n_samples:
            h1 = h1 + 1

        compo1 = self.vec_nchoosek(np.arange(1, h1+n_obj), n_obj-1)
        t = np.atleast_2d(np.arange(0, n_obj-1))
        compo2 = np.repeat(t, comb(h1 + n_obj - 1, n_obj - 1), axis=0)
        W = compo1 - compo2 - 1

        t = np.atleast_2d(np.zeros((W.shape[0], 1))) + h1
        w1 = np.hstack((W, t))
        t2 = np.atleast_2d(np.zeros((W.shape[0], 1)))
        w2 = np.hstack((t2, W))
        W = (w1 - w2)/h1

        if h1 < n_obj:
            h2 = 0
            while comb(h1 + n_obj - 1, n_obj - 1) + comb(h2 + n_obj, n_obj - 1) <= n_samples:
                h2 = h2 + 1

            if h2 > 0:
                t1 = self.vec_nchoosek(np.arange(1, h2 + n_obj), n_obj-1)
                t2 = np.atleast_2d(np.arange(0, n_obj-1))
                t2 = np.repeat(t2, comb(h2 + n_obj - 1,  n_obj-1), axis=0)
                W2 = t1 - t2 - 1

                t1 = np.atleast_2d(np.zeros((W2.shape[0], 1))) + h2
                t1 = np.hstack((W2, t1))
                t2 = np.atleast_2d(np.zeros((W2.shape[0], 1)))
                t2 = np.hstack((t2, W2))
                W2 = (t1 - t2)/h2

                t3 = W2/2 + 1/(2 * n_obj)
                W = np.vstack((W, t3))

        # fix 0
        W = np.maximum(W, 1e-6)
        new_sample_size = W.shape[0]
        return new_sample_size, W


    def vec_nchoosek(self,v, k):
        """v is a vector, k is the number of elements to be chosen"""
        k_obj = combinations(v, k)
        out = []
        for ki in k_obj:
            out = np.append(out, ki)

        out = np.atleast_2d(out).reshape(-1, k)
        return out










if __name__ == "__main__":
    pro = WFC4(n_var=6, n_obj=2, K=4)
    x = np.atleast_2d([[1.5, 2.5, 3.5, 4.5, 5.5, 6.5]])
    y = pro.evaluate(x)
    print(y)
    # n, w = pro._uniform_points(20, 3)
    # y = pro.pareto_front(n_pareto_points=100)

    # import matplotlib.pyplot as plt

    # plt.scatter(y[:, 0], y[:, 1])
    # plt.show()
