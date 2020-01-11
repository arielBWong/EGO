import autograd.numpy as anp
from surrogate_problems.sur_problem_base import Problem
from sklearn.utils.validation import check_array
import numpy as np
import pygmo as pg


class SMD1_F(Problem):

    def __init__(self, p, q, r):
        self.n_var = 2
        self.n_constr = 1
        self.n_obj = 1
        self.p = p
        self.q = q
        self.r = r

        xl1 = anp.array([10])

        self.xl = anp.array([-2, -1])
        self.xu = anp.array([2, 1])

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        x = check_array(x)

        F1 = np.sum(xu1**2, axis=1)
        F2 = np.sum(xl2 **2, axis=1)
        F3 = np.sum(xu2 ** 2, axis=1) + np.sum((xu2 -np.tanh(xl2)**2, axis=1)


        out["F"] = f


