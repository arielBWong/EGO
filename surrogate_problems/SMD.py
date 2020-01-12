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

        xu1_u = [10] * p
        xu1_l = [-5] * p

        xu2_u = [10] * r
        xu2_l = [-5] * r

        self.xl = anp.array(xu1_l + xu2_l)
        self.xu = anp.array(xu1_u + xu2_u)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        x = check_array(x)

        xu1 = x[:, np.arange(0, self.p)]
        xu2 = x[:,  np.arange(self.p, (self.p + self.r))]

        xl1 = x[:,  np.arange((self.p + self.r), (self.p + self.r + self.q))]
        xl2 = x[:,  np.arange((self.p + self.r + self.q), (self.p + self.r + self.q + self.r))]

        F1 = np.sum(xu1 ** 2, axis=1)
        F2 = np.sum(xl1 ** 2, axis=1)
        F3 = np.sum(xu2 ** 2, axis=1) + np.sum((xu2 - np.tanh(xl2))**2, axis=1)

        out["F"] = F1 + F2 + F3

class SMD1_f(Problem):

    def __init__(self, p, q, r):
        self.n_var = 2
        self.n_constr = 1
        self.n_obj = 1
        self.p = p
        self.q = q
        self.r = r

        xl1_u = [10] * q
        xl1_l = [-5] * q

        xl2_u = [np.pi/2] * r
        xl2_l = [-np.pi/2] * r

        self.xl = anp.array(xl1_l + xl2_l)
        self.xu = anp.array(xl1_u + xl2_u)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)


    def _evaluate(self, x, out, *args, **kwargs):
        x = check_array(x)

        xu1 = x[:, np.arange(0, self.p)]
        xu2 = x[:, np.arange(self.p, (self.p + self.r))]

        xl1 = x[:, np.arange((self.p + self.r), (self.p + self.r + self.q))]
        xl2 = x[:, np.arange((self.p + self.r + self.q), (self.p + self.r + self.q + self.r))]


        f1 = np.sum(xu1 ** 2, axis=1)
        f2 = np.sum(xl1 ** 2, axis=1)
        f3 = np.sum((xu2 - np.tanh(xl2))**2, axis=1)

        out["F"] = f1 + f2 + f3

