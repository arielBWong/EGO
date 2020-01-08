import autograd.numpy as anp
from surrogate_problems.sur_problem_base import Problem
from sklearn.utils.validation import check_array
import numpy as np
import pygmo as pg

# equation (6.4)


class ReverseMystery(Problem):

    # equation 5.11

    def __init__(self):
        self.n_var = 2
        self.n_constr = 1
        self.n_obj = 1
        self.xl = anp.array([0, 0])
        self.xu = anp.array([5, 5])
        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        x = check_array(x)

        x1 = np.atleast_2d(x[:, 0]).reshape(-1, 1)
        x2 = np.atleast_2d(x[:, 1]).reshape(-1, 1)

        part1 = 0.01 * (x2 - x1**2)**2
        part2 = (1 - x1)**2
        part3 = 2 * (2 - x2)**2
        part4 = 7 * np.sin(0.5 * x1) * np.sin(0.7 * x1 * x2)

        f = -np.sin(x1 - x2 - np.pi/8)

        g = -1 + part1 + part2 + part3 + part4

        out["F"] = f
        out["G"] = g

        return out["F"], out["G"]
