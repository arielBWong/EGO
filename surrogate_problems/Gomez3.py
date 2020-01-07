import autograd.numpy as anp
from surrogate_problems.sur_problem_base import Problem
from sklearn.utils.validation import check_array
import numpy as np
import pygmo as pg



class Gomez3(Problem):

    # equation 5.11 

    def __init__(self):
        self.n_var = 2
        self.n_constr = 1
        self.n_obj = 1
        self.xl = anp.array([-1, -1])
        self.xu = anp.array([1, 1])
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

        f = (4 - 2.1 * x1 ** 2 + x1 ** 4/3) * x1 ** 2 + \
            x1 * x1 + \
            (-4 + 4 * x2 ** 2) * x2**2
        g = -np.sin(4 * np.pi * x1) + 2 * (np.sin(2 * np.pi * x2)) ** 2

        out["F"] = f
        out["G"] = g

        return out["F"], out["G"]
