import autograd.numpy as anp
from surrogate_problems.sur_problem_base import Problem
from sklearn.utils.validation import check_array
import numpy as np
import pygmo as pg

# Example 1:
# Statistical surrogate model based sampling criterion
# for stochastic global optimization of problems with constraints


class SHCBc(Problem):

    def __init__(self):
        self.n_var = 2
        self.n_constr = 1
        self.n_obj = 1
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

        x1 = np.atleast_2d(x[:, 0]).reshape(-1, 1)
        x2 = np.atleast_2d(x[:, 1]).reshape(-1, 1)

        f = (x1 - 2)**2 + (x2 + 1)**2 - 3

        part1 = (4 - 2.1 * x1 ** 2 + x1 ** 4 / 3) * x1 ** 2
        part2 = x1 * x2
        part3 = (-4 + 4 * x2 ** 2) * x2 ** 2

        g = part1 + part2 + part3


        out["F"] = f
        out["G"] = g

        return out["F"], out["G"]

    def stop_criteria(self, x):
        x = check_array(x)
        if x.shape[0] > 1:
            raise ValueError(
                'comparison only between one vector and optimal solution'
            )
        d = np.sqrt((x[0, 0] - 1.8150) ** 2 + (x[0, 1] - (-0.8750) ** 2))
        if d < 1e-2:
            return True
        else:
            return False

