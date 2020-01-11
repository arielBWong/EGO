import autograd.numpy as anp
from surrogate_problems.sur_problem_base import Problem
from sklearn.utils.validation import check_array
import numpy as np
import pygmo as pg

# Example 2:
# Statistical surrogate model based sampling criterion
# for stochastic global optimization of problems with constraints


class Haupt_schewefel(Problem):

    def __init__(self):
        self.n_var = 2
        self.n_constr = 1
        self.n_obj = 1
        self.xl = anp.array([-15, -15])
        self.xu = anp.array([15, 15])
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

        f = -x1 * np.sin(x1 / 3) - 1.5 * x2 * np.sin(x2 / 3)

        g = -x1 * np.sin(np.sqrt(np.abs(x1))) - x2 * np.sin(np.sqrt(np.abs(x2)))

        out["F"] = f
        out["G"] = g

        return out["F"], out["G"]

    def stop_criteria(self, x):
        x = check_array(x)
        if x.shape[0] > 1:
            raise ValueError(
                'comparison only between one vector and optimal solution'
            )

        d1 = np.sqrt((x[0, 0] - 6.0860) ** 2 + (x[0, 1] - 6.0860) ** 2)
        d2 = np.sqrt((x[0, 0] - (-6.0860)) ** 2 + (x[0, 1] - 6.0860) ** 2)
        d3 = np.sqrt((x[0, 0] - 6.0860) ** 2 + (x[0, 1] - (-6.0860)) ** 2)
        if d1 < 1e-2 or d2 < 1e-2 or d3 < 1e-2:
            return True
        else:
            return False
