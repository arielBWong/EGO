import autograd.numpy as anp
from surrogate_problems.sur_problem_base import Problem
from sklearn.utils.validation import check_array
import numpy as np
import pygmo as pg

# Example 3:
# Statistical surrogate model based sampling criterion
# for stochastic global optimization of problems with constraints


class HS100(Problem):

    def __init__(self):
        self.n_var = 7
        self.n_constr = 1
        self.n_obj = 1
        self.xl = anp.array([-5, -5, -5, -5, -5, -5, -5])
        self.xu = anp.array([5, 5, 5, 5, 5, 5, 5])
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
        x3 = np.atleast_2d(x[:, 2]).reshape(-1, 1)
        x4 = np.atleast_2d(x[:, 3]).reshape(-1, 1)
        x5 = np.atleast_2d(x[:, 4]).reshape(-1, 1)
        x6 = np.atleast_2d(x[:, 5]).reshape(-1, 1)
        x7 = np.atleast_2d(x[:, 6]).reshape(-1, 1)

        f = (x1 - 10)**2 + \
            5 * (x2 - 12)**2 + \
            x3 ** 4 + \
            3 * (x4 - 11)**2 + \
            10 * x5 ** 6 + \
            7 * x6 ** 2 + \
            x7 ** 4 - \
            4 * x6 * x7 - \
            10 * x6 - \
            8 * x7

        g1 = 127 - 2 * x1 ** 2 - 3 * x2 ** 4 - \
            x3 - 4 * x4 ** 2 - 5 * x5
        g1 = -g1

        g2 = -4 * x1 ** 2 - x2 ** 2 + 3 * x1 * x2 - 2 * x3 ** 2 - 5 * x6 + 11 * x7
        g2 = -g2

        out["F"] = f
        out["G"] = g1

        return out["F"], out["G"]


    def stop_criteria(self, x):
        x = check_array(x)
        if x.shape[0] > 1:
            raise ValueError(
                'comparison only between one vector and optimal solution'
            )

        d = np.sqrt(
                    (x[0, 0] - 2.3305) ** 2 +
                    (x[0, 1] - 1.9514) ** 2 +
                    (x[0, 2] - (-0.4775)) ** 2 +
                    (x[0, 3] - 4.3657) ** 2 +
                    (x[0, 4] - (-0.6245)) ** 2 +
                    (x[0, 5] - 1.0381) ** 2 +
                    (x[0, 6] - 1.5942) ** 2
                    )

        if d < 1e-2:
            return True
        else:
            return False
