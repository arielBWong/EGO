import autograd.numpy as anp
from surrogate_problems.sur_problem_base import Problem
from sklearn.utils.validation import check_array
import numpy as np
import pygmo as pg



class GPc(Problem):

    # equation 6.6

    def __init__(self):
        self.n_var = 2
        self.n_constr = 2
        self.n_obj = 1
        self.xl = anp.array([-2, -2])
        self.xu = anp.array([2, 2])
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

        A = 19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2
        B = 18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2

        f = (1 + A * (x1 + x2 + 1)**2) * (30 + B * (2 * x1 - 3 * x2)**2)

        g1 = -3 * x1 + (-3 * x2)**3
        g2 = x1 - x2 - 1

        out["F"] = f
        out["G"] = anp.column_stack([g1, g2])

        return out["F"], out["G"]

    def stop_criteria(self, x: np.ndarray):
        x = check_array(x)
        if x.shape[0] > 1:
            raise ValueError(
                'comparison only between one vector and optimal solution'
            )
        d = np.sqrt((x[0, 0] - 0.5955) ** 2 + (x[0, 1] - (-0.4045)) ** 2)
        if d < 1e-2:
            return True
        else:
            return False
