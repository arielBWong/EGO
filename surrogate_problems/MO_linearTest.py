import autograd.numpy as anp
from surrogate_problems.sur_problem_base import Problem
from sklearn.utils.validation import check_array
import numpy as np
import pygmo as pg



class MO_test(Problem):

    def __init__(self):
        self.n_var = 2
        self.n_constr = 0
        self.n_obj = 2
        self.xl = anp.array([0, 0])
        self.xu = anp.array([2, 2])
        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        # input should be in the right range of defined problem
        x = check_array(x)

        x1 = np.atleast_2d(x[:, 0]).reshape(-1, 1)
        x2 = np.atleast_2d(x[:, 1]).reshape(-1, 1)

        # objective
        f1 = x1 + x2
        f2 = 4 - x1 - x2

        out["F"] = anp.column_stack([f1, f2])

        return out["F"]



    def stop_criteria(self, x):
        return False


