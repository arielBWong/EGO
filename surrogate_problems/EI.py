import autograd.numpy as anp
from surrogate_problems.sur_problem_base import Problem
from sklearn.utils.validation import check_array
import numpy as np
import pygmo as pg


from scipy.stats import norm
from sklearn.utils.validation import check_array




class EI(Problem):

    def __init__(self):
        self.n_var = 2
        self.n_constr = 1
        self.n_obj = 1
        self.xl = anp.array([-5, 0])
        self.xu = anp.array([10, 15])
        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        # input should be in the right range of defined problem

        out["F"] = 1
        out["G"] = 1

        return out["F"], out["G"]



