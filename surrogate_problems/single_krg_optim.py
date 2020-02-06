import autograd.numpy as anp
from surrogate_problems.sur_problem_base import Problem
from sklearn.utils.validation import check_array
import numpy as np
from utilities import save_data



class single_krg_optim(Problem):

    def __init__(self, krg, n_var, n_constr, n_obj, low, up):
        self.n_var = n_var
        self.n_constr = n_constr
        self.n_obj = n_obj
        self.xl = anp.array(low)
        self.xu = anp.array(up)
        self.model = krg
        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.atleast_2d(x).reshape(-1, self.n_var)
        y, _ = self.model.predict(x)
        out["F"] = y

        return out["F"]
