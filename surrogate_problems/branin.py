import autograd.numpy as anp
from surrogate_problems.sur_problem_base import Problem
from sklearn.utils.validation import check_array
import numpy as np
import pygmo as pg



class new_branin_5(Problem):

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
        x = check_array(x)

        x1 = np.atleast_2d(x[:, 0]).reshape(-1, 1)
        x2 = np.atleast_2d(x[:, 1]).reshape(-1, 1)

        # objective
        f = -(x1 - 10.) ** 2 - (x2 - 15.) ** 2

        a = 1.0
        b = 5.1 / (4 * np.pi ** 2)
        c = 5.0 / np.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8.0 * np.pi)

        part1 = a * (x2 - b * x1 ** 2 + c * x1 - 6.0) ** 2.0
        part2 = s * (1 - t) * np.cos(x1)
        part3 = s

        # constraint
        g = part1 + part2 + part3 - 5

        # not comfortable with this out, so add return instead
        out["F"] = f
        out["G"] = np.atleast_2d(g).reshape(-1, 1)

        return out["F"], out["G"]

    def _calc_pareto_front(self, *args, **kwargs):
        # args[0] should be f
        ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(args[0])
        p = args[0][ndf, :]

        # base class has this assignment
        return p

    def hyper_cube_sampling_convert(self, x):
        x = check_array(x)

        if x.shape[1] != self.n_var:
            print('sample data given do not fit the problem number of variables')
            exit(1)

        # assume that values in x is in range [0, 1]
        if np.any(x > 1) or np.any(x < 0):
            raise Exception('Input range error, this Branin input should be in range [0, 1]')
            exit(1)

        x_first = np.atleast_2d(x[:, 0]).reshape(-1, 1)
        x_first = self.xl[0] + x_first * (self.xu[0] - self.xl[0])
        for i in np.arange(1, self.n_var):
            x_next = np.atleast_2d(x[:, 1]).reshape(-1, 1)
            # convert to defined range
            x_next = self.xl[i] + x_next * (self.xu[i] - self.xl[i])
            x_first = np.hstack((x_first, x_next))

        return x_first

    def stop_criteria(self, x):
        x = check_array(x)
        if x.shape[0] > 1:
            raise ValueError(
                'comparison only between one vector and optimal solution'
            )

        d = np.sqrt((x[0, 0] - 3.2730)**2 + (x[0, 1] - 0.0489)**2)
        if d < 1e-2:
            return True
        else:
            return False



class new_branin_2(Problem):

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
        x = check_array(x)

        x1 = np.atleast_2d(x[:, 0]).reshape(-1, 1)
        x2 = np.atleast_2d(x[:, 1]).reshape(-1, 1)

        # objective
        f = -(x1 - 10.) ** 2 - (x2 - 15.) ** 2

        a = 1.0
        b = 5.1 / (4 * np.pi ** 2)
        c = 5.0 / np.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8.0 * np.pi)

        part1 = a * (x2 - b * x1 ** 2 + c * x1 - 6.0) ** 2.0
        part2 = s * (1 - t) * np.cos(x1)
        part3 = s

        # constraint
        g = part1 + part2 + part3 - 2

        # not comfortable with this out, so add return instead
        out["F"] = f
        out["G"] = np.atleast_2d(g).reshape(-1, 1)

        return out["F"], out["G"]

    def _calc_pareto_front(self, *args, **kwargs):
        # args[0] should be f
        ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(args[0])
        p = args[0][ndf, :]

        # base class has this assignment
        return p

    def hyper_cube_sampling_convert(self, x):
        x = check_array(x)

        if x.shape[1] != self.n_var:
            print('sample data given do not fit the problem number of variables')
            exit(1)

        # assume that values in x is in range [0, 1]
        if np.any(x > 1) or np.any(x < 0):
            raise Exception('Input range error, this Branin input should be in range [0, 1]')
            exit(1)

        x_first = np.atleast_2d(x[:, 0]).reshape(-1, 1)
        x_first = self.xl[0] + x_first * (self.xu[0] - self.xl[0])
        for i in np.arange(1, self.n_var):
            x_next = np.atleast_2d(x[:, 1]).reshape(-1, 1)
            # convert to defined range
            x_next = self.xl[i] + x_next * (self.xu[i] - self.xl[i])
            x_first = np.hstack((x_first, x_next))

        return x_first

    def stop_criteria(self, x):
        x = check_array(x)
        if x.shape[0] > 1:
            raise ValueError(
                'comparison only between one vector and optimal solution'
            )

        d = np.sqrt((x[0, 0] - 3.2143) ** 2 + (x[0, 1] - 0.9633) ** 2)
        if d < 1e-2:
            return True
        else:
            return False