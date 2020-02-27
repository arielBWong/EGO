import autograd.numpy as anp

from pymop.problem import Problem
import numpy as np


class DTLZ(Problem):
    def __init__(self, n_var, n_obj, k=None):

        if n_var:
            self.k = n_var - n_obj + 1
        elif k:
            self.k = k
            n_var = k + n_obj - 1
        else:
            raise Exception("Either provide number of variables or k!")

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=0, xu=1, type_var=anp.double)

    def g1(self, X_M):
        return 1 * (self.k + anp.sum(anp.square(X_M - 0.5) - anp.cos(2 * anp.pi * (X_M - 0.5)), axis=1))

    def g2(self, X_M):
        return anp.sum(anp.square(X_M - 0.5), axis=1)

    def obj_func(self, X_, g, alpha=1):
        f = []

        for i in range(0, self.n_obj):
            _f = (1 + g)
            _f *= anp.prod(anp.cos(anp.power(X_[:, :X_.shape[1] - i], alpha) * anp.pi / 2.0), axis=1)
            if i > 0:
                _f *= anp.sin(anp.power(X_[:, X_.shape[1] - i], alpha) * anp.pi / 2.0)

            f.append(_f)

        f = anp.column_stack(f)
        return f


def generic_sphere(ref_dirs):
    return ref_dirs / anp.tile(anp.linalg.norm(ref_dirs, axis=1)[:, None], (1, ref_dirs.shape[1]))


class DTLZ1(DTLZ):
    def __init__(self, n_var=7, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _calc_pareto_front(self, ref_dirs=None):
        return 0.5 * ref_dirs

    def _evaluate(self, x, out, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g1(X_M)

        f = []
        for i in range(0, self.n_obj):
            _f = 0.5 * (1 + g)
            _f *= anp.prod(X_[:, :X_.shape[1] - i], axis=1)
            if i > 0:
                _f *= 1 - X_[:, X_.shape[1] - i]
            f.append(_f)

        out["F"] = anp.column_stack(f)


class DTLZ2(DTLZ):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _calc_pareto_front(self, ref_dirs):
        return generic_sphere(ref_dirs)

    def _evaluate(self, x, out, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g2(X_M)
        out["F"] = self.obj_func(X_, g, alpha=1)


class DTLZ3(DTLZ):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _calc_pareto_front(self, ref_dirs):
        return generic_sphere(ref_dirs)

    def _evaluate(self, x, out, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g1(X_M)
        out["F"] = self.obj_func(X_, g, alpha=1)


class DTLZ4(DTLZ):
    def __init__(self, n_var=10, n_obj=3, alpha=100, d=100, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)
        self.alpha = alpha
        self.d = d

    def _calc_pareto_front(self, ref_dirs):
        return generic_sphere(ref_dirs)

    def _evaluate(self, x, out, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g2(X_M)
        out["F"] = self.obj_func(X_, g, alpha=self.alpha)


class DTLZ5(DTLZ):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        # raise Exception("Not implemented yet.")
        p1 = np.atleast_2d(np.arange(0, 1 + 1/n_pareto_points, 1/(n_pareto_points-1))).reshape(-1, 1)
        p2 = np.atleast_2d(np.arange(1, 0-1/n_pareto_points, -1/(n_pareto_points-1))).reshape(-1, 1)
        p = np.hstack((p1, p2))
        p3 = np.atleast_2d(np.sqrt(np.sum(p**2, axis=1))).reshape(-1, 1)
        p3 = np.repeat(p3, p.shape[1], axis=1)
        p = p/p3
        # p = np.hstack((p[:, ]))
        a = 0
        if self.n_obj - 2 > 0:
            select_columes = list(map(int, np.zeros(self.n_obj-2)))
            p = np.hstack((p[:, select_columes], p))
            n_p = len(p)  ## number of rows
            p4 = [self.n_obj-2]
            p4 = np.append(p4, np.arange(self.n_obj - 2, -0.01, -1))
            p4 = np.repeat(np.atleast_2d(p4), n_p, axis=0)
            p = p/np.sqrt(2)**p4
        return p




    def _evaluate(self, x, out, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g2(X_M)

        theta = 1 / (2 * (1 + g[:, None])) * (1 + 2 * g[:, None] * X_)
        theta = anp.column_stack([x[:, 0], theta[:, 1:]])

        out["F"] = self.obj_func(theta, g)


class DTLZ6(DTLZ):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _calc_pareto_front(self):
        raise Exception("Not implemented yet.")

    def _evaluate(self, x, out, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = anp.sum(anp.power(X_M, 0.1), axis=1)

        theta = 1 / (2 * (1 + g[:, None])) * (1 + 2 * g[:, None] * X_)
        theta = anp.column_stack([x[:, 0], theta[:, 1:]])

        out["F"] = self.obj_func(theta, g)


class DTLZ7(DTLZ):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _calc_pareto_front(self, n_pareto_points=100):
        interval = [0, 0.251412, 0.631627, 0.859401]
        median = (interval[1]-interval[0])/(interval[3]-interval[2]+interval[1]-interval[0])
        X = self._replicatepoint(n_pareto_points, self.n_obj-1)
        X[X <= median] = X[X <= median]*(interval[1]-interval[0])/median+interval[0]
        X[X > median] = (X[X > median] - median) * (interval[3]-interval[2])/(1-median)+interval[2]
        p2 = 2 * (self.n_obj - np.sum(X/2 * (1 + np.sin(3 * np.pi * X)), axis=1))
        p2 = np.atleast_2d(p2).reshape(-1, 1)
        p = np.hstack((X, p2))
        return p



    def _replicatepoint(self, sample_num, M):
        if M > 1 and M < 3:
            sample_num = np.ceil(sample_num**(1/M))**M
            gap = np.arange(0, 1 + 1e-7, 1/(sample_num**(1/M)-1))
            c1, c2 = np.meshgrid(gap, gap, indexing='ij')
            W = np.hstack((np.atleast_2d(c1.flatten(order='F')).reshape(-1, 1),
                           np.atleast_2d(c2.flatten(order='F')).reshape(-1, 1)))

        elif M == 1:
            W = np.arange(0, 1 + 1e-5, 1/(sample_num-1))
            W = np.atleast_2d(W).reshape(-1, 1)

        else:
            raise(
                "for number objectives greater than 3, not implemented"
            )
        return W




    def _evaluate(self, x, out, *args, **kwargs):
        f = []
        for i in range(0, self.n_obj - 1):
            f.append(x[:, i])
        f = anp.column_stack(f)

        g = 1 + 9 / self.k * anp.sum(x[:, -self.k:], axis=1)
        h = self.n_obj - anp.sum(f / (1 + g[:, None]) * (1 + anp.sin(3 * anp.pi * f)), axis=1)

        out["F"] = anp.column_stack([f, (1 + g) * h])


class ScaledProblem(Problem):

    def __init__(self, problem, scale_factor):
        super().__init__(n_var=problem.n_var, n_obj=problem.n_obj, n_constr=problem.n_constr,
                         xl=problem.xl, xu=problem.xu, type_var=problem.type_var)
        self.problem = problem
        self.scale_factor = scale_factor

    @staticmethod
    def get_scale(n, scale_factor):
        return anp.power(anp.full(n, scale_factor), anp.arange(n))

    def evaluate(self, X, *args, **kwargs):
        t = self.problem.evaluate(X, **kwargs)
        F = t[0] * ScaledProblem.get_scale(self.n_obj, self.scale_factor)
        return tuple([F] + list(t)[1:])

    def _calc_pareto_front(self, *args, **kwargs):
        return self.problem.pareto_front(*args, **kwargs) * ScaledProblem.get_scale(self.n_obj, self.scale_factor)


class ConvexProblem(Problem):

    def __init__(self, problem):
        super().__init__(problem.n_var, problem.n_obj, problem.n_constr, problem.xl, problem.xu)
        self.problem = problem

    @staticmethod
    def get_power(n):
        p = anp.full(n, 4.0)
        p[-1] = 2.0
        return p

    def evaluate(self, X, *args, **kwargs):
        t = self.problem.evaluate(X, **kwargs)
        F = anp.power(t[0], ConvexProblem.get_power(self.n_obj))
        return tuple([F] + list(t)[1:])

    def _calc_pareto_front(self, ref_dirs, *args, **kwargs):
        F = self.problem.pareto_front(ref_dirs)
        return anp.power(F, ConvexProblem.get_power(self.n_obj))



if __name__ == "__main__":
    pro = DTLZ7(n_var=6, n_obj=2)
    print(pro.name())
    y = pro.pareto_front(n_pareto_points=40)
    print(y)

    # bj = pro.evaluate(x)
    # print(obj)