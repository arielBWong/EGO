# EGO
Surrogate model optimization with Gaussian regressor

## Work around on pymop problem definition for multi-process evaluation of population fitness
When multi-process is used, buildin call of pickle does not allow class definition in class. <[I am reference]>(https://stackoverflow.com/questions/36994839/i-can-pickle-local-objects-if-i-use-a-derived-class)

Since pymop is used for the problem definition of Expectation improvement, the original library method **get_problem_from_func** was modified as follows:

```python

class MyProblem(Problem):
    def __init__(self, n_var, n_constr, n_obj, xl, xu, out, func):
        Problem.__init__(self)
        self.n_var = n_var
        self.n_constr = n_constr
        self.n_obj = n_obj
        # self.func = self._evaluate
        self.func = func
        self.xl = xl
        self.xu = xu

    def _evaluate(self, x, out, *args, **kwargs):
        self.func(x, out, *args, **kwargs)



def get_problem_from_func(func, xl, xu, n_var=None, func_args={}):
    if xl is None or xu is None:
        raise Exception("Please provide lower and upper bounds for the problem.")
    if isinstance(xl, (int, float)):
        xl = xl * anp.ones(n_var)
    if isinstance(xu, (int, float)):
        xu = xu * anp.ones(n_var)

    # determine through a test evaluation details about the problem
    n_var = xl.shape[0]
    n_obj = -1
    n_constr = 0

    out = {}
    func(xl[None, :], out, **func_args)
    at_least2d(out)

    n_obj = out["F"].shape[1]
    if out.get("G") is not None:
        n_constr = out["G"].shape[1]

    return MyProblem(n_var, n_constr, n_obj, xl, xu, out, func)

```



## Cross-validation on hyper-parameters in gaussian regression fitting
1. Training data is split into five-fold cross-validation fashion. 
2. If the number of samples is less than 5, then leave-one-out is used for the cross-validation
3. Multi-process calculating of cross-validation is enabled. The switch is in the method **cross_val_gpr** in file **cross_val_hyperp**
