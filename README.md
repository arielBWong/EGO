# EGO
Surrogate model optimization with Gaussian regressor
![Code flow](https://github.com/arielBWong/EGO/blob/master/images/struct.jpg)

## Expected improvement explained ##
Expected improvement EI is used for guiding search for next x. What EI is trying to look for can be explained as follows:
![EI preference explained](https://github.com/arielBWong/EGO/blob/master/images/ei_1_explan.jpg)
![EI value explained](https://github.com/arielBWong/EGO/blob/master/images/ei_1_details.jpg)
![EI2 explained](https://github.com/arielBWong/EGO/blob/master/images/ei_2_details.jpg)


## Work around on pymop problem definition for multi-process evaluation of population fitness
When multi-process is used, buildin call of pickle does not allow class definition in class. [I am reference](https://stackoverflow.com/questions/36994839/i-can-pickle-local-objects-if-i-use-a-derived-class)

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

## Work around the local search of fmin_l_bfgs_b used in gaussian regression of sklearn ##
Sklearn allows for external optimizer used for hyper-parameter optimization.
Since my case, I have found the fmin_l_bfgs_b keeps reporting abnormal warnings, in stead of taking time to look into fmin_l_bfgs_b,
I decided to use EA to work around it. Reasons are as follows:
1. It is easier for me to identify EA problems, if there is any;
2. Eventually, we will use global search for hyper-parameters,
3. Time is a bit tight for me to investigate fmin_l_bfgs_b

So the method, external_optimizer is located in cross_val_hyperp.py file.
Three bugs encountered:
1. since I used mymop library for constructing optimization problem, the built-in obj-func in
gpr is not compatible, a new wrap around the obj-func (wrap_obj_fun) is constructed to comply with mymop problem definition.
2. Again in order to comply with mymop, the gaussian regression code in sklearn is modified.

In method "log_marginal_likelihood" the last return value is formed into 2d array.

```python
    if eval_gradient:
        return log_likelihood, log_likelihood_gradient
    else:
        return np.atleast_2d(log_likelihood)
```


3. Given the current EGO code structure, gpr.fit is located in cross-validation part, which is a multiple-processing form.
Therefore, under Python's rules, this optimization cannot again use multiple-processing. Otherwise, it reports "daemonic processes are not allowed to have children"
error.


## Work around when Cholesky fails ##

3. There is also problem with the Cholesky decomposition for calculating the reverse of (K + phi*I). If its input is not
positive definite, then Cholesky fails. Therefore, I wrap a condition on it, which is if Cholesky fails, then not more
fitting, just return a prior for prediction, which is mean zeros prediction.
Changes are at around line 250.
The delete of self.X_train_ force the prediction of gpr to output zero mean.


```python
        try:
            self.L_ = cholesky(K, lower=True)  # Line 2
            # self.L_ changed, self._K_inv needs to be recomputed
            self._K_inv = None
        except np.linalg.LinAlgError as exc:
            exc.args = ("The kernel, %s, is not returning a "
                        "positive definite matrix. Try gradually "
                        "increasing the 'alpha' parameter of your "
                        "GaussianProcessRegressor estimator."
                        % self.kernel_,) + exc.args
            print ('np.linalg.LinAlgError')
            del self.X_train_
            return self
            #raise

```





## Cross-validation on hyper-parameters in gaussian regression fitting
1. Training data is split into five-fold cross-validation fashion. 
2. If the number of samples is less than 5, then leave-one-out is used for the cross-validation
3. Multi-process calculating of cross-validation is enabled. The switch is in the method **cross_val_gpr** in file **cross_val_hyperp**




## Confusion with GPR
Saved by this blog [Gaussian regression](https://cloud.tencent.com/developer/article/1353538)

For me, the first barrier concept of GPR is the confusion with p and f. 
p is the probability of an instance of a random variable, p(x).
f is the function value of a certain sample 
Most tutorials on GPR will tell us that a set of f from training samples follow multivariant gaussian distribution, which is often accompanied with a equation like: 

![multi variant gaussian example](https://github.com/arielBWong/EGO/blob/master/images/f5d7r7z9xt.png)

From this expression, I used to think that f should be a probability value, from distribution we can calculate the probability of a value in the supposed range, like 

![unly gaussian](https://github.com/arielBWong/EGO/blob/master/images/1d%20normal.png)

probability can only be in [0, 1], in prediction of f there is much more possible values than just [0, 1]

**No This was so wrong!**

By saying [f1...fn] is multivariant Gaussian distribution, one f is a 1D Gaussian. 
the catch is that there is a **sampling process** that can happen for distributions. 
We can drawn a sample from a distribution and where the sample locates, it is the value of f. 
Besides, mu can move all over the axis, so the value of f is not constrainted (compared to probability)
Like this:

![look at the correction direction](https://github.com/arielBWong/EGO/blob/master/images/correction.png)


