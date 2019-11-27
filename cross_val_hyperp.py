import numpy as np
import matplotlib.pyplot as plt
import optimizer_para_EI
from pymop.factory import get_problem_from_func
from EI_problem import acqusition_function
from unitFromGPR import f, mean_std_save, reverse_zscore
from scipy.stats import norm, zscore
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from EI_problem import expected_improvement
from sklearn.utils.validation import check_array
import pyDOE
import multiprocessing
from sklearn.metrics import mean_squared_error
from EA_optimizer_for_EI import train_data_norm, data_denorm


def cross_val_mse(val_x, val_y, gpr):
    val_x = check_array(val_x)
    val_y = check_array(val_y)

    val_x = val_x.reshape(-1, 1)
    pred_y = gpr.predict(val_x)

    mse = mean_squared_error(val_y, pred_y)
    return mse


def n_fold_cross_val(train_x, train_y):
    # the caller of this method has checked np.atleast_2d on variables
    # so no more check array needed
    n_samples = train_x.shape[0]

    # this n-fold probably needs some configuration
    n = 5

    # what is left over is included in the last folder
    fold_size = int(n_samples / n)

    # do I have to shuffle every time I extract a fold?
    # this is left for future change
    mse_list = []

    for i in range(n):

        temp_x = train_x
        temp_y = train_y

        # decide the index range that is used as validation set
        if i != n-1:
            sep_front = i * fold_size
            sep_back = (i+1) * fold_size
        else:
            sep_front = i * fold_size
            sep_back = n_samples - 1

        # select validation set
        val_fold_x = train_x[sep_front: sep_back, :]
        val_fold_y = train_y[sep_front: sep_back, :]

        # select train set
        train_fold_x = np.delete(temp_x, range(sep_front, sep_back), axis=0)
        train_fold_y = np.delete(temp_y, range(sep_front, sep_back), axis=0)

        # fit GPR
        # kernal initialization should also use external configuration
        kernel = RBF(1, (np.exp(-1), np.exp(3)))
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=0)

        gpr.fit(train_fold_x, train_fold_y)
        mse_list.append(cross_val_mse(val_fold_x, val_fold_y, gpr))

    # this only works on list type
    min_fold_index = mse_list.index(min(mse_list))

    # use this fold index to re-create the best gpr
    if min_fold_index != n-1:
        sep_front = min_fold_index * fold_size
        sep_back = (min_fold_index + 1) * fold_size
    else:
        sep_front = min_fold_index * fold_size
        sep_back = n_samples - 1

    # recover the training data
    train_fold_x = np.delete(temp_x, range(sep_front, sep_back), axis=0)
    train_fold_y = np.delete(temp_y, range(sep_front, sep_back), axis=0)

    # fit GPR
    # kernal initialization should also use external configuration
    kernel = RBF(1, (np.exp(-1), np.exp(3)))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=0)

    gpr.fit(train_fold_x, train_fold_y)

    return gpr


def leave_one_out_val(train_x, train_y):
    return None


def cross_val_gpr(train_x, train_y):

    train_x = check_array(train_x)
    train_y = check_array(train_y)

    n_vals = train_x.shape[1]

    if n_vals > 3:
        gpr = n_fold_cross_val(train_x, train_y)

    else:
        leave_one_out_val(train_x, train_y)

    return gpr