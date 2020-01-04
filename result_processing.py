import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.utils.validation import check_array
from sklearn.metrics import mean_squared_error
import multiprocessing as mp
from pymop.factory import get_problem_from_func
import optimizer
from joblib import dump, load
import cross_val_hyperp
from surrogate_problems import branin
from EA_optimizer_for_EI import norm_data
import pyDOE
import os

def reverse_zscore(data, m, s):
    return data * s + m

if __name__ == "__main__":

    diff = 0
    for output_index in range(2, 10):
        output_file_name = 'r_bset_f_seed_' + str(output_index) + '.joblib'
        best_f = load(output_file_name)

        diff = diff + np.abs(best_f - (-268.7879))


    print(diff/10)




    target_problem = branin.new_branin_5()
    number_of_initial_samples = 1000
    n_vals = target_problem.n_var

    sample_x = pyDOE.lhs(n_vals, number_of_initial_samples)
    sample_x = target_problem.hyper_cube_sampling_convert(sample_x)
    sample_y, sample_g = target_problem.evaluate(sample_x)

    mse_f_collect = 0
    mse_g_collect = 0
    for output_index in range(1):
        output_file_name = 'sample_x_seed_ ' + str(output_index) + '.joblib'
        train_x = load(output_file_name)

        out = {}
        train_y, cons_y = target_problem._evaluate(train_x, out)


        # normalize
        mean_cons_y, std_cons_y, norm_cons_y = norm_data(cons_y)
        mean_train_y, std_train_y, norm_train_y = norm_data(train_y)
        mean_train_x, std_train_x, norm_train_x = norm_data(train_x)

        # cross-validation:


        gpr, gpr_g = cross_val_hyperp. cross_val_gpr(norm_train_x, norm_train_y, norm_cons_y)
        n_obj = target_problem.n_obj
        n_cons = target_problem.n_constr

        # on prediction
        norm_sample_x = (sample_x - mean_train_x)/std_train_x
        norm_sample_y = (sample_y - mean_train_y)/std_train_y
        norm_sample_g = (sample_g - mean_cons_y)/std_cons_y

        mse_f = 0
        for j in range(n_obj):
            pred_y_norm = gpr[j].predict(norm_sample_x)
            pred_y = reverse_zscore(pred_y_norm, mean_train_y, std_train_y)
            mse_f = mse_f + mean_squared_error(pred_y, sample_y)

        mse_g = 0
        for j in range(n_cons):
            pred_g_norm = gpr_g[j].predict(norm_sample_x)
            pred_g = reverse_zscore(pred_g_norm, mean_cons_y, std_cons_y)
            mse_g = mse_g + mean_squared_error(pred_g, sample_g)

        mse_f_collect = mse_f_collect + mse_f
        mse_g_collect = mse_g_collect + mse_g


    print('mse of average f')
    print(mse_f_collect/20.0)

    print('mse of average g')
    print(mse_g_collect/20.0)






