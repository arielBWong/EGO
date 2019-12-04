import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.utils.validation import check_array
from sklearn.metrics import mean_squared_error
import multiprocessing as mp





def cross_val_mse_para(train_x, train_y, val_x, val_y):
    val_x = check_array(val_x)
    val_y = check_array(val_y)

    train_x = check_array(train_x)
    train_y = check_array(train_y)

    # fit GPR
    # kernal initialization should also use external configuration
    kernel = RBF(1, (np.exp(-1), np.exp(3)))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=0)
    gpr.fit(train_x, train_y)

    pred_y = gpr.predict(val_x)
    mse = mean_squared_error(val_y, pred_y)

    return mse





def n_fold_cross_val(train_x, train_y):
    # the caller of this method has checked np.atleast_2d on variables
    # so no more check array needed
    n_samples = train_x.shape[0]

    # this n-fold probably needs some configuration
    n = 5

    # in case there is zero fold_size outcome
    if n > n_samples:
        fold_size = 1
        # deal with situations where n_samples are not enough
        # change n fold to leave one out
        n = n_samples
    else:
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
            sep_back = n_samples

        # select validation set
        val_fold_x = train_x[sep_front: sep_back, :]
        val_fold_y = train_y[sep_front: sep_back, :]

        # select train set
        train_fold_x = np.delete(temp_x, range(sep_front, sep_back), axis=0)
        train_fold_y = np.delete(temp_y, range(sep_front, sep_back), axis=0)

        mse_list.append(cross_val_mse_para(train_fold_x, train_fold_y, val_fold_x, val_fold_y))

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




def n_fold_cross_val_para(train_x, train_y):

    # set up pool
    # number of processors probably need to be configurable
    pool = mp.Pool(processes=4)

    # the caller of this method has checked np.atleast_2d on variables
    # so no more check array needed
    n_samples = train_x.shape[0]

    # this n-fold probably needs some configuration
    n = 5

    # in case there is zero fold_size outcome
    if n > n_samples:
        fold_size = 1
        # deal with situations where n_samples are not enough
        # change n fold to leave one out
        n = n_samples
    else:
        # what is left over is included in the last folder
        fold_size = int(n_samples / n)

    # do I have to shuffle every time I extract a fold?
    # this is left for future change
    # yes first, let us shuffle the sample data
    index_samples = np.arange(n_samples)
    np.random.shuffle(index_samples)
    # print(index_samples)

    shaffled_train_x = train_x[index_samples, :]
    shaffled_train_y = train_y[index_samples, :]




    mse_list = []
    results = []

    for i in range(n):

        temp_x = train_x
        temp_y = train_y

        # decide the index range that is used as validation set
        if i != n - 1:
            sep_front = i * fold_size
            sep_back = (i + 1) * fold_size
        else:
            sep_front = i * fold_size
            sep_back = n_samples

        # select validation set
        val_fold_x = train_x[sep_front: sep_back, :]
        val_fold_y = train_y[sep_front: sep_back, :]

        # select train set
        train_fold_x = np.delete(temp_x, range(sep_front, sep_back), axis=0)
        train_fold_y = np.delete(temp_y, range(sep_front, sep_back), axis=0)

        # generate jobs for pool
        results.append(pool.apply_async(cross_val_mse_para, (train_fold_x, train_fold_y, val_fold_x, val_fold_y)))

    pool.close()
    pool.join()
    for i in results:
        mse_list.append(i.get())

    # this only works on list type
    min_fold_index = mse_list.index(min(mse_list))

    # use this fold index to re-create the best gpr
    if min_fold_index != n - 1:
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




def cross_val_gpr(train_x, train_y):

    train_x = check_array(train_x)
    train_y = check_array(train_y)
    # gpr = n_fold_cross_val(train_x, train_y)
    gpr = n_fold_cross_val_para(train_x, train_y)
    return gpr