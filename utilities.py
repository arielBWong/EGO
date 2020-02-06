import numpy as np
import matplotlib.pyplot as plt
import optimizer_EI
from pymop.factory import get_problem_from_func
from pymop import ZDT1, ZDT2, ZDT3, ZDT4, DTLZ1, G1, DTLZ2, BNH, Carside, Kursawe, OSY, Truss2D, WeldedBeam, TNK
from EI_krg import acqusition_function
from unitFromGPR import f, mean_std_save, reverse_zscore
from scipy.stats import norm, zscore
from sklearn.utils.validation import check_array
import pyDOE
import multiprocessing
from cross_val_hyperp import cross_val_krg
from joblib import dump, load
import time
from surrogate_problems import branin, GPc, Gomez3, Mystery, Reverse_Mystery, SHCBc, HS100, Haupt_schewefel, MO_linearTest, single_krg_optim
import os
import copy
import multiprocessing as mp
import pygmo as pg
from optimizer import optimizer
import EI_krg


def save_data(x, name):
    save_name = name + '.csv'
    np.savetxt(save_name, x, delimiter=',')

def save_pareto_front(train_y, filename):

    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(train_y)
    ndf = list(ndf)
    f_pareto = train_y[ndf[0], :]
    best_f_out = f_pareto
    dump(best_f_out, filename)
    '''
    with open(filename, 'w') as f_open:
        for f in best_f_out:
            for f_i in f:
                f_open.write(f_i)
                f_open.write('\t')

            f_open.write('\n')
    '''

def return_nd_front(train_y):
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(train_y)
    ndf = list(ndf)
    f_pareto = train_y[ndf[0], :]
    return f_pareto

def plot_each_pf(iter_list):
    for i in iter_list:
        filename = 'nd_iteration_' + str(i) + '_nd.joblib'
        fp = load(filename)
        ref = np.amax(fp, axis=0)
        ref = ref * 1.1

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(fp[:, 0], fp[:, 1], c='b', marker='o')
        ax.scatter(ref[0], ref[1], c='r', marker='x')
        # ax.scatter(best_f_ego[:, 0], best_f_ego[:, 1], c='g', marker='d')
        plt.legend(['fp','reference_point'])
        plt.show()

def samplex2f(f_pareto, n_obj, n_vals, krg, seed, method, nadir=None, ideal=None):

    n = 10000
    np.random.seed(seed)
    test_x = pyDOE.lhs(n_vals, n)

    f_a = np.zeros((n, 1))
    sig_a = np.zeros((n, 1))
    for k in krg:
        f, sig = k.predict(test_x)

        f = np.atleast_2d(f).reshape(-1, 1)
        sig = np.atleast_2d(sig).reshape(-1, 1)

        f_a = np.hstack((f_a, f))
        sig_a = np.hstack((sig_a, sig))

    fs = np.delete(f_a, 0, 1)
    sig = np.delete(sig_a, 0, 1)


    min_pf_by_feature = np.amin(f_pareto, axis=0)
    max_pf_by_feature = np.amax(f_pareto, axis=0)


    if len(f_pareto) > 1:
        norm_pf = (f_pareto - min_pf_by_feature) / (max_pf_by_feature - min_pf_by_feature)
        point_reference = np.atleast_2d([1.1] * n_obj)
        norm_mu = (fs - min_pf_by_feature) / (max_pf_by_feature - min_pf_by_feature)
    else:
        norm_pf = f_pareto
        point_reference = np.atleast_2d(abs(norm_pf) * 1.1)
        norm_mu = fs

    if method == 'eim':
        y = EI_krg.EIM_hv(norm_mu, sig, norm_pf, point_reference)
    elif method == 'hv':
        y = EI_krg.EI_hv(norm_mu, norm_pf, point_reference)
    elif method == 'hvr':
        y = EI_krg.HVR(ideal, nadir, f_pareto, fs, n_obj)
    else:
        raise (
            "samplex2f un-recognisable ei method"
        )

    '''
    single_mu = np.atleast_2d(norm_mu[0, :])
    single_sig = np.atleast_2d(sig[0, :])

    single_mu2 = np.atleast_2d(norm_mu[1, :])
    single_sig2 = np.atleast_2d(sig[1, :])

    two_mu = np.atleast_2d(norm_mu[0: 2, :])
    two_sig = np.atleast_2d(sig[0: 2, :])



    a1 = EI_krg.EIM_hv(single_mu, single_sig, norm_pf, point_reference)
    print(a1)
    a2 = EI_krg.EIM_hv(single_mu2, single_sig2, norm_pf, point_reference)
    print(a2)

    a12 = EI_krg.EIM_hv(two_mu, two_sig, norm_pf, point_reference)
    print(a12)
    
    '''

    y = y.ravel()

    y = y * -1.0
    y_index = np.argsort(y)
    pop_f = y[y_index[0: 100]]
    test_x = test_x[y_index[0: 100], :]

    f1 = (fs[:, 0]).ravel()
    f2 = (fs[:, 1]).ravel()

    return -y, f1, f2, fs, sig, test_x


def EIM_single_ins(mu, sig, f_pareto, n_obj):

    min_pf_by_feature = np.amin(f_pareto, axis=0)
    max_pf_by_feature = np.amax(f_pareto, axis=0)

    if len(f_pareto) > 1:
        norm_pf = (f_pareto - min_pf_by_feature) / (max_pf_by_feature - min_pf_by_feature)
        point_reference = np.atleast_2d([1.1] * n_obj)
        norm_mu = (mu - min_pf_by_feature) / (max_pf_by_feature - min_pf_by_feature)
    else:
        norm_pf = f_pareto
        point_reference = np.atleast_2d(norm_pf * 1.1)
        norm_mu = mu

    ref = point_reference
    out = EI_krg.EIM_hv(norm_mu, sig, norm_pf, ref)
    return out


def filter_func(x):
    a = -0.05
    b = 0.15
    c = -10
    d = -5
    if x[0] > a and x[0] < b and x[1] > c and x[1] < d:
        return True
    else:
        return False


def check_EIM_dynamic_direction(iter_list, problem, restart):
    # only 3 variables
    n_obj = problem.n_obj
    n_vals = problem.n_var
    prob = problem.name()
    seed = 0
    method = '_eim'
    for i in iter_list:

        filename = 'intermediate\\' + prob + method + '_seed_' + str(seed) + 'krg_iteration_' + str(i) + '.joblib'
        krg = load(filename)

        filename = 'intermediate\\' + prob + method + '_seed_' + str(seed) + 'nd_iteration_' + str(i) + '.joblib'
        nd_front = load(filename)

        filename = 'intermediate\\' + prob + method + '_seed_' + str(seed) + 'nextF_iteration_' + str(i) + '.joblib'
        nextF = load(filename)

        filename = 'intermediate\\' + prob + method + '_seed_' + str(seed) + 'search_record_iteration_' + str(i) + '_restart_0.joblib'
        record1 = load(filename)

        filename = 'intermediate\\' + prob + method + '_seed_' + str(seed) + 'search_record_iteration_' + str(i) + '_restart_1.joblib'
        record2 = load(filename)

        filename = 'intermediate\\' + prob + method + '_seed_' + str(seed) + 'search_record_iteration_' + str(i) + '_restart_2.joblib'
        record3 = load(filename)

        filename = 'intermediate\\' + prob + method + '_seed_' + str(seed) + 'search_record_iteration_' + str(i) + '_restart_3.joblib'
        record4 = load(filename)

        # compared with pareto front
        f_pareto = nd_front

        # process records
        gen_x = np.atleast_2d(record1[1]).reshape(-1, n_vals)
        gen_f = np.atleast_2d(record1[0]).reshape(-1, 1)

        # print(gen_f[-1, 0])
        a = np.atleast_2d(gen_x[-1, :]).reshape(1, -1)
        fs, sig = krg[0].predict(a)
        fs1, sig2 = krg[1].predict(a)
        fs1 = np.hstack((fs, fs1))
        fs = fs1.copy()
        sig = np.hstack((sig, sig2))

        # prepare data from optimization process
        n_re = gen_x.shape[0]
        order = np.arange(0, n_re)

        gen_x2 = np.atleast_2d(record2[1]).reshape(-1, n_vals)
        gen_f2 = np.atleast_2d(record2[0]).reshape(-1, 1)
        gen_x3 = np.atleast_2d(record3[1]).reshape(-1, n_vals)
        gen_f3 = np.atleast_2d(record3[0]).reshape(-1, 1)
        gen_x4 = np.atleast_2d(record4[1]).reshape(-1, n_vals)
        gen_f4 = np.atleast_2d(record4[0]).reshape(-1, 1)

        f_space = np.zeros((n_re, 1))
        s_space = np.zeros((n_re, 1))
        f_space2 = np.zeros((n_re, 1))
        s_space2 = np.zeros((n_re, 1))
        f_space3 = np.zeros((n_re, 1))
        s_space3 = np.zeros((n_re, 1))
        f_space4 = np.zeros((n_re, 1))
        s_space4 = np.zeros((n_re, 1))

        for k in krg:
            m, s = k.predict(gen_x)
            m2, s2 = k.predict(gen_x2)
            m3, s3 = k.predict(gen_x3)
            m4, s4 = k.predict(gen_x4)

            f_space = np.hstack((f_space, m))
            s_space = np.hstack((s_space, s))

            f_space2 = np.hstack((f_space2, m2))
            s_space2 = np.hstack((s_space2, s2))

            f_space3 = np.hstack((f_space3, m3))
            s_space3 = np.hstack((s_space3, s3))

            f_space4 = np.hstack((f_space4, m4))
            s_space4 = np.hstack((s_space4, s4))

        f_space1 = np.delete(f_space, 0, 1)
        s_space1 = np.delete(s_space, 0, 1)

        f_space2 = np.delete(f_space2, 0, 1)
        s_space2 = np.delete(s_space2, 0, 1)

        f_space3 = np.delete(f_space3, 0, 1)
        s_space3 = np.delete(s_space3, 0, 1)

        f_space4 = np.delete(f_space4, 0, 1)
        s_space4 = np.delete(s_space4, 0, 1)

        y, f1, f2, fs, sig, _ = samplex2f(f_pareto, n_obj, n_vals, krg, i, 'eim')

        '''
        if i == 18:
            f_again = list(filter(filter_func, fs))
            t = f_again[0]
            diff = fs - t
            diff = np.sum(diff, axis=1)
            index = np.argwhere(diff == 0)
            t_fs = fs[index[0], :]
            t_sig = sig[index[0], :]
            print('check location')
            print(t_fs)
            # print(t_sig)
            print('check eim')
            print(y[index[0][0]])
            print('best eim')
            print(np.max(y))

            a = EIM_single_ins(t_fs, t_sig, f_pareto, n_obj)
            print(a)

            single_mu = np.atleast_2d(fs[0, :])
            single_sig = np.atleast_2d(sig[0, :])

            two_mu = np.atleast_2d(fs[0: 2, :])
            two_sig = np.atleast_2d(sig[0: 2, :])

            a1 = EIM_single_ins(single_mu, single_sig, f_pareto, n_obj)
            a2 = EIM_single_ins(two_mu, two_sig, f_pareto, n_obj)

            print(a1)
            print(y[0])
            print(a2)
            print(y[0:2])

            a = 0
        '''


        w = np.argwhere(y == np.max(y))
        f1_max = f1[w[0]]
        f2_max = f2[w[0]]
        y_max = np.max(y)

        fig, [[ax1, ax2], [ax3, ax4], [ax5, ax6]] = plt.subplots(nrows=3, ncols=2, figsize=(20, 20))

        cm1 = plt.cm.get_cmap('RdYlBu')
        cm2 = plt.cm.get_cmap('winter')

        sc1 = ax1.scatter(f1, f2, c=y, s=0.02, cmap=cm1)
        # ax1.set(xlim=(0, up_fspace), ylim=(0, up_fspace2))
        t = 'EIM_hv max ' + "{:4.2f}".format(y_max) + '  ' + "{:4.2f}".format(f1_max[0]) + "/" + "{:4.2f}".format(f2_max[0])
        ax1.set_title(t)
        ax1.scatter(f_pareto[:, 0], f_pareto[:, 1], marker='^', color='black')
        ax1.scatter(f1_max, f2_max, marker='*', color='blue')
        ax1.scatter(nextF[:, 0], nextF[:, 1], marker='D', color='green')
        fig.colorbar(sc1, ax=ax1)

        sc3 = ax3.scatter(f_space1[:, 0], f_space1[:, 1], c=order, marker='X', cmap=cm2)
        # ax2.set(xlim=(0, up_fspace), ylim=(0, up_fspace2))

        t = 'searching1: best eim' + "{:4.2f}".format(gen_f[-1, :][0]) + '  ' + "{:4.2f}".format(f_space1[-1, 0]) + '/' + "{:4.2f}".format(f_space1[-1, 1])
        ax3.set_title(t)
        fig.colorbar(sc3, ax=ax3)

        sc4 = ax4.scatter(f_space2[:, 0], f_space2[:, 1], c=order, marker='X', cmap=cm2)
        t = 'searching2: best eim' + "{:4.2f}".format(gen_f2[-1, :][0]) + '  ' + "{:4.2f}".format(f_space2[-1, 0]) + '/' + "{:4.2f}".format(f_space2[-1, 1])
        ax4.set_title(t)
        fig.colorbar(sc4, ax=ax4)

        sc5 = ax5.scatter(f_space3[:, 0], f_space3[:, 1], c=order, marker='X', cmap=cm2)
        t = 'searching3: best eim' + "{:4.2f}".format(gen_f3[-1, :][0]) + '  ' + "{:4.2f}".format(f_space3[-1, 0]) + '/' + "{:4.2f}".format(f_space3[-1, 1])
        ax5.set_title(t)
        fig.colorbar(sc5, ax=ax5)

        sc6 = ax6.scatter(f_space4[:, 0], f_space4[:, 1], c=order, marker='X', cmap=cm2)
        t = 'searching4: best eim' + "{:4.2f}".format(gen_f4[-1, :][0]) + '  ' + "{:4.2f}".format(f_space4[-1, 0]) + '/' + "{:4.2f}".format(f_space4[-1, 1])
        ax6.set_title(t)
        fig.colorbar(sc6, ax=ax6)

        plt.subplots_adjust(hspace=1)
        t = problem.name() + 'EMI indication and corresponding ea search process'

        plt.title(t)
        saveName = 'visualization\\' + problem.name() + method + '_iteration_' + str(i) + '_EIM_process_visualization2_cheat_search.png'
        plt.savefig(saveName)

        # plt.show()
        a = 1

def check_EI_drag(iter_list, problem, method):

    # only 3 variables
    n_obj = problem.n_obj
    n_vals = problem.n_var
    pro = problem.name()
    seed = 0
    for p in iter_list:
        filename = 'intermediate\\' + pro + '_' + method + '_seed_' + str(seed) + 'krg_iteration_' + str(p) + '.joblib'
        krg = load(filename)

        filename = 'intermediate\\' + pro + '_' + method + '_seed_' + str(seed) + 'nd_iteration_' + str(p) + '.joblib'
        nd_front = load(filename)


        '''
        n = 100
        ff1 = np.linspace(0, 2, n).reshape(-1, 1)
        ff2 = np.linspace(0, 2, n).reshape(-1, 1)

        mu = np.hstack((ff1, ff2))

        f_pareto = nd_front
        min_pf_by_feature = np.amin(f_pareto, axis=0)
        max_pf_by_feature = np.amax(f_pareto, axis=0)
        # min_pf_by_feature = ideal
        # max_pf_by_feature = nadir
        if len(f_pareto) > 1:
            norm_pf = (f_pareto - min_pf_by_feature) / (max_pf_by_feature - min_pf_by_feature)
            point_reference = np.atleast_2d([1.1] * n_obj)
            norm_mu = (mu - min_pf_by_feature) / (max_pf_by_feature - min_pf_by_feature)
        else:
            norm_pf = f_pareto
            point_reference = np.atleast_2d(norm_pf * 1.1)
            norm_mu = mu


        # ref = np.max(nd_front, axis=0) * 1.1
        ref = point_reference


        # sigma = np.zeros((n, 2))
        # sigma = sigma * 0.1
        f1 = norm_mu[:, 0]
        f2 = norm_mu[:, 1]

        f1, f2 = np.meshgrid(f1, f2)

        k = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                mu = np.atleast_2d([f1[i, j], f2[i, j]])
                sigma = np.atleast_2d([0.1, 0.1]).reshape(-1, 2)
                # k[i, j] =EI_krg.EIM_hv(mu, sigma, norm_pf, ref)
                k[i, j] = EI_krg.EI_hv(mu, norm_pf, ref)

        fig, ax1 = plt.subplots(nrows=1)
        cm = plt.cm.get_cmap('RdYlBu')
            

        ff1, ff2 = np.meshgrid(ff1, ff2)

        '''

        y, f1, f2, _, _, _ = samplex2f(nd_front, n_obj, n_vals, krg, seed, 'hv')

        # ff1, ff2 = np.meshgrid(f1, f2)
        fig, ax1 = plt.subplots(nrows=1)
        cm = plt.cm.get_cmap('RdYlBu')
        sc = ax1.scatter(f1, f2, c=y, cmap=cm)
        ax1.scatter(nd_front[:, 0], nd_front[:, 1], marker='^', color='black')
        plt.colorbar(sc)

        # saveName = 'visualization\\' + problem.name()  + '_seed_' + str(seed) + '_iteration_' + str(p) + '_EI_visualization.png'
        t = problem.name() + '_iteration_' + str(p) + '_hv_visualization'

        plt.title(t)
        plt.show()
        # plt.savefig(saveName)

        a = 0

def check_EIM_3d_scatter(iter_list, problem, restart):
    # only 3 variables
    n_obj = problem.n_obj
    n_vals = problem.n_var
    prob = problem.name()
    seed = 0
    method = '_eim'
    for i in iter_list:
        filename = 'intermediate\\' + prob + method + '_seed_' + str(seed) + 'krg_iteration_' + str(i) + '.joblib'
        krg = load(filename)

        filename = 'intermediate\\' + prob + method + '_seed_' + str(seed) + 'nd_iteration_' + str(i) + '.joblib'
        nd_front = load(filename)

        filename = 'intermediate\\' + prob + method + '_seed_' + str(seed) + 'nextF_iteration_' + str(i) + '.joblib'
        nextF = load(filename)

        filename = 'intermediate\\' + prob + method + '_seed_' + str(seed) + 'search_record_iteration_' + str(
            i) + '_restart_0.joblib'
        record1 = load(filename)

        filename = 'intermediate\\' + prob + method + '_seed_' + str(seed) + 'search_record_iteration_' + str(
            i) + '_restart_1.joblib'
        record2 = load(filename)

        filename = 'intermediate\\' + prob + method + '_seed_' + str(seed) + 'search_record_iteration_' + str(
            i) + '_restart_2.joblib'
        record3 = load(filename)

        filename = 'intermediate\\' + prob + method + '_seed_' + str(seed) + 'search_record_iteration_' + str(
            i) + '_restart_3.joblib'
        record4 = load(filename)

        # compared with pareto front
        f_pareto = nd_front
        y, f1, f2, fs, _ = samplex2f(f_pareto, n_obj, n_vals, krg)

        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        cm = plt.cm.get_cmap('RdYlBu')
        ax.scatter(f1, f2, c=y, s=0.2, cmap=cm)
        plt.show()



if __name__ == "__main__":
    # check_EI_drag(np.arange(8, 59, 10), ZDT3(n_var=3), 'hv')
    check_EIM_dynamic_direction(np.arange(18, 59, 10), ZDT3(n_var=3), restart=4)
    # check_EIM_3d_scatter(np.arange(8, 59, 10), ZDT3(n_var=3), restart=4)

