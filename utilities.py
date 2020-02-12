import numpy as np
import matplotlib.pyplot as plt
from pymop import ZDT1, ZDT2, ZDT3, ZDT4, DTLZ1, G1, DTLZ2, BNH, Carside, Kursawe, OSY, Truss2D, WeldedBeam, TNK
import pyDOE
from joblib import dump, load
import pygmo as pg
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

def intermediate_save(target_problem, method_selection, seed_index,iteration, krg, train_y, nadir, ideal):
    saveName = 'intermediate\\' + target_problem.name() + '_' + method_selection + '_seed_' + str(
        seed_index) + 'krg_iteration_' + str(iteration) + '.joblib'
    dump(krg, saveName)

    saveName = 'intermediate\\' + target_problem.name() + '_' + method_selection + '_seed_' + str(
        seed_index) + 'nd_iteration_' + str(iteration) + '.joblib'
    save_pareto_front(train_y, saveName)

    saveName = 'intermediate\\' + target_problem.name() + '_' + method_selection + '_seed_' + str(
        seed_index) + '_nadir_iteration_' + str(iteration) + '.joblib'
    dump(nadir, saveName)

    saveName = 'intermediate\\' + target_problem.name() + '_' + method_selection + '_seed_' + str(
        seed_index) + '_ideal_iteration_' + str(iteration) + '.joblib'
    dump(ideal, saveName)

    return  True

def samplex2f(f_pareto, n_obj, n_vals, krg, seed, method, nadir=None, ideal=None):

    n = 100000
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
    point_reference = np.atleast_2d([1.1] * n_obj)
    if len(f_pareto) > 1:
        norm_pf = (f_pareto - min_pf_by_feature) / (max_pf_by_feature - min_pf_by_feature)
        point_reference = np.atleast_2d([1.1] * n_obj)
        norm_mu = (fs - min_pf_by_feature) / (max_pf_by_feature - min_pf_by_feature)
    else:
        norm_pf = f_pareto
        point_reference = np.atleast_2d(abs(norm_pf) * 1.1)
        norm_mu = fs

    if method == 'eim':
        y = EI_krg.EIM_hv(fs, sig, f_pareto, point_reference)
    elif method == 'hv':
        y = EI_krg.EI_hv(norm_mu, norm_pf, point_reference)
    elif method == 'hvr':
        y = EI_krg.HVR(ideal, nadir, f_pareto, fs, n_obj)
    else:
        raise (
            "samplex2f un-recognisable ei method"
        )


    y = y.ravel()

    y = y * -1.0
    y_index = np.argsort(y)
    pop_f = y[y_index[0: 100]]
    test_x1 = test_x[y_index[0: 100], :]

    f1 = (fs[:, 0]).ravel()
    f2 = (fs[:, 1]).ravel()

    return -y, f1, f2, fs, sig, test_x1, test_x


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


def check_EIM_dynamic_direction(iter_list, problem, method1):
    # only 3 variables
    n_obj = problem.n_obj
    n_vals = problem.n_var
    prob = problem.name()
    seed = 3
    method = '_' + method1
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

        y, f1, f2, fs, sig, _, samplex = samplex2f(f_pareto, n_obj, n_vals, krg, i, method1)

        f_true = problem.evaluate(samplex)
        if n_obj > 2:
            raise (
                "check_EIM_dynamic_direction is unable to process 3d plot"
            )
        f_min_s = np.atleast_2d(np.min(fs, axis=0))
        f_max_s = np.atleast_2d(np.max(fs, axis=0))

        f_min_true = np.atleast_2d(np.min(f_true, axis=0))
        f_max_true = np.atleast_2d(np.max(f_true, axis=0))

        f_min = np.min(np.vstack((f_min_s, f_min_true)), axis=0)
        f_max = np.max(np.vstack((f_max_s, f_max_true)), axis=0)

        true_pf = problem.pareto_front()



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

        nd_max = np.max(nd_front, axis = 0)

        fig, [[ax1, ax2], [ax3, ax4], [ax5, ax6], [ax7, ax8]] = plt.subplots(nrows=4, ncols=2, figsize=(20, 20))

        cm1 = plt.cm.get_cmap('RdYlBu')
        cm2 = plt.cm.get_cmap('winter')

        sc1 = ax1.scatter(f1, f2, c=y, s=0.02, cmap=cm1)
        # ax1.set(xlim=(0, up_fspace), ylim=(0, up_fspace2))
        t = method + ' max ' + "{:4.2f}".format(y_max) + '  ' + "{:4.2f}".format(f1_max[0]) + "/" + "{:4.2f}".format(f2_max[0])
        ax1.set_title(t)
        ax1.scatter(f_pareto[:, 0], f_pareto[:, 1], marker='^', color='black')
        ax1.scatter(f1_max, f2_max, marker='*', color='blue')
        ax1.scatter(nextF[:, 0], nextF[:, 1], marker='D', color='green')
        ax1.scatter(nd_max[0], nd_max[1], marker='d', color='black')
        fig.colorbar(sc1, ax=ax1)

        ax2.scatter(true_pf[:, 0], true_pf[:, 1], c='r', marker='x')
        ax2.set(xlim=(f_min[0], f_max[0]), ylim=(f_min[1], f_max[1]))
        t = problem.name() + 'pareto front'
        ax2.set_title(t)

        ax3.scatter(f1, f2, s=0.02, marker='o')
        t = 'krg prediction on f1 and f2'
        ax3.set_title(t)
        ax3.set(xlim=(f_min[0], f_max[0]), ylim=(f_min[1], f_max[1]))

        ax4.scatter(f_true[:, 0], f_true[:, 1], s=0.02, marker='o')
        t = 'true f on f1 and f2'
        ax4.set_title(t)
        ax4.set(xlim=(f_min[0], f_max[0]), ylim=(f_min[1], f_max[1]))
        
        
        sc5 = ax5.scatter(f_space1[:, 0], f_space1[:, 1], c=order, marker='X', cmap=cm2)
        # ax2.set(xlim=(0, up_fspace), ylim=(0, up_fspace2))

        t = 'searching1: best ' + method + "{:4.2f}".format(gen_f[-1, :][0]) + '  ' + "{:4.2f}".format(f_space1[-1, 0]) + '/' + "{:4.2f}".format(f_space1[-1, 1])
        ax5.set_title(t)
        fig.colorbar(sc5, ax=ax5)

        sc6 = ax6.scatter(f_space2[:, 0], f_space2[:, 1], c=order, marker='X', cmap=cm2)
        t = 'searching2: best ' + method + "{:4.2f}".format(gen_f2[-1, :][0]) + '  ' + "{:4.2f}".format(f_space2[-1, 0]) + '/' + "{:4.2f}".format(f_space2[-1, 1])
        ax6.set_title(t)
        fig.colorbar(sc6, ax=ax6)

        sc7 = ax7.scatter(f_space3[:, 0], f_space3[:, 1], c=order, marker='X', cmap=cm2)
        t = 'searching3: best'  + method + "{:4.2f}".format(gen_f3[-1, :][0]) + '  ' + "{:4.2f}".format(f_space3[-1, 0]) + '/' + "{:4.2f}".format(f_space3[-1, 1])
        ax5.set_title(t)
        fig.colorbar(sc7, ax=ax7)

        sc8 = ax8.scatter(f_space4[:, 0], f_space4[:, 1], c=order, marker='X', cmap=cm2)
        t = 'searching4: best ' + method + "{:4.2f}".format(gen_f4[-1, :][0]) + '  ' + "{:4.2f}".format(f_space4[-1, 0]) + '/' + "{:4.2f}".format(f_space4[-1, 1])
        ax8.set_title(t)
        fig.colorbar(sc8, ax=ax8)

        plt.subplots_adjust(hspace=1)
        t = problem.name() + method + 'indication and corresponding ea search process'

        plt.title(t)
        saveName = 'visualization\\' + problem.name() + method + '_iteration_' + str(i) + '_process_visualization_cheat_search.png'
        plt.savefig(saveName)

        # plt.show()
        a = 1

def check_EI_drag(iter_list, problem, method):

    # only 3 variables
    n_obj = problem.n_obj
    n_vals = problem.n_var
    pro = problem.name()
    seed = 99
    for p in iter_list:
        filename = 'intermediate\\' + pro + '_' + method + '_seed_' + str(seed) + 'krg_iteration_' + str(p) + '.joblib'
        krg = load(filename)

        filename = 'intermediate\\' + pro + '_' + method + '_seed_' + str(seed) + 'nd_iteration_' + str(p) + '.joblib'
        nd_front = load(filename)



        n = 10000

        test_y = pyDOE.lhs(n_obj, n)

        ff1 = np.atleast_2d(test_y[:, 0]).reshape(-1, 1)
        ff2 = np.atleast_2d(test_y[:, 1]).reshape(-1, 1)

        mu = np.hstack((ff1, ff2))

        f_pareto = nd_front
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

        sigma = np.atleast_2d([.1] * n).reshape(-1, 1)
        sigma = np.repeat(sigma, 2, axis=1)
        # ref = np.max(nd_front, axis=0) * 1.1
        ref = point_reference

        '''
        if method == 'eim':
            K = EI_krg.EIM_hv(norm_mu, sigma, norm_pf, point_reference)
        elif method == 'hv':
            K = EI_krg.EI_hv(norm_mu, norm_pf, point_reference)
        elif method == 'hvr':
            K = EI_krg.HVR(ideal, nadir, f_pareto, mu, n_obj)
        else:
            raise (
                'EI_krg MO process does not have this method'
            )
        '''
        K = EI_krg.EIM_hv(norm_mu, sigma, norm_pf, ref)


        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
        t = problem.name() + '_iteration_' + str(p) + '_' + method + '_visualization'
        plt.title(t)

        cm = plt.cm.get_cmap('RdYlBu')
        sc = ax1.scatter(ff1, ff2, c=K, cmap=cm)

        ax1.scatter(nd_front[:, 0], nd_front[:, 1], marker='^', color='black')

        ax1.set(xlim=(0, 1.0), ylim=(0, 1.0))
        fig.colorbar(sc, ax=ax1)

        sc2 = ax2.scatter(ff1, ff2, c=K, cmap=cm)
        ax2.scatter(nd_front[:, 0], nd_front[:, 1], marker='^', color='black')
        fig.colorbar(sc2, ax=ax2)

        saveName = 'visualization\\' + problem.name()  + '_seed_' + str(seed) + '_iteration_' + str(p) + method + ' indication visualization.png'
        #plt.show()
        plt.savefig(saveName)

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
        y, f1, f2, fs, _, _ = samplex2f(f_pareto, n_obj, n_vals, krg)

        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        cm = plt.cm.get_cmap('RdYlBu')
        ax.scatter(f1, f2, c=y, s=0.2, cmap=cm)
        plt.show()



if __name__ == "__main__":
    method = 'eim'
    # check_EI_drag(np.arange(5, 26, 10), ZDT1(n_var=6), method)
    check_EIM_dynamic_direction(np.arange(0, 32, 5), ZDT3(n_var=6), method)
    # check_EIM_3d_scatter(np.arange(8, 59, 10), ZDT3(n_var=3), restart=4)

