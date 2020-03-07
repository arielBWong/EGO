import numpy as np
import optimizer
from joblib import dump, load
import os
import pygmo as pg
from pymop import ZDT1, ZDT2, ZDT3, ZDT4, DTLZ1, DTLZ2, BNH, Carside, Kursawe, OSY, Truss2D, WeldedBeam, TNK
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
from surrogate_problems import WFG, iDTLZ, DTLZs
from scipy.stats import ranksums

def reverse_zscore(data, m, s):
    return data * s + m

def sample_ci(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h


def compare_somg():
    # single objective
    # multiple constraints

    diff = 0
    # problem_list = ['Gomez3', 'new_branin_5', 'Mystery', 'ReverseMystery', 'SHCBc', 'Haupt_schewefel', 'HS100', 'GPc']
    problem_list = ['HS100']
    problem_diff = {}
    for problem in problem_list:
        output_folder_name = 'outputs\\' + problem
        if os.path.exists(output_folder_name):
            # f_opt_name = output_folder_name + '\\' + problem + '.txt'
            # f_opt = genfromtxt(f_opt_name)
            diff = 0
            count = 0
            for output_index in range(20):
                output_f_name = output_folder_name + '\\' + 'best_f_seed_' + str(output_index) + '.joblib'
                output_x_name = output_folder_name + "\\" + 'best_x_seed_' + str(output_index) + '.joblib'
                best_f = load(output_f_name)
                best_x = load(output_x_name)
                print(best_f)
                print(best_x)
                # if os.path.exists(output_f_name):
                # best_f = load(output_f_name)
                # diff = np.abs(best_f - f_opt)
                # diff = diff + best_f
                # count = count + 1
        # print(problem)
        # print('f difference')
        # print(diff/count)
        # print(count)
        # problem_diff[problem] = diff/count

    # import json
    # with open('f_diff.json', 'w') as file:
    # file.write(json.dumps(problem_diff))


def ego_outputs_read(prob):
    output_folder_name = 'outputs\\' + prob
    output_f_name = output_folder_name + '\\' + 'best_f_seed_' + str(100) + '.joblib'
    best_f = load(output_f_name)
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(best_f)
    ndf = list(ndf)
    f_pareto = best_f[ndf[0], :]
    test_f = np.sum(f_pareto, axis=1)
    return f_pareto


def nsga2_outputs_read(prob):
    nsga_problem_save = 'NSGA2\\' + prob + '\\' + 'pareto_f.joblib'
    f_pareto2 = load(nsga_problem_save)
    return f_pareto2


def compare_save_ego2nsga(problem_list):
    save_compare = np.atleast_2d([0, 0])
    for p in problem_list:
        problem = p
        f_pareto = ego_outputs_read(problem)
        f_pareto2 = nsga2_outputs_read(problem)

        point_list = np.vstack((f_pareto, f_pareto2))
        point_nadir = np.max(point_list, axis=0)
        point_reference = point_nadir * 1.1

        hv_ego = pg.hypervolume(f_pareto)
        hv_nsga = pg.hypervolume(f_pareto2)

        hv_value_ego = hv_ego.compute(point_reference)
        hv_value_nsga = hv_nsga.compute(point_reference)

        new_compare = np.atleast_2d([hv_value_ego, hv_value_nsga])
        save_compare = np.vstack((save_compare, new_compare))

    save_compare = np.delete(save_compare, 0, 0).reshape(-1, 2)
    print(save_compare)
    with open('mo_compare.txt', 'w') as f:
        for i, p in enumerate(problem_list):
            f.write(p)
            f.write('\t')
            f.write(str(save_compare[i, 0]))
            f.write('\t')
            f.write(str(save_compare[i, 1]))
            f.write('\n')


def plot_pareto_vs_ouputs(prob, seed, method, run_signature, visualization_folder_name):

    from mpl_toolkits.mplot3d import Axes3D
    from pymop.factory import get_uniform_weights

    # read ouput f values
    problem = eval(prob)
    prob = problem.name()
    output_folder_name = 'outputs\\experiment_post_process_100_evaluation\\' + prob + '_' + run_signature

    if os.path.exists(output_folder_name):
        print(output_folder_name)
    else:
        raise ValueError(
            "results folder for EGO does not exist"
        )

    output_f_name = output_folder_name + '\\best_f_seed_' + str(seed[0]) + '_' + method + '.joblib'
    best_f_ego = load(output_f_name)
    for s in seed[1:]:
        output_f_name = output_folder_name + '\\best_f_seed_' + str(s) + '_' + method+ '.joblib'
        best_f_ego1 = load(output_f_name)
        best_f_ego = np.vstack((best_f_ego, best_f_ego1))

    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(best_f_ego)
    ndf = list(ndf)
    f_pareto = best_f_ego[ndf[0], :]
    best_f_ego = f_pareto
    n = len(best_f_ego)

    n_obj = problem.n_obj

    if n_obj == 2:
        if 'DTLZ' not in prob:
            true_pf = problem.pareto_front()
        else:
            ref_dir = get_uniform_weights(100, 2)
            true_pf = problem.pareto_front(ref_dir)

        max_by_truepf = np.amax(true_pf, axis=0)
        min_by_truepf = np.amin(true_pf, axis=0)

        # plot pareto front
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))

        ax1.scatter(best_f_ego[:, 0], best_f_ego[:, 1], marker='o')
        ax1.scatter(true_pf[:, 0], true_pf[:, 1], marker='x')
        ax1.legend([method, 'true_pf'])
        ax1.set_title(prob + ' ' + run_signature)

        for i in range(n):
            zuobiao = '[' + "{:4.2f}".format(f_pareto[i, 0]) + ', ' + "{:4.2f}".format(f_pareto[i, 1]) + ']'
            ax1.text(f_pareto[i, 0], f_pareto[i, 1], zuobiao)

        ax2.scatter(best_f_ego[:, 0], best_f_ego[:, 1],  marker='o')
        ax2.scatter(true_pf[:, 0], true_pf[:, 1], marker='x')
        ax2.set(xlim=(min_by_truepf[0], max_by_truepf[0]), ylim=(min_by_truepf[1], max_by_truepf[1]))
        ax2.legend([method, 'true_pf'])
        ax2.set_title(prob +' zoom in ' + run_signature)


        working_folder = os.getcwd()
        result_folder = working_folder + '\\' + visualization_folder_name
        if not os.path.isdir(result_folder):
            # shutil.rmtree(result_folder)
            # os.mkdir(result_folder)
            os.mkdir(result_folder)
        saveName = result_folder + '\\' + method + '_' + prob + '_seed_' + str(seed[0])  +  '.png'
        plt.savefig(saveName)

    else:

        ref_dir = get_uniform_weights(1000, 3)
        true_pf = problem.pareto_front(ref_dir)

        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(true_pf[:, 0], true_pf[:, 1], true_pf[:, 2],c='r', marker='x')
        ax.scatter(best_f_ego[:, 0], best_f_ego[:, 1], best_f_ego[:, 2], c='b', marker='o')
        ax.view_init(30, 60)
        ax.set_title(prob + ' ' + run_signature)
        ax.legend(['true_pf', method])

        saveName = 'visualization\\' + run_signature + prob + '_' + method + '_compare2pf.png'
        plt.savefig(saveName)
    # plt.show()
    a = 1




def run_extract_result(run_signature):

    problem_list = ['ZDT1', 'ZDT2', 'ZDT3',  'DTLZ2', 'DTLZ4', 'DTLZ1']
    method_list = ['hv', 'eim', 'hvr']
    seedlist = np.arange(0, 10)

    true_pf_zdt3 = ZDT3().pareto_front()
    true_pf_zdt3 = 1.1 * np.amax(true_pf_zdt3, axis=0)

    reference_dict = {'ZDT1': [1.1, 1.1],
                      'ZDT2': [1.1, 1.1],
                      'ZDT3': true_pf_zdt3,
                      'DTLZ2': [2.5, 2.5, 2.5],
                      'DTLZ4':  [1.1, 1.1, 1.1],
                      'DTLZ1': [0.5, 0.5]
                      }

    savefile = run_signature + 'hv_eim_hvr.csv'
    with open(savefile, 'w+') as f:
        for prob in problem_list:
            problem_save = []

            for method in method_list:
                hv = extract_results(method, prob, seedlist, reference_dict[prob], run_signature)
                problem_save.append(hv)

            for method_out in problem_save:
                for hv_element in method_out:
                    f.write(str(hv_element))
                    f.write(',')
            f.write('\n')


def extract_results(method, prob, seed_index, reference_point, run_signature):

    # read ouput f values
    output_folder_name = 'outputs\\' + prob + '_' + run_signature
    if os.path.exists(output_folder_name):
        print('output folder exists')
    else:
        raise ValueError(
            "results folder for EGO does not exist"
        )

    hv_all = []
    # reference_point = reference_point.ravel()
    for seed in seed_index:
        output_f_name = output_folder_name +'\\best_f_seed_' + str(seed) + '_' + method + '.joblib'
        print(output_f_name)
        best_f_ego = load(output_f_name)
        n_obj = best_f_ego.shape[1]

        # deal with out of range
        select = []
        for f in best_f_ego:
            if np.all(f <= reference_point):
                select = np.append(select, f)
        best_f_ego = np.atleast_2d(select).reshape(-1, n_obj)

        if len(best_f_ego) == 0:
            hv_all = np.append(hv_all, 0)
        else:
            hv = pg.hypervolume(best_f_ego)
            hv_value = hv.compute(reference_point)
            hv_all = np.append(hv_all, hv_value)


    hv_min = np.min(hv_all)
    hv_max = np.max(hv_all)
    hv_avg = np.average(hv_all)
    hv_std = np.std(hv_all)

    return hv_min, hv_max, hv_avg, hv_std





def parEGO_out_process():

    parEGO_folder_name = 'parEGO_out\\ZDT'
    for i in np.arange(1, 5):
        out_file = parEGO_folder_name + str(i) + '.txt'
        f = np.genfromtxt(out_file, delimiter='\t')

        f = np.atleast_2d(f).reshape(-1, 2)
        ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(f)
        ndf = list(ndf)
        f_pareto = f[ndf[0], :]


        #ego
        output_folder_name = 'outputs\\' + 'ZDT' + str(i)
        if os.path.exists(output_folder_name):
            output_f_name = output_folder_name + '\\best_f_seed_100.joblib'
            best_f_ego = load(output_f_name)
        else:
            raise ValueError(
                "results folder for EGO does not exist"
            )


        problem_obj = 'ZDT' + str(i) + '(n_var=3)'
        problem = eval(problem_obj)
        true_pf = problem.pareto_front()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(f_pareto[:, 0], f_pareto[:, 1], c='b', marker='o')
        ax.scatter(true_pf[:, 0], true_pf[:, 1], c='r', marker='x')
        ax.scatter(best_f_ego[:, 0], best_f_ego[:, 1], c='g', marker='d')
        plt.title(problem_obj)
        plt.show()


def plot_pareto_vs_ouputs_compare_hv_hvr(prob, seed, method, run_signature):
    from mpl_toolkits.mplot3d import Axes3D
    from pymop.factory import get_uniform_weights

    # read ouput f values
    output_folder_name = 'outputs\\' + prob + '_' + run_signature

    if os.path.exists(output_folder_name):
        print(output_folder_name)
    else:
        raise ValueError(
            "results folder for EGO does not exist"
        )

    output_f_name = output_folder_name + '\\best_f_seed_' + str(seed[0]) + '_' + method + '.joblib'
    best_f_ego = load(output_f_name)
    for s in seed[1:]:
        output_f_name = output_folder_name + '\\best_f_seed_' + str(s) + '_' + method + '.joblib'
        best_f_ego1 = load(output_f_name)
        best_f_ego = np.vstack((best_f_ego, best_f_ego1))

    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(best_f_ego)
    ndf = list(ndf)
    f_pareto = best_f_ego[ndf[0], :]
    best_f_ego = f_pareto
    n1 = len(best_f_ego)



    # read compare value hvr
    output_folder_name_r = 'outputs\\' + prob + '_' + run_signature + 'r'

    if os.path.exists(output_folder_name):
        print(output_folder_name)
    else:
        raise ValueError(
            "results folder for EGO does not exist"
        )

    output_f_name_r = output_folder_name + 'r' + '\\best_f_seed_' + str(seed[0]) + '_' + method + 'r' + '.joblib'
    best_f_ego_r = load(output_f_name_r)
    for s in seed[1:]:
        output_f_name = output_folder_name_r + '\\best_f_seed_' + str(s) + '_' + method + 'r' + '.joblib'
        best_f_ego1 = load(output_f_name)
        best_f_ego_r = np.vstack((best_f_ego_r, best_f_ego1))

    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(best_f_ego_r)
    ndf = list(ndf)
    f_pareto_r = best_f_ego_r[ndf[0], :]
    best_f_ego_r = f_pareto_r
    n2 = len(best_f_ego_r)

    # extract pareto front
    if 'ZDT' in prob:
        problem_obj = prob + '(n_var=6)'

    if 'DTLZ1' in prob:
        problem_obj = prob + '(n_var=6, n_obj=2)'

    if 'DTLZ2' in prob:
        problem_obj = prob + '(n_var=8, n_obj=3)'

    if 'DTLZ4' in prob:
        problem_obj = prob + '(n_var=8, n_obj=3)'

    if 'WFG4' in prob:
        problem_obj = prob + '.WFG4()'

    problem = eval(problem_obj)
    n_obj = problem.n_obj

    if n_obj == 2:
        if 'DTLZ' not in prob:
            true_pf = problem.pareto_front()
        else:
            ref_dir = get_uniform_weights(100, 2)
            true_pf = problem.pareto_front(ref_dir)

        max_by_truepf = np.amax(true_pf, axis=0)
        min_by_truepf = np.amin(true_pf, axis=0)

        # plot pareto front
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))

        ax1.scatter(best_f_ego[:, 0], best_f_ego[:, 1], c='b', marker='o')
        ax1.scatter(true_pf[:, 0], true_pf[:, 1], c='r', marker='x')
        ax1.legend([method, 'true_pf'])
        ax1.set_title(prob + ' hv')

        for i in range(n1):
            zuobiao = '[' + "{:4.2f}".format(f_pareto[i, 0]) + ', ' + "{:4.2f}".format(f_pareto[i, 1]) + ']'
            ax1.text(f_pareto[i, 0], f_pareto[i, 1], zuobiao)

        ax2.scatter(best_f_ego_r[:, 0], best_f_ego_r[:, 1], c='b', marker='o')
        ax2.scatter(true_pf[:, 0], true_pf[:, 1], c='r', marker='x')
        for i in range(n2):
            zuobiao = '[' + "{:4.2f}".format(f_pareto_r[i, 0]) + ', ' + "{:4.2f}".format(f_pareto_r[i, 1]) + ']'
            ax2.text(f_pareto_r[i, 0], f_pareto_r[i, 1], zuobiao)

        # ax2.set(xlim=(min_by_truepf[0], max_by_truepf[0]), ylim=(min_by_truepf[1], max_by_truepf[1]))
        ax2.legend([method+'r', 'true_pf'])
        ax2.set_title(prob + ' hvr')

        saveName = 'visualization\\' + prob + '_' + method + '_and_hvr_compare.png'
        plt.savefig(saveName)

    else:

        ref_dir = get_uniform_weights(1000, 3)
        true_pf = problem.pareto_front(ref_dir)

        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(true_pf[:, 0], true_pf[:, 1], true_pf[:, 2], c='r', marker='x')
        ax.scatter(best_f_ego[:, 0], best_f_ego[:, 1], best_f_ego[:, 2], c='b', marker='o')
        ax.view_init(30, 60)
        ax.set_title(prob + run_signature)
        ax.legend(['true_pf', method])

        saveName = 'visualization\\' + run_signature + prob + '_' + method + '_compare2pf.png'
        plt.savefig(saveName)
    plt.show()
    a = 1

def load_and_process():

    hv_igd = []
    for seed in np.arange(1, 31):
        filename = 'sample_out_freensga_' + str(seed)+'.csv'
        a = np.loadtxt(filename)
        print(a)
        hv_igd = np.append(hv_igd, a[1])
        hv_igd = np.append(hv_igd, a[2])

    hv_igd = np.atleast_2d(hv_igd).reshape(-1, 2)
    print(hv_igd)
    savename = 'nsga_free_seeding.csv'
    np.savetxt(savename, hv_igd, delimiter=',')
    a = 0

def load_hv_igd(prob, run_signature, seed):
    # output_folder_name = 'outputs\\experiment_post_process_100_evaluation\\' + prob + '_' + run_signature
    output_folder_name = 'outputs\\' + prob + '_' + run_signature

    if os.path.exists(output_folder_name):
        print(output_folder_name)
    else:
        print(output_folder_name)
        raise ValueError(
            "results folder for EGO does not exist"
        )

    output_f_name = output_folder_name + '\\hv_igd_' + str(seed) + '.csv'
    saved = np.loadtxt(output_f_name, delimiter=',')
    return saved[0], saved[1]



def combine_hv_igd_out(methods, seeds, problems, foldername):

    working_folder = os.getcwd()
    result_folder = working_folder + '\\' + foldername
    if not os.path.isdir(result_folder):
        # shutil.rmtree(result_folder)
        # os.mkdir(result_folder)
        os.mkdir(result_folder)


    n_seed = len(seeds)
    n_problem = len(problems)
    n_method = len(methods)

    problem_list = []
    output_matrix_h = np.zeros((n_problem, n_method * 3))  # column: median + index per method
    output_matrix_g = np.zeros((n_problem, n_method * 3))

    for problem_index, problem_str in enumerate(problems):
        problem = eval(problem_str)
        problem_name = problem.name()
        problem_list = np.append(problem_list, problem_name)

        save_file_hv = result_folder + '\\combined_hv_results_' + problem_name + '.csv'
        save_file_igd = result_folder + '\\combined_igd_results_' + problem_name + '.csv'

        hv_collection = []
        igd_collection = []
        for method in methods:
            for seed in seeds:
                hv, igd = load_hv_igd(problem_name, method, seed)
                hv_collection = np.append(hv_collection, hv)
                igd_collection = np.append(igd_collection, igd)

        hv_collection = np.atleast_2d(hv_collection).reshape(n_seed, -1, order='F')
        igd_collection = np.atleast_2d(igd_collection).reshape(n_seed, -1, order='F')



        median_h = []
        median_h_seed = []
        mean_h = []
        std_h = []
        ci_h = []


        median_g = []
        median_g_seed = []
        mean_g = []
        std_g = []
        ci_g = []
        for i, _ in enumerate(methods):
            mean_h.append(np.mean(hv_collection[:, i]))
            std_h.append(np.std(hv_collection[:, i]))
            cih = sample_ci(hv_collection[:, i])
            ci_h.append(cih)
            index = np.argsort(hv_collection[:, i])
            n = len(index)
            select_median_index = index[int(n/2)]
            median_h = np.append(median_h, hv_collection[:, i][select_median_index])
            median_h_seed = np.append(median_h_seed, select_median_index)

            mean_g.append(np.mean(igd_collection[:, i]))
            std_g.append(np.std(igd_collection[:, i]))
            cig = sample_ci(igd_collection[:, i])
            ci_g.append(cig)
            index = np.argsort(igd_collection[:, i])
            n = len(index)
            select_median_index = index[int(n / 2)]
            median_g = np.append(median_g, igd_collection[:, i][select_median_index])
            median_g_seed = np.append(median_g, select_median_index)

        mean_h = np.atleast_2d(mean_h).reshape(1, -1)
        std_h = np.atleast_2d(std_h).reshape(1, -1)
        ci_h = np.atleast_2d(ci_h).reshape(1, -1)
        median_h = np.atleast_2d(median_h).reshape(1, -1)
        hv_collection = np.vstack((hv_collection, mean_h, std_h, ci_h, median_h))

        mean_g = np.atleast_2d(mean_g).reshape(1, -1)
        std_g = np.atleast_2d(std_g).reshape(1, -1)
        ci_g = np.atleast_2d(ci_g).reshape(1, -1)
        median_g = np.atleast_2d(median_g).reshape(1, -1)
        igd_collection = np.vstack((igd_collection, mean_g, std_g, ci_g, median_g))

        # fill in output matrix
        for i, method in enumerate(methods):
            output_matrix_h[problem_index, 3 * i] = median_h[0, i]
            output_matrix_h[problem_index, 3 * i + 1] = mean_h[0, i]  # median_h_seed[i]
            output_matrix_h[problem_index, 3 * i + 2] = ci_h[0, i]  # median_h_seed[i]
            output_matrix_g[problem_index, 3 * i] = median_g[0, i]
            output_matrix_g[problem_index, 3 * i + 1] = mean_g[0, i]
            output_matrix_g[problem_index, 3 * i + 2] = ci_g[0, i]
            seed_median = int(median_h_seed[i])
            # plot_pareto_vs_ouputs(problem_str, [seed_median], method, method, 'ref_compare_visual')

        index_seeds = seeds.copy()
        for i, seed in enumerate(seeds):
            index_seeds[i] = str(seed)

        index_seeds = np.append(seeds, ['mean', 'std', 'ci', 'median'])

        h = pd.DataFrame(hv_collection, columns=methods, index=index_seeds)
        d = pd.DataFrame(igd_collection, columns=methods, index=index_seeds)

        h.to_csv(save_file_hv)
        d.to_csv(save_file_igd)
        a = 0

    # save output_matrix in to excel
    index_seeds = problems
    methods_ = np.atleast_2d(methods).reshape(1, -1)
    methods_ = np.repeat(methods_, 3, axis=1)
    methods_ = methods_.ravel()
    for i, method in enumerate(methods):
        methods_[3 * i] = methods_[3 * i] + 'median'
        methods_[3 * i + 1] = methods_[3 * i+1] + 'mean'
        methods_[3 * i + 2] = methods_[3 * i+2] + 'ci'


    h = pd.DataFrame(output_matrix_h, columns=methods_, index=index_seeds)
    g = pd.DataFrame(output_matrix_g, columns=methods_, index=index_seeds)

    save_file_hv = result_folder + '\\combined_hv_results_all_problems_mean_median.csv'
    save_file_igd = result_folder + '\\combined_igd_results_all_problems_mean_median.csv'

    h.to_csv(save_file_hv)
    g.to_csv(save_file_igd)



if __name__ == "__main__":

    run_signature = [# 'eim',
                     # 'eim_r',
                     # 'eim_nd',
                     # 'eim_r3'
                     'hvr',
                     'hv',
                     #'hv_r3'
                     ]

    # load_and_process()
    # run_extract_result(run_signature[2])

    MO_target_problems = [
          'ZDT1(n_var=6)',
          'ZDT2(n_var=6)',
          'ZDT3(n_var=6)',
          # 'WFG.WFG_1(n_var=6, n_obj=2, K=4)',
          # 'WFG.WFG_2(n_var=6, n_obj=2, K=4)',
         #  'WFG.WFG_3(n_var=6, n_obj=2, K=4)',
         #  'WFG.WFG_4(n_var=6, n_obj=2, K=4)',
         # #   'WFG.WFG_5(n_var=6, n_obj=2, K=4)',
         #  'WFG.WFG_6(n_var=6, n_obj=2, K=4)',
          # 'WFG.WFG_7(n_var=6, n_obj=2, K=4)',
         # 'WFG.WFG_8(n_var=6, n_obj=2, K=4)',
         # 'WFG.WFG_9(n_var=6, n_obj=2, K=4)',
         # 'DTLZ1(n_var=6, n_obj=2)',
        #  'DTLZ2(n_var=6, n_obj=2)',
         # 'DTLZs.DTLZ5(n_var=6, n_obj=2)',
        #  'DTLZs.DTLZ7(n_var=6, n_obj=2)',
        # # 'iDTLZ.IDTLZ1(n_var=6, n_obj=2)',
        # 'iDTLZ.IDTLZ2(n_var=6, n_obj=2)',
    ]

    for i in np.arange(30,40):
        seed = [i]
        for problem in MO_target_problems:
            for method in run_signature:
                a = 0
                # plot_pareto_vs_ouputs(problem, seed, method, method, 'ref_compare_visual')

    seeds = np.arange(0, 10)
    combine_hv_igd_out(run_signature, seeds, MO_target_problems, 'ref_compare_num')



    '''
    from pymop.factory import get_uniform_weights
    DTLZ2 = DTLZ2(n_var=8, n_obj=3)
    ref_dir = get_uniform_weights(1000, 3)
    pf = DTLZ2.pareto_front(ref_dir)

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cm = plt.cm.get_cmap('RdYlBu')
    ax.scatter(pf[:, 0], pf[:, 1], pf[:, 2])
    for angle in range(0, 360):
        ax.view_init(30, angle)
        plt.draw()
        plt.pause(.001)
    # plt.show()
    
    
   
    problem_list = ['ZDT1', 'ZDT3', 'ZDT2',  'DTLZ1', 'DTLZ2', 'DTLZ4']  # 'BNH', 'Kursawe', 'WeldedBeam']
    methods = ['eim', 'hv', 'hvr']
    

    for p in problem_list:
        for method in methods:
            plot_pareto_vs_ouputs(p, np.arange(0, 1), method, run_signature[3])


    # parEGO_out_process()
    '''


    # plot_pareto_vs_ouputs_compare_hv_hvr('ZDT1', np.arange(0, 10), 'hv', run_signature[6])
    # problem = ZDT3(n_var=6)
    # f = problem.pareto_front(10000)
    # np.savetxt('zdt3front.txt', f, delimiter=',')













