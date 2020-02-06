import numpy as np
import optimizer
from joblib import dump, load
import os
import pygmo as pg
from pymop import ZDT1, ZDT2, ZDT3, ZDT4, DTLZ1, G1, DTLZ2, BNH, Carside, Kursawe, OSY, Truss2D, WeldedBeam, TNK
import matplotlib.pyplot as plt

def reverse_zscore(data, m, s):
    return data * s + m



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


def plot_pareto_vs_ouputs(prob, seed, alg1, alg2=None, alg3=None):

    # read ouput f values
    output_folder_name = 'outputs\\' + prob

    if os.path.exists(output_folder_name):
        print('output folder exists')
    else:
        raise ValueError(
            "results folder for EGO does not exist"
        )

    output_f_name = output_folder_name + '\\best_f_seed_' + str(seed[0]) + '.joblib'
    best_f_ego = load(output_f_name)
    for s in seed[1:]:
        output_f_name = output_folder_name + '\\best_f_seed_' + str(s) + '.joblib'
        best_f_ego1 = load(output_f_name)
        best_f_ego = np.vstack((best_f_ego, best_f_ego1))

    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(best_f_ego)
    ndf = list(ndf)
    f_pareto = best_f_ego[ndf[0], :]
    best_f_ego = f_pareto



    # read ouput f from alg1
    if alg2:
        output_folder_name = 'parEGO_out\\' + prob
        # for now stop here as parEGO may be implemented in matlab

    # read pareto front from nsga2
    if alg3:
        output_folder_name = 'NSGA2\\' + prob
        if os.path.exists(output_folder_name):
            output_f_name = output_folder_name + 'pareto_f.joblib'
            best_f_nsga = load(output_f_name)

    # extract pareto front
    if 'ZDT' not in prob:
        problem_obj = prob + "()"
    else:
        problem_obj = prob + '(n_var=3)'

    problem = eval(problem_obj)
    true_pf = problem.pareto_front()
    n_obj = problem.n_obj

    max_by_truepf = np.amax(true_pf, axis=0)
    min_by_truepf = np.amin(true_pf, axis=0)


    # plot pareto front
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))

    ax1.scatter(best_f_ego[:, 0], best_f_ego[:, 1], c='b', marker='o')
    ax1.scatter(true_pf[:, 0], true_pf[:, 1], c='r', marker='x')
    ax1.legend(['hv_EIM', 'true_pf'])
    ax1.set_title(prob)

    ax2.scatter(best_f_ego[:, 0], best_f_ego[:, 1], c='b', marker='o')
    ax2.scatter(true_pf[:, 0], true_pf[:, 1], c='r', marker='x')
    ax2.set(xlim=(min_by_truepf[0], max_by_truepf[0]), ylim=(min_by_truepf[1], max_by_truepf[1]))
    ax2.legend(['hv_EIM', 'true_pf'])
    ax2.set_title(prob+' zoom in')

    saveName = 'visualization\\' + prob + '_ei_hv_compare2pf.png'
    plt.savefig(saveName)
    plt.show()

def run_extract_result():

    problem_list = ['ZDT1', 'ZDT2', 'ZDT3', 'DTLZ2','DTLZ4', 'DTLZ1']
    method_list = ['hv', 'eim', 'hvr']
    seedlist = np.arange(0, 10)

    true_pf = ZDT3.pareto_front()
    true_pf = 1.1 * np.amax(true_pf, axis=0)

    reference_dict = {'ZDT1': [1.1, 1.1],
                      'ZDT2': [1.1, 1.1],
                      'ZDT3': true_pf,
                      'DTLZ2': [1.1, 1.1, 1.1],
                      'DTLZ4':  [1.1, 1.1, 1.1],
                      'DTLZ1': [0.5, 0.5, 0.5]
                      }

    with open('\\data_processed\\hv_eim_hvr.csv', 'w+') as f:
        for prob in problem_list:
            problem_save = []

            for method in method_list:
                hv = extract_results(method, prob, seedlist, reference_dict[prob])
                problem_save.append(hv)

            for method_out in problem_save:
                for hv_element in method_out:
                    f.write(hv_element)
                    f.write(',')
            f.write('\n')







def extract_results(method, prob, seed_index, reference_point):

    # read ouput f values
    output_folder_name = 'outputs\\' + prob
    if os.path.exists(output_folder_name):
        print('output folder exists')
    else:
        raise ValueError(
            "results folder for EGO does not exist"
        )

    hv = []
    reference_point = reference_point.ravel()
    for seed in seed_index:
        output_f_name = output_folder_name + '\\best_x_seed_' + str(seed) + '_' + method + '.joblib'
        best_f_ego = load(output_f_name)
        n_obj = best_f_ego.shape[1]

        # deal with out of range
        select = []
        for f in best_f_ego:
            if np.all(f <= reference_point):
                np.append(select, f)
        best_f_ego = np.atleast_2d(select).reshape(-1, n_obj)

        hv = pg.hypervolume(best_f_ego)
        hv_value = hv.compute(reference_point)
        np.append(hv, hv_value)


    hv_min = np.min(hv)
    hv_max = np.max(hv)
    hv_avg = np.average(hv)
    hv_std = np.std(hv)

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





if __name__ == "__main__":

    run_extract_result()

    '''
    problem_list = ['ZDT3'] # 'ZDT3', 'ZDT4']  # 'BNH', 'Kursawe', 'WeldedBeam']
    # plot_pareto_vs_ouputs(problem_list[2], np.arange(100, 101), 'ego')
    for p in problem_list:
        plot_pareto_vs_ouputs(p, np.arange(0, 10), 'ego')

    # parEGO_out_process()

  
    output_f_name = 'outputs\\DTLZ2\\best_f_seed_100.joblib'
    best_f_ego = load(output_f_name)

    parEGO_folder_name = 'parEGO_out\\DTLZ2'
    out_file = parEGO_folder_name + '.txt'
    f = np.genfromtxt(out_file, delimiter='\t')
    f = np.atleast_2d(f).reshape(-1, 2)
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(f)
    ndf = list(ndf)
    f_pareto = f[ndf[0], :]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(best_f_ego[:, 0], best_f_ego[:, 1], c='g', marker='d')
    ax.scatter(f_pareto[:, 0], f_pareto[:, 1], c='b', marker='o')
    plt.legend(['EGO_new', 'parEGO'])
    plt.show()
    '''











