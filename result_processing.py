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


def plot_pareto_vs_ouputs(prob, alg1, alg2=None, alg3=None):

    # read ouput f values
    output_folder_name = 'outputs\\' + prob
    if os.path.exists(output_folder_name):
        output_f_name = output_folder_name + '\\best_f_seed_100.joblib'
        best_f_ego = load(output_f_name)
    else:
        raise ValueError(
            "results folder for EGO does not exist"
        )

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

    # normalize pareto front and output
    min_pf_by_feature = np.amin(true_pf, axis=0)
    max_pf_by_feature = np.amax(true_pf, axis=0)
    norm_true_pf = (true_pf - min_pf_by_feature)/(max_pf_by_feature - min_pf_by_feature)

    min_pf_by_feature = np.atleast_2d(min_pf_by_feature).reshape(1, -1)
    max_pf_by_feature = np.atleast_2d(max_pf_by_feature).reshape(1, -1)

    # normalize algorithm output
    best_f_ego = (best_f_ego - min_pf_by_feature)/(max_pf_by_feature - min_pf_by_feature)

    if alg3:
        best_f_nsga = (best_f_nsga -min_pf_by_feature)/(max_pf_by_feature - min_pf_by_feature)

    '''
    
    reference_point = [1.1] * n_obj

    best_f_ego = best_f_ego.tolist()
    if alg3:
        best_f_nsga = best_f_nsga.tolist()


    # calculate hypervolume index
    hv_ego = pg.hypervolume(best_f_ego)


    if alg3:
        hv_nsga = pg.hypervolume(best_f_nsga)

    hv_ego_value = hv_ego.compute(reference_point)

    if alg3:
        hv_nsga_value = hv_nsga.compute(reference_point)


   
    with open('mo_compare.txt', 'a') as f:
        p = prob
        f.write(p)
        f.write('\t')
        f.write(str(hv_ego_value))
        f.write('\t')
        if alg3:
            f.write(str(hv_nsga_value))

        f.write('\n')

    '''
    # plot pareto front
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(best_f_ego[:, 0], best_f_ego[:, 1], c='b', marker='o')
    ax.scatter(norm_true_pf[:, 0], norm_true_pf[:, 1], c='r', marker='x')
    plt.title(prob)
    saveName = 'visualization\\' + prob + '_compare.png'
    plt.savefig(saveName)
    plt.show()



if __name__ == "__main__":

    problem_list = ['ZDT1','ZDT2','ZDT3','ZDT4', 'BNH', 'Kursawe', 'WeldedBeam']
    for p in problem_list:
        plot_pareto_vs_ouputs(p, 'ego')











