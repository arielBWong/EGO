import numpy as np
from scipy.stats import norm
from sklearn.utils.validation import check_array
import pygmo as pg


# this will be the evaluation function that is called each time
from pymop.factory import get_problem_from_func


# calculate expected hv for multiple objective problems
def EIM_hv(mu, sig, nd_front, reference_point):
    # mu sig nu_front has to be np_2d

    mu = check_array(mu)
    sig = check_array(sig)
    nd_front = check_array(nd_front)
    n_nd = nd_front.shape[0]

    n_mu = mu.shape[0]
    mu_extend = np.repeat(mu, n_nd, axis=0)
    sig_extend = np.repeat(sig, n_nd, axis=0)
    r_extend = np.repeat(reference_point, n_nd * n_mu, axis=0)

    nd_front_extend = np.tile(nd_front, (n_mu, 1))

    imp = (nd_front_extend - mu_extend)/sig_extend
    EIM = (nd_front_extend - mu_extend) * norm.cdf(imp) + \
            sig_extend * norm.pdf(imp)
    y1 = np.prod(r_extend - nd_front_extend + EIM, axis=1)
    y2 = np.prod(r_extend - nd_front_extend, axis=1)

    y = np.atleast_2d((y1 - y2)).reshape(-1, n_nd)
    y = np.min(y, axis=1)

    return y



def expected_improvement(x,
                         train_x,
                         train_y,
                         feasible,
                         nadir,
                         ideal,
                         krg,
                         krg_g=None
                         ):

    n_samples = x.shape[0]
    n_obj = len(krg)

    mu_temp = np.zeros((n_samples, 1))
    sigma_temp = np.zeros((n_samples, 1))
    convert_index = 0
    for k in krg:
        mu, sigma = k.predict(x)

        sigma = np.atleast_2d(sigma)
        sigma_temp = np.hstack((sigma_temp, sigma))

        mu = np.atleast_2d(mu)
        mu_temp = np.hstack((mu_temp, mu))

        convert_index = convert_index + 1

    mu = np.delete(mu_temp, 0, 1).reshape(n_samples, n_obj)
    sigma = np.delete(sigma_temp, 0, 1).reshape(n_samples, n_obj)

    # change to matrix calculation
    pf = np.atleast_2d(np.ones((n_samples, 1)))

    if len(krg_g) > 0:
        # with constraint
        n_g = len(krg_g)
        mu_temp = np.zeros((n_samples, 1))
        sigma_temp = np.zeros((n_samples, 1))

        convert_index = 0
        for k in krg_g:
            mu_gx, sigma_gx = k.predict(x)


            # pf operate on denormalized range
            mu_gx = np.atleast_2d(mu_gx)
            mu_temp = np.hstack((mu_temp, mu_gx))

            sigma_gx = np.atleast_2d(sigma_gx)
            sigma_temp = np.hstack((sigma_temp, sigma_gx))

            convert_index = convert_index + 1

        # re-organise, and delete zero volume
        mu_gx = np.delete(mu_temp, 0, 1)
        sigma_gx = np.delete(sigma_temp, 0, 1)

        with np.errstate(divide='warn'):
            for each_g in range(n_g):
                pf_m = norm.cdf((0 - mu_gx[:, each_g])/sigma_gx[:, each_g])
                pf = pf * pf_m
            pf = np.atleast_2d(pf_m).reshape(-1, 1)

        if len(feasible) > 0:
            # If there is feasible solutions
            # EI to look for both feasible and EI preferred solution
            mu_sample_opt = np.min(feasible, axis=0)
        else:
            # If there is no feasible solution,
            # then EI go look for feasible solutions
            return pf
    else:
        # without constraint
        mu_sample_opt = np.min(train_y, axis=0)

    if len(krg) > 1:
        # multi-objective situation
        if len(krg_g) > 0:
            # this condition means mu_gx has been calculated
            if len(feasible) > 1:
                # ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(feasible)
                # f_pareto = feasible[ndf[0], :]
                f_pareto = feasible

                # normalize pareto front for ei
                min_pf_by_feature = np.amin(f_pareto, axis=0)
                max_pf_by_feature = np.amax(f_pareto, axis=0)
                norm_pf = (f_pareto - min_pf_by_feature) / (max_pf_by_feature - min_pf_by_feature)

                point_reference = [1.1] * n_obj
                norm_mu = (mu - min_pf_by_feature) / (max_pf_by_feature - min_pf_by_feature)

                compare_mu = np.all(norm_mu < point_reference, axis=1)
                ei = np.atleast_2d(np.zeros((n_samples, 1)))

                for index, compare in enumerate(compare_mu):
                    if compare:  # in bounding box
                        point_list = np.vstack((norm_pf, norm_mu[index, :]))
                        point_list = point_list.tolist()
                        hv = pg.hypervolume(point_list)
                        ei[index] = hv.compute(point_reference)

                '''
                # calculate hyper volume
                point_list = np.vstack((norm_pf, norm_mu))
                ei = 1.0
                for v in range(n_obj):
                    if norm_mu[0, v] >= point_reference[v]:
                        ei = 0
                        break
                if ei != 0:
                    # print(point_reference)
                    # print(point_list)
                    point_list = point_list.tolist()
                    hv = pg.hypervolume(point_list)
                    hv_value = hv.compute(point_reference)
                    ei = hv_value
                '''
            else:
                return pf
        else:
            # ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(train_y)
            # ndf = list(ndf)
            # f_pareto = train_y[ndf[0], :]
            f_pareto = train_y

            # normalize pareto front for ei
            # min_pf_by_feature = np.amin(f_pareto, axis=0)
            # max_pf_by_feature = np.amax(f_pareto, axis=0)
            min_pf_by_feature = ideal
            max_pf_by_feature = nadir

            norm_pf = (f_pareto - min_pf_by_feature) / (max_pf_by_feature - min_pf_by_feature)
            # test on suggested point
            point_reference = [1.1] * n_obj

            # norm_pf eliminate out of range points
            compare_pf = np.all(norm_pf < point_reference, axis=1)
            compare_pf = compare_pf.tolist()
            within_pf = compare_pf.index(True)
            norm_pf = norm_pf[within_pf, :]


            norm_mu = (mu - min_pf_by_feature) / (max_pf_by_feature - min_pf_by_feature)




            compare_mu = np.all(norm_mu < point_reference, axis=1)
            ei = np.atleast_2d(np.zeros((n_samples, 1)))

            for index, compare in enumerate(compare_mu):
                if compare:  # in bounding box
                    point_list = np.vstack((norm_pf, norm_mu[index, :]))
                    point_list = point_list.tolist()
                    hv = pg.hypervolume(point_list)
                    ei[index] = hv.compute(point_reference)

            # print(ei)

            '''
            # calculate hyper volume
            point_list = np.vstack((norm_pf, norm_mu))
            ei = 1.0
            for v in range(n_obj):
                if norm_mu[0, v] >= point_reference[v]:
                    ei = 0
                    break
            if ei != 0:
                # print(point_reference)
                # print(point_list)
                # print(norm_mu)
                point_list = point_list.tolist()
                hv = pg.hypervolume(point_list)
                hv_value = hv.compute(point_reference)

                ei = hv_value
            '''
    else:
        # single objective situation
        with np.errstate(divide='warn'):
            imp = mu_sample_opt - mu
            # print(imp.shape)
            # print(sigma.shape)
            Z = imp / sigma
            ei1 = imp * norm.cdf(Z)
            ei1[sigma == 0.0] = 0.0
            ei2 = sigma * norm.pdf(Z)
            ei = (ei1 + ei2)

    pena_ei = ei * pf
    pena_ei = np.atleast_2d(pena_ei)

    return pena_ei


# this acqusition function on G should be refactored
def acqusition_function(x,
                        out,
                        train_x,
                        train_y,
                        krg,
                        krg_g,
                        feasible,
                        nadir,
                        ideal,
                        ):

    dim = train_x.shape[1]
    x = np.atleast_2d(x).reshape(-1, dim)

    # wrap EI method, use minus to minimize
    out["F"] = -expected_improvement(x,
                                     train_x,
                                     train_y,
                                     feasible,
                                     nadir,
                                     ideal,
                                     krg,
                                     krg_g
                                     )



