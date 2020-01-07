import numpy as np
from scipy.stats import norm
from sklearn.utils.validation import check_array
import pygmo as pg


# this will be the evaluation function that is called each time
from pymop.factory import get_problem_from_func


def expected_improvement(X,
                         X_sample,
                         Y_sample,
                         y_mean,
                         y_std,
                         cons_g_mean,
                         cons_g_std,
                         feasible, gpr,
                         gpr_g=None,
                         xi=0.01):

    # X_sample/Y_sample, in the denormalized range
    n_samples = X.shape[0]
    n_obj = len(gpr)
    # mu, sigma = gpr.predict(X, return_std=True)
    mu_temp = np.zeros((n_samples, 1))
    sigma_temp = np.zeros((n_samples, 1))


    convert_index = 0
    for g in gpr:
        mu, sigma = g.predict(X, return_cov=True)

        # convert back to denormalized range
        sigma = np.atleast_2d(sigma)
        sigma = sigma * y_std[convert_index] + y_mean[convert_index]
        sigma_temp = np.hstack((sigma_temp, sigma))

        # convert back to denormalized range
        mu = np.atleast_2d(mu)
        mu = mu * y_std[convert_index] + y_mean[convert_index]
        mu_temp = np.hstack((mu_temp, mu))

        convert_index = convert_index + 1

    mu = np.delete(mu_temp, 0, 1).reshape(n_samples, n_obj)
    sigma = np.delete(sigma_temp, 0, 1).reshape(n_samples, n_obj)


    pf = 1.0

    if len(gpr_g) > 0:
        # with constraint
        n_g = len(gpr_g)
        mu_temp = np.zeros((n_samples, 1))
        sigma_temp = np.zeros((n_samples, 1))
        convert_index = 0
        for g in gpr_g:
            mu_gx, sigma_gx = g.predict(X, return_cov=True)

            # pf operate on denormalized range
            mu_gx = np.atleast_2d(mu_gx)
            mu_gx = mu_gx * cons_g_std[convert_index] + cons_g_mean[convert_index]
            mu_temp = np.hstack((mu_temp, mu_gx))

            # gpr prediction on sigma is not the same dimension as the mu
            # details have not been checked, here just make a conversion
            # on sigma
            sigma_gx = np.atleast_2d(sigma_gx)
            sigma_gx = sigma_gx * cons_g_std[convert_index] + cons_g_mean[convert_index]
            sigma_temp = np.hstack((sigma_temp, sigma_gx))

            convert_index = convert_index + 1

        # re-organise, and delete zero volume
        mu_gx = np.delete(mu_temp, 0, 1)
        sigma_gx = np.delete(sigma_temp, 0, 1)

        with np.errstate(divide='warn'):

            if sigma_gx == 0:
                z = 0
            pf = norm.cdf((0 - mu_gx) / sigma_gx)
            # create pf on multiple constraints (multiply over all constraints)
            pf_m = pf[:, 0]
            for i in np.arange(1, n_g):
                pf_m = pf_m * pf[:, i]
            pf = np.atleast_2d(pf_m).reshape(-1, 1)

        if feasible.size > 0:
            # If there is feasible solutions
            # EI to look for both feasible and EI preferred solution
            mu_sample_opt = np.min(feasible)
        else:
            # If there is no feasible solution,
            # then EI go look for feasible solutions
            return pf
    else:
        # without constraint
        mu_sample_opt = np.min(Y_sample)

    if len(gpr) > 1:
        # multi-objective situation
        if len(gpr_g) > 0:
            # this condition means mu_gx has been calculated
            if feasible.size > 0:
                ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(feasible)
                f_pareto = feasible[ndf, :]
                point_nadir = np.max(f_pareto, axis=0)
                point_reference = point_nadir * 1.1

                # calculate hyper volume
                point_list = np.vstack((f_pareto, mu))
                if mu[0][0] > point_reference[0][0] or mu[0][1] > point_reference[0][1]:
                    ei = 1e-5
                else:
                    hv = pg.hypervolume(point_list)
                    hv_value = hv.compute(point_reference)
                    ei = hv_value

            else:
                return pf
        else:
            ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(Y_sample)
            ndf = list(ndf)
            f_pareto = Y_sample[ndf[0], :]
            point_nadir = np.max(f_pareto, axis=0)
            point_reference = point_nadir * 1.1

            # calculate hyper volume
            point_list = np.vstack((f_pareto, mu))
            if mu[0, 0] > point_reference[0] or mu[0, 1] > point_reference[1]:
                ei = 1e-5
            else:
                hv = pg.hypervolume(point_list)
                hv_value = hv.compute(point_reference)
                ei = hv_value

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
    # print('return penalized ei')

    return pena_ei


# this acqusition function on G should be refactored
def acqusition_function(x,
                        out,
                        X_sample,
                        Y_sample,
                        y_mean,
                        y_std,
                        cons_g_mean,
                        cons_g_std,
                        gpr,
                        gpr_g,
                        feasible,
                        xi=0.01):

    dim = X_sample.shape[1]
    x = np.atleast_2d(x).reshape(-1, dim)

    # wrap EI method, use minus to minimize
    out["F"] = -expected_improvement(x,
                                     X_sample,
                                     Y_sample,
                                     y_mean,
                                     y_std,
                                     cons_g_mean,
                                     cons_g_std,
                                     feasible,
                                     gpr,
                                     gpr_g,
                                     xi=0.01)



