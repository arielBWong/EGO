import numpy as np
from scipy.stats import norm
from sklearn.utils.validation import check_array
import pygmo as pg
from scipy import special


# this will be the evaluation function that is called each time
from pymop.factory import get_problem_from_func

def close_adjustment(nd_front):

    # check if any of current nd front got too close to each other
    # align them together
    nd_front = check_array(nd_front)
    n_obj = nd_front.shape[1]
    n_nd = nd_front.shape[0]

    for i in range(0, n_obj):
        check_closeness = nd_front[:, i]
        for j in range(n_nd):
            for k in range(j + 1, n_nd):
                if abs(check_closeness[j] - check_closeness[k]) < 1e-5:
                    smaller = min([check_closeness[j], check_closeness[k]])
                    check_closeness[j] = check_closeness[k] = smaller

    return nd_front

def gaussiancdf(x):
    # x = check_array(x)
    y = 0.5 * (1 + special.erf(x / np.sqrt(2)))
    return y

def gausspdf(x):
    # x = check_array(x)
    y = 1/np.sqrt(2*np.pi) * np.exp(-np.square(x)/2)
    return y



# calculate expected hv for multiple objective problems
def EIM_hv(mu, sig, nd_front, reference_point):

    # mu sig nu_front has to be np_2d
    mu = check_array(mu)
    sig = check_array(sig)
    nd_front = check_array(nd_front)
    reference_point = check_array(reference_point)

    # np.savetxt('sig.csv', sig, delimiter=',')
    # np.savetxt('mu.csv', mu, delimiter=',')
    # np.savetxt('nd_front.csv', nd_front, delimiter=',')

    n_nd = nd_front.shape[0]
    n_mu = mu.shape[0]

    mu_extend = np.repeat(mu, n_nd, axis=0)
    sig_extend = np.repeat(sig, n_nd, axis=0)
    r_extend = np.repeat(reference_point, n_nd * n_mu, axis=0)

    nd_front_extend = np.tile(nd_front, (n_mu, 1))

    imp = (nd_front_extend - mu_extend)/sig_extend
    EIM = (nd_front_extend - mu_extend) * gaussiancdf(imp) + \
            sig_extend * gausspdf(imp)

    y1 = np.prod((r_extend - nd_front_extend + EIM), axis=1)
    y2 = np.prod((r_extend - nd_front_extend), axis=1)

    y = np.atleast_2d((y1 - y2)).reshape(-1, n_nd)
    y = np.min(y, axis=1)
    y = np.atleast_2d(y).reshape(-1, 1)

    # one beyond reference
    # diff = reference_point - mu
    # y_beyond = np.any(diff < 0, axis=1)
    # y_beyond = np.atleast_2d(y_beyond).reshape(-1, 1)
    # y = y * y_beyond

    return y


def EI_hv(mu_norm, nd_front_norm, reference_point_norm):

    # comply with pg settings
    reference_point_norm = reference_point_norm.ravel()
    n = mu_norm.shape[0]
    ei = []
    n_var = mu_norm.shape[1]
    for i in range(n):

        if np.any(np.atleast_2d(mu_norm[i, :]).reshape(-1, n_var) >
                  np.atleast_2d(reference_point_norm).reshape(-1, n_var),
                  axis=1):
            ei.append(0)
        else:
            if np.sum(np.isnan(mu_norm[i, :])) > 0:  # nan check
                ei.append(0)
            else:
                point_list = np.vstack((nd_front_norm, mu_norm[i, :]))
                hv = pg.hypervolume(point_list)
                hv_value = hv.compute(reference_point_norm)
                ei.append(hv_value)
    ei = np.atleast_2d(ei).reshape(n, -1)
    return ei


def HVR(ideal, nadir, nd_front, mu, n_obj):

    # use estimated ideal and nadir points from outside loop
    # to extract normalization boundary
    min_pf_by_feature = ideal
    max_pf_by_feature = nadir

    n_var = mu.shape[1]

    norm_nd = (nd_front - min_pf_by_feature) / (max_pf_by_feature - min_pf_by_feature)
    point_reference = np.atleast_2d([1.1] * n_obj)
    norm_mu = (mu - min_pf_by_feature) / (max_pf_by_feature - min_pf_by_feature)

    reference_point_norm = point_reference.ravel()
    n = norm_mu.shape[0]
    ei = []
    for i in range(n):
        if np.any(np.atleast_2d(norm_mu[i, :]).reshape(-1, n_var) > point_reference.reshape(-1, n_var),
                  axis=1):
            # print(norm_mu[i, :])
            # print('beyond reference point')
            ei.append(0)
        else:
            point_list = np.vstack((norm_nd, norm_mu[i, :]))
            if np.any(norm_mu[i, :] != norm_mu[i, :]):  # nan check
                ei.append(0)
            else:
                hv = pg.hypervolume(point_list)
                hv_value = hv.compute(reference_point_norm)
                ei.append(hv_value)
    ei = np.atleast_2d(ei).reshape(n, -1)
    return ei


def eim_infill_metric(x, nd_front_norm,krg):
    # for testing eim metric
    n_obj = len(krg)
    y_norm = []
    sig_norm = []
    for k in krg:
        y, sig = k.predict(x)
        y_norm = np.append(y_norm, y)
        sig_norm = np.append(sig_norm, sig)
    y_norm = np.atleast_2d(y_norm).reshape(-1, n_obj, order='F')
    sig_norm = np.atleast_2d(sig_norm).reshape(-1, n_obj, order='F')
    ei = EIM_hv(y_norm, sig_norm, nd_front_norm, np.atleast_2d([1.1]*n_obj))
    return -ei


def normalization_with_nd(mu, data_y):

    # using nd front as normalization boundary
    n_obj = data_y.shape[1]
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(data_y)
    ndf = list(ndf)
    ndf_size = len(ndf)
    # extract nd for normalization
    if len(ndf[0]) > 1:
        ndf_extend = ndf[0]
    else:
        ndf_extend = np.append(ndf[0], ndf[1])

    nd_front = data_y[ndf_extend, :]

    # normalization boundary
    min_nd_by_feature = np.amin(nd_front, axis=0)
    max_nd_by_feature = np.amax(nd_front, axis=0)

    if np.any(max_nd_by_feature - min_nd_by_feature < 1e-5):
        print('nd front aligned problem, re-select nd front')
        ndf_index = ndf[0]
        for k in np.arange(1, ndf_size):
            ndf_index = np.append(ndf_index, ndf[k])
            nd_front = data_y[ndf_index, :]
            min_nd_by_feature = np.amin(nd_front, axis=0)
            max_nd_by_feature = np.amax(nd_front, axis=0)
            if np.any(max_nd_by_feature - min_nd_by_feature < 1e-5):
                continue
            else:
                break

    # normalize nd front and x population for ei
    norm_nd = (nd_front - min_nd_by_feature) / (max_nd_by_feature - min_nd_by_feature)
    norm_mu = (mu - min_nd_by_feature) / (max_nd_by_feature - min_nd_by_feature)

    point_reference = np.atleast_2d([1.1] * n_obj)
    return norm_mu, norm_nd, point_reference



def expected_improvement(x,
                         train_x,
                         train_y,
                         norm_train_y,
                         feasible,
                         nadir,
                         ideal,
                         ei_method,
                         krg,
                         krg_g=None
                         ):

    n_samples = x.shape[0]
    n_obj = len(krg)
    n_g = len(krg_g)

    # predict f
    mu = []
    sigma = []
    for k in krg:
        mu_1, sigma_1 = k.predict(x)
        mu = np.append(mu, mu_1)
        sigma = np.append(sigma, sigma_1)
    mu = np.atleast_2d(mu).reshape(-1, n_obj, order='F')
    sigma = np.atleast_2d(sigma).reshape(-1, n_obj, order='F')

    # change to matrix calculation
    pf = np.atleast_2d(np.ones((n_samples, 1)))

    if n_g > 0:
        # with constraint
        mu_gx = []
        sigma_gx = []
        for k in krg_g:
            mu_g1, sigma_g1 = k.predict(x)
            # pf operate on denormalized range
            mu_gx = np.append(mu_gx, mu_g1)
            sigma_gx = np.append(sigma_gx, sigma_g1)

        # re-organise to 2d
        mu_gx = np.atleast_2d(mu_gx).reshape(-1, n_g, order='F')
        sigma_gx = np.atleast_2d(sigma_gx).reshape(-1, n_g, order='F')

        with np.errstate(divide='warn'):
            pf_m = norm.cdf((0 - mu_gx)/sigma_gx)
            pf = np.prod(pf_m, axis=1)
            pf = np.atleast_2d(pf).reshape(-1, 1)

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

    if n_obj > 1:
        # multi-objective situation
        if n_g > 0:
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
            else:
                return pf
        else:

            # using nd front as normalization boundary
            ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(train_y)
            ndf = list(ndf)
            # extract nd index
            if len(ndf[0]) > 1:
                ndf_extend = ndf[0]
            else:
                ndf_extend = np.append(ndf[0], ndf[1])

            # calculate eim metrics
            if ei_method == 'eim' or ei_method == 'eim_nd':
                nd_front_norm = norm_train_y[ndf[0], :]
                point_reference = np.atleast_2d([1.1] * n_obj)
                ei = EIM_hv(mu, sigma, nd_front_norm, point_reference)

            elif ei_method == 'eim_r' or ei_method == 'eim_r3':
                # reference point
                nd_front_norm = norm_train_y[ndf[0], :]
                point_reference = np.atleast_2d([1.1] * n_obj)
                ei = EIM_hv(mu, sigma, nd_front_norm, point_reference)

            elif ei_method == 'hv':
                norm_mu, norm_nd, point_reference = normalization_with_nd(mu, train_y)
                ei = EI_hv(norm_mu, norm_nd, point_reference)

            elif ei_method == 'hvr' or ei_method == 'hv_r3':
                # reference adjustment uses its own ideal and nadir for normalization
                nd_front = train_y[ndf_extend, :]
                # normalize with ideal nadir in HVR function
                ei = HVR(ideal, nadir, nd_front, mu, n_obj)

            else:
                raise(
                    'EI_krg MO process does not have this method'
                )



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
                        norm_train_y,
                        krg,
                        krg_g,
                        feasible,
                        nadir,
                        ideal,
                        ei_method
                        ):

    # redundent jump from before
    # wrap EI method, use minus to minimize
    out["F"] = -expected_improvement(x,
                                     train_x,
                                     train_y,
                                     norm_train_y,
                                     feasible,
                                     nadir,
                                     ideal,
                                     ei_method,
                                     krg,
                                     krg_g,
                                     )



if __name__ == "__main__":
    a = [[0, 1],[-0.5, 0.5]]
    a = np.atleast_2d(a)
    print(a)
    print(norm.cdf(a))