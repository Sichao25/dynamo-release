import numpy as np
from scipy.sparse import issparse
from tqdm import tqdm

from ...tools.utils import calc_norm_loglikelihood, calc_R2, elem_prod, find_extreme
from ..csc.utils_velocity import fit_linreg, fit_stochastic_linreg, fit_first_order_deg_lsq


def fit_beta_ss(t, U):
    """Estimate beta and gamma with the degradation data using the least squares method.

    Arguments
    ---------
        t: :class:`~numpy.ndarray`
            A vector of time points.
        U: :class:`~numpy.ndarray`
            A matrix of unspliced mRNA counts. Dimension: genes x cells.
        S: :class:`~numpy.ndarray`
            A matrix of spliced mRNA counts. Dimension: genes x cells.

    Returns
    -------
        beta: :class:`~numpy.ndarray`
            A vector of betas for all the genes.
        gamma: :class:`~numpy.ndarray`
            A vector of gammas for all the genes.
        u0: float
            Initial value of u.
        s0: float
            Initial value of s.
    """
    n = U.shape[0]  # self.get_n_genes(data=U)
    beta = np.zeros(n)
    u0, s0 = np.zeros(n), np.zeros(n)

    for i in tqdm(range(n), desc="estimating beta"):
        beta[i], u0[i] = fit_first_order_deg_lsq(t, U[i])
    return beta


def fit_slope_stochastic(S, U, US, S2, perc_left=None, perc_right=5):
    n_var = S.shape[0]
    k, all_r2, all_logLL = np.zeros(n_var), np.zeros(n_var), np.zeros(n_var)

    for i, s, u, us, s2 in tqdm(
        zip(np.arange(n_var), S, U, US, S2),
        "Estimate slope k via linear regression.",
    ):
        u = u.A.flatten() if issparse(u) else u.flatten()
        s = s.A.flatten() if issparse(s) else s.flatten()
        us = us.A.flatten() if issparse(us) else us.flatten()
        s2 = s2.A.flatten() if issparse(s2) else s2.flatten()

        mask = find_extreme(u, s, perc_left=perc_left, perc_right=perc_right)
        k[i] = fit_stochastic_linreg(u[mask], s[mask], us[mask], s2[mask])

        all_r2[i] = calc_R2(s, u, k[i])
        all_logLL[i] = calc_norm_loglikelihood(s, u, k[i])

    return k, 0, all_r2, all_logLL


def fit_labeling_synthesis(new, total, t, intercept=False, perc_left=None, perc_right=None):
    T = np.unique(t)
    K = np.zeros(len(T))
    R2 = np.zeros(len(T))
    for i in range(len(T)):
        n = new[t == T[i]]
        r = total[t == T[i]]
        eind = find_extreme(n, r, perc_left=perc_left, perc_right=perc_right)
        K[i], _, R2[i], _ = fit_linreg(r[eind], n[eind], intercept=intercept)
    return K, R2


def fit_total_to_spliced(total, spliced):
    return [fit_linreg(spliced[i], total[i], intercept=False, r2=False)[0] for i in range(total.shape[0])]


def compute_gamma_synthesis(K, T):
    gamma, _, r2, _ = fit_linreg(T, -np.log(1 - K))
    return gamma, r2


def compute_velocity_synthesis(N, R, gamma, t):
    k = 1 - np.exp(-np.einsum("i,j->ij", t, gamma))
    V = elem_prod(gamma, N) / k - elem_prod(gamma, R)
    return V


def lin_reg_gamma_synthesis(R, N, time, perc_right=100):
    n_var = R.shape[0]
    mean_R2, gamma, r2 = np.zeros(n_var), np.zeros(n_var), np.zeros(n_var)
    K_list, K_fit_list = [None] * n_var, [None] * n_var
    for i, r, n in tqdm(
        zip(np.arange(n_var), R, N),
        "Estimate gamma via linear regression of t vs. -ln(1-K)",
    ):
        r = r.A.flatten() if issparse(r) else r.flatten()
        n = n.A.flatten() if issparse(n) else n.flatten()

        K_list[i], R2 = fit_labeling_synthesis(n, r, time, perc_right=perc_right)
        gamma[i], r2[i] = compute_gamma_synthesis(K_list[i], np.unique(time))
        K_fit_list[i] = np.unique(time) * gamma[i]
        mean_R2[i] = np.mean(R2)

    return gamma, r2, K_list, mean_R2, K_fit_list
