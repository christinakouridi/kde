""""
Functions to perform Kernel Density Estimation using a Gaussian kernel.
Numbda is utilised to improve computationally efficiency, through prange() and the wrapper @njit.
"""

import numpy as np
from numba import njit, prange


@njit(parallel=True)
def compute_log_prob(x, mu, sigma):
    """
    Computes the log-probability of x, where log p(x) = Sum_i^d p(z_i)p(x|z_i). It follows equation X in our report.
    Makes use of the log-sum-exp trick to avoid numerical issues such as underflow in the exponential term for small
    sigmas, when its exponent is large and negative [ref: https://en.wikipedia.org/wiki/LogSumExp/]

    Parameters
    ----------
    x : sample in the validation / test set
    mu : mean values of Gaussian components p(x_j|z_i)
    sigma : standard deviation of all gaussian components p(.|z_i)

    Returns
    -------
    log_p : log-probability of sample x
    """
    k, d = mu.shape

    two_sigma_sqr = 2 * sigma ** 2
    prefactor = - 0.5 * np.log(np.pi * two_sigma_sqr)

    summand = - np.square(x-mu) / two_sigma_sqr + prefactor  # summation term with exp exponent in eq X in our report
    exponent = - np.log(k) + summand.sum(axis=1)  # exponent of exp term in eq X in our report

    # uses the log-sum-exp trick c = x.max();  logp = c + np.log(np.sum(np.exp(x - c)))
    c = np.max(exponent)
    exponent_scaled = exponent - c
    exp = np.exp(exponent_scaled).sum()
    log_p = np.log(exp) + c
    return log_p


@njit(parallel=True)
def kde_gauss(data_a: np.ndarray, data_b: np.ndarray, sigma: float) -> float:
    """
    Computes the mean log-probability of data_b using the Gaussian KDE method

    Parameters
    ----------
    data_a : first data split, 'training data'
    data_b : second data split, 'validation / test data'
    sigma : standard deviation of all gaussian components p(.|z_i)

    Returns
    -------
    mean_log_likelihood : mean log-probability of data_b
    """
    m = len(data_b)
    log_likelihood = 0.

    # get the log-probability of each sample in data_b
    for i in prange(m):
        log_likelihood_i = compute_log_prob(x=data_b[i], mu=data_a, sigma=sigma)
        log_likelihood += log_likelihood_i

    mean_log_likelihood = log_likelihood / m
    return mean_log_likelihood
