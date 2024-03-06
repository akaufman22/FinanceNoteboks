import numpy as np

def bsm_value(S, K, T, r, q, sigma, Flag):
    from scipy import stats

    d1 = (np.log(S/K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S/K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if Flag == 0:
        value = (S * np.exp(-q * T) * stats.norm.cdf(d1, 0.0, 1.0) -
                 K * np.exp(-r * T) * stats.norm.cdf(d2, 0.0, 1.0))
    elif Flag == 1:
        value = (K * np.exp(-r * T) * stats.norm.cdf(-d2, 0.0, 1.0) -
                 S * np.exp(-q * T) * stats.norm.cdf(-d1, 0.0, 1.0))
    else:
        value = 'NaN'
    return value


def bs76_value(F, K, T, r, sigma, Flag):
    from scipy import stats

    F = float(F)
    K = float(K)
    d1 = (np.log(F/K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if Flag == 0:
        value = (F * np.exp(-r * T) * stats.norm.cdf(d1, 0.0, 1.0) -
                 K * np.exp(-r * T) * stats.norm.cdf(d2, 0.0, 1.0))
    elif Flag == 1:
        value = (K * np.exp(-r * T) * stats.norm.cdf(-d2, 0.0, 1.0) -
                 F * np.exp(-r * T) * stats.norm.cdf(-d1, 0.0, 1.0))
    else:
        value = 'NaN'
    return value


def bsm_vega(S, K, T, r, q, sigma):
    from scipy import stats
    S = float(S)
    K = float(K)
    d1 = (np.log(S/K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    vega = S * stats.norm.pdf(d1, 0.0, 1.0) * np.sqrt(T)
    return vega


def bsm_delta(S, K, T, r, q, sigma, Flag):
    from scipy import stats
    d1 = (np.log(S/K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if Flag == 0:
        delta = np.exp(-q * T) * stats.norm.cdf(d1, 0.0, 1.0)
    elif Flag == 1:
        delta = np.exp(-q * T) * (stats.norm.cdf(d1, 0.0, 1.0) - 1)
    else:
        delta = 'NaN'
    return delta

def bsm_ivol(S, K, T, r, q, V, Flag, sigma_est, it=100, tol=0.001):
    for _ in range(it):
        sigma_prev = sigma_est
        sigma_est -= ((bsm_value(S, K, T, r, q, sigma_est, Flag) - V) /
                      bsm_vega(S, K, T, r, q, sigma_est))
        if abs(sigma_est - sigma_prev) < tol:
            break
    return sigma_est


def bsm_gamma(S, K, T, r, q, sigma):
    from scipy import stats
    d1 = (np.log(S/K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return (stats.norm.pdf(d1, 0, 1) * np.exp(-q * T) / (S * sigma * np.sqrt(T)))


def bsm_theta(S, K, T, r, q, sigma, Flag):
    from scipy import stats
    d1 = (np.log(S/K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S/K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if Flag == 0:
        theta = -np.exp(-q * T) * S * sigma * stats.norm.pdf(d1, 0, 1) / \
            (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * stats.norm.cdf(d2, 0, 1) + \
            q * S * np.exp(-q * T) * stats.norm.cdf(d1, 0, 1)
    elif Flag == 1:
        theta = -np.exp(-q * T) * S * sigma * stats.norm.pdf(-d1, 0, 1) / \
            (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * stats.norm.cdf(-d2, 0, 1) - \
            q * S * np.exp(-q * T) * stats.norm.cdf(-d1, 0, 1)
    else:
        theta = 'NaN'
    return theta


def bsm_rho(S, K, T, r, q, sigma, Flag):
    from scipy import stats
    d2 = (np.log(S/K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if Flag == 0:
        rho = K * T * np.exp(-r * T) * stats.norm.cdf(d2, 0, 1)
    elif Flag == 1:
        rho = -K * T * np.exp(-r * T) * stats.norm.cdf(-d2, 0, 1)
    else:
        rho = 'NaN'
    return rho
