#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 16:50:30 2023

@author: alex
"""


def bsm_value(S, K, T, r, q, sigma, Flag):
    from math import log, sqrt, exp
    from scipy import stats

    S = float(S)
    K = float(K)
    d1 = (log(S/K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = (log(S/K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    if Flag == 0:
        value = (S * exp(-q * T) * stats.norm.cdf(d1, 0.0, 1.0) -
                 K * exp(-r * T) * stats.norm.cdf(d2, 0.0, 1.0))
    elif Flag == 1:
        value = (K * exp(-r * T) * stats.norm.cdf(-d2, 0.0, 1.0) -
                 S * exp(-q * T) * stats.norm.cdf(-d1, 0.0, 1.0))
    else:
        value = 'NaN'
    return value


def bs76_value(F, K, T, r, sigma, Flag):
    from math import log, sqrt, exp
    from scipy import stats

    F = float(F)
    K = float(K)
    d1 = (log(F/K) + (0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if Flag == 0:
        value = (F * exp(-r * T) * stats.norm.cdf(d1, 0.0, 1.0) -
                 K * exp(-r * T) * stats.norm.cdf(d2, 0.0, 1.0))
    elif Flag == 1:
        value = (K * exp(-r * T) * stats.norm.cdf(-d2, 0.0, 1.0) -
                 F * exp(-r * T) * stats.norm.cdf(-d1, 0.0, 1.0))
    else:
        value = 'NaN'
    return value


def bsm_vega(S, K, T, r, q, sigma):
    from math import log, sqrt
    from scipy import stats
    S = float(S)
    K = float(K)
    d1 = (log(S/K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    vega = S * stats.norm.pdf(d1, 0.0, 1.0) * sqrt(T)
    return vega


def bsm_delta(S, K, T, r, q, sigma, Flag):
    from math import log, sqrt, exp
    from scipy import stats
    S = float(S)
    K = float(K)
    d1 = (log(S/K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    if Flag == 0:
        delta = exp(-q * T) * stats.norm.cdf(d1, 0.0, 1.0)
    elif Flag == 1:
        delta = exp(-q * T) * (stats.norm.cdf(d1, 0.0, 1.0) - 1)
    else:
        delta = 'NaN'
    return delta


def bsm_delta_prem(S, K, T, r, q, sigma, Flag):
    delta = (bsm_delta(S, K, T, r, q, sigma, Flag) -
             bsm_value(S, K, T, r, q, sigma, Flag) / S)
    return delta


def bsm_ivol(S, K, T, r, q, V, Flag, sigma_est, it=100, tol=0.001):
    for i in range(it):
        sigma_prev = sigma_est
        sigma_est -= ((bsm_value(S, K, T, r, q, sigma_est, Flag) - V) /
                      bsm_vega(S, K, T, r, q, sigma_est))
        if abs(sigma_est - sigma_prev) < tol:
            break
    return sigma_est


def bsm_gamma(S, K, T, r, q, sigma):
    from math import log, sqrt, exp
    from scipy import stats
    d1 = (log(S/K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    return (stats.norm.pdf(d1, 0, 1) * exp(-q * T) / (S * sigma * sqrt(T)))


def bsm_theta(S, K, T, r, q, sigma, Flag):
    from math import log, sqrt, exp
    from scipy import stats
    d1 = (log(S/K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = (log(S/K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    if Flag == 0:
        theta = -exp(-q * T) * S * sigma * stats.norm.pdf(d1, 0, 1) / \
            (2 * sqrt(T)) - r * K * exp(-r * T) * stats.norm.cdf(d2, 0, 1) + \
            q * S * exp(-q * T) * stats.norm.cdf(d1, 0, 1)
    elif Flag == 1:
        theta = -exp(-q * T) * S * sigma * stats.norm.pdf(-d1, 0, 1) / \
            (2 * sqrt(T)) + r * K * exp(-r * T) * stats.norm.cdf(-d2, 0, 1) - \
            q * S * exp(-q * T) * stats.norm.cdf(-d1, 0, 1)
    else:
        theta = 'NaN'
    return theta


def bsm_rho(S, K, T, r, q, sigma, Flag):
    from math import log, sqrt, exp
    from scipy import stats
    d2 = (log(S/K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    if Flag == 0:
        rho = K * T * exp(-r * T) * stats.norm.cdf(d2, 0, 1)
    elif Flag == 1:
        rho = -K * T * exp(-r * T) * stats.norm.cdf(-d2, 0, 1)
    else:
        rho = 'NaN'
    return rho
