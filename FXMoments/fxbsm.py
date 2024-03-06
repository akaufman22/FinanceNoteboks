"""
Extesion on standard bsm math for fx market
"""

from math import exp, log
import scipy.optimize as optimize
from bsm_math import bsm_value, bsm_delta

def bsm_delta_forward(S, K, T, r, q, sigma, Flag):
    return bsm_delta(S, K, T, r, q, sigma, Flag) * exp(q * T)

def bsm_delta_prem(S, K, T, r, q, sigma, Flag):
    return(bsm_delta(S, K, T, r, q, sigma, Flag) -
           bsm_value(S, K, T, r, q, sigma, Flag) / S)

def implied_foreign_depo(S, sw, T, depo):
    return depo - log(1 + (sw/S)) / T

def implied_domestic_depo(S, sw, T, fdepo):
    return fdepo + log(1 + (sw/S)) / T

def atm_dns(S, T, rd, rf, atm_vol,
            premium_included=1):
    assert premium_included in [0, 1], "Invalid premium_included given"
    if premium_included == 0:
        delta = bsm_delta
    else:
        delta = bsm_delta_prem
    y = lambda x: delta(S, x, T, rd, rf, atm_vol, 0) + delta(S, x, T, rd, rf, atm_vol, 1)
    dnk = optimize.newton(y, S)
    return dnk

def strike_from_delta_and_vol(S, T, rd, rf, vol, d, Flag,
                              premium_included=1, forward_delta=0, tol=0.00001, maxiter=100):
    assert premium_included in [0, 1], "Invalid premium_included given"
    assert forward_delta in [0, 1], "Invalid forward_delta given"
    if forward_delta == 1:
        multiplier = exp(rf * T)
    else:
        multiplier = 1
    if premium_included == 0:
        delta = bsm_delta
    else:
        delta = bsm_delta_prem
    y = lambda x: abs(multiplier * delta(S, x, T, rd, rf, vol, Flag) - d)
    strike = optimize.newton(y, S, tol=tol, maxiter=maxiter)
    return strike

def market_smile(S, T, rd, rf, quotes,
                 premium_included=1, forward_delta=0, atm_forward=0):
    """
    Procedure transforms market quotes into vols
    with abs strikes and put forward deltas as coordinates
    """
    assert premium_included in [0, 1], "Invalid premium_included given"
    assert forward_delta in [0, 1], "Invalid forward_delta given"
    assert atm_forward in [0, 1], "Invalid atm_forward given"
    atm = quotes[0] / 100
    put_10d = (quotes[0] + quotes[4] - quotes[3] / 2) / 100
    put_25d = (quotes[0] + quotes[2] - quotes[1] / 2) / 100
    call_25d = (quotes[0] + quotes[2] + quotes[1] / 2) / 100
    call_10d = (quotes[0] + quotes[4] + quotes[3] / 2) / 100

    vols = [put_10d, put_25d, atm, call_25d, call_10d]

    if atm_forward == 1:
        atm_strike = exp((rd- rf) * T) * S
    else:
        atm_strike = atm_dns(S, T, rd, rf, atm, premium_included)
    call_25d_strike = strike_from_delta_and_vol(
        S, T, rd, rf, call_25d, 0.25, 0, premium_included, forward_delta)
    call_10d_strike = strike_from_delta_and_vol(
        S, T, rd, rf, call_10d, 0.1, 0, premium_included, forward_delta)
    put_25d_strike = strike_from_delta_and_vol(
        S, T, rd, rf, put_25d, -0.25, 1, premium_included, forward_delta)
    put_10d_strike = strike_from_delta_and_vol(
        S, T, rd, rf, put_10d, -0.1, 1, premium_included, forward_delta)
    strikes = [put_10d_strike, put_25d_strike, atm_strike, call_25d_strike, call_10d_strike]

    pdeltas = [bsm_delta_forward(S, k, T, rd, rf, vol, 1) for k, vol in zip(strikes, vols)]
    return (strikes, pdeltas, vols)

