#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 00:08:57 2023

@author: alex
"""

from bsm_math import *


class VanillaOption():

    def __init__(self, K, T, Flag):
        self.K = K
        self.T = T
        self.Flag = Flag

    def bs_value(self, S, r, q, sigma):
        return bsm_value(S, self.K, self.T, r, q, sigma, self.Flag)

    def bs_vega(self, S, r, q, sigma):
        return bsm_vega(S, self.K, self.T, r, q, sigma)

    def bs_delta(self, S, r, q, sigma):
        return bsm_delta(S, self.K, self.T, r, q, sigma, self.Flag)

    def bs_gamma(self, S, r, q, sigma):
        return bsm_gamma(S, self.K, self.T, r, q, sigma)

    def bs_theta(self, S, r, q, sigma):
        return bsm_theta(S, self.K, self.T, r, q, sigma, self.Flag)

    def bs_rho(self, S, r, q, sigma):
        return bsm_rho(S, self.K, self.T, r, q, sigma, self.Flag)

    def bs_delta_prem(self, S, r, q, sigma):
        return bsm_delta_prem(S, self.K, self.T, r, q, sigma, self.Flag)

    def bs_ivol(self, S, r, q, V, sigma_est=0.3, it=100, tol=0.001):
        return bsm_ivol(S, self.K, self.T, r, q, V, self.Flag, sigma_est,
                        it=100, tol=0.001)
