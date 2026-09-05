"""Analytical and finite-difference Greeks for European vanilla options.

Each analytical Greek has a `numerical_*` counterpart computed via central
finite differences, so the two can be cross-checked against each other
(see tests/test_greeks.py).
"""

import numpy as np
from scipy.stats import norm

from .black_scholes import black_scholes_call, black_scholes_put, d1, d2


def delta_call(S, K, T, r, sigma, q=0.0):
    """Delta: sensitivity of option price to a $1 move in the underlying."""
    return np.exp(-q * T) * norm.cdf(d1(S, K, T, r, sigma, q))


def delta_put(S, K, T, r, sigma, q=0.0):
    return np.exp(-q * T) * (norm.cdf(d1(S, K, T, r, sigma, q)) - 1)


def gamma(S, K, T, r, sigma, q=0.0):
    """Gamma: rate of change of delta. Same for calls and puts, always positive."""
    _d1 = d1(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(_d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma, q=0.0):
    """Vega: sensitivity to volatility, per 1-unit (100%) move in sigma. Same for calls and puts."""
    _d1 = d1(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(_d1) * np.sqrt(T)


def theta_call(S, K, T, r, sigma, q=0.0):
    """Theta: sensitivity to time passage, per year. Long calls almost always have negative theta."""
    _d1 = d1(S, K, T, r, sigma, q)
    _d2 = d2(S, K, T, r, sigma, q)
    decay = -(S * np.exp(-q * T) * norm.pdf(_d1) * sigma) / (2 * np.sqrt(T))
    carry = -r * K * np.exp(-r * T) * norm.cdf(_d2)
    dividend = q * S * np.exp(-q * T) * norm.cdf(_d1)
    return decay + carry + dividend


def theta_put(S, K, T, r, sigma, q=0.0):
    _d1 = d1(S, K, T, r, sigma, q)
    _d2 = d2(S, K, T, r, sigma, q)
    decay = -(S * np.exp(-q * T) * norm.pdf(_d1) * sigma) / (2 * np.sqrt(T))
    carry = r * K * np.exp(-r * T) * norm.cdf(-_d2)
    dividend = -q * S * np.exp(-q * T) * norm.cdf(-_d1)
    return decay + carry + dividend


def rho_call(S, K, T, r, sigma, q=0.0):
    """Rho: sensitivity to the risk-free rate. Positive for calls, negative for puts."""
    _d2 = d2(S, K, T, r, sigma, q)
    return K * T * np.exp(-r * T) * norm.cdf(_d2)


def rho_put(S, K, T, r, sigma, q=0.0):
    _d2 = d2(S, K, T, r, sigma, q)
    return -K * T * np.exp(-r * T) * norm.cdf(-_d2)


# --- Finite-difference cross-checks -----------------------------------------
#
# NOTE: an earlier draft of these called black_scholes_call(S, K, r, sigma, T)
# instead of the correct (S, K, T, r, sigma) signature, which silently swapped
# T/r/sigma and made the numerical Greeks disagree with the analytical ones
# (e.g. numerical vega came out ~4x too small). Kept as a cautionary note
# since tests/test_greeks.py asserts these two families agree.


def numerical_delta(S, K, T, r, sigma, q=0.0, h=0.01):
    return (
        black_scholes_call(S + h, K, T, r, sigma, q) - black_scholes_call(S - h, K, T, r, sigma, q)
    ) / (2 * h)


def numerical_gamma(S, K, T, r, sigma, q=0.0, h=0.01):
    return (
        black_scholes_call(S + h, K, T, r, sigma, q)
        - 2 * black_scholes_call(S, K, T, r, sigma, q)
        + black_scholes_call(S - h, K, T, r, sigma, q)
    ) / (h**2)


def numerical_vega(S, K, T, r, sigma, q=0.0, h=0.001):
    return (
        black_scholes_call(S, K, T, r, sigma + h, q) - black_scholes_call(S, K, T, r, sigma - h, q)
    ) / (2 * h)


def numerical_theta(S, K, T, r, sigma, q=0.0, h=1 / 365):
    # calendar time t and time-to-expiry T run in opposite directions, so
    # theta = dV/dt = -dV/dT: take the central difference in T and negate it
    return -(
        black_scholes_call(S, K, T + h, r, sigma, q) - black_scholes_call(S, K, T - h, r, sigma, q)
    ) / (2 * h)


def numerical_rho(S, K, T, r, sigma, q=0.0, h=0.0001):
    return (
        black_scholes_call(S, K, T, r + h, sigma, q) - black_scholes_call(S, K, T, r - h, sigma, q)
    ) / (2 * h)
