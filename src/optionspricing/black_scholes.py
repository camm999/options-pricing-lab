"""Black-Scholes-Merton pricing for European vanilla options.

Includes a continuous dividend yield q (defaults to 0, which recovers the
plain Black-Scholes formula used for the non-dividend-paying examples).
"""

import numpy as np
from scipy.stats import norm


def d1(S, K, T, r, sigma, q=0.0):
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def d2(S, K, T, r, sigma, q=0.0):
    return d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)


def black_scholes_call(S, K, T, r, sigma, q=0.0):
    """
    Black-Scholes-Merton European call price.

    Parameters
    ----------
    S     : current underlying price
    K     : strike price
    T     : time to expiry in years
    r     : continuously compounded risk-free rate
    sigma : annualised volatility
    q     : continuous dividend yield (default 0)

    Returns
    -------
    Call option price (float)
    """
    _d1 = d1(S, K, T, r, sigma, q)
    _d2 = d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.cdf(_d1) - K * np.exp(-r * T) * norm.cdf(_d2)


def black_scholes_put(S, K, T, r, sigma, q=0.0):
    """
    Black-Scholes-Merton European put price.

    With q=0 this is equivalent to computing it via put-call parity:
    C - P = S - K*e^(-rT)  =>  P = C - S + K*e^(-rT)
    """
    _d1 = d1(S, K, T, r, sigma, q)
    _d2 = d2(S, K, T, r, sigma, q)
    return K * np.exp(-r * T) * norm.cdf(-_d2) - S * np.exp(-q * T) * norm.cdf(-_d1)
