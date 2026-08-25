"""Implied volatility via Newton-Raphson root finding.

Given a market price, solve BS(sigma) = C_market for sigma. The equation has
no closed-form inverse, so we iterate:

    sigma_{n+1} = sigma_n - f(sigma_n) / f'(sigma_n)

where f(sigma) = BS(sigma) - C_market and f'(sigma) is exactly vega, which we
already have a closed form for.
"""

import math

from .black_scholes import black_scholes_call, black_scholes_put
from .greeks import vega

# Newton-Raphson step size is diff/vega, and vega collapses towards zero for
# very short-dated or far-from-the-money contracts. Without a bound, a single
# unlucky step can fling sigma to absurd values (seen in practice: >1e10)
# from which it never recovers. Clamp iterates to a generous but sane range,
# matching the [0.01, 5] bounds used by scipy-based solvers for the same
# problem.
_MIN_SIGMA = 1e-4
_MAX_SIGMA = 5.0


def implied_volatility_newton(
    market_price,
    S,
    K,
    T,
    r,
    q=0.0,
    option_type="call",
    initial_guess=0.2,
    tolerance=1e-8,
    max_iterations=100,
):
    """
    Solve for implied volatility given a market option price.

    Returns the converged sigma, or NaN if it fails to converge within
    `max_iterations` (e.g. stale/illiquid quotes, or near-expiry contracts
    where vega is too small for Newton-Raphson to make useful progress).
    Callers building a surface/smile from many quotes should drop NaNs
    rather than plot them.
    """
    price_fn = black_scholes_call if option_type == "call" else black_scholes_put
    sigma = initial_guess

    for _ in range(max_iterations):
        price = price_fn(S, K, T, r, sigma, q)
        diff = price - market_price

        if abs(diff) < tolerance:
            return sigma

        v = vega(S, K, T, r, sigma, q)
        if v < 1e-12:
            return math.nan

        sigma = sigma - diff / v
        sigma = min(max(sigma, _MIN_SIGMA), _MAX_SIGMA)

    return math.nan
