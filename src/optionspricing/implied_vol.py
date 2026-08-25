"""Implied volatility via Newton-Raphson root finding.

Given a market price, solve BS(sigma) = C_market for sigma. The equation has
no closed-form inverse, so we iterate:

    sigma_{n+1} = sigma_n - f(sigma_n) / f'(sigma_n)

where f(sigma) = BS(sigma) - C_market and f'(sigma) is exactly vega, which we
already have a closed form for.
"""

from .black_scholes import black_scholes_call, black_scholes_put
from .greeks import vega


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

    Returns the converged sigma. Falls back to whatever the last iterate was
    if `max_iterations` is reached without hitting `tolerance` (e.g. for
    deep-OTM options where vega is tiny and Newton-Raphson stalls).
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
            break

        sigma = sigma - diff / v
        sigma = max(sigma, 1e-6)  # keep iterates in a sane, positive range

    return sigma
