"""Geometric Brownian Motion simulation and Monte Carlo option pricing.

Under the risk-neutral measure, S(t+dt) = S(t) * exp((r - q - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
with Z ~ N(0, 1). Simulating many terminal prices and averaging discounted
payoffs converges to the Black-Scholes price by the Law of Large Numbers.
"""

import numpy as np


def simulate_gbm_paths(S0, mu, sigma, T, n_steps, n_paths, rng=None):
    """
    Simulate `n_paths` GBM price paths under the real-world drift `mu`.

    Returns an array of shape (n_paths, n_steps + 1), including S0 at t=0.
    """
    rng = rng or np.random.default_rng()
    dt = T / n_steps

    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0

    Z = rng.standard_normal((n_paths, n_steps))
    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    paths[:, 1:] = S0 * np.exp(np.cumsum(increments, axis=1))

    return paths


def simulate_terminal_price(S0, r, sigma, T, n_simulations, q=0.0, rng=None):
    """Simulate terminal prices S(T) directly under the risk-neutral measure."""
    rng = rng or np.random.default_rng()
    Z = rng.standard_normal(n_simulations)
    return S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)


def monte_carlo_price(S0, K, T, r, sigma, option_type="call", q=0.0, n_simulations=100_000, rng=None):
    """
    Price a European option via Monte Carlo simulation of terminal prices.

    option_type: "call" or "put"
    """
    ST = simulate_terminal_price(S0, r, sigma, T, n_simulations, q=q, rng=rng)

    if option_type == "call":
        payoffs = np.maximum(ST - K, 0)
    elif option_type == "put":
        payoffs = np.maximum(K - ST, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return np.exp(-r * T) * np.mean(payoffs)
