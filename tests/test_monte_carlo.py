import numpy as np

from optionspricing import black_scholes_call, black_scholes_put, monte_carlo_price

S, K, T, r, sigma = 100, 100, 1, 0.05, 0.20


def test_monte_carlo_call_converges_to_black_scholes():
    rng = np.random.default_rng(0)
    bs_price = black_scholes_call(S, K, T, r, sigma)
    mc_price = monte_carlo_price(S, K, T, r, sigma, option_type="call", n_simulations=200_000, rng=rng)

    assert abs(mc_price - bs_price) / bs_price < 0.01


def test_monte_carlo_put_converges_to_black_scholes():
    rng = np.random.default_rng(0)
    bs_price = black_scholes_put(S, K, T, r, sigma)
    mc_price = monte_carlo_price(S, K, T, r, sigma, option_type="put", n_simulations=200_000, rng=rng)

    assert abs(mc_price - bs_price) / bs_price < 0.01


def test_monte_carlo_error_shrinks_with_more_simulations():
    bs_price = black_scholes_call(S, K, T, r, sigma)

    small_error = abs(monte_carlo_price(S, K, T, r, sigma, n_simulations=500, rng=np.random.default_rng(1)) - bs_price)
    large_error = abs(monte_carlo_price(S, K, T, r, sigma, n_simulations=200_000, rng=np.random.default_rng(1)) - bs_price)

    assert large_error < small_error


def test_invalid_option_type_raises():
    import pytest

    with pytest.raises(ValueError):
        monte_carlo_price(S, K, T, r, sigma, option_type="not_an_option")
