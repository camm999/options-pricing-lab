import numpy as np

from optionspricing import black_scholes_call, black_scholes_put, implied_volatility_newton

S, K, T, r = 100, 100, 1, 0.05


def test_recovers_known_call_volatility():
    true_sigma = 0.35
    market_price = black_scholes_call(S, K, T, r, true_sigma)

    recovered = implied_volatility_newton(market_price, S, K, T, r, option_type="call")

    assert np.isclose(recovered, true_sigma, atol=1e-4)


def test_recovers_known_put_volatility():
    true_sigma = 0.15
    market_price = black_scholes_put(S, K, T, r, true_sigma)

    recovered = implied_volatility_newton(market_price, S, K, T, r, option_type="put")

    assert np.isclose(recovered, true_sigma, atol=1e-4)


def test_recovers_volatility_across_a_range_of_strikes():
    true_sigma = 0.25
    for k in np.linspace(60, 140, 9):
        price = black_scholes_call(S, k, T, r, true_sigma)
        recovered = implied_volatility_newton(price, S, k, T, r, option_type="call")
        assert np.isclose(recovered, true_sigma, atol=1e-3)
