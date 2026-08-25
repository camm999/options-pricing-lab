import numpy as np

from optionspricing import black_scholes_call, black_scholes_put


def test_put_call_parity():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.20
    call = black_scholes_call(S, K, T, r, sigma)
    put = black_scholes_put(S, K, T, r, sigma)

    assert np.isclose(call - put, S - K * np.exp(-r * T))


def test_put_call_parity_with_dividends():
    S, K, T, r, sigma, q = 100, 90, 0.5, 0.03, 0.25, 0.02
    call = black_scholes_call(S, K, T, r, sigma, q)
    put = black_scholes_put(S, K, T, r, sigma, q)

    assert np.isclose(call - put, S * np.exp(-q * T) - K * np.exp(-r * T))


def test_deep_itm_call_approaches_intrinsic_value():
    S, K, T, r, sigma = 1000, 100, 1, 0.05, 0.20
    call = black_scholes_call(S, K, T, r, sigma)
    intrinsic = S - K * np.exp(-r * T)

    assert np.isclose(call, intrinsic, rtol=1e-3)


def test_deep_otm_call_is_near_zero():
    S, K, T, r, sigma = 10, 1000, 1, 0.05, 0.20
    call = black_scholes_call(S, K, T, r, sigma)

    assert call < 1e-6
