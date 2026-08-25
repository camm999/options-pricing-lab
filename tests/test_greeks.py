import numpy as np

from optionspricing.greeks import (
    delta_call,
    gamma,
    vega,
    theta_call,
    rho_call,
    numerical_delta,
    numerical_gamma,
    numerical_vega,
    numerical_theta,
    numerical_rho,
)

S, K, T, r, sigma = 100, 100, 1, 0.05, 0.20


def test_delta_matches_finite_difference():
    assert np.isclose(delta_call(S, K, T, r, sigma), numerical_delta(S, K, T, r, sigma), atol=1e-6)


def test_gamma_matches_finite_difference():
    assert np.isclose(gamma(S, K, T, r, sigma), numerical_gamma(S, K, T, r, sigma), atol=1e-4)


def test_vega_matches_finite_difference():
    assert np.isclose(vega(S, K, T, r, sigma), numerical_vega(S, K, T, r, sigma), rtol=1e-4)


def test_theta_matches_finite_difference():
    assert np.isclose(theta_call(S, K, T, r, sigma), numerical_theta(S, K, T, r, sigma), rtol=1e-3)


def test_rho_matches_finite_difference():
    assert np.isclose(rho_call(S, K, T, r, sigma), numerical_rho(S, K, T, r, sigma), rtol=1e-3)


def test_gamma_is_always_positive():
    for s in np.linspace(20, 180, 20):
        assert gamma(s, K, T, r, sigma) > 0


def test_delta_call_bounded_between_zero_and_one():
    for s in np.linspace(1, 500, 50):
        d = delta_call(s, K, T, r, sigma)
        assert 0 <= d <= 1
