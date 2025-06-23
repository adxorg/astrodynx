from astrodynx.twobody.universal_function import (
    _U0,
    _U1,
    _U2,
    _U3,
)

import jax.numpy as jnp


class TestU0:
    def test_U0_alpha_zero(self) -> None:
        chi = 2.5
        alpha = 0.0
        result = _U0(chi, alpha)
        assert jnp.allclose(result, 1.0)

    def test_U0_alpha_positive(self) -> None:
        chi = 1.0
        alpha = 0.5
        expected = jnp.cos(jnp.sqrt(alpha) * chi)
        result = _U0(chi, alpha)
        assert jnp.allclose(result, expected)

    def test_U0_alpha_negative(self) -> None:
        chi = 1.0
        alpha = -0.5
        expected = jnp.cosh(jnp.sqrt(-alpha) * chi)
        result = _U0(chi, alpha)
        assert jnp.allclose(result, expected)


class TestU1:
    def test_U1_alpha_zero(self) -> None:
        chi = 3.7
        alpha = 0.0
        result = _U1(chi, alpha)
        assert jnp.allclose(result, chi)

    def test_U1_alpha_positive(self) -> None:
        chi = 2.0
        alpha = 1.5
        expected = jnp.sin(jnp.sqrt(alpha) * chi) / jnp.sqrt(alpha)
        result = _U1(chi, alpha)
        assert jnp.allclose(result, expected)

    def test_U1_alpha_negative(self) -> None:
        chi = 2.0
        alpha = -1.5
        expected = jnp.sinh(jnp.sqrt(-alpha) * chi) / jnp.sqrt(-alpha)
        result = _U1(chi, alpha)
        assert jnp.allclose(result, expected)


class TestU2:
    def test_U2_alpha_zero(self) -> None:
        chi = 4.2
        alpha = 0.0
        expected = chi**2 / 2
        result = _U2(chi, alpha)
        assert jnp.allclose(result, expected)

    def test_U2_alpha_positive(self) -> None:
        chi = 1.5
        alpha = 2.0
        expected = (1 - jnp.cos(jnp.sqrt(alpha) * chi)) / alpha
        result = _U2(chi, alpha)
        assert jnp.allclose(result, expected)

    def test_U2_alpha_negative(self) -> None:
        chi = 1.5
        alpha = -2.0
        expected = (jnp.cosh(jnp.sqrt(-alpha) * chi) - 1) / -alpha
        result = _U2(chi, alpha)
        assert jnp.allclose(result, expected)


class TestU3:
    def test_U3_alpha_zero(self) -> None:
        chi = 2.0
        alpha = 0.0
        expected = chi**3 / 6
        result = _U3(chi, alpha)
        assert jnp.allclose(result, expected)

    def test_U3_alpha_positive(self) -> None:
        chi = 1.5
        alpha = 2.0
        sqrt_alpha = jnp.sqrt(alpha)
        expected = (sqrt_alpha * chi - jnp.sin(sqrt_alpha * chi)) / (alpha * sqrt_alpha)
        result = _U3(chi, alpha)
        assert jnp.allclose(result, expected)

    def test_U3_alpha_negative(self) -> None:
        chi = 1.5
        alpha = -2.0
        sqrt_neg_alpha = jnp.sqrt(-alpha)
        expected = (jnp.sinh(sqrt_neg_alpha * chi) - sqrt_neg_alpha * chi) / (
            -alpha * sqrt_neg_alpha
        )
        result = _U3(chi, alpha)
        assert jnp.allclose(result, expected)
