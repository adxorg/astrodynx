from astrodynx.twobody.kepler_equation import mean_anomaly_elps

from astrodynx.twobody.kepler_equation import (
    mean_anomaly_equ_elps,
    mean_anomaly_equ_hypb,
)

import jax.numpy as jnp


class TestMeanAnomalyElliptic:
    def test_scalar(self) -> None:
        # Example from docstring
        e = 0.1
        E = jnp.pi / 4
        expected = E - e * jnp.sin(E)
        result = mean_anomaly_equ_elps(e, E)
        assert jnp.allclose(result, expected)

    def test_zero_eccentricity(self) -> None:
        # If eccentricity is zero, mean anomaly equals eccentric anomaly
        e = 0.0
        E = jnp.linspace(0, 2 * jnp.pi, 5)
        result = mean_anomaly_equ_elps(e, E)
        assert jnp.allclose(result, E)

    def test_array_input(self) -> None:
        # Test with array inputs
        e = jnp.array([0.0, 0.1, 0.2])
        E = jnp.array([0.0, jnp.pi / 2, jnp.pi])
        expected = E - e * jnp.sin(E)
        result = mean_anomaly_equ_elps(e, E)
        assert jnp.allclose(result, expected)

    def test_negative_eccentricity(self) -> None:
        # Negative eccentricity is non-physical but should still compute
        e = -0.1
        E = jnp.pi / 3
        expected = E - e * jnp.sin(E)
        result = mean_anomaly_equ_elps(e, E)
        assert jnp.allclose(result, expected)

    def test_types(self) -> None:
        # Test with python floats and numpy floats
        e = 0.25
        E = 1.2
        result = mean_anomaly_equ_elps(e, E)
        expected = E - e * jnp.sin(E)
        assert jnp.allclose(result, expected)


class TestMeanAnomalyHyperbolic:
    def test_scalar(self) -> None:
        # Example from docstring
        e = 1.1
        H = jnp.pi / 4
        expected = e * jnp.sinh(H) - H
        result = mean_anomaly_equ_hypb(e, H)
        assert jnp.allclose(result, expected)

    def test_array_input(self) -> None:
        # Test with array inputs
        e = jnp.array([1.1, 2.0, 3.0])
        H = jnp.array([0.0, 1.0, 2.0])
        expected = e * jnp.sinh(H) - H
        result = mean_anomaly_equ_hypb(e, H)
        assert jnp.allclose(result, expected)

    def test_zero_hyperbolic_anomaly(self) -> None:
        # If H is zero, mean anomaly is -H = 0
        e = 2.0
        H = 0.0
        expected = e * jnp.sinh(H) - H
        result = mean_anomaly_equ_hypb(e, H)
        assert jnp.allclose(result, expected)

    def test_negative_hyperbolic_anomaly(self) -> None:
        # Should handle negative H
        e = 1.5
        H = -1.0
        expected = e * jnp.sinh(H) - H
        result = mean_anomaly_equ_hypb(e, H)
        assert jnp.allclose(result, expected)

    def test_types(self) -> None:
        # Test with python floats and numpy floats
        e = 2.5
        H = 1.2
        expected = e * jnp.sinh(H) - H
        result = mean_anomaly_equ_hypb(e, H)
        assert jnp.allclose(result, expected)


class TestMeanAnomalyElps:
    def test_scalar(self) -> None:
        # Example from docstring
        t = 1.0
        a = 1.0
        mu = 1.0
        expected = 1.0
        result = mean_anomaly_elps(t, a, mu)
        assert jnp.allclose(result, expected)

    def test_zero_time(self) -> None:
        # If t is zero, mean anomaly should be zero
        t = 0.0
        a = 2.0
        mu = 3.0
        expected = 0.0
        result = mean_anomaly_elps(t, a, mu)
        assert jnp.allclose(result, expected)

    def test_array_input(self) -> None:
        # Test with array inputs for t
        t = jnp.array([0.0, 1.0, 2.0])
        a = 2.0
        mu = 8.0
        expected = jnp.sqrt(mu / a**3) * t
        result = mean_anomaly_elps(t, a, mu)
        assert jnp.allclose(result, expected)

    def test_array_a_mu(self) -> None:
        # Test with array inputs for a and mu
        t = 1.0
        a = jnp.array([1.0, 2.0, 4.0])
        mu = jnp.array([1.0, 8.0, 64.0])
        expected = jnp.sqrt(mu / a**3) * t
        result = mean_anomaly_elps(t, a, mu)
        assert jnp.allclose(result, expected)

    def test_types(self) -> None:
        # Test with python floats and numpy floats
        t = 2.0
        a = 3.0
        mu = 27.0
        expected = jnp.sqrt(mu / a**3) * t
        result = mean_anomaly_elps(t, a, mu)
        assert jnp.allclose(result, expected)
