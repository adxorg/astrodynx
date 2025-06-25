from astrodynx.twobody import mean_anomaly_equ_elps

import jax.numpy as jnp


class TestMeanAnomalyEquElps:
    def test_scalar(self) -> None:
        e = 0.1
        E = jnp.pi / 4
        expected = E - e * jnp.sin(E)
        result = mean_anomaly_equ_elps(e, E)
        assert jnp.allclose(result, expected, rtol=1e-7)

    def test_array(self) -> None:
        e = jnp.array([0.1, 0.2])
        E = jnp.array([jnp.pi / 4, jnp.pi / 3])
        expected = E - e * jnp.sin(E)
        result = mean_anomaly_equ_elps(e, E)
        assert jnp.allclose(result, expected, rtol=1e-7)

    def test_broadcasting(self) -> None:
        e = jnp.array([0.1, 0.2])
        E = jnp.pi / 4
        expected = E - e * jnp.sin(E)
        result = mean_anomaly_equ_elps(e, E)
        assert jnp.allclose(result, expected, rtol=1e-7)

    def test_zero_eccentricity(self) -> None:
        e = 0.0
        E = jnp.linspace(0, 2 * jnp.pi, 5)
        expected = E
        result = mean_anomaly_equ_elps(e, E)
        assert jnp.allclose(result, expected, rtol=1e-7)

    def test_types(self) -> None:
        e = 0.1
        E = jnp.pi / 2
        result = mean_anomaly_equ_elps(e, E)
        assert isinstance(result, jnp.ndarray)
