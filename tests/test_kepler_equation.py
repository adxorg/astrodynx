import astrodynx as adx
import jax.numpy as jnp


class TestMeanAnomalyElps:
    def test_scalar_inputs(self) -> None:
        """Test with scalar inputs."""
        a = 1.0
        mu = 1.0
        deltat = 1.0
        expected = jnp.sqrt(mu / a**3) * deltat
        result = adx.mean_anomaly_elps(a, mu, deltat)
        assert jnp.allclose(result, expected)

    def test_array_inputs(self) -> None:
        """Test with array inputs."""
        a = jnp.array([1.0, 2.0])
        mu = jnp.array([1.0, 2.0])
        deltat = jnp.array([1.0, 1.0])
        expected = jnp.sqrt(mu / a**3) * deltat
        result = adx.mean_anomaly_elps(a, mu, deltat)
        assert jnp.allclose(result, expected)

    def test_broadcasting(self) -> None:
        """Test broadcasting capabilities."""
        a = jnp.array([1.0, 2.0])
        mu = 1.0
        deltat = 1.0
        expected = jnp.sqrt(mu / a**3) * deltat
        result = adx.mean_anomaly_elps(a, mu, deltat)
        assert jnp.allclose(result, expected)

    def test_types(self) -> None:
        """Test that the function returns the correct type."""
        a = 1.0
        mu = 1.0
        deltat = 1.0
        result = adx.mean_anomaly_elps(a, mu, deltat)
        assert isinstance(result, jnp.ndarray)


class TestMeanAnomalyHypb:
    def test_scalar_inputs(self) -> None:
        """Test with scalar inputs."""
        a = -1.0
        mu = 1.0
        deltat = 1.0
        expected = jnp.sqrt(mu / -(a**3)) * deltat
        result = adx.mean_anomaly_hypb(a, mu, deltat)
        assert jnp.allclose(result, expected)

    def test_array_inputs(self) -> None:
        """Test with array inputs."""
        a = jnp.array([-1.0, -2.0])
        mu = jnp.array([1.0, 2.0])
        deltat = jnp.array([1.0, 1.0])
        expected = jnp.sqrt(mu / -(a**3)) * deltat
        result = adx.mean_anomaly_hypb(a, mu, deltat)
        assert jnp.allclose(result, expected)

    def test_broadcasting(self) -> None:
        """Test broadcasting capabilities."""
        a = jnp.array([-1.0, -2.0])
        mu = 1.0
        deltat = 1.0
        expected = jnp.sqrt(mu / -(a**3)) * deltat
        result = adx.mean_anomaly_hypb(a, mu, deltat)
        assert jnp.allclose(result, expected)

    def test_types(self) -> None:
        """Test that the function returns the correct type."""
        a = -1.0
        mu = 1.0
        deltat = 1.0
        result = adx.mean_anomaly_hypb(a, mu, deltat)
        assert isinstance(result, jnp.ndarray)


class TestKeplerequElps:
    def test_scalar_inputs(self) -> None:
        """Test with scalar inputs."""
        E = 1.0
        e = 0.5
        a = 2.0
        mu = 1.0
        expected = (E - e * jnp.sin(E)) * jnp.sqrt(a**3 / mu)
        result = adx.keplerequ_elps(E, e, a, mu)
        assert jnp.allclose(result, expected)

    def test_array_inputs(self) -> None:
        """Test with array inputs."""
        E = jnp.array([1.0, 2.0])
        e = jnp.array([0.5, 0.3])
        a = jnp.array([2.0, 3.0])
        mu = jnp.array([1.0, 2.0])
        expected = (E - e * jnp.sin(E)) * jnp.sqrt(a**3 / mu)
        result = adx.keplerequ_elps(E, e, a, mu)
        assert jnp.allclose(result, expected)

    def test_broadcasting(self) -> None:
        """Test broadcasting capabilities."""
        E = jnp.array([1.0, 2.0])
        e = 0.5
        a = 2.0
        mu = 1.0
        expected = (E - e * jnp.sin(E)) * jnp.sqrt(a**3 / mu)
        result = adx.keplerequ_elps(E, e, a, mu)
        assert jnp.allclose(result, expected)

    def test_types(self) -> None:
        """Test that the function returns the correct type."""
        E = 1.0
        e = 0.5
        a = 2.0
        mu = 1.0
        result = adx.keplerequ_elps(E, e, a, mu)
        assert isinstance(result, jnp.ndarray)


class TestKeplerequHypb:
    def test_scalar_inputs(self) -> None:
        """Test with scalar inputs."""
        H = 1.0
        e = 1.5
        a = -2.0
        mu = 1.0
        expected = (e * jnp.sinh(H) - H) * jnp.sqrt(-(a**3) / mu)
        result = adx.keplerequ_hypb(H, e, a, mu)
        assert jnp.allclose(result, expected)

    def test_array_inputs(self) -> None:
        """Test with array inputs."""
        H = jnp.array([1.0, 2.0])
        e = jnp.array([1.5, 1.3])
        a = jnp.array([-2.0, -3.0])
        mu = jnp.array([1.0, 2.0])
        expected = (e * jnp.sinh(H) - H) * jnp.sqrt(-(a**3) / mu)
        result = adx.keplerequ_hypb(H, e, a, mu)
        assert jnp.allclose(result, expected)

    def test_broadcasting(self) -> None:
        """Test broadcasting capabilities."""
        H = jnp.array([1.0, 2.0])
        e = 1.5
        a = -2.0
        mu = 1.0
        expected = (e * jnp.sinh(H) - H) * jnp.sqrt(-(a**3) / mu)
        result = adx.keplerequ_hypb(H, e, a, mu)
        assert jnp.allclose(result, expected)

    def test_types(self) -> None:
        """Test that the function returns the correct type."""
        H = 1.0
        e = 1.5
        a = -2.0
        mu = 1.0
        result = adx.keplerequ_hypb(H, e, a, mu)
        assert isinstance(result, jnp.ndarray)
