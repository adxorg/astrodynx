import astrodynx as adx
import jax.numpy as jnp


class TestSigmaFunc:
    def test_scalar_inputs(self) -> None:
        """Test with scalar inputs."""
        r = jnp.array([1.0, 0.0, 0.0])
        v = jnp.array([0.0, 1.0, 0.0])
        mu = 1.0
        expected = 0.0
        result = adx.twobody.ivp.sigma_func(r, v, mu)
        assert jnp.allclose(result, expected)

    def test_orthogonal_vectors(self) -> None:
        """Test with orthogonal position and velocity vectors."""
        r = jnp.array([0.0, 2.0, 0.0])
        v = jnp.array([3.0, 0.0, 0.0])
        mu = 4.0
        expected = 0.0
        result = adx.twobody.ivp.sigma_func(r, v, mu)
        assert jnp.allclose(result, expected)

    def test_parallel_vectors(self) -> None:
        """Test with parallel position and velocity vectors."""
        r = jnp.array([2.0, 0.0, 0.0])
        v = jnp.array([3.0, 0.0, 0.0])
        mu = 4.0
        expected = 2.0 * 3.0 / jnp.sqrt(4.0)
        result = adx.twobody.ivp.sigma_func(r, v, mu)
        assert jnp.allclose(result, expected)

    def test_broadcasting(self) -> None:
        """Test broadcasting capabilities."""
        r = jnp.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        v = jnp.array([[1.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
        mu = jnp.array([1.0, 4.0])
        expected = jnp.array([1.0, 6.0 / 2.0])
        result = adx.twobody.ivp.sigma_func(r, v, mu)
        assert jnp.allclose(result, expected)

    def test_broadcasting_single_mu(self) -> None:
        """Test broadcasting with a single mu value."""
        r = jnp.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        v = jnp.array([[1.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
        mu = 4.0
        expected = jnp.array([1.0 / 2.0, 6.0 / 2.0])
        result = adx.twobody.ivp.sigma_func(r, v, mu)
        assert jnp.allclose(result, expected)

    def test_3d_vectors(self) -> None:
        """Test with 3D vectors."""
        r = jnp.array([1.0, 2.0, 3.0])
        v = jnp.array([4.0, 5.0, 6.0])
        mu = 9.0
        expected = (1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0) / 3.0
        result = adx.twobody.ivp.sigma_func(r, v, mu)
        assert jnp.allclose(result, expected)
