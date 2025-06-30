import astrodynx as adx
import jax.numpy as jnp


class TestSigmaFunc:
    def test_scalar_inputs(self) -> None:
        """Test with scalar inputs."""
        r = jnp.array([1.0, 0.0, 0.0])
        v = jnp.array([0.0, 1.0, 0.0])
        mu = 1.0
        expected = jnp.array([0.0])
        result = adx.twobody.ivp.sigma_func(r, v, mu)
        assert jnp.allclose(result, expected)

    def test_nonzero_result(self) -> None:
        """Test with inputs that produce non-zero result."""
        r = jnp.array([1.0, 1.0, 0.0])
        v = jnp.array([1.0, 0.0, 0.0])
        mu = 1.0
        expected = jnp.array([1.0])
        result = adx.twobody.ivp.sigma_func(r, v, mu)
        assert jnp.allclose(result, expected)

    def test_array_inputs(self) -> None:
        """Test with array inputs."""
        r = jnp.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        v = jnp.array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
        mu = jnp.array([[1.0], [2.0]])
        expected = jnp.array([[0.0], [0.0]])
        result = adx.twobody.ivp.sigma_func(r, v, mu)
        assert jnp.allclose(result, expected)

    def test_broadcasting(self) -> None:
        """Test broadcasting capabilities."""
        r = jnp.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        v = jnp.array([1.0, 1.0, 0.0])
        mu = 4.0
        expected = jnp.array([[0.5], [1.0]])
        result = adx.twobody.ivp.sigma_func(r, v, mu)
        assert jnp.allclose(result, expected)

    def test_different_mu_values(self) -> None:
        """Test with different gravitational parameter values."""
        r = jnp.array([3.0, 0.0, 0.0])
        v = jnp.array([0.0, 2.0, 0.0])
        mu_values = jnp.array([1.0, 4.0, 9.0])
        expected = jnp.array([[0.0], [0.0], [0.0]])
        results = jnp.stack([adx.twobody.ivp.sigma_func(r, v, mu) for mu in mu_values])
        assert jnp.allclose(results, expected)
