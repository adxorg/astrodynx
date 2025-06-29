import astrodynx as adx
import jax.numpy as jnp


class TestOrbPeriod:
    def test_scalar_inputs(self) -> None:
        a = 1.0
        mu = 1.0
        expected = 2 * jnp.pi
        result = adx.orb_period(a, mu)
        assert jnp.allclose(result, expected)

    def test_array_inputs(self) -> None:
        a = jnp.array([1.0, 4.0])
        mu = jnp.array([1.0, 1.0])
        expected = 2 * jnp.pi * jnp.sqrt(a**3 / mu)
        result = adx.orb_period(a, mu)
        assert jnp.allclose(result, expected)

    def test_broadcasting(self) -> None:
        a = jnp.array([1.0, 8.0])
        mu = 2.0
        expected = 2 * jnp.pi * jnp.sqrt(a**3 / mu)
        result = adx.orb_period(a, mu)
        assert jnp.allclose(result, expected)

    def test_zero_semimajor_axis(self) -> None:
        a = 0.0
        mu = 1.0
        result = adx.orb_period(a, mu)
        assert result == 0.0

    def test_negative_semimajor_axis(self):
        a = -1.0
        mu = 1.0
        result = adx.orb_period(a, mu)
        assert jnp.isnan(result)


class TestAngularMomentum:
    def test_basic_case(self) -> None:
        r = jnp.array([1.0, 0.0, 0.0])
        v = jnp.array([0.0, 1.0, 0.0])
        expected = jnp.array([0.0, 0.0, 1.0])
        result = adx.angular_momentum(r, v)
        assert jnp.allclose(result, expected)

    def test_negative_direction(self) -> None:
        r = jnp.array([0.0, 1.0, 0.0])
        v = jnp.array([1.0, 0.0, 0.0])
        expected = jnp.array([0.0, 0.0, -1.0])
        result = adx.angular_momentum(r, v)
        assert jnp.allclose(result, expected)

    def test_zero_vector(self) -> None:
        r = jnp.array([0.0, 0.0, 0.0])
        v = jnp.array([1.0, 2.0, 3.0])
        expected = jnp.array([0.0, 0.0, 0.0])
        result = adx.angular_momentum(r, v)
        assert jnp.allclose(result, expected)

    def test_broadcasting(self) -> None:
        r = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        v = jnp.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        expected = jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
        result = adx.angular_momentum(r, v)
        assert jnp.allclose(result, expected)

    def test_broadcasting_single_vector(self) -> None:
        r = jnp.array([1.0, 0.0, 0.0])
        v = jnp.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        expected = jnp.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
        result = adx.angular_momentum(r, v)
        assert jnp.allclose(result, expected)


class TestSemimajorAxis:
    def test_scalar_inputs(self) -> None:
        r = 1.0
        v = 1.0
        mu = 1.0
        expected = 1 / (2 / r - v**2 / mu)
        result = adx.semimajor_axis(r, v, mu)
        assert jnp.allclose(result, expected)

    def test_array_inputs(self) -> None:
        r = jnp.array([1.0, 2.0])
        v = jnp.array([1.0, 2.0])
        mu = jnp.array([1.0, 2.0])
        expected = 1 / (2 / r - v**2 / mu)
        result = adx.semimajor_axis(r, v, mu)
        assert jnp.allclose(result, expected)

    def test_broadcasting(self) -> None:
        r = jnp.array([1.0, 2.0])
        v = 1.0
        mu = 1.0
        expected = 1 / (2 / r - v**2 / mu)
        result = adx.semimajor_axis(r, v, mu)
        assert jnp.allclose(result, expected)

    def test_zero_velocity(self) -> None:
        r = 2.0
        v = 0.0
        mu = 1.0
        expected = 1 / (2 / r - v**2 / mu)
        result = adx.semimajor_axis(r, v, mu)
        assert jnp.allclose(result, expected)

    def test_negative_result(self) -> None:
        r = 1.0
        v = 2.0
        mu = 1.0
        expected = 1 / (2 / r - v**2 / mu)
        result = adx.semimajor_axis(r, v, mu)
        assert jnp.allclose(result, expected)


class TestEccentricityVector:
    def test_circular_orbit(self) -> None:
        r = jnp.array([1.0, 0.0, 0.0])
        v = jnp.array([0.0, 1.0, 0.0])
        mu = 1.0
        expected = jnp.array([0.0, 0.0, 0.0])
        result = adx.eccentricity_vector(r, v, mu)
        assert jnp.allclose(result, expected)

    def test_elliptical_orbit(self) -> None:
        r = jnp.array([1.0, 1.0, 0.0])
        v = jnp.array([0.0, 1.0, 0.0])
        mu = 2.0
        h = adx.angular_momentum(r, v)
        expected = jnp.cross(v, h) / mu - r / jnp.linalg.norm(r)
        result = adx.eccentricity_vector(r, v, mu)
        assert jnp.allclose(result, expected)

    def test_broadcasting(self) -> None:
        r = jnp.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        v = jnp.array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
        mu = jnp.array([[1.0], [2.0]])
        expected = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        result = adx.eccentricity_vector(r, v, mu)
        assert jnp.allclose(result, expected)

    def test_zero_velocity(self) -> None:
        r = jnp.array([1.0, 0.0, 0.0])
        v = jnp.array([0.0, 0.0, 0.0])
        mu = 1.0
        expected = jnp.array([-1.0, 0.0, 0.0])
        result = adx.eccentricity_vector(r, v, mu)
        assert jnp.allclose(result, expected)
