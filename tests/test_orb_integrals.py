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
        expected = jnp.cross(v, h) / mu - r / jnp.linalg.vector_norm(r)
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


class TestSemiparameter:
    def test_scalar_inputs(self) -> None:
        h = 1.0
        mu = 1.0
        expected = h**2 / mu
        result = adx.semiparameter(h, mu)
        assert jnp.allclose(result, expected)

    def test_array_inputs(self) -> None:
        h = jnp.array([1.0, 2.0])
        mu = jnp.array([1.0, 2.0])
        expected = h**2 / mu
        result = adx.semiparameter(h, mu)
        assert jnp.allclose(result, expected)

    def test_broadcasting(self) -> None:
        h = jnp.array([1.0, 2.0])
        mu = 1.0
        expected = h**2 / mu
        result = adx.semiparameter(h, mu)
        assert jnp.allclose(result, expected)

    def test_zero_angular_momentum(self) -> None:
        h = 0.0
        mu = 1.0
        expected = 0.0
        result = adx.semiparameter(h, mu)
        assert jnp.allclose(result, expected)

    def test_large_values(self) -> None:
        h = 1e3
        mu = 1e-3
        expected = h**2 / mu
        result = adx.semiparameter(h, mu)
        assert jnp.allclose(result, expected)


class TestMeanMotion:
    def test_scalar_inputs(self) -> None:
        """Test with scalar input."""
        P = 1.0
        expected = 2 * jnp.pi
        result = adx.mean_motion(P)
        assert jnp.allclose(result, expected)

    def test_array_inputs(self) -> None:
        """Test with array inputs."""
        P = jnp.array([1.0, 2.0])
        expected = 2 * jnp.pi / P
        result = adx.mean_motion(P)
        assert jnp.allclose(result, expected)

    def test_broadcasting(self) -> None:
        """Test broadcasting capabilities."""
        P = jnp.array([1.0, 2.0, 4.0])
        expected = 2 * jnp.pi / P
        result = adx.mean_motion(P)
        assert jnp.allclose(result, expected)

    def test_large_period(self) -> None:
        """Test with large orbital period."""
        P = 1e6
        expected = 2 * jnp.pi / P
        result = adx.mean_motion(P)
        assert jnp.allclose(result, expected)

    def test_small_period(self) -> None:
        """Test with small orbital period."""
        P = 1e-6
        expected = 2 * jnp.pi / P
        result = adx.mean_motion(P)
        assert jnp.allclose(result, expected)


class TestEquOfOrbit:
    def test_scalar_inputs(self) -> None:
        """Test with scalar inputs."""
        p = 1.0
        e = 0.0
        f = 0.0
        expected = 1.0
        result = adx.equ_of_orbit(p, e, f)
        assert jnp.allclose(result, expected)

    def test_array_inputs(self) -> None:
        """Test with array inputs."""
        p = jnp.array([1.0, 2.0])
        e = jnp.array([0.5, 0.3])
        f = jnp.array([0.0, jnp.pi])
        expected = p / (1 + e * jnp.cos(f))
        result = adx.equ_of_orbit(p, e, f)
        assert jnp.allclose(result, expected)

    def test_broadcasting(self) -> None:
        """Test broadcasting capabilities."""
        p = jnp.array([1.0, 2.0])
        e = 0.5
        f = 0.0
        expected = p / (1 + e * jnp.cos(f))
        result = adx.equ_of_orbit(p, e, f)
        assert jnp.allclose(result, expected)

    def test_circular_orbit(self) -> None:
        """Test with circular orbit (e=0)."""
        p = 2.0
        e = 0.0
        f_values = jnp.linspace(0, 2 * jnp.pi, 10)
        expected = jnp.full_like(f_values, p)
        result = adx.equ_of_orbit(p, e, f_values)
        assert jnp.allclose(result, expected)

    def test_elliptical_orbit(self) -> None:
        """Test with elliptical orbit."""
        p = 1.0
        e = 0.5
        f = jnp.array([0.0, jnp.pi])
        expected = jnp.array([p / (1 + e), p / (1 - e)])
        result = adx.equ_of_orbit(p, e, f)
        assert jnp.allclose(result, expected)


class TestRadiusPeriapsis:
    def test_scalar_inputs(self) -> None:
        """Test with scalar inputs."""
        p = 1.0
        e = 0.5
        expected = p / (1 + e)
        result = adx.radius_periapsis(p, e)
        assert jnp.allclose(result, expected)

    def test_array_inputs(self) -> None:
        """Test with array inputs."""
        p = jnp.array([1.0, 2.0])
        e = jnp.array([0.5, 0.3])
        expected = p / (1 + e)
        result = adx.radius_periapsis(p, e)
        assert jnp.allclose(result, expected)

    def test_broadcasting(self) -> None:
        """Test broadcasting capabilities."""
        p = jnp.array([1.0, 2.0])
        e = 0.5
        expected = p / (1 + e)
        result = adx.radius_periapsis(p, e)
        assert jnp.allclose(result, expected)

    def test_circular_orbit(self) -> None:
        """Test with circular orbit (e=0)."""
        p = 2.0
        e = 0.0
        expected = p
        result = adx.radius_periapsis(p, e)
        assert jnp.allclose(result, expected)

    def test_high_eccentricity(self) -> None:
        """Test with high eccentricity."""
        p = 1.0
        e = 0.9
        expected = p / (1 + e)
        result = adx.radius_periapsis(p, e)
        assert jnp.allclose(result, expected)


class TestRadiusApoapsis:
    def test_scalar_inputs(self) -> None:
        """Test with scalar inputs."""
        p = 1.0
        e = 0.5
        expected = p / (1 - e)
        result = adx.radius_apoapsis(p, e)
        assert jnp.allclose(result, expected)

    def test_array_inputs(self) -> None:
        """Test with array inputs."""
        p = jnp.array([1.0, 2.0])
        e = jnp.array([0.5, 0.3])
        expected = p / (1 - e)
        result = adx.radius_apoapsis(p, e)
        assert jnp.allclose(result, expected)

    def test_broadcasting(self) -> None:
        """Test broadcasting capabilities."""
        p = jnp.array([1.0, 2.0])
        e = 0.5
        expected = p / (1 - e)
        result = adx.radius_apoapsis(p, e)
        assert jnp.allclose(result, expected)

    def test_circular_orbit(self) -> None:
        """Test with circular orbit (e=0)."""
        p = 2.0
        e = 0.0
        expected = p
        result = adx.radius_apoapsis(p, e)
        assert jnp.allclose(result, expected)

    def test_approaching_unity(self) -> None:
        """Test with eccentricity approaching 1."""
        p = 1.0
        e = 0.99
        expected = p / (1 - e)
        result = adx.radius_apoapsis(p, e)
        assert jnp.allclose(result, expected)
