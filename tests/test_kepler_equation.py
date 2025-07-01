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


class TestKeplerEquElps:
    def test_scalar_inputs(self) -> None:
        """Test with scalar inputs."""
        E = jnp.pi / 4
        e = 0.1
        M = 0.7
        expected = E - e * jnp.sin(E) - M
        result = adx.kepler_equ_elps(E, e, M)
        assert jnp.allclose(result, expected)

    def test_array_inputs(self) -> None:
        """Test with array inputs."""
        E = jnp.array([jnp.pi / 4, jnp.pi / 2])
        e = jnp.array([0.1, 0.2])
        M = jnp.array([0.7, 0.8])
        expected = E - e * jnp.sin(E) - M
        result = adx.kepler_equ_elps(E, e, M)
        assert jnp.allclose(result, expected)

    def test_broadcasting(self) -> None:
        """Test broadcasting capabilities."""
        E = jnp.array([jnp.pi / 4, jnp.pi / 2])
        e = 0.1
        M = 0.7
        expected = E - e * jnp.sin(E) - M
        result = adx.kepler_equ_elps(E, e, M)
        assert jnp.allclose(result, expected)

    def test_circular_orbit(self) -> None:
        """Test with circular orbit (e=0)."""
        E = jnp.linspace(0, 2 * jnp.pi, 10)
        e = 0.0
        M = E  # For circular orbits, E = M
        expected = jnp.zeros_like(E)
        result = adx.kepler_equ_elps(E, e, M)
        assert jnp.allclose(result, expected)


class TestKeplerEquHypb:
    def test_scalar_inputs(self) -> None:
        """Test with scalar inputs."""
        H = 1.0
        e = 1.5
        N = 1.0
        expected = e * jnp.sinh(H) - H - N
        result = adx.kepler_equ_hypb(H, e, N)
        assert jnp.allclose(result, expected)

    def test_array_inputs(self) -> None:
        """Test with array inputs."""
        H = jnp.array([1.0, 2.0])
        e = jnp.array([1.5, 1.5])
        N = jnp.array([1.0, 1.0])
        expected = e * jnp.sinh(H) - H - N
        result = adx.kepler_equ_hypb(H, e, N)
        assert jnp.allclose(result, expected)

    def test_broadcasting(self) -> None:
        """Test broadcasting capabilities."""
        H = jnp.array([1.0, 2.0])
        e = 1.5
        N = 1.0
        expected = e * jnp.sinh(H) - H - N
        result = adx.kepler_equ_hypb(H, e, N)
        assert jnp.allclose(result, expected)

    def test_different_eccentricities(self) -> None:
        """Test with different eccentricity values."""
        H = 1.0
        e_values = jnp.array([1.1, 1.5, 2.0])
        N = 1.0
        expected = e_values * jnp.sinh(H) - H - N
        results = jnp.stack([adx.kepler_equ_hypb(H, e, N) for e in e_values])
        assert jnp.allclose(results, expected)


class TestKeplerequUni:
    def test_scalar_inputs(self) -> None:
        """Test with scalar inputs."""
        chi = 1.0
        alpha = 1.0
        sigma0 = 0.0
        r0 = 1.0
        deltat = 1.0
        mu = 1.0
        expected = (
            r0 * adx.twobody.ivp.U1(chi, alpha)
            + sigma0 * adx.twobody.ivp.U2(chi, alpha)
            + adx.twobody.ivp.U3(chi, alpha)
            - jnp.sqrt(mu) * deltat
        )
        result = adx.kepler_equ_uni(chi, alpha, sigma0, r0, mu, deltat)
        assert jnp.allclose(result, expected)

    def test_array_inputs(self) -> None:
        """Test with array inputs."""
        chi = jnp.array([1.0, 2.0])
        alpha = jnp.array([1.0, 1.0])
        sigma0 = jnp.array([0.0, 0.0])
        r0 = jnp.array([1.0, 1.0])
        deltat = jnp.array([1.0, 1.0])
        mu = jnp.array([1.0, 1.0])
        expected = (
            r0 * adx.twobody.ivp.U1(chi, alpha)
            + sigma0 * adx.twobody.ivp.U2(chi, alpha)
            + adx.twobody.ivp.U3(chi, alpha)
            - jnp.sqrt(mu) * deltat
        )
        result = adx.kepler_equ_uni(chi, alpha, sigma0, r0, mu, deltat)
        assert jnp.allclose(result, expected)

    def test_broadcasting(self) -> None:
        """Test broadcasting capabilities."""
        chi = jnp.array([1.0, 2.0])
        alpha = 1.0
        sigma0 = 0.0
        r0 = 1.0
        deltat = 1.0
        mu = 1.0
        expected = (
            r0 * adx.twobody.ivp.U1(chi, alpha)
            + sigma0 * adx.twobody.ivp.U2(chi, alpha)
            + adx.twobody.ivp.U3(chi, alpha)
            - jnp.sqrt(mu) * deltat
        )
        result = adx.kepler_equ_uni(chi, alpha, sigma0, r0, mu, deltat)
        assert jnp.allclose(result, expected)

    def test_default_parameters(self) -> None:
        """Test with default parameter values."""
        chi = 1.0
        alpha = 1.0
        sigma0 = 0.0
        r0 = 1.0
        # Using default deltat=0 and mu=1
        expected = (
            r0 * adx.twobody.ivp.U1(chi, alpha)
            + sigma0 * adx.twobody.ivp.U2(chi, alpha)
            + adx.twobody.ivp.U3(chi, alpha)
        )
        result = adx.kepler_equ_uni(chi, alpha, sigma0, r0)
        assert jnp.allclose(result, expected)

    def test_known_values(self) -> None:
        """Test with known values from the docstring example."""
        chi = 1.0
        alpha = 1.0
        sigma0 = 0.0
        r0 = 1.0
        deltat = 1.0
        mu = 1.0
        # The docstring example shows this should be 0
        expected = 0.0
        result = adx.kepler_equ_uni(chi, alpha, sigma0, r0, mu, deltat)
        assert jnp.allclose(result, expected, atol=1e-7)
