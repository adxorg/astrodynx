import astrodynx as adx
import jax
import jax.numpy as jnp


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


class TestMeanAnomalyElps:
    def test_scalar_inputs(self) -> None:
        """Test with scalar inputs."""
        a = 1.0
        mu = 1.0
        deltat = 1.0
        expected = jnp.sqrt(mu / a**3) * deltat
        result = adx.mean_anomaly_elps(a, deltat, mu)
        assert jnp.allclose(result, expected)

    def test_array_inputs(self) -> None:
        """Test with array inputs."""
        a = jnp.array([1.0, 2.0])
        mu = jnp.array([1.0, 2.0])
        deltat = jnp.array([1.0, 1.0])
        expected = jnp.sqrt(mu / a**3) * deltat
        result = adx.mean_anomaly_elps(a, deltat, mu)
        assert jnp.allclose(result, expected)

    def test_broadcasting(self) -> None:
        """Test broadcasting capabilities."""
        a = jnp.array([1.0, 2.0])
        mu = 1.0
        deltat = 1.0
        expected = jnp.sqrt(mu / a**3) * deltat
        result = adx.mean_anomaly_elps(a, deltat, mu)
        assert jnp.allclose(result, expected)


class TestMeanAnomalyHypb:
    def test_scalar_inputs(self) -> None:
        """Test with scalar inputs."""
        a = -1.0
        mu = 1.0
        deltat = 1.0
        expected = jnp.sqrt(mu / -(a**3)) * deltat
        result = adx.mean_anomaly_hypb(a, deltat, mu)
        assert jnp.allclose(result, expected)

    def test_array_inputs(self) -> None:
        """Test with array inputs."""
        a = jnp.array([-1.0, -2.0])
        mu = jnp.array([1.0, 2.0])
        deltat = jnp.array([1.0, 1.0])
        expected = jnp.sqrt(mu / -(a**3)) * deltat
        result = adx.mean_anomaly_hypb(a, deltat, mu)
        assert jnp.allclose(result, expected)

    def test_broadcasting(self) -> None:
        """Test broadcasting capabilities."""
        a = jnp.array([-1.0, -2.0])
        mu = 1.0
        deltat = 1.0
        expected = jnp.sqrt(mu / -(a**3)) * deltat
        result = adx.mean_anomaly_hypb(a, deltat, mu)
        assert jnp.allclose(result, expected)


class TestKeplerEquUni:
    def test_scalar_inputs(self) -> None:
        """Test with scalar inputs."""
        chi = 1.0
        alpha = 1.0
        r0 = 1.0
        sigma0 = 0.0
        deltat = 1.0
        mu = 1.0
        expected = (
            r0 * adx.twobody.ufunc1(chi, alpha)
            + sigma0 * adx.twobody.ufunc2(chi, alpha)
            + adx.twobody.ufunc3(chi, alpha)
            - jnp.sqrt(mu) * deltat
        )
        result = adx.kepler_equ_uni(chi, alpha, r0, sigma0, deltat, mu)
        assert jnp.allclose(result, expected)

    def test_array_inputs(self) -> None:
        """Test with array inputs."""
        chi = jnp.array([1.0, 2.0])
        alpha = 1.0
        r0 = jnp.array([1.0, 1.0])
        sigma0 = jnp.array([0.0, 0.0])
        deltat = jnp.array([1.0, 1.0])
        mu = jnp.array([1.0, 1.0])
        expected = (
            r0 * adx.twobody.ufunc1(chi, alpha)
            + sigma0 * adx.twobody.ufunc2(chi, alpha)
            + adx.twobody.ufunc3(chi, alpha)
            - jnp.sqrt(mu) * deltat
        )
        result = adx.kepler_equ_uni(chi, alpha, r0, sigma0, deltat, mu)
        assert jnp.allclose(result, expected)

    def test_gradient_wrt_chi(self) -> None:
        """Test gradient with respect to chi."""
        chi = jnp.pi * 0.6
        alpha = jnp.array([1.0, 0, -1.0])
        r0, sigma0, deltat, mu = 1.0, 0.0, 1.0, 1.0
        U0 = jax.vmap(adx.twobody.ufunc0, in_axes=(None, 0))(chi, alpha)
        U1 = jax.vmap(adx.twobody.ufunc1, in_axes=(None, 0))(chi, alpha)
        U2 = jax.vmap(adx.twobody.ufunc2, in_axes=(None, 0))(chi, alpha)
        expected = adx.equ_of_orb_uvi(U0, U1, U2, r0, sigma0) / jnp.sqrt(
            mu
        )  # dt/d chi = r / sqrt(mu), Battin 1999, pp.174.

        result = jax.vmap(
            jax.grad(adx.kepler_equ_uni, argnums=0),
            in_axes=(None, 0, None, None, None, None),
        )(chi, alpha, r0, sigma0, deltat, mu)
        assert jnp.allclose(result, expected)


class TestGeneralizedAnomaly:
    def test_scalar_inputs(self) -> None:
        """Test with scalar inputs."""
        alpha = 1.0
        sigma = 1.0
        sigma0 = 0.0
        deltat = 1.0
        mu = 1.0
        expected = alpha * jnp.sqrt(mu) * deltat + sigma - sigma0
        result = adx.generalized_anomaly(alpha, sigma, sigma0, deltat, mu)
        assert jnp.allclose(result, expected)

    def test_array_inputs(self) -> None:
        """Test with array inputs."""
        alpha = jnp.array([1.0, 1.0])
        sigma = jnp.array([1.0, 2.0])
        sigma0 = jnp.array([0.0, 0.0])
        deltat = jnp.array([1.0, 1.0])
        mu = jnp.array([1.0, 1.0])
        expected = alpha * jnp.sqrt(mu) * deltat + sigma - sigma0
        result = adx.generalized_anomaly(alpha, sigma, sigma0, deltat, mu)
        assert jnp.allclose(result, expected)

    def test_broadcasting(self) -> None:
        """Test broadcasting capabilities."""
        alpha = 1.0
        sigma = jnp.array([1.0, 2.0])
        sigma0 = 0.0
        deltat = 1.0
        mu = 1.0
        expected = alpha * jnp.sqrt(mu) * deltat + sigma - sigma0
        result = adx.generalized_anomaly(alpha, sigma, sigma0, deltat, mu)
        assert jnp.allclose(result, expected)


class TestSolveKeplerUni:
    def test_basic_functionality(self) -> None:
        """Test basic functionality with simple inputs."""
        # Setup simple inputs
        alpha = 1.0
        r0 = 1.0
        sigma0 = 0.0
        deltat = jnp.pi / 2
        mu = 1.0

        chi = adx.solve_kepler_uni(deltat, alpha, r0, sigma0, mu)

        # Verify the solution satisfies the universal Kepler equation
        residual = adx.kepler_equ_uni(chi, alpha, r0, sigma0, deltat, mu)
        assert jnp.abs(residual) < 1e-6

    def test_elliptical_orbit(self) -> None:
        """Test with parameters for an elliptical orbit."""
        # Setup elliptical orbit
        r0_vec = jnp.array([1.0, 0.0, 0.0])
        v0_vec = jnp.array([0.0, 0.8, 0.0])  # v < sqrt(mu/r) for elliptical
        mu = 1.0
        deltat = jnp.pi / 2

        r0 = jnp.linalg.norm(r0_vec)
        v0 = jnp.linalg.norm(v0_vec)
        alpha = 1.0 / adx.semimajor_axis(r0, v0, mu)
        sigma0 = adx.twobody.sigma_fn(r0_vec, v0_vec, mu)

        chi = adx.solve_kepler_uni(deltat, alpha.item(), r0.item(), sigma0.item(), mu)

        # Verify the solution satisfies the universal Kepler equation
        residual = adx.kepler_equ_uni(chi, alpha, r0, sigma0, deltat, mu)
        assert jnp.abs(residual) < 1e-6

    def test_hyperbolic_orbit(self) -> None:
        """Test with parameters for a hyperbolic orbit."""
        # Setup hyperbolic orbit
        r0_vec = jnp.array([1.0, 0.0, 0.0])
        v0_vec = jnp.array([0.0, 1.5, 0.0])  # v > sqrt(mu/r) for hyperbolic
        mu = 1.0
        deltat = 1.0

        r0 = jnp.linalg.norm(r0_vec)
        v0 = jnp.linalg.norm(v0_vec)
        alpha = 1.0 / adx.semimajor_axis(r0, v0, mu)
        sigma0 = adx.twobody.sigma_fn(r0_vec, v0_vec, mu)

        chi = adx.solve_kepler_uni(deltat, alpha.item(), r0.item(), sigma0.item(), mu)

        # Verify the solution satisfies the universal Kepler equation
        residual = adx.kepler_equ_uni(chi, alpha, r0, sigma0, deltat, mu)
        assert jnp.abs(residual) < 1e-6

    def test_parabolic_orbit(self) -> None:
        """Test with parameters for a parabolic orbit."""
        # Setup parabolic orbit (alpha ≈ 0)
        r0_vec = jnp.array([1.0, 0.0, 0.0])
        v0_vec = jnp.array([0.0, 1.0, 0.0])  # v ≈ sqrt(mu/r) for parabolic
        mu = 1.0
        deltat = 1.0

        r0 = jnp.linalg.norm(r0_vec)
        v0 = jnp.linalg.norm(v0_vec)
        alpha = 1.0 / adx.semimajor_axis(r0, v0, mu)
        sigma0 = adx.twobody.sigma_fn(r0_vec, v0_vec, mu)

        chi = adx.solve_kepler_uni(deltat, alpha.item(), r0.item(), sigma0.item(), mu)

        # Verify the solution satisfies the universal Kepler equation
        residual = adx.kepler_equ_uni(chi, alpha, r0, sigma0, deltat, mu)
        assert jnp.abs(residual) < 1e-6
