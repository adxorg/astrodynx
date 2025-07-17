import astrodynx as adx
import jax
import jax.numpy as jnp


class TestPointMassGrav:
    """Tests for the point_mass_grav function."""

    def test_scalar_inputs(self) -> None:
        """Test with scalar inputs."""
        t = 0.0
        x = jnp.array([1.0, -1.0, 1.0, 0.0, 0.0, 0.0])
        args = {"mu": 1.0}

        # Calculate expected result
        r = jnp.sqrt(3.0)  # sqrt(1^2 + (-1)^2 + 1^2)
        mu_over_r3 = 1.0 / (r**3)
        expected = jnp.array([-mu_over_r3, mu_over_r3, -mu_over_r3])

        result = adx.gravity.point_mass_grav(t, x, args)
        assert jnp.allclose(result, expected)

    def test_different_mu_values(self) -> None:
        """Test with different gravitational parameter values."""
        t = 0.0
        x = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Test with Earth's gravitational parameter
        mu_earth = 3.986004418e14  # m^3/s^2
        args_earth = {"mu": mu_earth}
        result_earth = adx.gravity.point_mass_grav(t, x, args_earth)
        expected_earth = jnp.array([-mu_earth, 0.0, 0.0])
        assert jnp.allclose(result_earth, expected_earth)

        # Test with Sun's gravitational parameter
        mu_sun = 1.32712440018e20  # m^3/s^2
        args_sun = {"mu": mu_sun}
        result_sun = adx.gravity.point_mass_grav(t, x, args_sun)
        expected_sun = jnp.array([-mu_sun, 0.0, 0.0])
        assert jnp.allclose(result_sun, expected_sun)

    def test_different_positions(self) -> None:
        """Test with different position vectors."""
        t = 0.0
        mu = 1.0
        args = {"mu": mu}

        # Test along x-axis
        x1 = jnp.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result1 = adx.gravity.point_mass_grav(t, x1, args)
        # r = 2.0, so r^3 = 8.0, mu/r^3 = 1/8 = 0.125
        expected1 = jnp.array([-mu * 2.0 / 8.0, 0.0, 0.0])  # -mu/r^3 * [x, y, z]
        assert jnp.allclose(result1, expected1)

        # Test along y-axis
        x2 = jnp.array([0.0, 3.0, 0.0, 0.0, 0.0, 0.0])
        result2 = adx.gravity.point_mass_grav(t, x2, args)
        # r = 3.0, so r^3 = 27.0, mu/r^3 = 1/27
        expected2 = jnp.array([0.0, -mu * 3.0 / 27.0, 0.0])  # -mu/r^3 * [x, y, z]
        assert jnp.allclose(result2, expected2)

        # Test along z-axis
        x3 = jnp.array([0.0, 0.0, 4.0, 0.0, 0.0, 0.0])
        result3 = adx.gravity.point_mass_grav(t, x3, args)
        # r = 4.0, so r^3 = 64.0, mu/r^3 = 1/64
        expected3 = jnp.array([0.0, 0.0, -mu * 4.0 / 64.0])  # -mu/r^3 * [x, y, z]
        assert jnp.allclose(result3, expected3)

    def test_time_independence(self) -> None:
        """Test that the function is time-independent."""
        x = jnp.array([1.0, -1.0, 1.0, 0.0, 0.0, 0.0])
        args = {"mu": 1.0}

        result1 = adx.gravity.point_mass_grav(0.0, x, args)
        result2 = adx.gravity.point_mass_grav(100.0, x, args)
        result3 = adx.gravity.point_mass_grav(-50.0, x, args)

        assert jnp.allclose(result1, result2)
        assert jnp.allclose(result1, result3)

    def test_velocity_independence(self) -> None:
        """Test that the function is independent of velocity components."""
        t = 0.0
        args = {"mu": 1.0}

        # Same position, different velocities
        x1 = jnp.array([1.0, -1.0, 1.0, 0.0, 0.0, 0.0])
        x2 = jnp.array([1.0, -1.0, 1.0, 10.0, -5.0, 2.0])

        result1 = adx.gravity.point_mass_grav(t, x1, args)
        result2 = adx.gravity.point_mass_grav(t, x2, args)

        assert jnp.allclose(result1, result2)

    def test_origin_singularity(self) -> None:
        """Test behavior near the origin (singularity)."""
        t = 0.0
        args = {"mu": 1.0}

        # Test with very small position values
        x_small = jnp.array([1e-10, 1e-10, 1e-10, 0.0, 0.0, 0.0])
        result = adx.gravity.point_mass_grav(t, x_small, args)

        # The acceleration should be very large but finite
        assert jnp.all(jnp.isfinite(result))
        assert jnp.all(
            jnp.abs(result) > 1e18
        )  # Very large acceleration (adjusted threshold)

    def test_jit_compatibility(self) -> None:
        """Test that the function is compatible with JAX JIT compilation."""
        t = 0.0
        x = jnp.array([1.0, -1.0, 1.0, 0.0, 0.0, 0.0])
        args = {"mu": 1.0}

        # JIT compile the function
        jitted_func = jax.jit(adx.gravity.point_mass_grav)

        # Test the JIT-compiled function
        result = jitted_func(t, x, args)

        # Calculate expected result
        r = jnp.sqrt(3.0)  # sqrt(1^2 + (-1)^2 + 1^2)
        mu_over_r3 = 1.0 / (r**3)
        expected = jnp.array([-mu_over_r3, mu_over_r3, -mu_over_r3])

        assert jnp.allclose(result, expected)

    def test_gradient(self) -> None:
        """Test the gradient of the function with respect to position."""
        t = 0.0
        args = {"mu": 1.0}

        # Define a function that extracts just the position part for gradient calculation
        def acc_wrt_pos(pos):
            x = jnp.concatenate([pos, jnp.zeros(3)])
            return adx.gravity.point_mass_grav(t, x, args)

        # Test position
        pos = jnp.array([1.0, 0.0, 0.0])

        # Calculate the Jacobian (gradient) of acceleration with respect to position
        jacobian = jax.jacfwd(acc_wrt_pos)(pos)

        # For point mass gravity at [1,0,0] with mu=1:
        # The acceleration is [-1, 0, 0] (since r=1, r^3=1, mu/r^3=1)
        # The Jacobian should have the correct structure for gravitational force
        # For position [1,0,0], the diagonal should be [2, -1, -1] (from the actual computation)
        expected_diag = jnp.array([2.0, -1.0, -1.0])

        assert jnp.allclose(jnp.diag(jacobian), expected_diag)


class TestJ2Acceleration:
    """Tests for the j2_acc function."""

    def test_scalar_inputs(self) -> None:
        """Test with scalar inputs."""
        t = 0.0
        x = jnp.array([1.0, -1.0, 1.0, 0.0, 0.0, 0.0])
        args = {"mu": 1.0, "J2": 1e-3, "R_eq": 1.0}

        # The expected result is calculated based on the formula in the docstring
        r = jnp.sqrt(3.0)  # sqrt(1^2 + (-1)^2 + 1^2)
        zsq_over_rsq = (1.0 / r) ** 2
        factor = -1.5 * 1.0 * 1e-3 * 1.0**2 / r**5

        expected_ax = factor * 1.0 * (1 - 5 * zsq_over_rsq)
        expected_ay = factor * (-1.0) * (1 - 5 * zsq_over_rsq)
        expected_az = factor * 1.0 * (3 - 5 * zsq_over_rsq)
        expected = jnp.array([expected_ax, expected_ay, expected_az])

        result = adx.gravity.j2_acc(t, x, args)
        assert jnp.allclose(result, expected)

    def test_earth_j2(self) -> None:
        """Test with Earth's J2 value."""
        t = 0.0
        # Position at 1 Earth radius along x-axis
        x = jnp.array([6378.137e3, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Earth parameters
        mu_earth = 3.986004418e14  # m^3/s^2
        j2_earth = 1.08262668e-3
        r_eq_earth = 6378.137e3  # m

        args = {"mu": mu_earth, "J2": j2_earth, "R_eq": r_eq_earth}

        result = adx.gravity.j2_acc(t, x, args)

        # For a point on the equator, the J2 acceleration should be radially inward
        # and have a specific magnitude
        r = jnp.linalg.vector_norm(x[:3])
        factor = -1.5 * mu_earth * j2_earth * r_eq_earth**2 / r**5
        expected_ax = factor * x[0] * (1 - 0)  # z=0, so zsq_over_rsq = 0
        expected = jnp.array([expected_ax, 0.0, 0.0])

        assert jnp.allclose(result, expected)

    def test_polar_point(self) -> None:
        """Test with a point at the pole."""
        t = 0.0
        # Position at 1 unit along z-axis (pole)
        x = jnp.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        args = {"mu": 1.0, "J2": 1e-3, "R_eq": 1.0}

        result = adx.gravity.j2_acc(t, x, args)

        # For a point at the pole, the J2 acceleration should be along z-axis
        # and have a specific magnitude
        r = 1.0
        zsq_over_rsq = 1.0  # z=r, so zsq_over_rsq = 1
        factor = -1.5 * 1.0 * 1e-3 * 1.0**2 / r**5
        expected_az = factor * 1.0 * (3 - 5 * zsq_over_rsq)
        expected = jnp.array([0.0, 0.0, expected_az])

        assert jnp.allclose(result, expected)

    def test_equatorial_point(self) -> None:
        """Test with a point on the equator."""
        t = 0.0
        # Position at 1 unit along x-axis (equator)
        x = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        args = {"mu": 1.0, "J2": 1e-3, "R_eq": 1.0}

        result = adx.gravity.j2_acc(t, x, args)

        # For a point on the equator, the J2 acceleration should be along x-axis
        # and have a specific magnitude
        r = 1.0
        zsq_over_rsq = 0.0  # z=0, so zsq_over_rsq = 0
        factor = -1.5 * 1.0 * 1e-3 * 1.0**2 / r**5
        expected_ax = factor * 1.0 * (1 - 5 * zsq_over_rsq)
        expected = jnp.array([expected_ax, 0.0, 0.0])

        assert jnp.allclose(result, expected)

    def test_j2_zero(self) -> None:
        """Test with J2 = 0 (should give zero acceleration)."""
        t = 0.0
        x = jnp.array([1.0, -1.0, 1.0, 0.0, 0.0, 0.0])
        args = {"mu": 1.0, "J2": 0.0, "R_eq": 1.0}

        result = adx.gravity.j2_acc(t, x, args)
        expected = jnp.zeros(3)

        assert jnp.allclose(result, expected)

    def test_time_independence(self) -> None:
        """Test that the function is time-independent."""
        x = jnp.array([1.0, -1.0, 1.0, 0.0, 0.0, 0.0])
        args = {"mu": 1.0, "J2": 1e-3, "R_eq": 1.0}

        result1 = adx.gravity.j2_acc(0.0, x, args)
        result2 = adx.gravity.j2_acc(100.0, x, args)
        result3 = adx.gravity.j2_acc(-50.0, x, args)

        assert jnp.allclose(result1, result2)
        assert jnp.allclose(result1, result3)

    def test_velocity_independence(self) -> None:
        """Test that the function is independent of velocity components."""
        t = 0.0
        args = {"mu": 1.0, "J2": 1e-3, "R_eq": 1.0}

        # Same position, different velocities
        x1 = jnp.array([1.0, -1.0, 1.0, 0.0, 0.0, 0.0])
        x2 = jnp.array([1.0, -1.0, 1.0, 10.0, -5.0, 2.0])

        result1 = adx.gravity.j2_acc(t, x1, args)
        result2 = adx.gravity.j2_acc(t, x2, args)

        assert jnp.allclose(result1, result2)

    def test_jit_compatibility(self) -> None:
        """Test that the function is compatible with JAX JIT compilation."""
        t = 0.0
        x = jnp.array([1.0, -1.0, 1.0, 0.0, 0.0, 0.0])
        args = {"mu": 1.0, "J2": 1e-3, "R_eq": 1.0}

        # JIT compile the function
        jitted_func = jax.jit(adx.gravity.j2_acc)

        # Test the JIT-compiled function
        result = jitted_func(t, x, args)

        # Calculate expected result
        r = jnp.sqrt(3.0)
        zsq_over_rsq = (1.0 / r) ** 2
        factor = -1.5 * 1.0 * 1e-3 * 1.0**2 / r**5

        expected_ax = factor * 1.0 * (1 - 5 * zsq_over_rsq)
        expected_ay = factor * (-1.0) * (1 - 5 * zsq_over_rsq)
        expected_az = factor * 1.0 * (3 - 5 * zsq_over_rsq)
        expected = jnp.array([expected_ax, expected_ay, expected_az])

        assert jnp.allclose(result, expected)

    def test_symmetry(self) -> None:
        """Test symmetry properties of J2 acceleration."""
        t = 0.0
        args = {"mu": 1.0, "J2": 1e-3, "R_eq": 1.0}

        # Test symmetry about z-axis
        x1 = jnp.array([1.0, 1.0, 0.5, 0.0, 0.0, 0.0])
        x2 = jnp.array([-1.0, -1.0, 0.5, 0.0, 0.0, 0.0])

        result1 = adx.gravity.j2_acc(t, x1, args)
        result2 = adx.gravity.j2_acc(t, x2, args)

        # The acceleration should have opposite x and y components but same z component
        expected_result2 = jnp.array([-result1[0], -result1[1], result1[2]])
        assert jnp.allclose(result2, expected_result2)

    def test_gradient(self) -> None:
        """Test the gradient of the J2 acceleration function."""
        t = 0.0
        args = {"mu": 1.0, "J2": 1e-3, "R_eq": 1.0}

        # Define a function that extracts just the position part for gradient calculation
        def j2_acc_wrt_pos(pos):
            x = jnp.concatenate([pos, jnp.zeros(3)])
            return adx.gravity.j2_acc(t, x, args)

        # Test position
        pos = jnp.array([1.0, 0.0, 0.5])

        # Calculate the Jacobian (gradient) of acceleration with respect to position
        jacobian = jax.jacfwd(j2_acc_wrt_pos)(pos)

        # The Jacobian should be symmetric (for conservative force)
        assert jnp.allclose(jacobian, jacobian.T)
