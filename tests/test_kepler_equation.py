from astrodynx.twobody import mean_anomaly_equ_elps, mean_anomaly_equ_hypb

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


class TestMeanAnomalyEquHypb:
    def test_scalar(self) -> None:
        """Test scalar inputs with basic hyperbolic orbit parameters."""
        e = 1.1
        H = 1.0
        expected = e * jnp.sinh(H) - H
        result = mean_anomaly_equ_hypb(e, H)
        assert jnp.allclose(result, expected, rtol=1e-7)

    def test_array(self) -> None:
        """Test array inputs with multiple eccentricities and hyperbolic anomalies."""
        e = jnp.array([1.1, 1.2])
        H = jnp.array([1.0, 2.0])
        expected = e * jnp.sinh(H) - H
        result = mean_anomaly_equ_hypb(e, H)
        assert jnp.allclose(result, expected, rtol=1e-7)

    def test_broadcasting(self) -> None:
        """Test broadcasting between scalar and array inputs."""
        e = jnp.array([1.1, 1.2])
        H = 1.0
        expected = e * jnp.sinh(H) - H
        result = mean_anomaly_equ_hypb(e, H)
        assert jnp.allclose(result, expected, rtol=1e-7)

    def test_broadcasting_reverse(self) -> None:
        """Test broadcasting with scalar eccentricity and array hyperbolic anomaly."""
        e = 1.5
        H = jnp.array([0.5, 1.0, 1.5])
        expected = e * jnp.sinh(H) - H
        result = mean_anomaly_equ_hypb(e, H)
        assert jnp.allclose(result, expected, rtol=1e-7)

    def test_minimum_eccentricity(self) -> None:
        """Test with minimum valid eccentricity for hyperbolic orbits (e > 1)."""
        e = 1.001  # Just above 1
        H = 0.5
        expected = e * jnp.sinh(H) - H
        result = mean_anomaly_equ_hypb(e, H)
        assert jnp.allclose(result, expected, rtol=1e-7)

    def test_large_eccentricity(self) -> None:
        """Test with large eccentricity values."""
        e = 10.0
        H = 0.5
        expected = e * jnp.sinh(H) - H
        result = mean_anomaly_equ_hypb(e, H)
        assert jnp.allclose(result, expected, rtol=1e-7)

    def test_zero_hyperbolic_anomaly(self) -> None:
        """Test with zero hyperbolic anomaly."""
        e = jnp.array([1.1, 1.5, 2.0])
        H = 0.0
        expected = e * jnp.sinh(H) - H  # Should be 0 since sinh(0) = 0
        result = mean_anomaly_equ_hypb(e, H)
        assert jnp.allclose(result, expected, rtol=1e-7)
        assert jnp.allclose(result, 0.0, atol=1e-7)

    def test_negative_hyperbolic_anomaly(self) -> None:
        """Test with negative hyperbolic anomaly values."""
        e = 1.5
        H = jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        expected = e * jnp.sinh(H) - H
        result = mean_anomaly_equ_hypb(e, H)
        assert jnp.allclose(result, expected, rtol=1e-7)

    def test_large_hyperbolic_anomaly(self) -> None:
        """Test with large hyperbolic anomaly values."""
        e = 1.2
        H = 5.0
        expected = e * jnp.sinh(H) - H
        result = mean_anomaly_equ_hypb(e, H)
        assert jnp.allclose(result, expected, rtol=1e-7)

    def test_multidimensional_arrays(self) -> None:
        """Test with multidimensional array inputs."""
        e = jnp.array([[1.1, 1.2], [1.3, 1.4]])
        H = jnp.array([[0.5, 1.0], [1.5, 2.0]])
        expected = e * jnp.sinh(H) - H
        result = mean_anomaly_equ_hypb(e, H)
        assert jnp.allclose(result, expected, rtol=1e-7)
        assert result.shape == e.shape

    def test_types(self) -> None:
        """Test that the function returns the correct type."""
        e = 1.1
        H = 1.0
        result = mean_anomaly_equ_hypb(e, H)
        assert isinstance(result, jnp.ndarray)

    def test_mathematical_properties(self) -> None:
        """Test mathematical properties of the hyperbolic mean anomaly equation."""
        e = 1.5

        # Test that the function is monotonic in H for fixed e
        H_values = jnp.linspace(-2, 2, 10)
        results = mean_anomaly_equ_hypb(e, H_values)
        # Check that results are monotonically increasing
        assert jnp.all(jnp.diff(results) > 0)

    def test_symmetry_properties(self) -> None:
        """Test symmetry properties: N(-H) = -N(H) for the hyperbolic case."""
        e = 1.5
        H = 1.0

        result_pos = mean_anomaly_equ_hypb(e, H)
        result_neg = mean_anomaly_equ_hypb(e, -H)

        # For hyperbolic orbits: N(-H) = e*sinh(-H) - (-H) = -e*sinh(H) + H = -(e*sinh(H) - H) = -N(H)
        assert jnp.allclose(result_neg, -result_pos, rtol=1e-7)

    def test_docstring_examples(self) -> None:
        """Test the examples provided in the function docstring."""
        # First example
        e = 1.1
        H = 1.0
        result = mean_anomaly_equ_hypb(e, H)
        expected = 0.29272127  # From corrected docstring
        assert jnp.allclose(result, expected, rtol=1e-6)

        # Second example with broadcasting
        e = jnp.array([1.1, 1.2])
        H = jnp.array([1.0, 2.0])
        result = mean_anomaly_equ_hypb(e, H)
        expected = jnp.array([0.29272127, 2.3522325])  # From corrected docstring
        assert jnp.allclose(result, expected, rtol=1e-6)

    def test_edge_case_very_small_hyperbolic_anomaly(self) -> None:
        """Test with very small hyperbolic anomaly values."""
        e = 1.1
        H = 1e-10
        expected = e * jnp.sinh(H) - H
        result = mean_anomaly_equ_hypb(e, H)
        assert jnp.allclose(result, expected, rtol=1e-7)
        # For small H, sinh(H) ≈ H, so result ≈ e*H - H = H*(e-1)
        assert jnp.allclose(result, H * (e - 1), rtol=1e-6)

    def test_performance_large_arrays(self) -> None:
        """Test performance with large arrays."""
        import time

        # Create large arrays
        size = 10000
        e = jnp.full(size, 1.5)
        H = jnp.linspace(-5, 5, size)

        start_time = time.time()
        result = mean_anomaly_equ_hypb(e, H)
        end_time = time.time()

        # Check that computation completes and results are reasonable
        assert result.shape == (size,)
        assert jnp.all(jnp.isfinite(result))

        # Performance should be reasonable (less than 1 second for 10k elements)
        computation_time = end_time - start_time
        assert computation_time < 1.0, (
            f"Computation took {computation_time:.3f} seconds, which is too slow"
        )

    def test_invalid_eccentricity_boundary(self) -> None:
        """Test behavior at the boundary of valid eccentricity (e = 1)."""
        # Note: e = 1 is parabolic, not hyperbolic, but test the boundary behavior
        e = 1.0
        H = 1.0
        result = mean_anomaly_equ_hypb(e, H)
        expected = e * jnp.sinh(H) - H
        assert jnp.allclose(result, expected, rtol=1e-7)

    def test_consistency_with_formula(self) -> None:
        """Test consistency with the mathematical formula across various inputs."""
        # Test multiple combinations
        e_values = jnp.array([1.01, 1.1, 1.5, 2.0, 5.0])
        H_values = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        for e in e_values:
            for H in H_values:
                result = mean_anomaly_equ_hypb(e, H)
                expected = e * jnp.sinh(H) - H
                assert jnp.allclose(result, expected, rtol=1e-7), (
                    f"Failed for e={e}, H={H}"
                )
