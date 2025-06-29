import astrodynx as adx
import jax.numpy as jnp


class TestRotmat3dx:
    def test_zero_angle(self):
        """Test that zero angle returns the identity matrix."""
        angle = 0.0
        expected = jnp.eye(3)
        result = adx.utils.rotmat3dx(angle)
        assert jnp.allclose(result, expected, atol=1e-7)

    def test_pi_over_two(self):
        """Test rotation by 90 degrees (π/2 radians)."""
        angle = jnp.pi / 2
        expected = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        result = adx.utils.rotmat3dx(angle)
        assert jnp.allclose(result, expected, atol=1e-7)

    def test_pi(self):
        """Test rotation by 180 degrees (π radians)."""
        angle = jnp.pi
        expected = jnp.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        result = adx.utils.rotmat3dx(angle)
        assert jnp.allclose(result, expected, atol=1e-7)

    def test_negative_angle(self):
        """Test rotation by a negative angle (-π/2 radians)."""
        angle = -jnp.pi / 2
        expected = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
        result = adx.utils.rotmat3dx(angle)
        assert jnp.allclose(result, expected, atol=1e-7)

    def test_broadcasting(self):
        """Test broadcasting with an array of angles."""
        angles = jnp.array([0.0, jnp.pi / 2])
        results = jnp.stack([adx.utils.rotmat3dx(a) for a in angles])
        expected0 = jnp.eye(3)
        expected1 = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        assert jnp.allclose(results[0], expected0, atol=1e-7)
        assert jnp.allclose(results[1], expected1, atol=1e-7)


class TestRotmat3dy:
    def test_zero_angle(self):
        """Test that zero angle returns the identity matrix."""
        angle = 0.0
        expected = jnp.eye(3)
        result = adx.utils.rotmat3dy(angle)
        assert jnp.allclose(result, expected, atol=1e-7)

    def test_pi_over_two(self):
        """Test rotation by 90 degrees (π/2 radians)."""
        angle = jnp.pi / 2
        expected = jnp.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        result = adx.utils.rotmat3dy(angle)
        assert jnp.allclose(result, expected, atol=1e-7)

    def test_pi(self):
        """Test rotation by 180 degrees (π radians)."""
        angle = jnp.pi
        expected = jnp.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        result = adx.utils.rotmat3dy(angle)
        assert jnp.allclose(result, expected, atol=1e-7)

    def test_negative_angle(self):
        """Test rotation by a negative angle (-π/2 radians)."""
        angle = -jnp.pi / 2
        expected = jnp.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        result = adx.utils.rotmat3dy(angle)
        assert jnp.allclose(result, expected, atol=1e-7)

    def test_broadcasting(self):
        """Test broadcasting with an array of angles."""
        angles = jnp.array([0.0, jnp.pi / 2])
        results = jnp.stack([adx.utils.rotmat3dy(a) for a in angles])
        expected0 = jnp.eye(3)
        expected1 = jnp.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        assert jnp.allclose(results[0], expected0, atol=1e-7)
        assert jnp.allclose(results[1], expected1, atol=1e-7)


class TestRotmat3dz:
    def test_zero_angle(self):
        """Test that zero angle returns the identity matrix."""
        angle = 0.0
        expected = jnp.eye(3)
        result = adx.utils.rotmat3dz(angle)
        assert jnp.allclose(result, expected, atol=1e-7)

    def test_pi_over_two(self):
        """Test rotation by 90 degrees (π/2 radians)."""
        angle = jnp.pi / 2
        expected = jnp.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        result = adx.utils.rotmat3dz(angle)
        assert jnp.allclose(result, expected, atol=1e-7)

    def test_pi(self):
        """Test rotation by 180 degrees (π radians)."""
        angle = jnp.pi
        expected = jnp.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
        result = adx.utils.rotmat3dz(angle)
        assert jnp.allclose(result, expected, atol=1e-7)

    def test_negative_angle(self):
        """Test rotation by a negative angle (-π/2 radians)."""
        angle = -jnp.pi / 2
        expected = jnp.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        result = adx.utils.rotmat3dz(angle)
        assert jnp.allclose(result, expected, atol=1e-7)

    def test_broadcasting(self):
        """Test broadcasting with an array of angles."""
        angles = jnp.array([0.0, jnp.pi / 2])
        results = jnp.stack([adx.utils.rotmat3dz(a) for a in angles])
        expected0 = jnp.eye(3)
        expected1 = jnp.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        assert jnp.allclose(results[0], expected0, atol=1e-7)
        assert jnp.allclose(results[1], expected1, atol=1e-7)
