import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array


def rotmat3dx(angle: ArrayLike) -> Array:
    r"""Returns a 3x3 rotation matrix for a given angle around the x-axis.

    Args:
        angle: The angle in radians to rotate around the x-axis.

    Returns:
        A 3x3 rotation matrix that rotates vectors around the x-axis by the specified angle.

    Notes:
        The rotation matrix is defined as:
        $$
        R_x(\theta) = \begin{bmatrix}
        1 & 0 & 0 \\
        0 & \cos(\theta) & -\sin(\theta) \\
        0 & \sin(\theta) & \cos(\theta)
        \end{bmatrix}
        $$
        where $\theta$ is the angle of rotation.

    References:
        Battin, 1999, pp.85.

    Examples:
        Creating a rotation matrix for a 90-degree rotation (π/2 radians):

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> angle = jnp.pi / 2
        >>> jnp.allclose(adx.utils.rotmat3dx(angle), jnp.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]), atol=1e-7)
        Array(True, dtype=bool)

        Broadcasting with an array of angles:

        >>> angles = jnp.array([0.0, jnp.pi / 2])
        >>> results = jnp.stack([adx.utils.rotmat3dx(a) for a in angles])
        >>> expected0 = jnp.eye(3)
        >>> expected1 = jnp.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        >>> jnp.allclose(results[0], expected0, atol=1e-7)
        Array(True, dtype=bool)
        >>> jnp.allclose(results[1], expected1, atol=1e-7)
        Array(True, dtype=bool)
    """
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    z = jnp.zeros_like(angle)
    o = jnp.ones_like(angle)
    return jnp.stack(
        [
            jnp.stack([o, z, z], axis=-1),
            jnp.stack([z, c, -s], axis=-1),
            jnp.stack([z, s, c], axis=-1),
        ],
        axis=-2,
    )


def rotmat3dy(angle: ArrayLike) -> Array:
    r"""Returns a 3x3 rotation matrix for a given angle around the y-axis.

    Args:
        angle: The angle in radians to rotate around the y-axis.

    Returns:
        A 3x3 rotation matrix that rotates vectors around the y-axis by the specified angle.

    Notes:
        The rotation matrix is defined as:
        $$
        R_y(\theta) = \begin{bmatrix}
        \cos(\theta) & 0 & \sin(\theta) \\
        0 & 1 & 0 \\
        -\sin(\theta) & 0 & \cos(\theta)
        \end{bmatrix}
        $$
        where $\theta$ is the angle of rotation.

    References:
        Battin, 1999, pp.85.

    Examples:
        Creating a rotation matrix for a 90-degree rotation (π/2 radians):

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> angle = jnp.pi / 2
        >>> jnp.allclose(adx.utils.rotmat3dy(angle), jnp.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]), atol=1e-7)
        Array(True, dtype=bool)

        Broadcasting with an array of angles:

        >>> angles = jnp.array([0.0, jnp.pi / 2])
        >>> results = jnp.stack([adx.utils.rotmat3dy(a) for a in angles])
        >>> expected0 = jnp.eye(3)
        >>> expected1 = jnp.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
        >>> jnp.allclose(results[0], expected0, atol=1e-7)
        Array(True, dtype=bool)
        >>> jnp.allclose(results[1], expected1, atol=1e-7)
        Array(True, dtype=bool)
    """
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    z = jnp.zeros_like(angle)
    o = jnp.ones_like(angle)
    return jnp.stack(
        [
            jnp.stack([c, z, s], axis=-1),
            jnp.stack([z, o, z], axis=-1),
            jnp.stack([-s, z, c], axis=-1),
        ],
        axis=-2,
    )


def rotmat3dz(angle: ArrayLike) -> Array:
    r"""Returns a 3x3 rotation matrix for a given angle around the z-axis.

    Args:
        angle: The angle in radians to rotate around the z-axis.

    Returns:
        A 3x3 rotation matrix that rotates vectors around the z-axis by the specified angle.

    Notes:
        The rotation matrix is defined as:
        $$
        R_z(\theta) = \begin{bmatrix}
        \cos(\theta) & -\sin(\theta) & 0 \\
        \sin(\theta) & \cos(\theta) & 0 \\
        0 & 0 & 1
        \end{bmatrix}
        $$
        where $\theta$ is the angle of rotation.

    References:
        Battin, 1999, pp.85.

    Examples:
        Creating a rotation matrix for a 90-degree rotation (π/2 radians):

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> angle = jnp.pi / 2
        >>> jnp.allclose(adx.utils.rotmat3dz(angle), jnp.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]), atol=1e-7)
        Array(True, dtype=bool)

        Broadcasting with an array of angles:

        >>> angles = jnp.array([0.0, jnp.pi / 2])
        >>> results = jnp.stack([adx.utils.rotmat3dz(a) for a in angles])
        >>> expected0 = jnp.eye(3)
        >>> expected1 = jnp.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        >>> jnp.allclose(results[0], expected0, atol=1e-7)
        Array(True, dtype=bool)
        >>> jnp.allclose(results[1], expected1, atol=1e-7)
        Array(True, dtype=bool)
    """
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    z = jnp.zeros_like(angle)
    o = jnp.ones_like(angle)
    return jnp.stack(
        [
            jnp.stack([c, -s, z], axis=-1),
            jnp.stack([s, c, z], axis=-1),
            jnp.stack([z, z, o], axis=-1),
        ],
        axis=-2,
    )
