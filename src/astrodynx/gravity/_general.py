from jax import numpy as jnp
from jaxtyping import ArrayLike, DTypeLike, PyTree
from typing import Any

"""General gravity perturbations"""


def point_mass_grav(
    t: DTypeLike, x: ArrayLike, args: PyTree[Any] = {"mu": 1.0}
) -> ArrayLike:
    r"""Returns the acceleration due to a point mass.

    Args:
        t: The time.
        x: The state vector.
        args: Static arguments.

    Returns:
        The acceleration due to a point mass.

    Notes:
        The acceleration due to a point mass is defined as:
        $$
        \boldsymbol{a} = -\frac{\mu}{r^3} \boldsymbol{r}
        $$
        where $\boldsymbol{a}$ is the acceleration, $\mu$ is the gravitational parameter, and $\boldsymbol{r}$ is the position vector.

    References:
        Battin, 1999, pp.114.

    Examples:
        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> t = 0.0
        >>> x = jnp.array([1.0, -1.0, 1.0, 0.0, 0.0, 0.0])
        >>> args = {"mu": 1.0}
        >>> adx.gravity.point_mass_grav(t, x, args)
        Array([-0.1924...,  0.1924..., -0.1924...], dtype=float32)
    """
    mu = args["mu"]
    mu_ = mu / jnp.linalg.vector_norm(x[:3]) ** 3
    return jnp.stack([-mu_ * x[0], -mu_ * x[1], -mu_ * x[2]])


def j2_acc(
    t: DTypeLike, x: ArrayLike, args: PyTree[Any] = {"mu": 1.0, "J2": 0.0, "R_eq": 1.0}
) -> ArrayLike:
    r"""Returns the acceleration due to J2 perturbation.

    Args:
        t: The time.
        x: The state vector.
        args: Static arguments.

    Returns:
        The acceleration due to J2 perturbation.

    Notes:
        The acceleration due to J2 perturbation is defined as:
        $$
        \begin{align*}
        a_x &= -\frac{3}{2} \frac{\mu J_2 R_{eq}^2}{r^5} x \left( 1 - 5 \frac{z^2}{r^2} \right) \\
        a_y &= -\frac{3}{2} \frac{\mu J_2 R_{eq}^2}{r^5} y \left( 1 - 5 \frac{z^2}{r^2} \right) \\
        a_z &= -\frac{3}{2} \frac{\mu J_2 R_{eq}^2}{r^5} z \left( 3 - 5 \frac{z^2}{r^2} \right)
        \end{align*}
        $$
        where $\boldsymbol{a}$ is the acceleration, $\mu$ is the gravitational parameter, $J_2$ is the second zonal harmonic, $R_{eq}$ is the equatorial radius, and $\boldsymbol{r} = [x, y, z]$ is the position vector.

    References:
        Vallado, 2013, pp.594.

    Examples:
        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> t = 0.0
        >>> x = jnp.array([1.0, -1.0, 1.0, 0.0, 0.0, 0.0])
        >>> args = {"mu": 1.0, "J2": 1e-3, "R_eq": 1.0}
        >>> expected = jnp.array([ 6.4150023e-05, -6.4150023e-05, -1.2830009e-04])
        >>> actual = adx.gravity.j2_acc(t, x, args)
        >>> jnp.allclose(expected, actual)
        Array(True, dtype=bool)
    """
    mu = args["mu"]
    J2 = args["J2"]
    R_eq = args["R_eq"]
    r = jnp.linalg.vector_norm(x[:3])
    zsq_over_rsq = (x[2] / r) ** 2
    factor = -1.5 * mu * J2 * R_eq**2 / r**5
    ax = factor * x[0] * (1 - 5 * zsq_over_rsq)
    ay = factor * x[1] * (1 - 5 * zsq_over_rsq)
    az = factor * x[2] * (3 - 5 * zsq_over_rsq)
    return jnp.stack([ax, ay, az])
