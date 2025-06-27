import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array


def orb_period(a: ArrayLike, mu: ArrayLike) -> Array:
    r"""
    Returns the orbital period of a two-body system.

    Args:
        a: Semimajor axis of the orbit.
        mu: Gravitational parameter of the central body; shape broadcast-compatible with `a`.

    Returns:
        The orbital period of the object in the two-body system.

    Notes:
        The orbital period is calculated using Kepler's third law:
        $$
        P = 2\pi \sqrt{\frac{a^3}{\mu}}
        $$
        where $P$ is the orbital period, $a$ is the semimajor axis, and $\mu$ is the gravitational parameter.

    References:
        Battin, 1999, pp.119.

    Examples:
        A simple example of calculating the orbital period for a circular orbit with a semimajor axis of 1.0 and a gravitational parameter of 1.0:

        >>> import jax.numpy as jnp
        >>> from astrodynx.twobody import orb_period
        >>> a = 1.0
        >>> mu = 1.0
        >>> orb_period(a, mu)
        Array(6.2831855, dtype=float32, weak_type=True)

        With broadcasting, you can calculate the orbital period for multiple semimajor axes and gravitational parameters:

        >>> a = jnp.array([1.0, 2.0])
        >>> mu = jnp.array([1.0, 2.0])
        >>> orb_period(a, mu)
        Array([ 6.2831855, 12.566371 ], dtype=float32)
    """
    return 2 * jnp.pi * jnp.sqrt(a**3 / mu)


def angular_momentum(r: ArrayLike, v: ArrayLike) -> Array:
    r"""
    Returns the specific angular momentum of a two-body system.

    Args:
        r: (..., 3) position vector of the object in the two-body system.
        v: (..., 3) velocity vector of the object in the two-body system, which shape broadcast-compatible with `r`.

    Returns:
        The specific angular momentum vector of the object in the two-body system.

    Notes
        The specific angular momentum is calculated using the cross product of the position and velocity vectors:
        $$
        \boldsymbol{h} = \boldsymbol{r} \times \boldsymbol{v}
        $$
        where $\boldsymbol{h}$ is the specific angular momentum, $\boldsymbol{r}$ is the position vector, and $\boldsymbol{v}$ is the velocity vector.

    References
        Battin, 1999, pp.115.

    Examples
        A simple example of calculating the specific angular momentum for a position vector [1, 0, 0] and velocity vector [0, 1, 0]:

        >>> import jax.numpy as jnp
        >>> from astrodynx.twobody import angular_momentum
        >>> r = jnp.array([1.0, 0.0, 0.0])
        >>> v = jnp.array([0.0, 1.0, 0.0])
        >>> angular_momentum(r, v)
        Array([0., 0., 1.], dtype=float32)

        With broadcasting, you can calculate the specific angular momentum for multiple position and velocity vectors:

        >>> r = jnp.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        >>> v = jnp.array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
        >>> angular_momentum(r, v)
        Array([[0., 0., 1.],
               [0., 0., 4.]], dtype=float32)
    """
    return jnp.cross(r, v)


def semimajor_axis(r: ArrayLike, v: ArrayLike, mu: ArrayLike) -> ArrayLike:
    r"""
    Returns the semimajor axis of a two-body orbit.

    Args:
        r: Norm of the object's position vector in the two-body system.
        v: Norm of the object's velocity vector in the two-body system, which shape broadcast-compatible with `r`.
        mu: Gravitational parameter of the central body; shape broadcast-compatible with `r` and `v`.

    Returns:
        The semimajor axis of the orbit.

    Notes
        The semimajor axis is calculated using equation (3.16):
        $$
        a = \left( \frac{2}{r} - \frac{v^2}{\mu} \right)^{-1}
        $$
        where $a$ is the semimajor axis, $r$ is the norm of the position vector, $v$ is the norm of the velocity vector, and $\mu$ is the gravitational parameter.

    References
        Battin, 1999, pp.116.

    Examples
        A simple example of calculating the semimajor axis with a position vector norm of 1.0, velocity vector norm of 1.0, and gravitational parameter of 1.0:

        >>> import jax.numpy as jnp
        >>> from astrodynx.twobody import semimajor_axis
        >>> r = 1.0
        >>> v = 1.0
        >>> mu = 1.0
        >>> semimajor_axis(r, v, mu)
        1.0

        With broadcasting, you can calculate the semimajor axis for multiple position and velocity vectors:

        >>> r = jnp.array([1.0, 2.0])
        >>> v = jnp.array([1.0, 2.0])
        >>> mu = jnp.array([1.0, 2.0])
        >>> semimajor_axis(r, v, mu)
        Array([ 1., -1.], dtype=float32)
    """
    return 1 / (2 / r - v**2 / mu)


def eccentricity_vector(r: ArrayLike, v: ArrayLike, mu: ArrayLike) -> Array:
    r"""
    Returns the eccentricity vector of a two-body orbit.

    Args:
        r: (..., 3) position vector of the object in the two-body system.
        v: (..., 3) velocity vector of the object in the two-body system, which shape broadcast-compatible with `r`.
        mu: Gravitational parameter of the central body; shape broadcast-compatible with `r` and `v`.

    Returns:
        The eccentricity vector of the orbit.

    Notes
        The eccentricity vector is calculated using equation (3.14):
        $$
        \boldsymbol{e} = \frac{\boldsymbol{v} \times \boldsymbol{h}}{\mu} - \frac{\boldsymbol{r}}{r}
        $$
        where $\boldsymbol{e}$ is the eccentricity vector, $\boldsymbol{v}$ is the velocity vector, $\boldsymbol{h}$ is the specific angular momentum vector, $\mu$ is the gravitational parameter, and $\boldsymbol{r}$ is the position vector.

    References
        Battin, 1999, pp.116.

    Examples
        A simple example of calculating the eccentricity vector for a position vector [1, 0, 0] and velocity vector [0, 1, 0]:

        >>> import jax.numpy as jnp
        >>> from astrodynx.twobody import eccentricity_vector
        >>> r = jnp.array([1.0, 0.0, 0.0])
        >>> v = jnp.array([0.0, 1.0, 0.0])
        >>> mu = 1.0
        >>> eccentricity_vector(r, v, mu)
        Array([0., 0., 0.], dtype=float32)

        With broadcasting, you can calculate the eccentricity vector for multiple position and velocity vectors:

        >>> r = jnp.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        >>> v = jnp.array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
        >>> mu = jnp.array([[1.0],[2.0]])
        >>> eccentricity_vector(r, v, mu)
        Array([[0., 0., 0.],
            [3., 0., 0.]], dtype=float32)
    """
    h = angular_momentum(r, v)
    return jnp.cross(v, h) / mu - r / jnp.linalg.norm(r, axis=-1, keepdims=True)
