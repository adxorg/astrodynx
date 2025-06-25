import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array


def orb_period(a: ArrayLike, mu: ArrayLike) -> Array:
    r"""Returns the orbital period of a two-body system.

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
    ----------
    The specific angular momentum is calculated using the cross product of the position and velocity vectors:
    $$
    h = r \times v
    $$
    where $h$ is the specific angular momentum, $r$ is the position vector, and $v$ is the velocity vector.

    References
    ----------
    Battin, 1999, pp.115.

    Examples
    --------
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
