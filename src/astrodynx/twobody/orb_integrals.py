import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array

"""Orbital integrals and elements for two-body orbital mechanics."""


def orb_period(a: ArrayLike, mu: ArrayLike = 1) -> Array:
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
        >>> import astrodynx as adx
        >>> a = 1.0
        >>> mu = 1.0
        >>> adx.orb_period(a, mu)
        Array(6.2831855, dtype=float32, weak_type=True)

        With broadcasting, you can calculate the orbital period for multiple semimajor axes and gravitational parameters:

        >>> a = jnp.array([1.0, 2.0])
        >>> mu = jnp.array([1.0, 2.0])
        >>> adx.orb_period(a, mu)
        Array([ 6.2831855, 12.566371 ], dtype=float32)
    """
    return 2 * jnp.pi * jnp.sqrt(a**3 / mu)


def angular_momentum(pos_vec: ArrayLike, vel_vec: ArrayLike) -> Array:
    r"""
    Returns the specific angular momentum of a two-body system.

    Args:
        pos_vec: (..., 3) position vector of the object in the two-body system.
        vel_vec: (..., 3) velocity vector of the object in the two-body system, which shape broadcast-compatible with `pos_vec`.

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
        >>> import astrodynx as adx
        >>> pos_vec = jnp.array([1.0, 0.0, 0.0])
        >>> vel_vec = jnp.array([0.0, 1.0, 0.0])
        >>> adx.angular_momentum(pos_vec, vel_vec)
        Array([0., 0., 1.], dtype=float32)

        With broadcasting, you can calculate the specific angular momentum for multiple position and velocity vectors:

        >>> pos_vec = jnp.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        >>> vel_vec = jnp.array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
        >>> adx.angular_momentum(pos_vec, vel_vec)
        Array([[0., 0., 1.],
               [0., 0., 4.]], dtype=float32)
    """
    return jnp.cross(pos_vec, vel_vec)


def semimajor_axis(r: ArrayLike, v: ArrayLike, mu: ArrayLike = 1) -> ArrayLike:
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
        >>> import astrodynx as adx
        >>> r = 1.0
        >>> v = 1.0
        >>> mu = 1.0
        >>> adx.semimajor_axis(r, v, mu)
        1.0

        With broadcasting, you can calculate the semimajor axis for multiple position and velocity vectors:

        >>> r = jnp.array([1.0, 2.0])
        >>> v = jnp.array([1.0, 2.0])
        >>> mu = jnp.array([1.0, 2.0])
        >>> adx.semimajor_axis(r, v, mu)
        Array([ 1., -1.], dtype=float32)
    """
    return 1 / (2 / r - v**2 / mu)


def eccentricity_vector(
    pos_vec: ArrayLike, vel_vec: ArrayLike, mu: ArrayLike = 1
) -> Array:
    r"""
    Returns the eccentricity vector of a two-body orbit.

    Args:
        pos_vec: (..., 3) position vector of the object in the two-body system.
        vel_vec: (..., 3) velocity vector of the object in the two-body system, which shape broadcast-compatible with `pos_vec`.
        mu: Gravitational parameter of the central body; shape broadcast-compatible with `pos_vec` and `vel_vec`.

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
        >>> import astrodynx as adx
        >>> pos_vec = jnp.array([1.0, 0.0, 0.0])
        >>> vel_vec = jnp.array([0.0, 1.0, 0.0])
        >>> mu = 1.0
        >>> adx.eccentricity_vector(pos_vec, vel_vec, mu)
        Array([0., 0., 0.], dtype=float32)

        With broadcasting, you can calculate the eccentricity vector for multiple position and velocity vectors:

        >>> pos_vec = jnp.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        >>> vel_vec = jnp.array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
        >>> mu = jnp.array([[1.0],[2.0]])
        >>> adx.eccentricity_vector(pos_vec, vel_vec, mu)
        Array([[0., 0., 0.],
               [3., 0., 0.]], dtype=float32)
    """
    h = angular_momentum(pos_vec, vel_vec)
    return jnp.cross(vel_vec, h) / mu - pos_vec / jnp.linalg.norm(
        pos_vec, axis=-1, keepdims=True
    )


def semiparameter(h: ArrayLike, mu: ArrayLike = 1) -> ArrayLike:
    r"""
    Returns the semiparameter of a two-body orbit.

    Args:
        h: Norm of the specific angular momentum vector of the object in the two-body system.
        mu: Gravitational parameter of the central body; shape broadcast-compatible with `h`.

    Returns:
        The semiparameter of the orbit.

    Notes
        The semiparameter is calculated using equation (3.15):
        $$
        p = \frac{h^2}{\mu}
        $$
        where $p$ is the semiparameter, $h$ is the norm of the specific angular momentum vector, and $\mu$ is the gravitational parameter.

    References
        Battin, 1999, pp.116.

    Examples
        A simple example of calculating the semiparameter with a specific angular momentum norm of 1.0 and gravitational parameter of 1.0:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> h = 1.0
        >>> mu = 1.0
        >>> adx.semiparameter(h, mu)
        1.0

        With broadcasting, you can calculate the semiparameter for multiple specific angular momentum norms and gravitational parameters:

        >>> h = jnp.array([1.0, 2.0])
        >>> mu = jnp.array([1.0, 2.0])
        >>> adx.semiparameter(h, mu)
        Array([1., 2.], dtype=float32)
    """
    return h**2 / mu


def mean_motion(P: ArrayLike) -> ArrayLike:
    r"""
    Returns the mean motion of a two-body orbit.

    Args:
        P: Orbital period of the object in the two-body system.

    Returns:
        The mean motion of the orbit.

    Notes
        The mean motion is calculated using equation (3.24):
        $$
        n = \frac{2\pi}{P}
        $$
        where $n$ is the mean motion and $P$ is the orbital period.

    References
        Battin, 1999, pp.119.

    Examples
        A simple example of calculating the mean motion with an orbital period of 1.0:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> P = 1.0
        >>> adx.mean_motion(P)
        6.2831...

        With broadcasting, you can calculate the mean motion for multiple orbital periods:

        >>> P = jnp.array([1.0, 2.0])
        >>> adx.mean_motion(P)
        Array([6.2831..., 3.1415...], dtype=float32)
    """
    return 2 * jnp.pi / P


def equ_of_orbit(p: ArrayLike, e: ArrayLike, f: ArrayLike) -> Array:
    r"""
    Returns the equation of the orbit in the two-body system.

    Args:
        p: Semiparameter of the orbit.
        e: Eccentricity of the orbit; shape broadcast-compatible with `p`.
        f: True anomaly of the orbit; shape broadcast-compatible with `p` and `e`.

    Returns:
        The equation of the orbit.

    Notes
        The equation of the orbit is calculated using equation (3.20):
        $$
        r = \frac{p}{1 + e \cos(f)}
        $$
        where $r$ is the norm of the position vector, $p$ is the semiparameter, $e$ is the eccentricity, and $f$ is the true anomaly.

    References
        Battin, 1999, pp.117.

    Examples
        A simple example of calculating the equation of the orbit with a semiparameter of 1.0, eccentricity of 0.0, and true anomaly of 0.0:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> p = 1.0
        >>> e = 0.0
        >>> f = 0.0
        >>> adx.equ_of_orbit(p, e, f)
        Array(1., dtype=float32, weak_type=True)

        With broadcasting, you can calculate the equation of the orbit for multiple semiparameters, eccentricities, and true anomalies:

        >>> p = jnp.array([1.0, 2.0])
        >>> e = jnp.array([0.0, 0.0])
        >>> f = jnp.array([0.0, 0.0])
        >>> adx.equ_of_orbit(p, e, f)
        Array([1., 2.], dtype=float32)
    """
    return p / (1 + e * jnp.cos(f))


def radius_periapsis(p: ArrayLike, e: ArrayLike) -> ArrayLike:
    r"""
    Returns the radius of periapsis of the orbit.

    Args:
        p: Semiparameter of the orbit.
        e: Eccentricity of the orbit; shape broadcast-compatible with `p`.

    Returns:
        The radius of periapsis of the orbit.

    Notes
        The radius of periapsis is calculated using equation:
        $$
        r_p = \frac{p}{1 + e}
        $$
        where $r_p$ is the radius of periapsis, $p$ is the semiparameter, and $e$ is the eccentricity.

    References
        Battin, 1999, pp.117.

    Examples
        A simple example of calculating the radius of periapsis with a semiparameter of 1.0 and eccentricity of 0.0:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> p = 1.0
        >>> e = 0.0
        >>> adx.radius_periapsis(p, e)
        1.0

        With broadcasting, you can calculate the radius of periapsis for multiple semiparameters and eccentricities:

        >>> p = jnp.array([1.0, 2.0])
        >>> e = jnp.array([0.0, 0.0])
        >>> adx.radius_periapsis(p, e)
        Array([1., 2.], dtype=float32)
    """
    return p / (1 + e)


def radius_apoapsis(p: ArrayLike, e: ArrayLike) -> ArrayLike:
    r"""
    Returns the radius of apoapsis of the orbit.

    Args:
        p: Semiparameter of the orbit.
        e: Eccentricity of the orbit; shape broadcast-compatible with `p`.

    Returns:
        The radius of apoapsis of the orbit.

    Notes
        The radius of apoapsis is calculated using equation:
        $$
        r_a = \frac{p}{1 - e}
        $$
        where $r_a$ is the radius of apoapsis, $p$ is the semiparameter, and $e$ is the eccentricity.

    References
        Battin, 1999, pp.117.

    Examples
        A simple example of calculating the radius of apoapsis with a semiparameter of 1.0 and eccentricity of 0.0:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> p = 1.0
        >>> e = 0.0
        >>> adx.radius_apoapsis(p, e)
        1.0

        With broadcasting, you can calculate the radius of apoapsis for multiple semiparameters and eccentricities:

        >>> p = jnp.array([1.0, 2.0])
        >>> e = jnp.array([0.0, 0.0])
        >>> adx.radius_apoapsis(p, e)
        Array([1., 2.], dtype=float32)
    """
    return p / (1 - e)
