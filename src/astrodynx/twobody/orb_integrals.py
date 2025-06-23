import jax.numpy as jnp
from jax.typing import ArrayLike, DTypeLike
from jax import Array


def ang_momentum_vec(r: ArrayLike, v: ArrayLike) -> Array:
    r"""
    Returns the angular momentum vector of a two-body system.

    Parameters
    ----------
    r : (3,) ArrayLike
        Position vector of the object.
    v : (3,) ArrayLike
        Velocity vector of the object.

    Returns
    -------
    out : (3,) Array
        Angular momentum vector of the object.

    Notes
    -----
    The angular momentum vector is calculated using the cross product of the position and velocity vectors:
    $$
    \mathbf{h} = \mathbf{r} \times \mathbf{v}
    $$
    where $\mathbf{h}$ is the angular momentum vector, $\mathbf{r}$ is the position vector, $\mathbf{v}$ is the velocity vector.

    References
    ----------
    Battin, 1999, pp.115.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from astrodynx.twobody.orb_integrals import ang_momentum
    >>> r = jnp.array([1.0, 0.0, 0.0])
    >>> v = jnp.array([0.0, 1.0, 0.0])
    >>> ang_momentum(r, v)
    Array([0., 0., 1.])
    """
    return jnp.cross(r, v)


def eccentricity_vec(r: ArrayLike, v: ArrayLike, mu: DTypeLike) -> Array:
    r"""
    Returns the eccentricity vector of a two-body system.

    Parameters
    ----------
    r : (3,) ArrayLike
        Position vector of the object.
    v : (3,) ArrayLike
        Velocity vector of the object.
    mu : DTypeLike
        Gravitational parameter of the central body.

    Returns
    -------
    out : (3,) Array
        Eccentricity vector of the object.

    Notes
    -----
    The eccentricity vector is calculated using the formula:
    $$
    \mathbf{e} = \frac{\mathbf{v} \times \mathbf{h}}{\mu} - \frac{\mathbf{r}}{r}
    $$
    where $\mathbf{e}$ is the eccentricity vector, $\mathbf{v}$ is the velocity vector, $\mathbf{h}$ is the angular momentum vector, $\mu$ is the gravitational parameter, and $\mathbf{r}$ is the position vector.

    The result is a 3D vector representing the eccentricity of the object's orbit in the two-body system.

    References
    ----------
    Battin, 1999, pp.115.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from astrodynx.twobody.orb_integrals import eccentricity_vec, ang_momentum_vec
    >>> r = jnp.array([1.0, 0.0, 0.0])
    >>> v = jnp.array([0.0, 1.0, 0.0])
    >>> mu = 1.0
    >>> h = ang_momentum_vec(r, v)
    >>> eccentricity_vec(r, v, mu)
    Array([-1.,  0.,  0.])
    """
    h = ang_momentum_vec(r, v)
    return (jnp.cross(v, h) / mu) - (r / jnp.linalg.norm(r))


def semiparameter(h: DTypeLike, mu: DTypeLike) -> DTypeLike:
    r"""
    Returns the semiparameter of a two-body system.

    Parameters
    ----------
    h : DTypeLike
        Magnitude of the angular momentum vector of the object.
    mu : DTypeLike
        Gravitational parameter of the central body.

    Returns
    -------
    out : DTypeLike
        Semiparameter of the object's orbit.

    Notes
    -----
    The semiparameter is calculated using the formula:
    $$
    p = \frac{h^2}{\mu}
    $$
    where $p$ is the semiparameter, $h$ is the magnitude of the angular momentum vector, and $\mu$ is the gravitational parameter.
    The result is a scalar value representing the semiparameter of the object's orbit in the two-body system.

    References
    ----------
    Battin, 1999, pp.116.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from astrodynx.twobody.orb_integrals import semiparameter
    >>> h = 1.0
    >>> mu = 1.0
    >>> semiparameter(h, mu)
    1.0
    """
    return h**2 / mu


def semimajor_axis(r: DTypeLike, v: DTypeLike, mu: DTypeLike) -> DTypeLike:
    r"""
    Returns the semimajor axis of a two-body system.

    Parameters
    ----------
    r : DTypeLike
        Magnitude of the position vector of the object.
    v : DTypeLike
        Magnitude of the velocity vector of the object.
    mu : DTypeLike
        Gravitational parameter of the central body.

    Returns
    -------
    out : DTypeLike
        Semimajor axis of the object's orbit.

    Notes
    -----
    The semimajor axis is calculated using the formula:
    $$
    a = \left(\frac{2}{r} - \frac{v^2}{\mu}\right)^{-1}
    $$
    where $a$ is the semimajor axis, $r$ is the magnitude of the position vector, $v$ is the magnitude of the velocity vector, and $\mu$ is the gravitational parameter.
    The result is a scalar value representing the semimajor axis of the object's orbit in the two-body system.

    References
    ----------
    Battin, 1999, pp.116.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from astrodynx.twobody.orb_integrals import semimajor_axis
    >>> r = 1.0
    >>> v = 1.0
    >>> mu = 1.0
    >>> semimajor_axis(r, v, mu)
    1.0
    """
    return 1 / (2 / r - (v**2) / mu)


def equation_of_orbit(p: DTypeLike, e: DTypeLike, f: DTypeLike) -> DTypeLike:
    r"""
    Returns the equation of the orbit for a two-body system.

    Parameters
    ----------
    p : DTypeLike
        Semiparameter of the object's orbit.
    e : DTypeLike
        Eccentricity of the object's orbit.
    f : DTypeLike
        True anomaly of the object's orbit.

    Returns
    -------
    out : DTypeLike
        radius.

    Notes
    -----
    The equation of the orbit is calculated using the formula:
    $$
    r = \frac{p}{1 + e \cos(f)}
    $$
    where $r$ is the distance from the central body, $p$ is the semiparameter, $e$ is the eccentricity, and $f$ is the true anomaly.

    References
    ----------
    Battin, 1999, pp.116.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from astrodynx.twobody.orb_integrals import equation_of_orbit
    >>> p = 1.0
    >>> e = 0.5
    >>> f = jnp.pi / 4  # 45 degrees in radians
    >>> equation_of_orbit(p, e, f)
    0.7071067811865476
    """
    return p / (1 + e * jnp.cos(f))


def orb_period(a: DTypeLike, mu: DTypeLike) -> DTypeLike:
    r"""
    Returns the period of elliptic motion.

    Parameters
    ----------
    a : DTypeLike
        Semimajor axis of the object's orbit.
    mu : DTypeLike
        Gravitational parameter of the central body.

    Returns
    -------
    out : DTypeLike
        Period of the object's orbit.

    Notes
    -----
    The period is calculated using the formula:
    $$
    P = 2\pi \sqrt{\frac{a^3}{\mu}}
    $$
    where $P$ is the period, $a$ is the semimajor axis, and $\mu$ is the gravitational parameter.

    References
    ----------
    Battin, 1999, pp.119.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from astrodynx.twobody.orb_integrals import orb_period
    >>> a = 1.0
    >>> mu = 1.0
    >>> orb_period(a, mu)
    6.283185307179586
    """
    return 2 * jnp.pi * jnp.sqrt(a**3 / mu)
