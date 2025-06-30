import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array


def keplerequ_elps(E: ArrayLike, e: ArrayLike, a: ArrayLike, mu: ArrayLike) -> Array:
    r"""Returns the flight time from periapsis for an elliptical orbit.

    Args:
        E: Eccentric anomaly of the orbit.
        e: Eccentricity of the orbit; $e < 1$.
        a: Semimajor axis of the orbit, a > 0.
        mu: Gravitational parameter of the central body.

    Returns:
        The flight time from periapsis for an elliptical orbit.

    Notes:
        The flight time from periapsis for an elliptical orbit is calculated using the formula:
        $$
        \Delta t = (E - e \sin E) \sqrt{\frac{a^3}{\mu}}
        $$
        where $\Delta t$ is the flight time, $E$ is the eccentric anomaly, $e$ is the eccentricity, $a$ is the semimajor axis, and $\mu$ is the gravitational parameter.

    References:
        Battin, 1999, pp.160.

    Examples:
        A simple example of calculating the flight time for an orbit with eccentricity 0.1, eccentric anomaly π/4, semimajor axis 1.0, and gravitational parameter 1.0:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> e = 0.1
        >>> E = jnp.pi / 4
        >>> a = 1.0
        >>> mu = 1.0
        >>> adx.keplerequ_elps(E, e, a, mu)
        Array(0.7146875, dtype=float32, weak_type=True)

        With broadcasting, you can calculate the flight time for multiple eccentricities, eccentric anomalies, semimajor axes, and gravitational parameters:

        >>> e = jnp.array([0.1, 0.2])
        >>> E = jnp.array([jnp.pi / 4, jnp.pi / 3])
        >>> a = jnp.array([1.0, 2.0])
        >>> mu = jnp.array([1.0, 2.0])
        >>> adx.keplerequ_elps(E, e, a, mu)
        Array([0.7146875, 1.747985 ], dtype=float32)
    """
    return (E - e * jnp.sin(E)) * jnp.sqrt(a**3 / mu)


def keplerequ_hypb(H: ArrayLike, e: ArrayLike, a: ArrayLike, mu: ArrayLike) -> Array:
    r"""Returns the flight time from periapsis for a hyperbolic orbit.

    Args:
        H: Hyperbolic eccentric anomaly of the orbit.
        e: Eccentricity of the orbit; $e > 1$.
        a: Semimajor axis of the orbit, a < 0.
        mu: Gravitational parameter of the central body.

    Returns:
        The flight time from periapsis for a hyperbolic orbit.

    Notes:
        The flight time from periapsis for a hyperbolic orbit is calculated using the formula:
        $$
        \Delta t = (e \sinh H - H) \sqrt{\frac{-a^3}{\mu}}
        $$
        where $\Delta t$ is the flight time, $H$ is the hyperbolic eccentric anomaly, $e > 1$ is the eccentricity, $a < 0$ is the semimajor axis, and $\mu$ is the gravitational parameter.

    References:
        Battin, 1999, pp.166.

    Examples:
        A simple example of calculating the flight time for an orbit with eccentricity 1.1, hyperbolic eccentric anomaly 1.0, semimajor axis -1.0, and gravitational parameter 1.0:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> e = 1.1
        >>> H = 1.0
        >>> a = -1.0
        >>> mu = 1.0
        >>> adx.keplerequ_hypb(H, e, a, mu)
        Array(0.29272127, dtype=float32, weak_type=True)

        With broadcasting, you can calculate the flight time for multiple eccentricities, hyperbolic eccentric anomalies, semimajor axes, and gravitational parameters:

        >>> e = jnp.array([1.1, 1.2])
        >>> H = jnp.array([1.0, 1.0])
        >>> a = jnp.array([-1.0, -2.0])
        >>> mu = jnp.array([1.0, 2.0])
        >>> adx.keplerequ_hypb(H, e, a, mu)
        Array([0.29272127, 0.82048297], dtype=float32)
    """
    return (e * jnp.sinh(H) - H) * jnp.sqrt(-(a**3) / mu)


def mean_anomaly_elps(a: ArrayLike, mu: ArrayLike, deltat: ArrayLike) -> Array:
    r"""Returns the mean anomaly for an elliptical orbit.

    Args:
        a: Semimajor axis of the orbit, a > 0.
        mu: Gravitational parameter of the central body.
        deltat: Time since periapsis passage.

    Returns:
        The mean anomaly for an elliptical orbit.

    Notes:
        The mean anomaly for an elliptical orbit is calculated using the formula:
        $$
        M = \sqrt{\frac{\mu}{a^3}} \Delta t
        $$
        where $M$ is the mean anomaly, $a>0$ is the semimajor axis, $\mu$ is the gravitational parameter, and $\Delta t$ is the time since periapsis passage.

    References:
        Battin, 1999, pp.160.

    Examples:
        A simple example of calculating the mean anomaly for an orbit with semimajor axis 1.0, gravitational parameter 1.0, and time since periapsis passage 1.0:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> a = 1.0
        >>> mu = 1.0
        >>> deltat = 1.0
        >>> adx.mean_anomaly_elps(a, mu, deltat)
        Array(1., dtype=float32, weak_type=True)

        With broadcasting, you can calculate the mean anomaly for multiple semimajor axes, gravitational parameters, and times since periapsis passage:

        >>> a = jnp.array([1.0, 2.0])
        >>> mu = jnp.array([1.0, 2.0])
        >>> deltat = jnp.array([1.0, 1.0])
        >>> adx.mean_anomaly_elps(a, mu, deltat)
        Array([1. , 0.5], dtype=float32)
    """
    return jnp.sqrt(mu / a**3) * deltat


def mean_anomaly_hypb(a: ArrayLike, mu: ArrayLike, deltat: ArrayLike) -> Array:
    r"""Returns the mean anomaly for a hyperbolic orbit.

    Args:
        a: Semimajor axis of the orbit, a < 0.
        mu: Gravitational parameter of the central body.
        deltat: Time since periapsis passage.

    Returns:
        The mean anomaly for a hyperbolic orbit.

    Notes:
        The mean anomaly for a hyperbolic orbit is calculated using the formula:
        $$
        N = \sqrt{\frac{\mu}{-a^3}} \Delta t
        $$
        where $N$ is the mean anomaly, $a<0$ is the semimajor axis, $\mu$ is the gravitational parameter, and $\Delta t$ is the time since periapsis passage.

    References:
        Battin, 1999, pp.166.

    Examples:
        A simple example of calculating the mean anomaly for an orbit with semimajor axis -1.0, gravitational parameter 1.0, and time since periapsis passage 1.0:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> a = -1.0
        >>> mu = 1.0
        >>> deltat = 1.0
        >>> adx.mean_anomaly_hypb(a, mu, deltat)
        Array(1., dtype=float32, weak_type=True)

        With broadcasting, you can calculate the mean anomaly for multiple semimajor axes, gravitational parameters, and times since periapsis passage:

        >>> a = jnp.array([-1.0, -2.0])
        >>> mu = jnp.array([1.0, 2.0])
        >>> deltat = jnp.array([1.0, 1.0])
        >>> adx.mean_anomaly_hypb(a, mu, deltat)
        Array([1. , 0.5], dtype=float32)
    """
    return jnp.sqrt(mu / -(a**3)) * deltat
