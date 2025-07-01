import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array
from astrodynx.twobody.ivp import U1, U2, U3, sigma_func
import astrodynx as adx


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


def keplerequ_uni(
    chi: ArrayLike, pos0_vec: ArrayLike, vel0_vec: ArrayLike, mu: ArrayLike
) -> Array:
    r"""Returns the flight time from $t_0$ to the moment when the generalized anomaly is $\chi$.

    Args:
        chi: The generalized anomaly.
        pos0_vec: (..., 3) The position vector at $t_0$.
        vel0_vec: (..., 3) The velocity vector at $t_0$.
        mu: The gravitational parameter.

    Returns:
        The flight time from $t_0$ to the moment when the generalized anomaly is $\chi$.

    Notes:
        The flight time from $t_0$ to the moment when the generalized anomaly is $\chi$ is calculated using the formula:
        $$
        \Delta t = \frac{r_0 U_1(\chi, \alpha) + \sigma_0 U_2(\chi, \alpha) + U_3(\chi, \alpha)}{\sqrt{\mu}}
        $$
        where $\Delta t = t - t_0$ is the flight time, $r_0$ is the norm of the position vector at $t_0$, $v_0$ is the norm of the velocity vector at $t_0$, $\mu$ is the gravitational parameter, $U_1$ is the universal function U1, $U_2$ is the universal function U2, $U_3$ is the universal function U3, and $\sigma_0$ is the sigma function at $t_0$.

    References:
        Battin, 1999, pp.178.

    Examples:
        A simple example of calculating the flight time for an orbit with a generalized anomaly of 1.0, a position vector of [1, 0, 0], a velocity vector of [0, 1, 0], and a gravitational parameter of 1.0:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> chi = 1.0
        >>> pos0_vec = jnp.array([1.0, 0.0, 0.0])
        >>> vel0_vec = jnp.array([0.0, 1.0, 0.0])
        >>> mu = 1.0
        >>> adx.keplerequ_uni(chi, pos0_vec, vel0_vec, mu)
        Array([1.], dtype=float32)

        With broadcasting, you can calculate the flight time for multiple generalized anomalies, position and velocity vectors, and gravitational parameters:

        >>> chi = jnp.array([[1.0], [1.0]])
        >>> pos0_vec = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        >>> vel0_vec = jnp.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        >>> mu = jnp.array([[1.0], [1.0]])
        >>> adx.keplerequ_uni(chi, pos0_vec, vel0_vec, mu)
        Array([[1.],
               [1.]], dtype=float32)
    """
    r0_norm = jnp.linalg.norm(pos0_vec, axis=-1, keepdims=True)
    alpha = 1.0 / adx.semimajor_axis(
        r0_norm, jnp.linalg.norm(vel0_vec, axis=-1, keepdims=True), mu
    )
    return (
        r0_norm * U1(chi, alpha)
        + sigma_func(pos0_vec, vel0_vec, mu) * U2(chi, alpha)
        + U3(chi, alpha)
    ) / jnp.sqrt(mu)


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
