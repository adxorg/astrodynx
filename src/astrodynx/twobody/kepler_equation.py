import jax.numpy as jnp
from jax.typing import ArrayLike, DTypeLike
from jax import Array
from astrodynx.twobody.uniformulas import U1, U2, U3

"""Kepler's equations and generalized anomaly for two-body orbital mechanics."""


def kepler_equ_elps(E: ArrayLike, e: ArrayLike, M: ArrayLike = 0) -> Array:
    r"""Returns the Kepler's equation for elliptical orbits in the form f(E) = 0.

    Args:
        E: Eccentric anomaly.
        e: Eccentricity of the orbit, 0 <= e < 1.
        M: (optional) Mean anomaly.

    Returns:
        The value of Kepler's equation for elliptical orbits: E - e*sin(E) - M.

    Notes:
        Kepler's equation for elliptical orbits relates the eccentric anomaly E to the mean anomaly M:
        $$
        E - e \sin E = M
        $$
        This function returns the equation in the form f(E) = 0, which is useful for root-finding algorithms.

    References:
        Battin, 1999, pp.160.

    Examples:
        A simple example:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> E = jnp.pi/4
        >>> e = 0.1
        >>> M = 0.7
        >>> adx.kepler_equ_elps(E, e, M)
        Array(0.01468..., dtype=float32, weak_type=True)

        With broadcasting, you can calculate the Kepler's equation for multiple eccentric anomalies, eccentricities, and mean anomalies:

        >>> E = jnp.array([jnp.pi/4, jnp.pi/2])
        >>> e = jnp.array([0.1, 0.2])
        >>> M = jnp.array([0.7, 0.8])
        >>> adx.kepler_equ_elps(E, e, M)
        Array([0.01468..., 0.5707...], dtype=float32)
    """
    return E - e * jnp.sin(E) - M


def kepler_equ_hypb(H: ArrayLike, e: ArrayLike, N: ArrayLike = 0) -> Array:
    r"""Returns the Kepler's equation for hyperbolic orbits in the form f(H) = 0.

    Args:
        H: Hyperbolic eccentric anomaly.
        e: Eccentricity of the orbit, e > 1.
        N: (optional) Hyperbolic mean anomaly.

    Returns:
        The value of Kepler's equation for hyperbolic orbits: e*sinh(H) - H - N.

    Notes:
        Kepler's equation for hyperbolic orbits relates the hyperbolic eccentric anomaly H to the hyperbolic mean anomaly N:
        $$
        e \sinh H - H = N
        $$
        This function returns the equation in the form f(H) = 0, which is useful for root-finding algorithms.

    References:
        Battin, 1999, pp.168.

    Examples:
        A simple example:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> H = 1.0
        >>> e = 1.5
        >>> N = 1.0
        >>> adx.kepler_equ_hypb(H, e, N)
        Array(-0.2371..., dtype=float32, weak_type=True)

        With broadcasting, you can calculate the Kepler's equation for multiple hyperbolic eccentric anomalies, eccentricities, and hyperbolic mean anomalies:

        >>> H = jnp.array([1.0, 2.0])
        >>> e = jnp.array([1.5, 1.5])
        >>> N = jnp.array([1.0, 1.0])
        >>> adx.kepler_equ_hypb(H, e, N)
        Array([-0.2371...,  2.4402...], dtype=float32)
    """
    return e * jnp.sinh(H) - H - N


def mean_anomaly_elps(a: ArrayLike, deltat: ArrayLike, mu: ArrayLike = 1) -> Array:
    r"""Returns the mean anomaly for an elliptical orbit.

    Args:
        a: Semimajor axis of the orbit, a > 0.
        deltat: Time since periapsis passage.
        mu: (optional) Gravitational parameter of the central body.

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
        >>> adx.mean_anomaly_elps(a, deltat, mu)
        Array(1., dtype=float32, weak_type=True)

        With broadcasting, you can calculate the mean anomaly for multiple semimajor axes, gravitational parameters, and times since periapsis passage:

        >>> a = jnp.array([1.0, 2.0])
        >>> mu = jnp.array([1.0, 2.0])
        >>> deltat = jnp.array([1.0, 1.0])
        >>> adx.mean_anomaly_elps(a, deltat, mu)
        Array([1. , 0.5], dtype=float32)
    """
    return jnp.sqrt(mu / a**3) * deltat


def mean_anomaly_hypb(a: ArrayLike, deltat: ArrayLike, mu: ArrayLike = 1) -> Array:
    r"""Returns the mean anomaly for a hyperbolic orbit.

    Args:
        a: Semimajor axis of the orbit, a < 0.
        deltat: Time since periapsis passage.
        mu: (optional) Gravitational parameter of the central body.

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
        >>> adx.mean_anomaly_hypb(a, deltat, mu)
        Array(1., dtype=float32, weak_type=True)

        With broadcasting, you can calculate the mean anomaly for multiple semimajor axes, gravitational parameters, and times since periapsis passage:

        >>> a = jnp.array([-1.0, -2.0])
        >>> mu = jnp.array([1.0, 2.0])
        >>> deltat = jnp.array([1.0, 1.0])
        >>> adx.mean_anomaly_hypb(a, deltat, mu)
        Array([1. , 0.5], dtype=float32)
    """
    return jnp.sqrt(mu / -(a**3)) * deltat


def kepler_equ_uni(
    chi: ArrayLike,
    alpha: DTypeLike = 1,
    r0: ArrayLike = 1,
    sigma0: ArrayLike = 0,
    deltat: ArrayLike = 0,
    mu: ArrayLike = 1,
) -> Array:
    r"""Returns the universal Kepler's equation in the form f(chi) = 0.

    Args:
        chi: The generalized anomaly.
        alpha: (optional) The reciprocal of the semimajor axis.
        r0: (optional) The radius at the initial time.
        sigma0: (optional) The sigma function at the initial time.
        deltat: (optional) The time since the initial time.
        mu: (optional) The gravitational parameter.

    Returns:
        The value of the universal Kepler's equation.

    Notes:
        The universal Kepler's equation is defined as:
        $$
        r_0 U_1(\chi, \alpha) + \sigma_0 U_2(\chi, \alpha) + U_3(\chi, \alpha) - \sqrt{\mu} \Delta t = 0
        $$
        where $\chi$ is the generalized anomaly, $\alpha = \frac{1}{a}$ is the reciprocal of semimajor axis, $\sigma_0$ is the sigma function at the initial time, $r_0$ is the norm of the position vector at the initial time, $\mu$ is the gravitational parameter, $U_1$ is the universal function U1, $U_2$ is the universal function U2, $U_3$ is the universal function U3, and $\Delta t$ is the time since the initial time.

    References:
        Battin, 1999, pp.178.

    Examples:
        A simple example:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> chi = 1.0
        >>> alpha = 1.0
        >>> sigma0 = 0.0
        >>> r0 = 1.0
        >>> mu = 1.0
        >>> deltat = 1.0
        >>> adx.kepler_equ_uni(chi, alpha, r0, sigma0, deltat, mu)
        Array(0., dtype=float32, weak_type=True)

        With broadcasting:

        >>> chi = jnp.array([1.0, 2.0])
        >>> alpha = 1.
        >>> sigma0 = jnp.array([0.0, 0.0])
        >>> r0 = jnp.array([1.0, 1.0])
        >>> deltat = jnp.array([1.0, 1.0])
        >>> mu = jnp.array([1.0, 1.0])
        >>> adx.kepler_equ_uni(chi, alpha, r0, sigma0, deltat, mu)
        Array([0., 1.], dtype=float32)
    """
    return (
        r0 * U1(chi, alpha)
        + sigma0 * U2(chi, alpha)
        + U3(chi, alpha)
        - jnp.sqrt(mu) * deltat
    )


def generalized_anomaly(
    alpha: ArrayLike,
    sigma: ArrayLike,
    sigma0: ArrayLike,
    deltat: ArrayLike = 0,
    mu: ArrayLike = 1,
) -> Array:
    r"""Returns the generalized anomaly.

    Args:
        alpha: The reciprocal of the semimajor axis.
        sigma: The sigma function at the current time.
        sigma0: The sigma function at the initial time.
        deltat: (optional) The time since the initial time.
        mu: (optional) The gravitational parameter.

    Returns:
        The generalized anomaly.

    Notes:
        The generalized anomaly is defined as:
        $$
        \chi = \alpha \sqrt{\mu} \Delta t + \sigma - \sigma_0
        $$
        where $\chi$ is the generalized anomaly, $\alpha = \frac{1}{a}$ is the reciprocal of semimajor axis, $\sigma$ is the sigma function at the current time, $\sigma_0$ is the sigma function at the initial time, $\mu$ is the gravitational parameter, and $\Delta t$ is the time since the initial time.

    References:
        Battin, 1999, pp.179.

    Examples:
        A simple example:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> alpha = 1.0
        >>> sigma = 1.0
        >>> sigma0 = 0.0
        >>> mu = 1.0
        >>> deltat = 1.0
        >>> adx.generalized_anomaly(alpha, sigma, sigma0, deltat, mu)
        Array(2., dtype=float32, weak_type=True)

        With broadcasting:

        >>> alpha = jnp.array([1.0, 1.0])
        >>> sigma = jnp.array([1.0, 2.0])
        >>> sigma0 = jnp.array([0.0, 0.0])
        >>> mu = jnp.array([1.0, 1.0])
        >>> deltat = jnp.array([1.0, 1.0])
        >>> adx.generalized_anomaly(alpha, sigma, sigma0, deltat, mu)
        Array([2., 3.], dtype=float32)
    """
    return alpha * jnp.sqrt(mu) * deltat + sigma - sigma0
