import jax.numpy as jnp
from jax.typing import DTypeLike
from jax import Array


def mean_anomaly_equ_elps(
    eccentricity: DTypeLike, eccentric_anomaly: DTypeLike
) -> Array:
    r"""
    Returns the mean anomaly in the ecliptic frame.

    Parameters
    ----------
    eccentricity : DTypeLike
        Eccentricity of the orbit.
    eccentric_anomaly : DTypeLike
        Eccentric anomaly of the orbit.

    Returns
    -------
    out : Array
        Mean anomaly in the ecliptic frame.

    Notes
    -----
    The mean anomaly is calculated using the formula:
    $$
    M = E - e \sin(E)
    $$
    where $M$ is the mean anomaly, $E$ is the eccentric anomaly, and $e<1$ is the eccentricity.

    References
    ----------
    Battin, 1999, pp.160.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from astrodynx.twobody.kepler_equation import mean_anomaly_equ_elps
    >>> mean_anomaly_equ_elps(0.1, jnp.pi/4)
    Array(0.78539816)
    """
    return eccentric_anomaly - eccentricity * jnp.sin(eccentric_anomaly)


def mean_anomaly_equ_hypb(
    eccentricity: DTypeLike, hyperbolic_anomaly: DTypeLike
) -> Array:
    r"""
    Returns the mean anomaly in the hyperbolic frame.

    Parameters
    ----------
    eccentricity : DTypeLike
        Eccentricity of the orbit.
    hyperbolic_anomaly : DTypeLike
        Hyperbolic anomaly of the orbit.

    Returns
    -------
    out : Array
        Mean anomaly in the hyperbolic frame.

    Notes
    -----
    The mean anomaly is calculated using the formula:
    $$
    M_h = e \sinh(H) - H
    $$
    where $M_h$ is the mean anomaly, $H$ is the hyperbolic anomaly, and $e>1$ is the eccentricity.

    References
    ----------
    Battin, 1999, pp.168.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from astrodynx.twobody.kepler_equation import mean_anomaly_equ_hypb
    >>> mean_anomaly_equ_hypb(1.1, jnp.pi/4)
    Array(0.78539816)
    """
    return eccentricity * jnp.sinh(hyperbolic_anomaly) - hyperbolic_anomaly


def mean_anomaly_elps(t: DTypeLike, a: DTypeLike, mu: DTypeLike) -> Array:
    r"""
    Returns the mean anomaly in the ecliptic frame.

    Parameters
    ----------
    t : DTypeLike
        Time since periapsis passage.
    a : DTypeLike
        Semimajor axis of the orbit.
    mu : DTypeLike
        Standard gravitational parameter of the central body.

    Returns
    -------
    out : Array
        Mean anomaly in the ecliptic frame.

    Notes
    -----
    The mean anomaly is calculated using the formula:
    $$
    M = \sqrt{\frac{\mu}{a^3}} t
    $$
    where $M$ is the mean anomaly, $\mu$ is the standard gravitational parameter, and $a$ is the semimajor axis.

    References
    ----------
    Battin, 1999, pp.160.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from astrodynx.twobody.kepler_equation import mean_anomaly_elps
    >>> mean_anomaly_elps(1.0, 1.0, 1.0)
    Array(1.0)
    """
    return jnp.sqrt(mu / a**3) * t
