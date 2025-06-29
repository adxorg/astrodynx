import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array


def mean_anomaly_equ_elps(e: ArrayLike, E: ArrayLike) -> Array:
    r"""Returns the mean anomaly for an elliptical orbit.

    Args:
        e: Eccentricity of the orbit; $e < 1$.
        E: Eccentric anomaly of the orbit; shape broadcast-compatible with `e`.

    Returns:
        The mean anomaly for an elliptical orbit.

    Notes:
        The mean anomaly is calculated using the formula:
        $$
        M = E - e \sin(E)
        $$
        where $M$ is the mean anomaly, $E$ is the eccentric anomaly, and $e < 1$ is the eccentricity.

    References:
        Battin, 1999, pp.160.

    Examples:
        A simple example of calculating the mean anomaly for an orbit with eccentricity 0.1 and eccentric anomaly π/4:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> e = 0.1
        >>> E = jnp.pi / 4
        >>> adx.mean_anomaly_equ_elps(e, E)
        Array(0.7146875, dtype=float32, weak_type=True)

        With broadcasting, you can calculate the mean anomaly for multiple eccentricities and eccentric anomalies:

        >>> e = jnp.array([0.1, 0.2])
        >>> E = jnp.array([jnp.pi / 4, jnp.pi / 3])
        >>> adx.mean_anomaly_equ_elps(e, E)
        Array([0.7146875, 0.8739925], dtype=float32)
    """
    return E - e * jnp.sin(E)


def mean_anomaly_equ_hypb(e: ArrayLike, H: ArrayLike) -> Array:
    r"""Returns the mean anomaly for a hyperbolic orbit.

    Args:
        e: Eccentricity of the orbit; $e > 1$.
        H: Hyperbolic eccentric anomaly of the orbit; shape broadcast-compatible with `e`.

    Returns:
        The mean anomaly for a hyperbolic orbit.

    Notes:
        The mean anomaly is calculated using the formula:
        $$
        N = e \sinh(H) - H
        $$
        where $N$ is the mean anomaly, $H$ is the hyperbolic eccentric anomaly, and $e > 1$ is the eccentricity.

    References:
        Battin, 1999, pp.168.

    Examples:
        A simple example of calculating the mean anomaly for an orbit with eccentricity 1.1 and hyperbolic eccentric anomaly 1.0:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> e = 1.1
        >>> H = 1.0
        >>> adx.mean_anomaly_equ_hypb(e, H)
        Array(0.29272127, dtype=float32, weak_type=True)

        With broadcasting, you can calculate the mean anomaly for multiple eccentricities and hyperbolic eccentric anomalies:

        >>> e = jnp.array([1.1, 1.2])
        >>> H = jnp.array([1.0, 2.0])
        >>> adx.mean_anomaly_equ_hypb(e, H)
        Array([0.29272127, 2.3522325 ], dtype=float32)
    """
    return e * jnp.sinh(H) - H
