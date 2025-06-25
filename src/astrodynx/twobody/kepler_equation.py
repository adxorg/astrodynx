import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array


def mean_anomaly_equ_elps(e: ArrayLike, E: ArrayLike) -> Array:
    r"""Returns the mean anomaly in the ecliptic frame.

    Args:
        e: Eccentricity of the orbit.
        E: Eccentric anomaly of the orbit; shape broadcast-compatible with `e`.

    Returns:
        The mean anomaly in the ecliptic frame.

    Notes:
        The mean anomaly is calculated using the formula:
        $$
        M = E - e \sin(E)
        $$
        where $M$ is the mean anomaly, $E$ is the eccentric anomaly, and $e$ is the eccentricity.

    References:
        Battin, 1999, pp.160.

    Examples:
        A simple example of calculating the mean anomaly for an orbit with eccentricity 0.1 and eccentric anomaly π/4:

        >>> import jax.numpy as jnp
        >>> from astrodynx.twobody import mean_anomaly_equ_elps
        >>> e = 0.1
        >>> E = jnp.pi / 4
        >>> mean_anomaly_equ_elps(e, E)
        Array(0.7146875, dtype=float32, weak_type=True)

        With broadcasting, you can calculate the mean anomaly for multiple eccentricities and eccentric anomalies:

        >>> e = jnp.array([0.1, 0.2])
        >>> E = jnp.array([jnp.pi / 4, jnp.pi / 3])
        >>> mean_anomaly_equ_elps(e, E)
        Array([0.7146875, 0.8739925], dtype=float32)
    """
    return E - e * jnp.sin(E)
