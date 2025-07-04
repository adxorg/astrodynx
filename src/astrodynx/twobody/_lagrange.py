import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array

"""Lagrange coefficients for two-body orbital mechanics."""


def lagrange_F(U2: ArrayLike, r0_mag: ArrayLike = 1) -> ArrayLike:
    r"""The Lagrange F function

    Args:
        U2: The universal function U2.
        r0_mag: The radius at the initial time.

    Returns:
        The value of the Lagrange F function.

    Notes:
        The Lagrange F function is defined as:
        $$
        F = 1 - \frac{U_2}{r_0}
        $$
        where $U_2$ is the universal function U2 and $r_0$ is the radius at the initial time.

    References:
        Battin, 1999, pp.179.

    Examples:
        A simple example:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> U2 = 1.0
        >>> r0 = 1.0
        >>> adx.twobody.lagrange_F(U2, r0)
        0.0

        With broadcasting:

        >>> U2 = jnp.array([1.0, 2.0])
        >>> r0 = jnp.array([1.0, 1.0])
        >>> adx.twobody.lagrange_F(U2, r0)
        Array([ 0., -1.], dtype=float32)
    """
    return 1 - U2 / r0_mag


def lagrange_G(
    U1: ArrayLike,
    U2: ArrayLike,
    sigma0: ArrayLike = 0,
    r0_mag: ArrayLike = 1,
    mu: ArrayLike = 1,
) -> Array:
    r"""The Lagrange G function

    Args:
        U1: The universal function U1.
        U2: The universal function U2.
        sigma0: The sigma function at the initial time.
        r0_mag: The radius at the initial time.
        mu: The gravitational parameter.

    Returns:
        The value of the Lagrange G function.

    Notes:
        The Lagrange G function is defined as:
        $$
        G = \frac{r_0 U_1 + \sigma_0 U_2}{\sqrt{\mu}}
        $$
        where $U_1$ is the universal function U1, $U_2$ is the universal function U2, $\sigma_0$ is the sigma function at the initial time, $r_0$ is the radius at the initial time, and $\mu$ is the gravitational parameter.

    References:
        Battin, 1999, pp.179.

    Examples:
        A simple example:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> U1 = 1.0
        >>> U2 = 1.0
        >>> sigma0 = 0.0
        >>> r0 = 1.0
        >>> mu = 1.0
        >>> adx.twobody.lagrange_G(U1, U2, sigma0, r0, mu)
        Array(1., dtype=float32, weak_type=True)

        With broadcasting:

        >>> U1 = jnp.array([1.0, 2.0])
        >>> U2 = jnp.array([1.0, 1.0])
        >>> sigma0 = jnp.array([0.0, 0.0])
        >>> r0 = jnp.array([1.0, 1.0])
        >>> mu = jnp.array([1.0, 1.0])
        >>> adx.twobody.lagrange_G(U1, U2, sigma0, r0, mu)
        Array([1., 2.], dtype=float32)
    """
    return (r0_mag * U1 + sigma0 * U2) / jnp.sqrt(mu)


def lagrange_Ft(
    U1: ArrayLike, r_mag: ArrayLike, r0_mag: ArrayLike = 1, mu: ArrayLike = 1
) -> Array:
    r"""The Lagrange Ft function

    Args:
        U1: The universal function U1.
        r_mag: The radius at the current time.
        r0_mag: The radius at the initial time.
        mu: The gravitational parameter.

    Returns:
        The value of the Lagrange Ft function.

    Notes:
        The Lagrange Ft function is defined as:
        $$
        F_t = -\frac{\sqrt{\mu}}{r r_0} U_1
        $$
        where $U_1$ is the universal function U1, $r$ is the radius at the current time, $r_0$ is the radius at the initial time, and $\mu$ is the gravitational parameter.

    References:
        Battin, 1999, pp.179.

    Examples:
        A simple example:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> U1 = 1.0
        >>> r = 1.0
        >>> r0 = 1.0
        >>> mu = 1.0
        >>> adx.twobody.lagrange_Ft(U1, r, r0, mu)
        Array(-1., dtype=float32, weak_type=True)

        With broadcasting:

        >>> U1 = jnp.array([1.0, 2.0])
        >>> r = jnp.array([1.0, 1.0])
        >>> r0 = jnp.array([1.0, 1.0])
        >>> mu = jnp.array([1.0, 1.0])
        >>> adx.twobody.lagrange_Ft(U1, r, r0, mu)
        Array([-1., -2.], dtype=float32)
    """
    return -jnp.sqrt(mu) / r_mag * U1 / r0_mag


def lagrange_Gt(U2: ArrayLike, r_mag: ArrayLike) -> ArrayLike:
    r"""The Lagrange Gt function

    Args:
        U2: The universal function U2.
        r_mag: The radius at the current time.

    Returns:
        The value of the Lagrange Gt function.

    Notes:
        The Lagrange Gt function is defined as:
        $$
        G_t = 1 - \frac{U_2}{r}
        $$
        where $U_2$ is the universal function U2 and $r$ is the radius at the current time.

    References:
        Battin, 1999, pp.179.

    Examples:
        A simple example:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> U2 = 1.0
        >>> r = 1.0
        >>> adx.twobody.lagrange_Gt(U2, r)
        0.0

        With broadcasting:

        >>> U2 = jnp.array([1.0, 2.0])
        >>> r = jnp.array([1.0, 1.0])
        >>> adx.twobody.lagrange_Gt(U2, r)
        Array([ 0., -1.], dtype=float32)
    """
    return 1 - U2 / r_mag
