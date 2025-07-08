import jax.numpy as jnp
import jax
from jax.typing import ArrayLike, DTypeLike
from jax import Array

"""Universal functions for two-body orbital mechanics."""


def sigma_fn(pos_vec: ArrayLike, vel_vec: ArrayLike, mu: ArrayLike = 1) -> Array:
    r"""The sigma function

    Args:
        pos_vec: (...,3) The position vector.
        vel_vec: (...,3) The velocity vector.
        mu: The gravitational parameter.

    Returns:
        The value of the sigma function.

    Notes:
        The sigma function is defined as:
        $$
        \sigma = \frac{\boldsymbol{r} \cdot \boldsymbol{v}}{\sqrt{\mu}}
        $$
        where $\boldsymbol{r}$ is the position vector, $\boldsymbol{v}$ is the velocity vector, and $\mu$ is the gravitational parameter.

    References:
        Battin, 1999, pp.174.

    Examples:
        A simple example of calculating the sigma function with a position vector of [1, 0, 0], a velocity vector of [0, 1, 0], and a gravitational parameter of 1.0:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> pos_vec = jnp.array([1.0, 0.0, 0.0])
        >>> vel_vec = jnp.array([0.0, 1.0, 0.0])
        >>> mu = 1.0
        >>> adx.twobody.sigma_fn(pos_vec, vel_vec, mu)
        Array([0.], dtype=float32)

        With broadcasting, you can calculate the sigma function for multiple position and velocity vectors:

        >>> pos_vec = jnp.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        >>> vel_vec = jnp.array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
        >>> mu = jnp.array([[1.0], [2.0]])
        >>> adx.twobody.sigma_fn(pos_vec, vel_vec, mu)
        Array([[0.],
               [0.]], dtype=float32)
    """
    return jnp.sum(pos_vec * vel_vec, axis=-1, keepdims=True) / jnp.sqrt(mu)


def ufunc0(chi: ArrayLike, alpha: DTypeLike) -> Array:
    r"""The universal function U0

    Args:
        chi: The generalized anomaly.
        alpha: The reciprocal of the semimajor axis.

    Returns:
        The value of the universal function U0.

    Notes:
        The universal function U0 is defined as:
        $$
        U_0(\chi, \alpha) = \begin{cases}
        1 & \alpha = 0 \\
        \cos(\sqrt{\alpha} \chi) & \alpha > 0 \\
        \cosh(\sqrt{-\alpha} \chi) & \alpha < 0
        \end{cases}
        $$
        where $\chi$ is the generalized anomaly and $\alpha = \frac{1}{a}$ is the reciprocal of semimajor axis.

    References:
        Battin, 1999, pp.180.

    Examples:
        A simple example:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> chi = 1.0
        >>> alpha = 1.0
        >>> adx.twobody.ufunc0(chi, alpha)
        Array(0.5403..., dtype=float32, weak_type=True)

        With broadcasting:

        >>> chi = jnp.array([1.0, 2.0])
        >>> alpha = 1.0
        >>> adx.twobody.ufunc0(chi, alpha)
        Array([ 0.5403..., -0.4161...], dtype=float32)
    """
    return jax.lax.cond(
        alpha > 0,
        lambda: jnp.cos(jnp.sqrt(alpha) * chi),
        lambda: jax.lax.cond(
            alpha < 0,
            lambda: jnp.cosh(jnp.sqrt(-alpha) * chi),
            lambda: jnp.ones_like(chi),
        ),
    )


def ufunc1(chi: ArrayLike, alpha: DTypeLike) -> Array:
    r"""The universal function U1

    Args:
        chi: The generalized anomaly.
        alpha: The reciprocal of the semimajor axis.

    Returns:
        The value of the universal function U1.

    Notes:
        The universal function U1 is defined as:
        $$
        U_1(\chi, \alpha) = \begin{cases}
        \frac{\sin(\sqrt{\alpha} \chi)}{\sqrt{\alpha}} & \alpha > 0 \\
        \frac{\sinh(\sqrt{-\alpha} \chi)}{\sqrt{-\alpha}} & \alpha < 0 \\
        \chi & \alpha = 0
        \end{cases}
        $$
        where $\chi$ is the generalized anomaly and $\alpha = \frac{1}{a}$ is the reciprocal of semimajor axis.

    References:
        Battin, 1999, pp.180.

    Examples:
        A simple example:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> chi = 1.0
        >>> alpha = 1.0
        >>> adx.twobody.ufunc1(chi, alpha)
        Array(0.8414..., dtype=float32, weak_type=True)

        With broadcasting:

        >>> chi = jnp.array([1.0, 2.0])
        >>> alpha = 1.0
        >>> adx.twobody.ufunc1(chi, alpha)
        Array([0.8414..., 0.9092...], dtype=float32)
    """
    return jax.lax.cond(
        alpha > 0,
        lambda: jnp.sin(jnp.sqrt(alpha) * chi) / jnp.sqrt(alpha),
        lambda: jax.lax.cond(
            alpha < 0,
            lambda: jnp.sinh(jnp.sqrt(-alpha) * chi) / jnp.sqrt(-alpha),
            lambda: chi,
        ),
    )


def ufunc2(chi: ArrayLike, alpha: DTypeLike) -> Array:
    r"""The universal function U2

    Args:
        chi: The generalized anomaly.
        alpha: The reciprocal of the semimajor axis.

    Returns:
        The value of the universal function U2.

    Notes:
        The universal function U2 is defined as:
        $$
        U_2(\chi, \alpha) = \begin{cases}
        \frac{1 - \cos(\sqrt{\alpha} \chi)}{\alpha} & \alpha > 0 \\
        \frac{\cosh(\sqrt{-\alpha} \chi) - 1}{-\alpha} & \alpha < 0 \\
        \frac{\chi^2}{2} & \alpha = 0
        \end{cases}
        $$
        where $\chi$ is the generalized anomaly and $\alpha = \frac{1}{a}$ is the reciprocal of semimajor axis.

    References:
        Battin, 1999, pp.180.

    Examples:
        A simple example:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> chi = 1.0
        >>> alpha = 1.0
        >>> adx.twobody.ufunc2(chi, alpha)
        Array(0.4596..., dtype=float32, weak_type=True)

        With broadcasting:

        >>> chi = jnp.array([1.0, 2.0])
        >>> alpha = 1.0
        >>> adx.twobody.ufunc2(chi, alpha)
        Array([0.4596..., 1.4161...], dtype=float32)
    """
    return jax.lax.cond(
        alpha > 0,
        lambda: (1 - jnp.cos(jnp.sqrt(alpha) * chi)) / alpha,
        lambda: jax.lax.cond(
            alpha < 0,
            lambda: (jnp.cosh(jnp.sqrt(-alpha) * chi) - 1) / -alpha,
            lambda: chi**2 / 2,
        ),
    )


def ufunc3(chi: ArrayLike, alpha: DTypeLike) -> Array:
    r"""The universal function U3

    Args:
        chi: The generalized anomaly.
        alpha: The reciprocal of the semimajor axis.

    Returns:
        The value of the universal function U3.

    Notes:
        The universal function U3 is defined as:
        $$
        U_3(\chi, \alpha) = \begin{cases}
        \frac{\sqrt{\alpha} \chi - \sin(\sqrt{\alpha} \chi)}{\alpha \sqrt{\alpha}} & \alpha > 0 \\
        \frac{\sqrt{-\alpha} \chi - \sinh(\sqrt{-\alpha} \chi)}{\alpha \sqrt{-\alpha}} & \alpha < 0 \\
        \frac{\chi^3}{6} & \alpha = 0
        \end{cases}
        $$
        where $\chi$ is the generalized anomaly and $\alpha = \frac{1}{a}$ is the reciprocal of semimajor axis.

    References:
        Battin, 1999, pp.180.

    Examples:
        A simple example:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> chi = 1.0
        >>> alpha = 1.0
        >>> adx.twobody.ufunc3(chi, alpha)
        Array(0.1585..., dtype=float32, weak_type=True)

        With broadcasting:

        >>> chi = jnp.array([1.0, 2.0])
        >>> alpha = 1.0
        >>> adx.twobody.ufunc3(chi, alpha)
        Array([0.1585..., 1.0907...], dtype=float32)
    """
    return jax.lax.cond(
        alpha > 0,
        lambda: (jnp.sqrt(alpha) * chi - jnp.sin(jnp.sqrt(alpha) * chi))
        / alpha
        / jnp.sqrt(alpha),
        lambda: jax.lax.cond(
            alpha < 0,
            lambda: (jnp.sqrt(-alpha) * chi - jnp.sinh(jnp.sqrt(-alpha) * chi))
            / alpha
            / jnp.sqrt(-alpha),
            lambda: chi**3 / 6,
        ),
    )


def ufunc4(chi: ArrayLike, alpha: DTypeLike) -> Array:
    r"""The universal function U4

    Args:
        chi: The generalized anomaly.
        alpha: The reciprocal of the semimajor axis.

    Returns:
        The value of the universal function U4.

    Notes:
        The universal function U4 is defined as:
        $$
        U_4(\chi, \alpha) = \begin{cases}
        \frac{\alpha \chi^2 - 2 + 2 \cos(\sqrt{\alpha} \chi)}{2 \alpha^2} & \alpha > 0 \\
        \frac{\alpha \chi^2 - 2 + 2 \cosh(\sqrt{-\alpha} \chi)}{2 \alpha^2} & \alpha < 0 \\
        \frac{\chi^4}{24} & \alpha = 0
        \end{cases}
        $$
        where $\chi$ is the generalized anomaly and $\alpha = \frac{1}{a}$ is the reciprocal of semimajor axis.

    References:
        Battin, 1999, pp.183.

    Examples:
        A simple example:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> chi = 1.0
        >>> alpha = 1.0
        >>> adx.twobody.ufunc4(chi, alpha)
        Array(0.0403..., dtype=float32, weak_type=True)

        With broadcasting:

        >>> chi = jnp.array([1.0, 2.0])
        >>> alpha = 1.0
        >>> adx.twobody.ufunc4(chi, alpha)
        Array([0.0403..., 0.5838...], dtype=float32)
    """
    return jax.lax.cond(
        alpha > 0,
        lambda: (alpha * chi**2 - 2 + 2 * jnp.cos(jnp.sqrt(alpha) * chi))
        / (2 * alpha**2),
        lambda: jax.lax.cond(
            alpha < 0,
            lambda: (alpha * chi**2 - 2 + 2 * jnp.cosh(jnp.sqrt(-alpha) * chi))
            / (2 * alpha**2),
            lambda: chi**4 / 24,
        ),
    )


def ufunc5(chi: ArrayLike, alpha: DTypeLike) -> Array:
    r"""The universal function U5

    Args:
        chi: The generalized anomaly.
        alpha: The reciprocal of the semimajor axis.

    Returns:
        The value of the universal function U5.

    Notes:
        The universal function U5 is defined as:
        $$
        U_5(\chi, \alpha) = \begin{cases}
        \frac{\alpha^2 \chi^3 - 6\alpha \chi + 6\sqrt{\alpha} \sin(\sqrt{\alpha} \chi)}{6\alpha^3} & \alpha > 0 \\
        \frac{\alpha^2 \chi^3 - 6\alpha \chi - 6\sqrt{-\alpha} \sinh(\sqrt{-\alpha} \chi)}{6\alpha^3} & \alpha < 0 \\
        \frac{\chi^5}{120} & \alpha = 0
        \end{cases}
        $$
        where $\chi$ is the generalized anomaly and $\alpha = \frac{1}{a}$ is the reciprocal of semimajor axis.

    References:
        Battin, 1999, pp.183.

    Examples:
        A simple example:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> chi = 1.0
        >>> alpha = 1.0
        >>> adx.twobody.ufunc5(chi, alpha)
        Array(0.0081..., dtype=float32, weak_type=True)

        With broadcasting:

        >>> chi = jnp.array([2.0, 3.0])
        >>> alpha = 1.0
        >>> adx.twobody.ufunc5(chi, alpha)
        Array([0.2426..., 1.6411...], dtype=float32)
    """
    return jax.lax.cond(
        alpha > 0,
        lambda: (
            alpha**2 * chi**3
            - 6 * alpha * chi
            + 6 * jnp.sqrt(alpha) * jnp.sin(jnp.sqrt(alpha) * chi)
        )
        / (6 * alpha**3),
        lambda: jax.lax.cond(
            alpha < 0,
            lambda: (
                alpha**2 * chi**3
                - 6 * alpha * chi
                - 6 * jnp.sqrt(-alpha) * jnp.sinh(jnp.sqrt(-alpha) * chi)
            )
            / (6 * alpha**3),
            lambda: chi**5 / 120,
        ),
    )
