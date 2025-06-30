import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array


def sigma_func(r: ArrayLike, v: ArrayLike, mu: ArrayLike) -> Array:
    r"""The sigma function

    Args:
        r: (...,3) The position vector.
        v: (...,3) The velocity vector.
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
        >>> r = jnp.array([1.0, 0.0, 0.0])
        >>> v = jnp.array([0.0, 1.0, 0.0])
        >>> mu = 1.0
        >>> adx.twobody.ivp.sigma_func(r, v, mu)
        Array(0., dtype=float32)

        With broadcasting, you can calculate the sigma function for multiple position and velocity vectors:

        >>> r = jnp.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        >>> v = jnp.array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
        >>> mu = jnp.array([1.0, 2.0])
        >>> adx.twobody.ivp.sigma_func(r, v, mu)
        Array([0., 0.], dtype=float32)
    """
    return jnp.sum(r * v, axis=-1) / jnp.sqrt(mu)


def U0(chi: ArrayLike, alpha: ArrayLike) -> Array:
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
        A simple example of calculating the universal function U0 with a argument of 1.0 and a parameter of 1.0:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> chi = 1.0
        >>> alpha = 1.0
        >>> adx.twobody.ivp.U0(chi, alpha)
        Array(0.5403..., dtype=float32, weak_type=True)

        With broadcasting, you can calculate the universal function U0 for multiple arguments and parameters:

        >>> chi = jnp.array([1.0, 2.0])
        >>> alpha = jnp.array([1.0, -2.0])
        >>> adx.twobody.ivp.U0(chi, alpha)
        Array([0.5403..., 8.4889...], dtype=float32)
    """
    conds = [
        alpha > 0,
        alpha < 0,
    ]
    choices = [
        jnp.cos(jnp.sqrt(alpha) * chi),
        jnp.cosh(jnp.sqrt(-alpha) * chi),
    ]
    return jnp.select(conds, choices, default=1.0)


def U1(chi: ArrayLike, alpha: ArrayLike) -> Array:
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
        \chi & \alpha = 0 \\
        \frac{\sin(\sqrt{\alpha} \chi)}{\sqrt{\alpha}} & \alpha > 0 \\
        \frac{\sinh(\sqrt{-\alpha} \chi)}{\sqrt{-\alpha}} & \alpha < 0
        \end{cases}
        $$
        where $\chi$ is the generalized anomaly and $\alpha = \frac{1}{a}$ is the reciprocal of semimajor axis.

    References:
        Battin, 1999, pp.180.

    Examples:
        A simple example of calculating the universal function U1 with a argument of 1.0 and a parameter of 1.0:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> chi = 1.0
        >>> alpha = 1.0
        >>> adx.twobody.ivp.U1(chi, alpha)
        Array(0.8414..., dtype=float32, weak_type=True)

        With broadcasting, you can calculate the universal function U1 for multiple arguments and parameters:

        >>> chi = jnp.array([1.0, 2.0])
        >>> alpha = jnp.array([1.0, -2.0])
        >>> adx.twobody.ivp.U1(chi, alpha)
        Array([0.8414..., 5.9608...], dtype=float32)
    """
    conds = [
        alpha > 0,
        alpha < 0,
    ]
    choices = [
        jnp.sin(jnp.sqrt(alpha) * chi) / jnp.sqrt(alpha),
        jnp.sinh(jnp.sqrt(-alpha) * chi) / jnp.sqrt(-alpha),
    ]
    return jnp.select(conds, choices, default=chi)


def U2(chi: ArrayLike, alpha: ArrayLike) -> Array:
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
        \frac{\chi^2}{2} & \alpha = 0 \\
        \frac{1 - \cos(\sqrt{\alpha} \chi)}{\alpha} & \alpha > 0 \\
        \frac{1 - \cosh(\sqrt{-\alpha} \chi)}{\alpha} & \alpha < 0
        \end{cases}
        $$
        where $\chi$ is the generalized anomaly and $\alpha = \frac{1}{a}$ is the reciprocal of semimajor axis.

    References:
        Battin, 1999, pp.180.

    Examples:
        A simple example of calculating the universal function U2 with a argument of 1.0 and a parameter of 1.0:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> chi = 1.0
        >>> alpha = 1.0
        >>> adx.twobody.ivp.U2(chi, alpha)
        Array(0.4596..., dtype=float32, weak_type=True)

        With broadcasting, you can calculate the universal function U2 for multiple arguments and parameters:

        >>> chi = jnp.array([1.0, 2.0])
        >>> alpha = jnp.array([1.0, -2.0])
        >>> adx.twobody.ivp.U2(chi, alpha)
        Array([0.4596..., 3.7444...], dtype=float32)
    """
    conds = [
        alpha > 0,
        alpha < 0,
    ]
    choices = [
        (1 - jnp.cos(jnp.sqrt(alpha) * chi)) / alpha,
        (1 - jnp.cosh(jnp.sqrt(-alpha) * chi)) / alpha,
    ]
    return jnp.select(conds, choices, default=chi**2 / 2.0)


def U3(chi: ArrayLike, alpha: ArrayLike) -> Array:
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
        \frac{\chi^3}{6} & \alpha = 0 \\
        \frac{\sqrt{\alpha} \chi - \sin(\sqrt{\alpha} \chi)}{\alpha \sqrt{\alpha}} & \alpha > 0 \\
        \frac{\sqrt{-\alpha} \chi - \sinh(\sqrt{-\alpha} \chi)}{\alpha \sqrt{-\alpha}} & \alpha < 0
        \end{cases}
        $$
        where $\chi$ is the generalized anomaly and $\alpha = \frac{1}{a}$ is the reciprocal of semimajor axis.

    References:
        Battin, 1999, pp.180.

    Examples:
        A simple example of calculating the universal function U3 with a argument of 1.0 and a parameter of 1.0:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> chi = 1.0
        >>> alpha = 1.0
        >>> adx.twobody.ivp.U3(chi, alpha)
        Array(0.1585..., dtype=float32, weak_type=True)

        With broadcasting, you can calculate the universal function U3 for multiple arguments and parameters:

        >>> chi = jnp.array([1.0, 2.0])
        >>> alpha = jnp.array([1.0, -2.0])
        >>> adx.twobody.ivp.U3(chi, alpha)
        Array([0.1585..., 1.9804...], dtype=float32)
    """

    conds = [
        alpha > 0,
        alpha < 0,
    ]
    choices = [
        (jnp.sqrt(alpha) * chi - jnp.sin(jnp.sqrt(alpha) * chi))
        / alpha
        / jnp.sqrt(alpha),
        (jnp.sqrt(-alpha) * chi - jnp.sinh(jnp.sqrt(-alpha) * chi))
        / alpha
        / jnp.sqrt(-alpha),
    ]
    return jnp.select(conds, choices, default=chi**3 / 6.0)


def U4(chi: ArrayLike, alpha: ArrayLike) -> Array:
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
        \frac{\chi^4}{24} & \alpha = 0 \\
        \frac{\chi^2 / 2 - U_2(\chi, \alpha)}{\alpha} & \alpha \neq 0
        \end{cases}
        $$
        where $\chi$ is the generalized anomaly and $\alpha = \frac{1}{a}$ is the reciprocal of semimajor axis.

    References:
        Battin, 1999, pp.183.

    Examples:
        A simple example of calculating the universal function U4 with a argument of 1.0 and a parameter of 1.0:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> chi = 1.0
        >>> alpha = 1.0
        >>> adx.twobody.ivp.U4(chi, alpha)
        Array(0.0403..., dtype=float32, weak_type=True)

        With broadcasting, you can calculate the universal function U4 for multiple arguments and parameters:

        >>> chi = jnp.array([1.0, 2.0])
        >>> alpha = jnp.array([1.0, -2.0])
        >>> adx.twobody.ivp.U4(chi, alpha)
        Array([0.0403..., 0.8722...], dtype=float32)
    """
    return jnp.where(
        jnp.isclose(alpha, 0.0), chi**4 / 24.0, (chi**2 / 2 - U2(chi, alpha)) / alpha
    )


def U5(chi: ArrayLike, alpha: ArrayLike) -> Array:
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
        \frac{\chi^5}{120} & \alpha = 0 \\
        \frac{\chi^3 / 6 - U_3(\chi, \alpha)}{\alpha} & \alpha \neq 0
        \end{cases}
        $$
        where $\chi$ is the generalized anomaly and $\alpha = \frac{1}{a}$ is the reciprocal of semimajor axis.

    References:
        Battin, 1999, pp.183.

    Examples:
        A simple example of calculating the universal function U5 with a argument of 1.0 and a parameter of 1.0:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> chi = 1.0
        >>> alpha = 1.0
        >>> adx.twobody.ivp.U5(chi, alpha)
        Array(0.008137..., dtype=float32, weak_type=True)

        With broadcasting, you can calculate the universal function U5 for multiple arguments and parameters:

        >>> chi = jnp.array([1.0, 2.0])
        >>> alpha = jnp.array([1.0, -2.0])
        >>> adx.twobody.ivp.U5(chi, alpha)
        Array([0.008137..., 0.323536...], dtype=float32)
    """
    return jnp.where(
        jnp.isclose(alpha, 0.0), chi**5 / 120.0, (chi**3 / 6 - U3(chi, alpha)) / alpha
    )
