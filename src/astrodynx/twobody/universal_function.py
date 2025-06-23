import jax.numpy as jnp
from jax.typing import DTypeLike
from jax import lax


def _U0(chi: DTypeLike, alpha: DTypeLike) -> DTypeLike:
    r"""
    Returns the universal function U0.

    Parameters
    ----------
    chi : DTypeLike
        Universal variable.
    alpha : DTypeLike
        Reciprocal of semimajor axis $a$.

    Returns
    -------
    out : DTypeLike
        Value of the universal function U0.

    Notes
    -----
    The universal function U0 is defined as:
    $$
    U_0(\chi; \alpha) =
    \begin{cases}
    1 & \text{if } \alpha = 0 \\
    \cos(\sqrt{\alpha} \chi) & \text{if } \alpha > 0 \\
    \cosh(\sqrt{-\alpha} \chi) & \text{if } \alpha < 0
    \end{cases}
    $$

    References
    ----------
    Battin, 1999, pp. 180.

    Examples
    --------
    >>> from astrodynx.twobody.universal_function import _U0
    >>> chi = 1.0
    >>> alpha = 0.5
    >>> _U0(chi, alpha)
    0.7602445970756301
    """
    sqrt_alpha = jnp.sqrt(jnp.abs(alpha))

    def case_zero(_: None) -> DTypeLike:
        return jnp.ones_like(chi)

    def case_positive(_: None) -> DTypeLike:
        return jnp.cos(sqrt_alpha * chi)

    def case_negative(_: None) -> DTypeLike:
        return jnp.cosh(sqrt_alpha * chi)

    return lax.cond(
        alpha == 0,
        case_zero,
        lambda _: lax.cond(alpha > 0, case_positive, case_negative, operand=None),
        operand=None,
    )


def _U1(chi: DTypeLike, alpha: DTypeLike) -> DTypeLike:
    r"""
    Returns the universal function U1.

    Parameters
    ----------
    chi : DTypeLike
        Universal variable.
    alpha : DTypeLike
        Reciprocal of semimajor axis $a$.

    Returns
    -------
    out : DTypeLike
        Value of the universal function U1.

    Notes
    -----
    The universal function U1 is defined as:
    $$
    U_1(\chi; \alpha) =
    \begin{cases}
    \chi & \text{if } \alpha = 0 \\
    \frac{\sin(\sqrt{\alpha} \chi)}{\sqrt{\alpha}} & \text{if } \alpha > 0 \\
    \frac{\sinh(\sqrt{-\alpha} \chi)}{\sqrt{-\alpha}} & \text{if } \alpha < 0
    \end{cases}
    $$

    References
    ----------
    Battin, 1999, pp. 180.

    Examples
    --------
    >>> from astrodynx.twobody.universal_function import _U1
    >>> chi = 1.0
    >>> alpha = 0.5
    >>> _U1(chi, alpha)
    0.9187253698655684
    """
    sqrt_alpha = jnp.sqrt(jnp.abs(alpha))

    def case_zero(_: None) -> DTypeLike:
        return chi

    def case_positive(_: None) -> DTypeLike:
        return jnp.sin(sqrt_alpha * chi) / sqrt_alpha

    def case_negative(_: None) -> DTypeLike:
        return jnp.sinh(sqrt_alpha * chi) / sqrt_alpha

    return lax.cond(
        alpha == 0,
        case_zero,
        lambda _: lax.cond(alpha > 0, case_positive, case_negative, operand=None),
        operand=None,
    )


def _U2(chi: DTypeLike, alpha: DTypeLike) -> DTypeLike:
    r"""
    Returns the universal function U2.

    Parameters
    ----------
    chi : DTypeLike
        Universal variable.
    alpha : DTypeLike
        Reciprocal of semimajor axis $a$.

    Returns
    -------
    out : DTypeLike
        Value of the universal function U2.

    Notes
    -----
    The universal function U2 is defined as:
    $$
    U_2(\chi; \alpha) =
    \begin{cases}
    \frac{\chi^2}{2} & \text{if } \alpha = 0 \\
    \frac{1 - \cos(\sqrt{\alpha} \chi)}{\alpha} & \text{if } \alpha > 0 \\
    \frac{\cosh(\sqrt{-\alpha} \chi) - 1}{-\alpha} & \text{if } \alpha < 0
    \end{cases}
    $$

    References
    ----------
    Battin, 1999, pp. 180.

    Examples
    --------
    >>> from astrodynx.twobody.universal_function import _U2
    >>> chi = 1.0
    >>> alpha = 0.5
    >>> _U2(chi, alpha)
    0.47951080584873984
    """

    sqrt_alpha = jnp.sqrt(jnp.abs(alpha))

    def case_zero(_: None) -> DTypeLike:
        return chi**2 / 2

    def case_positive(_: None) -> DTypeLike:
        return (1 - jnp.cos(sqrt_alpha * chi)) / alpha

    def case_negative(_: None) -> DTypeLike:
        return (jnp.cosh(sqrt_alpha * chi) - 1) / -alpha

    return lax.cond(
        alpha == 0,
        case_zero,
        lambda _: lax.cond(alpha > 0, case_positive, case_negative, operand=None),
        operand=None,
    )


def _U3(chi: DTypeLike, alpha: DTypeLike) -> DTypeLike:
    r"""
    Returns the universal function U3.

    Parameters
    ----------
    chi : DTypeLike
        Universal variable.
    alpha : DTypeLike
        Reciprocal of semimajor axis $a$.

    Returns
    -------
    out : DTypeLike
        Value of the universal function U3.

    Notes
    -----
    The universal function U3 is defined as:
    $$
    U_3(\chi; \alpha) =
    \begin{cases}
    \frac{\chi^3}{6} & \text{if } \alpha = 0 \\
    \frac{\sqrt{\alpha} \chi - \sin(\sqrt{\alpha} \chi)}{\alpha \sqrt{\alpha}} & \text{if } \alpha > 0 \\
    \frac{\sinh(\sqrt{-\alpha} \chi) - \sqrt{-\alpha} \chi}{-\alpha \sqrt{-\alpha}} & \text{if } \alpha < 0
    \end{cases}
    $$

    References
    ----------
    Battin, 1999, pp. 180.

    Examples
    --------
    >>> from astrodynx.twobody.universal_function import _U3
    >>> chi = 1.0
    >>> alpha = 0.5
    >>> _U3(chi, alpha)
    0.16254926026886315
    """
    sqrt_alpha = jnp.sqrt(jnp.abs(alpha))

    def case_zero(_: None) -> DTypeLike:
        return chi**3 / 6

    def case_positive(_: None) -> DTypeLike:
        return (sqrt_alpha * chi - jnp.sin(sqrt_alpha * chi)) / (
            alpha * jnp.sqrt(alpha)
        )

    def case_negative(_: None) -> DTypeLike:
        return (jnp.sinh(sqrt_alpha * chi) - sqrt_alpha * chi) / (
            -alpha * jnp.sqrt(-alpha)
        )

    return lax.cond(
        alpha == 0,
        case_zero,
        lambda _: lax.cond(alpha > 0, case_positive, case_negative, operand=None),
        operand=None,
    )
