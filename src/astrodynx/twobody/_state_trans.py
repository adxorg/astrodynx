import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array
from jax.numpy.linalg import vector_norm
import astrodynx as adx
from astrodynx.twobody._uniformulas import ufunc1, ufunc2, ufunc4, ufunc5, sigma_fn
from astrodynx.twobody._lagrange import lagrange_F, lagrange_G, lagrange_Ft, lagrange_Gt

"""State transition matrices for two-body orbital mechanics."""


def _C_func(
    chi: ArrayLike,
    U2: ArrayLike,
    U4: ArrayLike,
    U5: ArrayLike,
    deltat: ArrayLike = 3.14,
    mu: ArrayLike = 1,
) -> ArrayLike:
    return (3 * U5 - chi * U4 - jnp.sqrt(mu) * deltat * U2) / jnp.sqrt(mu)


def prpr0(
    r_vec: ArrayLike,
    v_vec: ArrayLike,
    r0_vec: ArrayLike,
    v0_vec: ArrayLike,
    F: ArrayLike,
    C: ArrayLike,
    mu: ArrayLike = 1,
) -> Array:
    r"""The State Transition Matrix for position with respect to initial position.

    Args:
        r_vec: (3,) The position vector at the current time.
        v_vec: (3,) The velocity vector at the current time.
        r0_vec: (3,) The position vector at the initial time.
        v0_vec: (3,) The velocity vector at the initial time.
        F: (3,) The Lagrange F function.
        C: (3,) The C function.
        mu: (optional) The gravitational parameter.

    Returns:
        The State Transition Matrix for position with respect to initial position.

    Notes:
        The State Transition Matrix for position with respect to initial position is defined as:
        $$
        \frac{d\boldsymbol{r}}{d\boldsymbol{r}_0} =
        \frac{r}{\mu} (\boldsymbol{v} - \boldsymbol{v}_0) (\boldsymbol{v} - \boldsymbol{v}_0)^T
        + (1 - F) \frac{\boldsymbol{r} \boldsymbol{r}_0^T}{r_0^2}
        + \frac{C \boldsymbol{v} \boldsymbol{r}_0^T}{r_0^3}
        + F \boldsymbol{I}
        $$
        where $\boldsymbol{r}$ is the position vector at current time, $\boldsymbol{v}$ is the velocity vector at the current time, $\boldsymbol{r}_0$ is the position vector at the initial time, $\boldsymbol{v}_0$ is the velocity vector at the initial time, $F$ is the Lagrange F function, $C$ is the C function, $\mu$ is the gravitational parameter, and $\boldsymbol{I}$ is the identity matrix.

    References:
        Battin, 1999, pp.467.
    """
    r0_mag = vector_norm(r0_vec)
    return (
        vector_norm(r_vec) / mu * jnp.outer(v_vec - v0_vec, v_vec - v0_vec)
        + (1 - F) * jnp.outer(r_vec / r0_mag, r0_vec / r0_mag)
        + C / r0_mag**2 * jnp.outer(v_vec, r0_vec / r0_mag)
        + F * jnp.eye(3)
    )


def prpv0(
    r_vec: ArrayLike,
    v_vec: ArrayLike,
    r0_vec: ArrayLike,
    v0_vec: ArrayLike,
    F: ArrayLike,
    G: ArrayLike,
    C: ArrayLike,
    mu: ArrayLike = 1,
) -> Array:
    r"""The State Transition Matrix for position with respect to initial velocity.

    Args:
        r_vec: (3,) The position vector at the current time.
        v_vec: (3,) The velocity vector at the current time.
        r0_vec: (3,) The position vector at the initial time.
        v0_vec: (3,) The velocity vector at the initial time.
        F: (3,) The Lagrange F function.
        G: (3,) The Lagrange G function.
        C: (3,) The C function.
        mu: (optional) The gravitational parameter.

    Returns:
        The State Transition Matrix for position with respect to initial velocity.

    Notes:
        The State Transition Matrix for position with respect to initial velocity is defined as:
        $$
        \frac{d\boldsymbol{r}}{d\boldsymbol{v}_0} =
        \frac{r_0}{\mu} (1 - F) [(\boldsymbol{r} - \boldsymbol{r}_0) \boldsymbol{v}_0^T - (\boldsymbol{v} - \boldsymbol{v}_0) \boldsymbol{r}_0^T]
        + \frac{C}{\mu} \boldsymbol{v} \boldsymbol{v}_0^T
        + G \boldsymbol{I}
        $$
        where $\boldsymbol{r}$ is the position vector at current time, $\boldsymbol{v}$ is the velocity vector at the current time, $\boldsymbol{r}_0$ is the position vector at the initial time, $\boldsymbol{v}_0$ is the velocity vector at the initial time, $F$ is the Lagrange F function, $G$ is the Lagrange G function, $C$ is the C function, $\mu$ is the gravitational parameter, and $\boldsymbol{I}$ is the identity matrix.

    References:
        Battin, 1999, pp.467.
    """
    return (
        vector_norm(r0_vec)
        / mu
        * (1 - F)
        * (jnp.outer(r_vec - r0_vec, v0_vec) - jnp.outer(v_vec - v0_vec, r0_vec))
        + C / mu * jnp.outer(v_vec, v0_vec)
        + G * jnp.eye(3)
    )


def pvpr0(
    r_vec: ArrayLike,
    v_vec: ArrayLike,
    r0_vec: ArrayLike,
    v0_vec: ArrayLike,
    Ft: ArrayLike,
    C: ArrayLike,
    mu: ArrayLike = 1,
) -> Array:
    r"""The State Transition Matrix for velocity with respect to initial position.

    Args:
        r_vec: (3,) The position vector at the current time.
        v_vec: (3,) The velocity vector at the current time.
        r0_vec: (3,) The position vector at the initial time.
        v0_vec: (3,) The velocity vector at the initial time.
        Ft: (3,) The Lagrange Ft function.
        C: (3,) The C function.
        mu: (optional) The gravitational parameter.

    Returns:
        The State Transition Matrix for velocity with respect to initial position.

    Notes:
        The State Transition Matrix for velocity with respect to initial position is defined as:
        $$
        \begin{split}
        \frac{d\boldsymbol{v}}{d\boldsymbol{r}_0} =
        & -\frac{1}{r_0^2} (\boldsymbol{v} - \boldsymbol{v}_0) \boldsymbol{r}_0^T
        - \frac{1}{r^2} \boldsymbol{r} (\boldsymbol{v} - \boldsymbol{v}_0)^T  \\
        & + F_t \left(
        \boldsymbol{I} - \frac{\boldsymbol{r} \boldsymbol{r}^T}{r^2}
        + \frac{1}{\mu r} (\boldsymbol{r} \boldsymbol{v}^T - \boldsymbol{v} \boldsymbol{r}^T) \boldsymbol{r}  (\boldsymbol{v} - \boldsymbol{v}_0)^T
        \right)
        - \frac{\mu C}{r^2 r_0^2} \boldsymbol{r} \boldsymbol{r}_0^T
        \end{split}
        $$
        where $\boldsymbol{r}$ is the position vector at current time, $\boldsymbol{v}$ is the velocity vector at the current time, $\boldsymbol{r}_0$ is the position vector at the initial time, $\boldsymbol{v}_0$ is the velocity vector at the initial time, $F_t$ is the Lagrange Ft function, $C$ is the C function, $\mu$ is the gravitational parameter, and $\boldsymbol{I}$ is the identity matrix.

    References:
        Battin, 1999, pp.467.
    """
    r0_mag = vector_norm(r0_vec)
    r_mag = vector_norm(r_vec)
    return (
        -jnp.outer(v_vec - v0_vec, r0_vec / r0_mag**2)
        - jnp.outer(r_vec / r_mag**2, v_vec - v0_vec)
        + Ft
        * (
            jnp.eye(3)
            - jnp.outer(r_vec / r_mag, r_vec / r_mag)
            + (jnp.outer(r_vec, v_vec) - jnp.outer(v_vec, r_vec))
            / mu
            @ jnp.outer(r_vec / r_mag, v_vec - v0_vec)
        )
        - mu / r_mag**2 * C / r0_mag**2 * jnp.outer(r_vec / r_mag, r0_vec / r0_mag)
    )


def pvpv0(
    r_vec: ArrayLike,
    v_vec: ArrayLike,
    r0_vec: ArrayLike,
    v0_vec: ArrayLike,
    F: ArrayLike,
    Gt: ArrayLike,
    C: ArrayLike,
    mu: ArrayLike = 1,
) -> Array:
    r"""The State Transition Matrix for velocity with respect to initial velocity.

    Args:
        r_vec: (3,) The position vector at the current time.
        v_vec: (3,) The velocity vector at the current time.
        r0_vec: (3,) The position vector at the initial time.
        v0_vec: (3,) The velocity vector at the initial time.
        F: (3,) The Lagrange F function.
        Gt: (3,) The Lagrange Gt function.
        C: (3,) The C function.
        mu: (optional) The gravitational parameter.

    Returns:
        The State Transition Matrix for velocity with respect to initial velocity.

    Notes:
        The State Transition Matrix for velocity with respect to initial velocity is defined as:
        $$
        \frac{d\boldsymbol{v}}{d\boldsymbol{v}_0} =
        \frac{r_0}{\mu} (\boldsymbol{v} - \boldsymbol{v}_0) (\boldsymbol{v} - \boldsymbol{v}_0)^T
        + \frac{1}{r_0^3}[r_0 (1-F) \boldsymbol{r} \boldsymbol{r}_0^T - C \boldsymbol{r} \boldsymbol{v}_0^T]
        + G_t \boldsymbol{I}
        $$
    """
    r0_mag = vector_norm(r0_vec)
    r_mag = vector_norm(r_vec)
    return (
        r0_mag / mu * jnp.outer(v_vec - v0_vec, v_vec - v0_vec)
        + (1 - F) * jnp.outer(r_vec / r_mag, r0_vec / r_mag) * r0_mag / r_mag
        - C / r_mag**2 * jnp.outer(r_vec / r_mag, v0_vec)
        + Gt * jnp.eye(3)
    )


def dxdx0(
    r_vec: ArrayLike,
    v_vec: ArrayLike,
    r0_vec: ArrayLike,
    v0_vec: ArrayLike,
    deltat: ArrayLike = 3.14,
    mu: ArrayLike = 1,
) -> Array:
    r"""The State Transition Matrix.

    Args:
        r_vec: (3,) The position vector at the current time.
        v_vec: (3,) The velocity vector at the current time.
        r0_vec: (3,) The position vector at the initial time.
        v0_vec: (3,) The velocity vector at the initial time.
        deltat: (optional) The time since the initial time.
        mu: (optional) The gravitational parameter.

    Returns:
        The State Transition Matrix.

    Notes:
        The State Transition Matrix is defined as:
        $$
        \frac{d\boldsymbol{x}}{d\boldsymbol{x}_0} =
        \begin{bmatrix}
        \frac{d\boldsymbol{r}}{d\boldsymbol{r}_0} & \frac{d\boldsymbol{r}}{d\boldsymbol{v}_0} \\
        \frac{d\boldsymbol{v}}{d\boldsymbol{r}_0} & \frac{d\boldsymbol{v}}{d\boldsymbol{v}_0}
        \end{bmatrix}
        $$
        where $\boldsymbol{x} = [\boldsymbol{r}, \boldsymbol{v}]^T$ is the state vector, $\boldsymbol{x}_0 = [\boldsymbol{r}_0, \boldsymbol{v}_0]^T$ is the initial state vector, and $\frac{d\boldsymbol{x}}{d\boldsymbol{x}_0}$ is the State Transition Matrix.

    References:
        Battin, 1999, pp.180.

    Examples:
        A simple example:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> r_vec = jnp.array([-0.24986234273434585, -0.69332384278075210, 4.9599012168662551e-3])
        >>> v_vec = jnp.array([ 1.2189179487500401,  0.05977450696618754, -0.007101943980682161])
        >>> r0_vec = jnp.array([-0.66234662571997105, 0.74919751798749190, -1.6259997018919074e-4])
        >>> v0_vec = jnp.array([-0.8166746784630675, -0.32961417380268476,  0.006248107587795581])
        >>> deltat = 2.5803148345055149
        >>> mu = 1.0
        >>> expected = jnp.array([[ 6.96499107e+00, -4.49913836e+00, -1.89246497e-02,  7.59769798e+00, -5.23072188e-01, -3.53190183e-02,],
        ...                       [ 1.51857748e+00, -1.52804032e+00, -3.60174714e-03,  2.52883087e+00,  1.17277875e+00, -1.61237785e-02,],
        ...                       [-4.98856694e-02,  2.99417750e-02, -5.82723061e-01, -5.10209945e-02,  8.87921668e-04,  7.78936443e-01,],
        ...                       [ 4.75150081e+00, -3.53238392e+00, -1.06813429e-02,  6.32244923e+00,  1.13091953e+00, -4.74582385e-02,],
        ...                       [ 8.76182663e+00, -6.69223965e+00, -1.65955664e-02,  1.00231736e+01, -5.74210092e-02, -6.14800215e-02,],
        ...                       [-7.39347940e-02,  5.19339444e-02, -4.25016825e-01, -9.32086959e-02, -1.19134537e-02, -1.14713870e+00]])
        >>> jnp.allclose(adx.twobody.dxdx0(r_vec, v_vec, r0_vec, v0_vec, deltat), expected)
        Array(True, dtype=bool)
    """
    r0_mag, r_mag = vector_norm(r0_vec), vector_norm(r_vec)
    alpha = 1.0 / adx.semimajor_axis(r0_mag, vector_norm(v0_vec), mu)
    chi = adx.generalized_anomaly(
        alpha, sigma_fn(r_vec, v_vec, mu), sigma_fn(r0_vec, v0_vec, mu), deltat, mu
    )
    C = _C_func(
        chi, ufunc2(chi, alpha), ufunc4(chi, alpha), ufunc5(chi, alpha), deltat, mu
    )
    F = lagrange_F(ufunc2(chi, alpha), r0_mag)
    G = lagrange_G(
        ufunc1(chi, alpha),
        ufunc2(chi, alpha),
        sigma_fn(r0_vec, v0_vec, mu),
        r0_mag,
        mu,
    )
    Ft = lagrange_Ft(ufunc1(chi, alpha), r_mag, r0_mag, mu)
    Gt = lagrange_Gt(ufunc2(chi, alpha), r_mag)
    return jnp.block(
        [
            [
                prpr0(r_vec, v_vec, r0_vec, v0_vec, F, C, mu),
                prpv0(r_vec, v_vec, r0_vec, v0_vec, F, G, C, mu),
            ],
            [
                pvpr0(r_vec, v_vec, r0_vec, v0_vec, Ft, C, mu),
                pvpv0(r_vec, v_vec, r0_vec, v0_vec, F, Gt, C, mu),
            ],
        ]
    )
