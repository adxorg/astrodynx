import jax.numpy as jnp
import jax
from jax.typing import ArrayLike, DTypeLike
from jax import Array
from astrodynx.twobody._kep_equ import solve_kepler_uni
from astrodynx.twobody._uniformulas import sigma_fn, ufunc0, ufunc1, ufunc2
from astrodynx.twobody._orb_integrals import semimajor_axis, equ_of_orb_uvi
from astrodynx.twobody._lagrange import lagrange_F, lagrange_G, lagrange_Ft, lagrange_Gt


def lagrange_prop(
    F: ArrayLike,
    G: ArrayLike,
    Ft: ArrayLike,
    Gt: ArrayLike,
    r0_vec: ArrayLike,
    v0_vec: ArrayLike,
) -> tuple[Array, Array]:
    r"""The Lagrange propagator.

    Args:
        F: The Lagrange F function.
        G: The Lagrange G function.
        Ft: The Lagrange Ft function.
        Gt: The Lagrange Gt function.
        r0_vec: (...,3) The position vector at the initial time.
        v0_vec: (...,3) The velocity vector at the initial time.

    Returns:
        The propagated state vector.

    Notes:
        The Lagrange propagator is defined as:
        $$
        \begin{align*}
        \boldsymbol{r} &= F \boldsymbol{r}_0 + G \boldsymbol{v}_0 \\
        \boldsymbol{v} &= F_t \boldsymbol{r}_0 + G_t \boldsymbol{v}_0
        \end{align*}
        $$
        where $\boldsymbol{x}$ is the propagated state vector, $\boldsymbol{r}$ is the position vector at the current time, $\boldsymbol{v}$ is the velocity vector at the current time, $\boldsymbol{r}_0$ is the position vector at the initial time, $\boldsymbol{v}_0$ is the velocity vector at the initial time, $F$ is the Lagrange F function, $G$ is the Lagrange G function,
        $F_t$ is the Lagrange Ft function, and $G_t$ is the Lagrange Gt function.

    References:
        Battin, 1999, pp.129.

    Examples:
        A simple example:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> F = 1.0
        >>> G = 1.0
        >>> Ft = 1.0
        >>> Gt = 1.0
        >>> r0_vec = jnp.array([1.0, 0.0, 0.0])
        >>> v0_vec = jnp.array([0.0, 1.0, 0.0])
        >>> r_vec, v_vec = adx.lagrange_prop(F, G, Ft, Gt, r0_vec, v0_vec)
        >>> assert jnp.allclose(r_vec, jnp.array([1.0, 1.0, 0.0]))
        >>> assert jnp.allclose(v_vec, jnp.array([1.0, 1.0, 0.0]))

        With broadcasting:

        >>> F = jnp.array([[1.0], [2.0]])
        >>> G = jnp.array([[1.0], [1.0]])
        >>> Ft = jnp.array([[1.0], [1.0]])
        >>> Gt = jnp.array([[1.0], [1.0]])
        >>> r0_vec = jnp.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        >>> v0_vec = jnp.array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
        >>> r_vec, v_vec = adx.lagrange_prop(F, G, Ft, Gt, r0_vec, v0_vec)
        >>> assert jnp.allclose(r_vec, jnp.array([[1.0, 1.0, 0.0], [4.0, 2.0, 0.0]]))
        >>> assert jnp.allclose(v_vec, jnp.array([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0]]))
    """
    return F * r0_vec + G * v0_vec, Ft * r0_vec + Gt * v0_vec


def kepler_prop(
    ts: ArrayLike, r0_vec: ArrayLike, v0_vec: ArrayLike, mu: DTypeLike = 1.0
) -> tuple[Array, Array]:
    r"""The Kepler propagator for a single initial state.

    Args:
        ts: (n,)The time steps to propagate to.
        r0_vec: (3,) The position vector at the initial time.
        v0_vec: (3,) The velocity vector at the initial time.
        mu: (optional) The gravitational parameter.

    Returns:
        The propagated state vector.

    Notes:
        The Kepler propagator is a wrapper function for the :func:`lagrange_prop` function.

    Examples:
        A simple example:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> r0_vec = jnp.array([1.0, 0.0, 0.0])
        >>> v0_vec = jnp.array([0.0, 1.0, 0.0])
        >>> mu = 1.0
        >>> ts = jnp.pi
        >>> r_vec,v_vec = adx.kepler_prop(ts, r0_vec, v0_vec, mu)
        >>> assert jnp.allclose(r_vec, jnp.array([-1.0, 0.0, 0.0]), atol=1e-6)
        >>> assert jnp.allclose(v_vec, jnp.array([0.0, -1.0, 0.0]), atol=1e-6)

        With broadcasting:

        >>> r0_vec = jnp.array([1.0, 0.0, 0.0])
        >>> v0_vec = jnp.array([0.0, 1.0, 0.0])
        >>> mu = 1.0
        >>> ts = jnp.linspace(0, 2*jnp.pi, 12)
        >>> r_vec,v_vec = adx.kepler_prop(ts, r0_vec, v0_vec, mu)
        >>> assert r_vec.shape == (12, 3)
        >>> assert v_vec.shape == (12, 3)
    """
    r0 = jnp.linalg.norm(r0_vec)
    v0 = jnp.linalg.norm(v0_vec)
    alpha = 1.0 / semimajor_axis(r0, v0, mu)
    sigma0 = sigma_fn(r0_vec, v0_vec, mu)

    chi = jax.vmap(solve_kepler_uni, in_axes=(0, None, None, None, None))(
        jnp.atleast_1d(ts), alpha.item(), r0.item(), sigma0.item(), mu
    )
    F = lagrange_F(ufunc2(chi, alpha), r0)
    G = lagrange_G(
        ufunc1(chi, alpha),
        ufunc2(chi, alpha),
        sigma0,
        r0,
        mu,
    )
    r = equ_of_orb_uvi(
        ufunc0(chi, alpha), ufunc1(chi, alpha), ufunc2(chi, alpha), r0, sigma0
    )
    Ft = lagrange_Ft(ufunc1(chi, alpha), r, r0, mu)
    Gt = lagrange_Gt(ufunc2(chi, alpha), r)
    return lagrange_prop(
        F[:, jnp.newaxis],
        G[:, jnp.newaxis],
        Ft[:, jnp.newaxis],
        Gt[:, jnp.newaxis],
        r0_vec,
        v0_vec,
    )
