import jax.numpy as jnp
from jax.typing import ArrayLike, DTypeLike
from jax import Array
from astrodynx.twobody._uniformulas import ufunc0, ufunc1, ufunc2, ufunc3
import jax
from astrodynx.twobody._orb_integrals import equ_of_orb_uvi

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


def dE(E: ArrayLike, e: ArrayLike) -> Array:
    r"""Returns the derivative of Kepler's equation for elliptical orbits with respect to the eccentric anomaly.

    Args:
        E: Eccentric anomaly.
        e: Eccentricity of the orbit, 0 <= e < 1.

    Returns:
        The derivative of Kepler's equation for elliptical orbits with respect to the eccentric anomaly.

    Notes:
        The derivative of Kepler's equation for elliptical orbits with respect to the eccentric anomaly is:
        $$
        dE = 1 - e \cos E
        $$
        where $E$ is the eccentric anomaly and $e$ is the eccentricity.
    """
    return 1 - e * jnp.cos(E)


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


def dH(H: ArrayLike, e: ArrayLike) -> Array:
    r"""Returns the derivative of Kepler's equation for hyperbolic orbits with respect to the hyperbolic eccentric anomaly.

    Args:
        H: Hyperbolic eccentric anomaly.
        e: Eccentricity of the orbit, e > 1.

    Returns:
        The derivative of Kepler's equation for hyperbolic orbits with respect to the hyperbolic eccentric anomaly.

    Notes:
        The derivative of Kepler's equation for hyperbolic orbits with respect to the hyperbolic eccentric anomaly is:
        $$
        dH = e \cosh H - 1
        $$
        where $H$ is the hyperbolic eccentric anomaly and $e$ is the eccentricity.
    """
    return e * jnp.cosh(H) - 1


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
        r0 * ufunc1(chi, alpha)
        + sigma0 * ufunc2(chi, alpha)
        + ufunc3(chi, alpha)
        - jnp.sqrt(mu) * deltat
    )


def dchi(chi: ArrayLike, alpha: DTypeLike, r0: ArrayLike, sigma0: ArrayLike) -> Array:
    return equ_of_orb_uvi(
        ufunc0(chi, alpha), ufunc1(chi, alpha), ufunc2(chi, alpha), r0, sigma0
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


def solve_kepler_elps(
    M: DTypeLike, e: DTypeLike, tol: DTypeLike = 1e-6, max_iter: int = 20
) -> Array:
    r"""Returns the eccentric anomaly for an elliptical orbit.

    Args:
        M: Mean anomaly.
        e: Eccentricity of the orbit, 0 <= e < 1.
        tol: (optional) Tolerance for convergence.
        max_iter: (optional) Maximum number of iterations.

    Returns:
        The eccentric anomaly for an elliptical orbit.

    Notes:
        The eccentric anomaly is calculated by solving Kepler's equation for elliptical orbits:
        $$
        E - e \sin E = M
        $$
        where $E$ is the eccentric anomaly, $e$ is the eccentricity, and $M$ is the mean anomaly.

    References:
        Vallado, 2013, pp.76.

    Examples:
        A simple example:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> e = 0.37255
        >>> M = 3.6029
        >>> adx.solve_kepler_elps(M,e)
        Array(3.479..., dtype=float32, weak_type=True)
    """
    E0 = M + e * jnp.sin(M) + e**2 * jnp.sin(2 * M) / 2

    def cond_fn(val: tuple[int, Array, Array]) -> Array:
        iter_count, E, E_prev = val
        not_converged = jnp.abs(E - E_prev) > tol
        under_max_iter = iter_count < max_iter
        return not_converged & under_max_iter

    def body_fn(val: tuple[int, Array, Array]) -> Array:
        iter_count, E, _ = val
        f = E - e * jnp.sin(E) - M
        E_new = E - f / dE(E, e)
        return (iter_count + 1, E_new, E)

    _, E, _ = jax.lax.while_loop(cond_fn, body_fn, (0, E0, E0 + 2 * tol))
    return E


def solve_kepler_hypb(
    N: DTypeLike, e: DTypeLike, tol: DTypeLike = 1e-6, max_iter: int = 50
) -> Array:
    r"""Returns the hyperbolic eccentric anomaly for a hyperbolic orbit.

    Args:
        N: Hyperbolic mean anomaly.
        e: Eccentricity of the orbit, e > 1.
        tol: (optional) Tolerance for convergence.
        max_iter: (optional) Maximum number of iterations.

    Returns:
        The hyperbolic eccentric anomaly for a hyperbolic orbit.

    Notes:
        The hyperbolic eccentric anomaly is calculated by solving Kepler's equation for hyperbolic orbits:
        $$
        e \sinh H - H = N
        $$
        where $H$ is the hyperbolic eccentric anomaly, $e$ is the eccentricity, and $N$ is the hyperbolic mean anomaly.

    References:
        Battin, 1999, pp.168.

    Examples:
        A simple example:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> e = 2.7696
        >>> N = 40.69
        >>> adx.solve_kepler_hypb(N,e)
        Array(3.463..., dtype=float32, weak_type=True)
    """
    H0 = jnp.log(2 * N / e + jnp.sqrt((2 * N / e) ** 2 + 1))

    def cond_fn(val: tuple[int, Array, Array]) -> Array:
        iter_count, H, _ = val
        not_converged = jnp.abs(kepler_equ_hypb(H, e, N)) > tol
        under_max_iter = iter_count < max_iter
        return not_converged & under_max_iter

    def body_fn(val: tuple[int, Array, Array]) -> Array:
        iter_count, H, _ = val
        H_new = H - kepler_equ_hypb(H, e, N) / dH(H, e)
        return (iter_count + 1, H_new, H)

    _, H, _ = jax.lax.while_loop(cond_fn, body_fn, (0, H0, H0 + 2 * tol))
    return H


def solve_kepler_uni(
    deltat: DTypeLike,
    alpha: DTypeLike,
    r0: DTypeLike,
    sigma0: DTypeLike,
    mu: DTypeLike = 1,
    tol: DTypeLike = 1e-6,
    max_iter: int = 50,
) -> Array:
    r"""Returns the generalized anomaly for a universal orbit equation.

    Args:
        deltat: The time since the initial time.
        alpha: The reciprocal of the semimajor axis.
        r0: The radius at the initial time.
        sigma0: The sigma function at the initial time.
        mu: (optional) The gravitational parameter.
        tol: (optional) Tolerance for convergence.
        max_iter: (optional) Maximum number of iterations.

    Returns:
        The generalized anomaly for a universal orbit equation.

    Notes:
        The generalized anomaly is calculated by solving the universal orbit equation:
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
        >>> r0_vec = jnp.array([1.0, 0.0, 0.0])
        >>> v0_vec = jnp.array([0.0, 1.1, 0.0])
        >>> mu = 1.0
        >>> deltat = jnp.pi*0.5
        >>> r0 = jnp.linalg.norm(r0_vec)
        >>> v0 = jnp.linalg.norm(v0_vec)
        >>> alpha = 1.0 / adx.semimajor_axis(r0, v0, mu)
        >>> sigma0 = adx.twobody.sigma_fn(r0_vec, v0_vec, mu)
        >>> chi = adx.solve_kepler_uni(deltat, alpha.item(), r0.item(), sigma0.item(), mu)
        >>> assert adx.kepler_equ_uni(chi,alpha,r0, sigma0, deltat, mu) < 1e-6
    """

    chi0 = jnp.sqrt(mu) * jnp.abs(alpha) * deltat

    def cond_fn(val: tuple[int, Array, Array]) -> Array:
        iter_count, chi, _ = val
        not_converged = (
            jnp.abs(kepler_equ_uni(chi, alpha, r0, sigma0, deltat, mu)) > tol
        )
        under_max_iter = iter_count < max_iter
        return not_converged & under_max_iter

    def body_fn(val: tuple[int, Array, Array]) -> Array:
        iter_count, chi, _ = val
        chi_new = chi - kepler_equ_uni(chi, alpha, r0, sigma0, deltat, mu) / dchi(
            chi, alpha, r0, sigma0
        )
        return (iter_count + 1, chi_new, chi)

    _, chi, _ = jax.lax.while_loop(cond_fn, body_fn, (0, chi0, chi0 + 2 * tol))
    return chi
