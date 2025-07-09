from jax import numpy as jnp
from jax.typing import ArrayLike
from jax import Array
from astrodynx.twobody._orb_integrals import (
    angular_momentum,
    eccentricity_vector,
    semiparameter,
)
from jax.numpy.linalg import vector_norm
from astrodynx.utils import rotmat3dx, rotmat3dz


def true_anomaly(pos_vec: ArrayLike, e_vec: ArrayLike) -> Array:
    true_anom = jnp.arctan2(
        vector_norm(jnp.cross(e_vec, pos_vec), axis=-1, keepdims=True),
        jnp.sum(e_vec * pos_vec, axis=-1, keepdims=True),
    )
    return jnp.where(true_anom < 0, 2 * jnp.pi + true_anom, true_anom)


def right_ascension(h_vec: ArrayLike) -> Array:
    N = jnp.cross(jnp.array([0, 0, 1]), h_vec)
    raan = jnp.arctan2(N[..., 1:2], N[..., 0:1])
    return jnp.where(raan < 0, 2 * jnp.pi + raan, raan)


def argument_of_periapsis(h_vec: ArrayLike, e_vec: ArrayLike) -> Array:
    N = jnp.cross(jnp.array([0, 0, 1]), h_vec)
    argp = jnp.arctan2(
        vector_norm(jnp.cross(N, e_vec), axis=-1, keepdims=True),
        jnp.sum(N * e_vec, axis=-1, keepdims=True),
    )
    return jnp.where(argp < 0, 2 * jnp.pi + argp, argp)


def rv2coe(
    pos_vec: ArrayLike, vel_vec: ArrayLike, mu: ArrayLike = 1
) -> tuple[Array, Array, Array, Array, Array, Array]:
    r"""Transfer position and velocity vectors to classical orbital elements.

    Args:
        pos_vec: (..., 3) position vector of the object in the two-body system.
        vel_vec: (..., 3) velocity vector of the object in the two-body system, which shape broadcast-compatible with `pos_vec`.
        mu: Gravitational parameter of the central body; shape broadcast-compatible with `pos_vec` and `vel_vec`.

    Returns:
        (p, e, incl, raan, argp, true_anom) The classical orbital elements of the object in the two-body system.

    Notes
        The classical orbital elements are calculated using the following equations:
        $$
        \begin{aligned}
        \boldsymbol{h} &= \boldsymbol{r} \times \boldsymbol{v} \\
        p &= \frac{h^2}{\mu} \\
        \boldsymbol{e} &= \frac{\boldsymbol{v} \times \boldsymbol{h}}{\mu} - \frac{\boldsymbol{r}}{r} \\
        i &= \arccos(h_z / h) \\
        \boldsymbol{N} &= \boldsymbol{k} \times \boldsymbol{h} \\
        \Omega &= \arctan2(N_y, N_x) \\
        \omega &= \arctan2(\|\boldsymbol{N}\times \boldsymbol{e}\|, \boldsymbol{N} \cdot \boldsymbol{e}) \\
        f &= \arctan2(\|\boldsymbol{e}\times \boldsymbol{r}\|, \boldsymbol{e} \cdot \boldsymbol{r})
        \end{aligned}
        $$
        where $p$ is the semiparameter, $e$ is the eccentricity vector, $i$ is the inclination, $\Omega$ is the right ascension of the ascending node, $\omega$ is the argument of periapsis, and $f$ is the true anomaly.

    References
        Vallado, 2013, pp.114.

    Examples
        A simple example:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> pos_vec = jnp.array([1.0, 0.0, 0.0])
        >>> vel_vec = jnp.array([0.0, 1.0, 0.0])
        >>> mu = 1.0
        >>> p, e, incl, raan, argp, true_anom = adx.rv2coe(pos_vec, vel_vec, mu)
        >>> print(p, e, incl, raan, argp, true_anom)
        1.0 0.0 0.0 0.0 0.0 0.0

        With broadcasting, you can calculate the classical orbital elements for multiple position and velocity vectors:

        >>> pos_vec = jnp.array([[-4039.8965, 4814.5605, 3628.625], [6525.3677, 6861.5317, 6449.117]])
        >>> vel_vec = jnp.array([[-10.385988, -4.771922, 1.7438745], [4.902279, 5.5331397, -1.9757109]])
        >>> mu = jnp.array([[398600],[398600.4418]])
        >>> p, e, incl, raan, argp, true_anom = adx.rv2coe(pos_vec, vel_vec, mu)
        >>> assert jnp.allclose(p, jnp.array([16056.1966,11067.79]))
        >>> assert jnp.allclose(e, jnp.array([1.4,0.83285]))
        >>> assert jnp.allclose(incl, jnp.array([jnp.deg2rad(30),jnp.deg2rad(87.87)]))
        >>> assert jnp.allclose(raan, jnp.array([jnp.deg2rad(40),jnp.deg2rad(227.89)]))
        >>> assert jnp.allclose(argp, jnp.array([jnp.deg2rad(60),jnp.deg2rad(53.38)]))
        >>> assert jnp.allclose(true_anom, jnp.array([jnp.deg2rad(30),jnp.deg2rad(92.335)]))
    """
    h_vec = angular_momentum(pos_vec, vel_vec)
    h_mag = vector_norm(h_vec, axis=-1, keepdims=True)
    e_vec = eccentricity_vector(pos_vec, vel_vec, mu)
    e_mag = vector_norm(e_vec, axis=-1, keepdims=True)
    p = semiparameter(h_mag, mu)
    incl = jnp.arccos(h_vec[..., 2:3] / h_mag)
    raan = right_ascension(h_vec)
    argp = argument_of_periapsis(h_vec, e_vec)
    true_anom = true_anomaly(pos_vec, e_vec)
    return (
        jnp.squeeze(p, axis=-1),
        jnp.squeeze(e_mag, axis=-1),
        jnp.squeeze(incl, axis=-1),
        jnp.squeeze(raan, axis=-1),
        jnp.squeeze(argp, axis=-1),
        jnp.squeeze(true_anom, axis=-1),
    )


def coe2rv(
    p: ArrayLike,
    e: ArrayLike,
    incl: ArrayLike,
    raan: ArrayLike,
    argp: ArrayLike,
    true_anom: ArrayLike,
    mu: ArrayLike = 1,
) -> tuple[Array, Array]:
    r"""Transfer classical orbital elements to position and velocity vectors.

    Args:
        p: Semiparameter of the orbit.
        e: Eccentricity of the orbit; shape broadcast-compatible with `p`.
        incl: Inclination of the orbit; shape broadcast-compatible with `p` and `e`.
        raan: Right ascension of the ascending node; shape broadcast-compatible with `p`, `e`, and `incl`.
        argp: Argument of periapsis; shape broadcast-compatible with `p`, `e`, `incl`, and `raan`.
        true_anom: True anomaly; shape broadcast-compatible with `p`, `e`, `incl`, `raan`, and `argp`.
        mu: Gravitational parameter of the central body; shape broadcast-compatible with `p`, `e`, `incl`, `raan`, `argp`, and `true_anom`.

    Returns:
        The position and velocity vectors of the object in the two-body system.

    Notes
        The position and velocity vectors are calculated using the following equations:
        $$
        \begin{aligned}
        \boldsymbol{r} &=  \boldsymbol{R}_z(-\Omega) \boldsymbol{R}_x(-i) \boldsymbol{R}_z(-\omega) \boldsymbol{r}_{pf} \\
        \boldsymbol{v} &= \boldsymbol{R}_z(-\Omega) \boldsymbol{R}_x(-i) \boldsymbol{R}_z(-\omega) \boldsymbol{v}_{pf}
        \end{aligned}
        $$
        where $\boldsymbol{R}_z$ and $\boldsymbol{R}_x$ are the rotation matrices about the z-axis and x-axis.
        And, \boldsymbol{r}_{pf} and \boldsymbol{v}_{pf} are the position and velocity vectors in the perifocal frame:
        $$
        \begin{aligned}
        \boldsymbol{r}_{pf} &= \frac{p}{1 + e \cos(f)} \begin{bmatrix} \cos(f) \\ \sin(f) \\ 0 \end{bmatrix} \\
        \boldsymbol{v}_{pf} &= \sqrt{\frac{\mu}{p}} \begin{bmatrix} -\sin(f) \\ e + \cos(f) \\ 0 \end{bmatrix}
        \end{aligned}
        $$
        where $p$ is the semiparameter, $e$ is the eccentricity, $f$ is the true anomaly, and $\mu$ is the gravitational parameter.

    References
        Vallado, 2013, pp.119.

    Examples
        A simple example:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> p = 1.0
        >>> e = 0.0
        >>> incl = 0.0
        >>> raan = 0.0
        >>> argp = 0.0
        >>> true_anom = 0.0
        >>> mu = 1.0
        >>> adx.coe2rv(p, e, incl, raan, argp, true_anom, mu)
        (Array([1., 0., 0.], dtype=float32, weak_type=True), Array([0., 1., 0.], dtype=float32, weak_type=True))

        With broadcasting, you can calculate the position and velocity vectors for multiple sets of orbital elements:

        >>> mu = jnp.array([398600,398600.4418])
        >>> p =jnp.array([16056.196688409433,11067.79])
        >>> e = jnp.array([1.4,0.83285])
        >>> incl = jnp.array([jnp.deg2rad(30),jnp.deg2rad(87.87)])
        >>> raan = jnp.array([jnp.deg2rad(40),jnp.deg2rad(227.89)])
        >>> argp = jnp.array([jnp.deg2rad(60),jnp.deg2rad(53.38)])
        >>> true_anom = jnp.array([jnp.deg2rad(30),jnp.deg2rad(92.335)])
        >>> pos_vec, vel_vec = adx.coe2rv(p, e, incl, raan, argp, true_anom, mu)
        >>> assert jnp.allclose(pos_vec[0], jnp.array([-4039.8965, 4814.5605, 3628.625]))
        >>> assert jnp.allclose(vel_vec[0], jnp.array([-10.385988, -4.771922, 1.7438745]))
        >>> assert jnp.allclose(pos_vec[1], jnp.array([6525.3677, 6861.5317, 6449.117]))
        >>> assert jnp.allclose(vel_vec[1], jnp.array([4.902279, 5.5331397, -1.9757109]))
    """
    r_pf = jnp.expand_dims(p / (1 + e * jnp.cos(true_anom)), axis=-1) * jnp.stack(
        [jnp.cos(true_anom), jnp.sin(true_anom), jnp.zeros_like(true_anom)], axis=-1
    )
    v_pf = jnp.expand_dims(jnp.sqrt(mu / p), axis=-1) * jnp.stack(
        [-jnp.sin(true_anom), e + jnp.cos(true_anom), jnp.zeros_like(true_anom)],
        axis=-1,
    )

    rotmat = rotmat3dz(raan) @ rotmat3dx(incl) @ rotmat3dz(argp)
    r_ijk = rotmat @ jnp.expand_dims(r_pf, axis=-1)
    v_ijk = rotmat @ jnp.expand_dims(v_pf, axis=-1)
    return jnp.squeeze(r_ijk, axis=-1), jnp.squeeze(v_ijk, axis=-1)
