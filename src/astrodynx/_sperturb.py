from jaxtyping import ArrayLike, DTypeLike, PyTree
import diffrax
from typing import Any

"""Cowell's method for propagating orbits under the influence of perturbing forces."""


def cowell_method(
    term: diffrax.AbstractTerm,
    x0: ArrayLike,
    t1: DTypeLike,
    *,
    t0: DTypeLike = 0.0,
    args: PyTree[Any] = {"mu": 1.0},
    dt0: DTypeLike = 0.01,
    event: diffrax.Event = None,
    solver: diffrax.AbstractSolver = diffrax.Tsit5(),
    saveat: diffrax.SaveAt = diffrax.SaveAt(subs=diffrax.SubSaveAt(t1=True)),
    max_steps: int | None = 4096,
    stepsize_controller: diffrax.AbstractStepSizeController = diffrax.PIDController(
        rtol=1e-8, atol=1e-8
    ),
) -> diffrax.Solution:
    r"""Returns the solution to the differential equation using Cowell's method.

    Args:
        term: The differential equation to solve.
        x0: The initial state.
        t1: The final time.
        t0: The initial time.
        args: Any additional arguments to pass to the differential equation.
        dt0: The step size to use for the first step. If using fixed step sizes then this will also be the step size for all other steps. (Except the last one, which may be slightly smaller and clipped to t1.) If set as None then the initial step size will be determined automatically.
        event: An event at which to terminate the solve early. See the `diffrax events <https://docs.kidger.site/diffrax/api/events/>`_ documentation for more information.
        solver: The solver for the differential equation. See the `diffrax solver <https://docs.kidger.site/diffrax/usage/how-to-choose-a-solver/>`_ guide on how to choose a solver.
        saveat: The times to save the solution.
        max_steps: The maximum number of steps to take before quitting the computation unconditionally.
        stepsize_controller: The stepsize controller to use.

    Returns:
        The solution to the differential equation. See the `diffrax solution <https://docs.kidger.site/diffrax/api/solution/>`_ documentation for more information.

    Notes:
        Cowell's method is a numerical method for propagating orbits under the influence of perturbing forces. It is based on solving the differential equation:
        $$
        \left[\begin{matrix}
        \dot{\boldsymbol{r}}\\
        \dot{\boldsymbol{v}}
        \end{matrix} \right]
        = \left[\begin{matrix}
        \boldsymbol{v}\\
        \boldsymbol{a}
        \end{matrix} \right]
        $$
        where $\boldsymbol{r}$ is the position vector, $\boldsymbol{v}$ is the velocity vector, and $\boldsymbol{a}$ is the acceleration vector.

    References:
        Vallado, 2013, pp.526.

    Examples:
        A simple example of propagating an orbit under the influence of a central body and J2 perturbations:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> import diffrax
        >>> def vector_field(t, x, args):
        ...     acc = adx.gravity.point_mass_grav(t, x, args)
        ...     acc += adx.gravity.j2_acc(t, x, args)
        ...     return jnp.concatenate([x[3:], acc])
        >>> args = {"mu": 1.0, "rmin": 0.7, "J2": 1e-6, "R_eq": 1.0}
        >>> t1 = 3.14
        >>> x0 = jnp.array([1.0, 0.0, 0.0, 0.0, 0.9, 0.0])
        >>> event = diffrax.Event(adx.events.radius_toolow)
        >>> term = diffrax.ODETerm(vector_field)
        >>> sol = adx.cowell_method(term, x0, t1, args=args, event=event)
        >>> print(sol.ts[-1])
        2.13...
    """
    return diffrax.diffeqsolve(
        term,
        solver,
        t0,
        t1,
        dt0,
        x0,
        args=args,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        event=event,
        max_steps=max_steps,
    )


def spprop_steps(
    term: diffrax.AbstractTerm,
    x0: ArrayLike,
    t1: DTypeLike,
    ts: ArrayLike,
    *,
    args: PyTree[Any] = {"mu": 1.0},
    event: diffrax.Event = None,
    solver: diffrax.AbstractSolver = diffrax.Tsit5(),
    stepsize_controller: diffrax.AbstractStepSizeController = diffrax.PIDController(
        rtol=1e-8, atol=1e-8
    ),
) -> diffrax.Solution:
    """Returns the solution to the differential equation using Cowell's method with fixed step sizes.

    Args:
        term: The differential equation to solve.
        x0: The initial state.
        t1: The final time.
        ts: The times to save the solution.
        args: Any additional arguments to pass to the differential equation.
        event: An event at which to terminate the solve early. See the `diffrax events <https://docs.kidger.site/diffrax/api/events/>`_ documentation for more information.
        solver: The solver for the differential equation. See the `diffrax solver <https://docs.kidger.site/diffrax/usage/how-to-choose-a-solver/>`_ guide on how to choose a solver.
        stepsize_controller: The stepsize controller to use.

    Returns:
        The solution to the differential equation. See the `diffrax solution <https://docs.kidger.site/diffrax/api/solution/>`_ documentation for more information.

    Notes:
        This is a wrapper around the :func:`cowell_method` function. See the :func:`cowell_method` function for more information.

    Examples:
        A simple example of propagating an orbit under the influence of a central body and J2 perturbations:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> import diffrax
        >>> def vector_field(t, x, args):
        ...     acc = adx.gravity.point_mass_grav(t, x, args)
        ...     acc += adx.gravity.j2_acc(t, x, args)
        ...     return jnp.concatenate([x[3:], acc])
        >>> args = {"mu": 1.0, "rmin": 0.7, "J2": 1e-6, "R_eq": 1.0}
        >>> t1 = 3.14
        >>> ts = jnp.linspace(0, t1, 100)
        >>> x0 = jnp.array([1.0, 0.0, 0.0, 0.0, 0.9, 0.0])
        >>> event = diffrax.Event(adx.events.radius_toolow)
        >>> term = diffrax.ODETerm(vector_field)
        >>> sol = adx.spprop_steps(term, x0, t1, ts, args=args, event=event)
        >>> tsmask = jnp.isfinite(sol.ts)
        >>> xf = sol.ys[tsmask][-1]
        >>> expected = jnp.array([-0.589,0.37, 0.,-0.59,-1.15, 0.])
        >>> jnp.allclose(xf, expected, atol=1e-2)
        Array(True, dtype=bool)

    """
    return cowell_method(
        term,
        x0,
        t1,
        args=args,
        event=event,
        solver=solver,
        max_steps=None,
        stepsize_controller=stepsize_controller,
        saveat=diffrax.SaveAt(ts=ts),
    )


def spprop_varstep(
    term: diffrax.AbstractTerm,
    x0: ArrayLike,
    t1: DTypeLike,
    *,
    args: PyTree[Any] = {"mu": 1.0},
    event: diffrax.Event = None,
    solver: diffrax.AbstractSolver = diffrax.Tsit5(),
    max_steps: int = 4096,
    stepsize_controller: diffrax.AbstractStepSizeController = diffrax.PIDController(
        rtol=1e-8, atol=1e-8
    ),
) -> diffrax.Solution:
    """Returns the solution to the differential equation using Cowell's method with variable step sizes.

    Args:
        term: The differential equation to solve.
        x0: The initial state.
        t1: The final time.
        args: Any additional arguments to pass to the differential equation.
        event: An event at which to terminate the solve early. See the `diffrax events <https://docs.kidger.site/diffrax/api/events/>`_ documentation for more information.
        solver: The solver for the differential equation. See the `diffrax solver <https://docs.kidger.site/diffrax/usage/how-to-choose-a-solver/>`_ guide on how to choose a solver.
        max_steps: The maximum number of steps to take before quitting the computation unconditionally.
        stepsize_controller: The stepsize controller to use.

    Returns:
        The solution to the differential equation. See the `diffrax solution <https://docs.kidger.site/diffrax/api/solution/>`_ documentation for more information.

    Notes:
        This is a wrapper around the :func:`cowell_method` function. See the :func:`cowell_method` function for more information.

    Examples:
        A simple example of propagating an orbit under the influence of a central body and J2 perturbations:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> import diffrax
        >>> def vector_field(t, x, args):
        ...     acc = adx.gravity.point_mass_grav(t, x, args)
        ...     acc += adx.gravity.j2_acc(t, x, args)
        ...     return jnp.concatenate([x[3:], acc])
        >>> args = {"mu": 1.0, "rmin": 0.7, "J2": 1e-6, "R_eq": 1.0}
        >>> t1 = 3.14
        >>> x0 = jnp.array([1.0, 0.0, 0.0, 0.0, 0.9, 0.0])
        >>> event = diffrax.Event(adx.events.radius_toolow)
        >>> term = diffrax.ODETerm(vector_field)
        >>> sol = adx.spprop_varstep(term, x0, t1, args=args, event=event)
        >>> tsmask = jnp.isfinite(sol.ts)
        >>> xf = sol.ys[tsmask][-1]
        >>> expected = jnp.array([-0.59,0.36, 0.,-0.58,-1.16, 0.])
        >>> jnp.allclose(xf, expected, atol=1e-2)
        Array(True, dtype=bool)
    """
    return cowell_method(
        term,
        x0,
        t1,
        args=args,
        event=event,
        solver=solver,
        max_steps=max_steps,
        stepsize_controller=stepsize_controller,
        saveat=diffrax.SaveAt(t0=True, steps=True),
    )


def spprop_finnal(
    term: diffrax.AbstractTerm,
    x0: ArrayLike,
    t1: DTypeLike,
    *,
    args: PyTree[Any] = {"mu": 1.0},
    event: diffrax.Event = None,
    solver: diffrax.AbstractSolver = diffrax.Tsit5(),
    max_steps: int | None = 4096,
    stepsize_controller: diffrax.AbstractStepSizeController = diffrax.PIDController(
        rtol=1e-8, atol=1e-8
    ),
) -> diffrax.Solution:
    """Returns the solution to the differential equation using Cowell's method with variable step sizes.

    Args:
        term: The differential equation to solve.
        x0: The initial state.
        t1: The final time.
        args: Any additional arguments to pass to the differential equation.
        event: An event at which to terminate the solve early. See the `diffrax events <https://docs.kidger.site/diffrax/api/events/>`_ documentation for more information.
        solver: The solver for the differential equation. See the `diffrax solver <https://docs.kidger.site/diffrax/usage/how-to-choose-a-solver/>`_ guide on how to choose a solver.
        max_steps: The maximum number of steps to take before quitting the computation unconditionally.
        stepsize_controller: The stepsize controller to use.

    Returns:
        The solution to the differential equation. See the `diffrax solution <https://docs.kidger.site/diffrax/api/solution/>`_ documentation for more information.

    Notes:
        This is a wrapper around the :func:`cowell_method` function. See the :func:`cowell_method` function for more information.

    Examples:
        A simple example of propagating an orbit under the influence of a central body and J2 perturbations:

        >>> import jax.numpy as jnp
        >>> import astrodynx as adx
        >>> import diffrax
        >>> def vector_field(t, x, args):
        ...     acc = adx.gravity.point_mass_grav(t, x, args)
        ...     acc += adx.gravity.j2_acc(t, x, args)
        ...     return jnp.concatenate([x[3:], acc])
        >>> args = {"mu": 1.0, "rmin": 0.7, "J2": 1e-6, "R_eq": 1.0}
        >>> t1 = 3.14
        >>> x0 = jnp.array([1.0, 0.0, 0.0, 0.0, 0.9, 0.0])
        >>> event = diffrax.Event(adx.events.radius_toolow)
        >>> term = diffrax.ODETerm(vector_field)
        >>> sol = adx.spprop_finnal(term, x0, t1, args=args, event=event)
        >>> expected = jnp.array([-0.59,0.36, 0.,-0.58,-1.16, 0.])
        >>> jnp.allclose(sol.ys[-1], expected, atol=1e-2)
        Array(True, dtype=bool)
    """
    return cowell_method(
        term,
        x0,
        t1,
        args=args,
        event=event,
        solver=solver,
        max_steps=max_steps,
        stepsize_controller=stepsize_controller,
    )
