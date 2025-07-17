from jax import numpy as jnp
from jaxtyping import ArrayLike, DTypeLike, PyTree, Array
from typing import Any

"""Events for propagating orbits."""


def radius_islow(
    t: DTypeLike, x: ArrayLike, args: PyTree[Any] = {"rmin": 0.0}, **kwargs: Any
) -> Array:
    r"""Returns the difference between the radius and the minimum radius.

    Args:
        t: The time.
        x: (N,) The state vector, where the first 3 elements are the position vector.
        args: Static arguments, which must contain a key "rmin" with the minimum radius.
        kwargs: Any additional arguments.

    Returns:
        The difference between the radius and the minimum radius.
        The propagation will stop when this value is negative.

    Notes:
        The difference between the radius and the minimum radius is calculated using the following equation:
        $$
        r - r_{min}
        $$
        where $r$ is the radius and $r_{min}$ is the minimum radius.

        This event can be used to terminate the propagation of an orbit when the radius falls below a certain threshold.
    """
    rmin = args["rmin"]
    return jnp.linalg.vector_norm(x[:3]) - rmin
