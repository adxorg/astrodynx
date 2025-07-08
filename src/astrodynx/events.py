from jax import numpy as jnp
from jaxtyping import ArrayLike, DTypeLike, PyTree, Array
from typing import Any

"""Events for propagating orbits."""


def radius_toolow(
    t: DTypeLike, x: ArrayLike, args: PyTree[Any] = {"rmin": 0.0}, **kwargs: Any
) -> Array:
    r"""Returns True if the radius is lower than the minimum radius.

    Args:
        t: The time.
        x: The state vector.
        args: Static arguments.
        kwargs: Any additional arguments.

    Returns:
        True if the radius is lower than the minimum radius, False otherwise.

    Notes:
        This event can be used to terminate the propagation of an orbit when the radius falls below a certain threshold.
    """
    rmin = args["rmin"]
    return jnp.linalg.norm(x[:3]) < rmin
