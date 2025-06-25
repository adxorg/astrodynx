# Copilot Instructions
AstroDynX is a modern astrodynamics library powered by JAX, designed for high-performance scientific computing, automatic differentiation, and GPU/TPU acceleration.

## General Guidelines
- Follow the existing code style and conventions in this repository.
- Prefer clear, concise, and well-documented code.
- Use type annotations where appropriate.
- Write docstrings for all public functions and classes, following the Google style guide.
- Prefer `jax.numpy` over `numpy` for numerical operations.
- Use `Array`, `ArrayLike` and `DTypeLike` for type hints. Prefer `ArrayLike` for array inputs and `Array` for array outputs.
- Ensure compatibility with JAX and avoid using features incompatible with JAX transformations (e.g., in-place array modifications).
- Functional programming paradigms are encouraged, such as using `jax.lax` for control flow and `jax.vmap` for vectorization.

## Testing
- Place all tests in the `tests/` directory.
- Tests for a function should be wrapped in a single classes named `Test<FunctionName>`.

## Documentation
- Write all docstrings in English.
- Use LaTeX syntax for mathematical expressions in docstrings.
- Reference relevant literature where appropriate.
- Add examples in docstrings using the `Examples` section, including a broadcasting example if applicable.

## Imports
- Use absolute imports within the `src/astrodynx` package.

## Miscellaneous
- Avoid hardcoding constants; use parameters or configuration files where possible.
- Ensure all new code passes linting and tests before committing.
