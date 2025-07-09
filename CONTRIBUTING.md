# Contributing to AstroDynX

Thank you for your interest in contributing to AstroDynX! This guide will help you get started with contributing to our modern astrodynamics library.

## Related Guides

This document is part of our comprehensive contribution documentation:

- **[NEWCOMER_GUIDE.md](NEWCOMER_GUIDE.md)** - Perfect starting point for new contributors
- **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)** - Detailed technical guidelines for developers

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment Setup](#development-environment-setup)
- [Development Workflow](#development-workflow)
- [Submitting Changes](#submitting-changes)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Contributing Areas](#contributing-areas)

## Code of Conduct

By participating in this project, you agree to abide by our code of conduct. Please be respectful and constructive in all interactions.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Basic knowledge of JAX and astrodynamics concepts

### Fork and Clone Repository

1. Fork the repository on GitHub by clicking the "Fork" button
2. Clone your fork to your local machine:

```bash
git clone https://github.com/YOUR_USERNAME/astrodynx.git
cd astrodynx
```

3. Add the original repository as an upstream remote:

```bash
git remote add upstream https://github.com/adxorg/astrodynx.git
git fetch upstream
```

## Development Environment Setup

### Devcontainer Setup (Recommended ✅)

The easiest way to set up your development environment is to use the provided devcontainer. This will automatically install all dependencies and set up the environment for you.

1. Install [Visual Studio Code](https://code.visualstudio.com/) (if you haven't already)
2. Install the [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension
3. Open the repository in Visual Studio Code
4. Press `F1`, then select `Remote-Containers: Reopen in Container`
5. Wait for the container to build and start


### Virtual Environment Setup (Optional :😐)

1. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the package in development mode with all dependencies:

```bash
pip install -e .[dev,docs]
```

3. Set up pre-commit hooks (IMPORTANT):

```bash
pre-commit install
```

### Verify Your Environment

Run the test suite to ensure everything is set up correctly:

```bash
pytest
```



Test your JAX installation:

```python
import jax
print(f"JAX version: {jax.__version__}")
print(f"Available devices: {jax.devices()}")
```

## Development Workflow

### 1. Fetch and Rebase
Before starting any work, ensure your local repository is up to date with the upstream repository:

```bash
git fetch upstream
git checkout main
git rebase upstream/main
```
> [!NOTE]
> Always fetch and rebase before starting your work. This will help keep linear history and avoid merge commits.

### 2. Make Your Changes

- Follow the [code standards](#code-standards)
- Write tests for new functionality
- Update documentation as needed
- Keep commits focused and atomic

### 3. Test Your Changes

Run the full test suite:

```bash
pytest
```

Run specific tests:

```bash
pytest tests/test_kepler_equation.py
pytest tests/test_kepler_equation.py::test_specific_function
```

### 4. Check Code Quality

Pre-commit hooks will run automatically, but you can also run them manually:

```bash
pre-commit run --all-files
```

## Submitting Changes

### Before Committing

1. **Run all checks**:

```bash
pytest           # Run tests
git add .        # Stage changes
pre-commit       # Run code quality checks
```

2. **Stash any uncommitted changes**:

```bash
git stash
```

3. **Sync with upstream**:

```bash
git fetch upstream
git rebase upstream/main
```
If there are conflicts, resolve them and continue the rebase with `git rebase --continue`.

4. **Unstash your changes**:

```bash
git stash pop
```


### Commit your Changes

We use [Conventional Commits](https://www.conventionalcommits.org/) format.
Use `cz c` (commitizen) for interactive commit message creation:

```bash
git add .
cz c
```

### Creating a Pull Request
Once your changes are ready and tested, it's time to create a pull request (PR).
1. **Push your changes**:

```bash
git push origin main -f
```

2. **Create PR on GitHub**:
   - Go to your fork on GitHub
   - Click "Compare & pull request"
   - Fill out the PR template
   - Link any related issues

### Review Process

1. **Automated Checks**: GitHub Actions will run tests and checks
2. **Code Review**: Maintainers will review your code
3. **Address Feedback**: Make requested changes
4. **Approval**: Once approved, your PR will be merged



## Code Standards

### Code Style

We use several tools to maintain code quality:

- **Ruff**: For linting and code formatting
- **MyPy**: For static type checking
- **Pre-commit**: For automated checks

### Coding Guidelines

1. **Type Hints**: All functions should have type hints
2. **Docstrings**: Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for all public functions
3. **JAX Best Practices**:
   - Use `jax.numpy` instead of `numpy`
   - Write pure functions when possible
4. **Naming Conventions**:
   - Functions: `snake_case`
   - Classes: `PascalCase`
   - Constants: `UPPER_CASE`
   - Private functions: `_leading_underscore`

## Testing

### Test Structure

Tests are organized in the `tests/` directory, mirroring the source structure:

```
tests/
├── test_import.py              # Basic import tests
├── test_kepler_equation.py     # Kepler equation tests
├── test_orb_integrals.py       # Orbital integrals tests
├── test_rotation_matrix.py     # Rotation matrix tests
```

### Writing Tests

1. **Test Naming**: Use descriptive names starting with `test_`
2. **Test Coverage**: Aim for high test coverage
3. **JAX Testing**: Use `jax.numpy` arrays in tests
4. **Parametrized Tests**: Use `pytest.mark.parametrize` for multiple test cases

### Example Test

```python
import pytest
import jax.numpy as jnp
from astrodynx.twobody import solve_kepler_equation

class TestKeplerEquation:
    """Test suite for Kepler equation solver."""

    @pytest.mark.parametrize("mean_anomaly,eccentricity,expected", [
        (0.0, 0.0, 0.0),
        (jnp.pi/2, 0.1, 1.6709637),
        (jnp.pi, 0.5, jnp.pi),
    ])
    def test_solve_kepler_equation(self, mean_anomaly, eccentricity, expected):
        """Test Kepler equation solver with known values."""
        result = solve_kepler_equation(mean_anomaly, eccentricity)
        assert jnp.allclose(result, expected, rtol=1e-10)
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/astrodynx --cov-report=html

# Run specific test file
pytest tests/test_kepler_equation.py

# Run with verbose output
pytest -v

# Run doctests
pytest --doctest-modules src/astrodynx
```

## Documentation

### Building Documentation

The documentation is built using Sphinx and hosted on GitHub Pages.

1. Install documentation dependencies:

```bash
pip install -e .[docs]
```

2. Build the documentation:

```bash
sphinx-build -b html docs docs/_build/html
```

3. View the documentation locally:

```bash
# Open docs/_build/html/index.html in your browser
python -m http.server 8000 --directory docs/_build/html
```

### Documentation Guidelines

1. **API Documentation**: All public functions must have comprehensive docstrings
2. **Examples**: Include practical examples in docstrings
3. **Tutorials**: Add tutorials for new major features
4. **Mathematical Notation**: Use LaTeX for mathematical expressions

## Contributing Areas

We welcome contributions in various areas:

### Code Contributions

- **New Features**: Orbital mechanics algorithms, coordinate transformations
- **Performance**: JAX optimizations, GPU/TPU support
- **Bug Fixes**: Numerical accuracy, edge cases

### Documentation

- **API Documentation**: Improve docstrings and examples
- **Tutorials**: Step-by-step guides for common tasks
- **Examples**: Practical use cases and best practices
- **Theory**: Mathematical background and references

### Testing

- **Unit Tests**: Increase test coverage
- **Integration Tests**: End-to-end workflows
- **Benchmarks**: Performance testing

### Infrastructure

- **CI/CD**: Improve automation
- **Packaging**: Distribution and installation
- **Tools**: Development workflow improvements

Thank you for contributing to AstroDynX! 🚀
