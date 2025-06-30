# Welcome to AstroDynX! 🚀

Thank you for your interest in contributing to AstroDynX! This guide will help you get started as a new contributor.

## What is AstroDynX?

AstroDynX is a modern astrodynamics library built with JAX, designed for:
- High-performance orbital mechanics computations
- Automatic differentiation for optimization problems
- GPU/TPU acceleration for large-scale simulations
- Modern Python development practices

## How Can You Contribute?

There are many ways to contribute to AstroDynX, regardless of your experience level:

### 🐛 Bug Reports
- Found something that doesn't work? Report it!
- Use our [bug report template](.github/ISSUE_TEMPLATE/bug_report.md)
- Include minimal code examples that reproduce the issue

### 📚 Documentation
- Improve API documentation
- Write tutorials and examples
- Fix typos and clarify explanations
- Add mathematical background for algorithms

### 🧪 Testing
- Write tests for existing functionality
- Add edge case testing
- Improve test coverage
- Create performance benchmarks

### ✨ New Features
- Implement new orbital mechanics algorithms
- Add coordinate system transformations
- Optimize existing functions
- Enhance GPU/TPU support

### 🎨 Code Quality
- Improve code style and readability
- Add type hints
- Optimize performance
- Refactor complex functions

## Getting Started

### 1. Choose Your First Contribution

#### For Beginners
Look for issues labeled with:
- `good first issue` - Perfect for newcomers
- `documentation` - Improve docs and examples
- `help wanted` - Community input needed

#### For Experienced Developers
Consider:
- `enhancement` - New features and improvements
- `performance` - Optimization opportunities
- `algorithm` - New orbital mechanics implementations

### 2. Set Up Your Development Environment

Follow our [CONTRIBUTING.md](CONTRIBUTING.md) guide:

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/astrodynx.git
cd astrodynx

# Set up development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .[dev,docs]
pre-commit install

# Verify installation
pytest
```

### 3. Understand the Codebase

#### Project Structure
```
astrodynx/
├── src/astrodynx/           # Main package
│   ├── twobody/            # Two-body orbital mechanics
│   └── utils/              # Utility functions
├── tests/                  # Test suite
├── docs/                   # Documentation
└── benchmark/              # Performance benchmarks
```

#### Key Technologies
- **JAX**: For numerical computing and automatic differentiation
- **Python 3.10+**: Modern Python features
- **Pytest**: For testing
- **Sphinx**: For documentation
- **Pre-commit**: For code quality

### 4. Make Your First Contribution

#### Small Documentation Fix
Perfect first contribution:

1. Find a typo or unclear explanation
2. Fork the repository
3. Make the fix
4. Submit a pull request

#### Add a Simple Test
Great way to understand the codebase:

1. Look at existing tests in `tests/`
2. Find a function that needs more test coverage
3. Write a test following existing patterns
4. Submit your improvement

#### Example: Adding a Test

```python
# In tests/test_kepler_equation.py
import pytest
import jax.numpy as jnp
from astrodynx.twobody import solve_kepler_equation

def test_kepler_equation_zero_eccentricity():
    """Test Kepler equation for circular orbit (e=0)."""
    mean_anomaly = jnp.pi / 2
    eccentricity = 0.0

    eccentric_anomaly = solve_kepler_equation(mean_anomaly, eccentricity)

    # For circular orbits, E = M
    assert jnp.allclose(eccentric_anomaly, mean_anomaly)
```

## Learning Resources

### Astrodynamics Background
- **Curtis, H. D.** - "Orbital Mechanics for Engineering Students"
- **Vallado, D. A.** - "Fundamentals of Astrodynamics and Applications"
- **Battin, R. H.** - "An Introduction to the Mathematics and Methods of Astrodynamics"

### JAX Resources
- [JAX Documentation](https://jax.readthedocs.io/)
- [JAX Tutorial](https://jax.readthedocs.io/en/latest/jax-101/index.html)
- [JAX for Scientific Computing](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html)

### Python Development
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [Pytest Documentation](https://docs.pytest.org/)
- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/en/latest/format.html)

## Common Beginner Questions

### Q: I'm new to astrodynamics. Can I still contribute?
**A:** Absolutely! You can help with:
- Documentation improvements
- Testing and bug reports
- Code quality improvements
- Learning alongside the community

### Q: I don't know JAX. Is that a problem?
**A:** Not at all! JAX is similar to NumPy with some additional features. You can:
- Start with documentation contributions
- Learn JAX through our examples
- Ask questions in issues or discussions

### Q: How do I know if my contribution is valuable?
**A:** All contributions are valuable! Even small improvements like:
- Fixing typos
- Adding comments
- Improving error messages
- Writing examples

### Q: What if I make a mistake?
**A:** Mistakes are part of learning! Our review process will help you:
- Learn best practices
- Improve your code
- Understand the project better

## Getting Help

### Communication Channels
- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Pull Request Comments**: For code-specific discussions

### Asking Good Questions
When asking for help:

1. **Be Specific**: Include error messages, code snippets, and expected behavior
2. **Provide Context**: Explain what you're trying to achieve
3. **Show Your Work**: Share what you've already tried
4. **Be Patient**: Remember this is a volunteer-driven project

### Example Good Question
> "I'm trying to add a test for the `orbital_period` function, but I'm getting a shape mismatch error. Here's my code:
>
> ```python
> def test_orbital_period():
>     a = jnp.array([7000e3, 8000e3])
>     mu = 3.986004418e14
>     T = orbital_period(a, mu)
>     # Error: shapes don't match
> ```
>
> I expected this to work with array inputs. What am I missing?"

## Recognition and Growth

### Contributor Recognition
- Contributors are acknowledged in release notes
- Significant contributions may lead to maintainer opportunities
- Your GitHub profile will show your open source contributions

### Learning Opportunities
Contributing to AstroDynX helps you:
- Learn modern Python development practices
- Understand JAX and high-performance computing
- Gain experience with astrodynamics algorithms
- Build your open source portfolio

## Next Steps

1. **Explore the Codebase**: Browse the source code and tests
2. **Read the Documentation**: Understand the API and examples
3. **Find an Issue**: Look for `good first issue` labels
4. **Join the Community**: Participate in discussions
5. **Make Your First PR**: Start with something small and manageable

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:
- Be respectful and constructive
- Help others learn and grow
- Focus on the code, not the person
- Celebrate diverse perspectives and experiences

## Thank You!

Your interest in contributing to AstroDynX is greatly appreciated. Whether you're fixing a typo, adding a feature, or helping other contributors, you're making the project better for everyone.

Welcome to the AstroDynX community! 🌟

---

**Ready to contribute?** Check out our [CONTRIBUTING.md](CONTRIBUTING.md) for detailed instructions, or browse [open issues](https://github.com/adxorg/astrodynx/issues) to find something that interests you.
