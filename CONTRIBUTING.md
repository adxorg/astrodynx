# Contributing Guide

## Clone repository
Fork the source repository to your repository on GitHub.

Then clone it to your local machine, and add the original repository as an upstream remote to keep your fork up to date.
```bash
git clone <your-fork-url>
cd astrodynx
git remote add upstream https://github.com/adxorg/astrodynx.git -t main
```

## Setup development environment
Install development dependencies.
```bash
pip install -e .[dev]
pre-commit install # IMPORTANT: This sets up pre-commit hooks to run checks before committing.
```
If you want to run on NVIDIA GPU, ensure you have the necessary drivers and CUDA toolkit installed. Then install the JAX library with CUDA support:
```bash
pip install -U "jax[cuda12]"
```
If you want to run on GOOGLE TPU, ensure you have the necessary drivers and TPU runtime installed. Then install the JAX library with TPU support:
```bash
pip install -U "jax[tpu]"
```

Test the installation.
```bash
pytest
```

## Development workflow
Make changes to the codebase, then test your changes.
```bash
pytest
```

## Commit changes

Stash any uncommitted changes
```bash
git stash
```

Before committing, ensure your local repository is still up to date with the upstream repository.
```bash
git fetch upstream
git checkout main
git rebase upstream/main
```
If there are conflicts, resolve them and continue the rebase.
```bash
git rebase --continue
```
Then, pop your stashed changes
```bash
git stash pop
```
If you have made changes to the code, run the tests again to ensure everything is working.
```bash
pytest
```
If all tests pass, you can proceed to commit your changes.
```bash
git add .
pre-commit
cz c
```
Push your changes to your fork
```bash
git push -f origin main
```

## Create a pull request
Go to the original repository on GitHub and create a pull request from your fork's `main` branch to the original repository's `main` branch. Provide a clear description of your changes and any relevant information.

## Check github actions
After creating the pull request, GitHub Actions will automatically run tests and checks on your code. Ensure that all checks pass before the pull request can be merged.


## Build docs
```bash
pip install -e .[docs]
sphinx-build -b html docs docs/_build/html
```
