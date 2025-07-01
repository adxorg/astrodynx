Welcome to AstroDynX!
=====================

A modern astrodynamics library powered by JAX: differentiate, vectorize, JIT to GPU/TPU, and more.

.. image:: https://img.shields.io/pypi/v/astrodynx
   :target: https://pypi.org/project/astrodynx/
.. image:: https://img.shields.io/github/license/adxorg/astrodynx
   :target: https://github.com/adxorg/astrodynx/blob/main/LICENSE
.. image:: https://github.com/adxorg/astrodynx/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/adxorg/astrodynx/actions/workflows/ci.yml
.. image:: https://codecov.io/gh/adxorg/astrodynx/graph/badge.svg?token=azxgWzPIIU
   :target: https://codecov.io/gh/adxorg/astrodynx

What is AstroDynX?
------------------
AstroDynX is a modern astrodynamics library powered by JAX, designed for high-performance scientific computing, automatic differentiation, and GPU/TPU acceleration.

Key Features
------------

**JAX-Powered Performance**
   - Automatic differentiation for sensitivity analysis
   - Vectorization for batch computations
   - JIT compilation for high-performance execution
   - GPU/TPU acceleration support

**Comprehensive Orbital Mechanics**
   - Kepler's equation solvers for elliptical and hyperbolic orbits
   - Orbital element calculations and transformations
   - Two-body dynamics and orbital integrals

**Modern Python Design**
   - Type hints for better code clarity
   - Broadcasting support for array operations
   - Clean, well-documented API

.. warning::
   This project is still experimental, APIs could change between releases without notice.

Installation
------------
Default installation for CPU usage:

.. code-block:: bash

   pip install astrodynx

.. hint::

   AstroDynX is written in pure Python build with JAX, so it is compatible with any platform that supports JAX, including CPU, GPU, and TPU. By default, it installs the CPU version. If you want to use AstroDynX on GPU/TPU, follow the `instructions <https://jax.readthedocs.io/en/latest/installation.html>`_ to install the appropriate JAX backend for your hardware.


Quickstart
----------

Get started with AstroDynX by exploring some basic orbital mechanics calculations:

.. code-block:: python

   import astrodynx as adx
   import jax.numpy as jnp

   # Check the version
   print(f"AstroDynX version: {adx.__version__}")

   # Example 1: Compute the orbital period of an elliptical orbit
   a = 1.0  # semimajor axis
   mu = 1.0  # gravitational parameter
   period = adx.orb_period(a, mu)
   print(f"Orbital period: {period:.4f}")

   # Example 2: Calculate mean anomaly for elliptical orbit
   e = 0.1  # eccentricity
   E = jnp.pi / 4  # eccentric anomaly
   M = adx.kepler_equ_elps(e, E)
   print(f"Mean anomaly: {M:.4f}")

   # Example 3: Calculate mean anomaly for hyperbolic orbit
   e_hyp = 1.5  # hyperbolic eccentricity
   H = 1.0  # hyperbolic eccentric anomaly
   N = adx.kepler_equ_hypb(e_hyp, H)
   print(f"Hyperbolic mean anomaly: {N:.4f}")

   # Example 4: Compute angular momentum
   r = jnp.array([1.0, 0.0, 0.0])  # position vector
   v = jnp.array([0.0, 1.0, 0.0])  # velocity vector
   h = adx.angular_momentum(r, v)
   print(f"Angular momentum: {h}")

Citation
--------
If you use AstroDynX in your academic work, please cite our project:

.. code-block:: bibtex

   @misc{astrodynx2025,
     title={AstroDynX: Modern Astrodynamics with JAX},
     author={Peng SHU and contributors},
     year={2025},
     howpublished={\url{https://github.com/adxorg/astrodynx}}
   }

.. toctree::
   :maxdepth: 2
   :caption: Documentation
   :hidden:

   tutorials/index
   tests
   api/index
   changelog

.. toctree::
   :caption: Indices and tables
   :hidden:

   Index <genindex>
   Module Index <modindex>
