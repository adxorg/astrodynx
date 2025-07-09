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
.. image:: https://app.readthedocs.org/projects/astrodynx/badge/?version=latest
   :target: https://app.readthedocs.org/projects/astrodynx/builds/

What is AstroDynX?
------------------
AstroDynX is a modern astrodynamics library powered by JAX, designed for high-performance scientific computing, automatic differentiation, and GPU/TPU acceleration. Whether you're analyzing satellite orbits, designing interplanetary trajectories, or conducting orbital mechanics research, AstroDynX provides the tools you need with the performance benefits of JAX.

Key Features
------------

**🚀 JAX-Powered Performance**
   - **Automatic differentiation**: Compute gradients for optimization and sensitivity analysis
   - **Vectorization**: Process multiple orbits simultaneously with ``jax.vmap``
   - **JIT compilation**: Achieve near-C performance with ``jax.jit``
   - **GPU/TPU acceleration**: Scale computations to modern hardware

**🛰️ Comprehensive Orbital Mechanics**
   - **Kepler's equation solvers**: Support for elliptical, hyperbolic, and universal formulations
   - **Orbital elements**: Calculate and transform between different orbital representations
   - **Two-body dynamics**: Classical orbital mechanics with modern numerical methods
   - **Orbital propagation**: Both analytical (Kepler) and numerical (Cowell) propagators

**🔧 Advanced Capabilities**
   - **Perturbation modeling**: J2 gravitational harmonics and custom force models
   - **Event detection**: Collision avoidance, ground station passes, and custom events
   - **State transformations**: Position/velocity to orbital elements and vice versa
   - **Coordinate systems**: Rotation matrices and reference frame transformations

**💻 Modern Python Design**
   - **Type hints**: Full type annotations for better IDE support and code clarity
   - **Broadcasting support**: Work with arrays of orbital states naturally
   - **Clean API**: Intuitive function names and consistent parameter conventions
   - **Extensive documentation**: Comprehensive examples and mathematical references

.. warning::
   This project is still experimental, APIs could change between releases without notice.

Installation
------------

**Quick Installation (CPU)**

.. code-block:: bash

   pip install astrodynx

**GPU/TPU Installation**

For GPU or TPU acceleration, install the appropriate JAX backend first:

.. code-block:: bash

   # For NVIDIA GPUs (CUDA 12)
   pip install "jax[cuda12]"
   pip install astrodynx

   # For Google TPUs
   pip install "jax[tpu]"
   pip install astrodynx

.. hint::

   AstroDynX is written in pure Python with JAX, making it compatible with any platform that supports JAX. For detailed GPU/TPU setup instructions, see the `JAX installation guide <https://jax.readthedocs.io/en/latest/installation.html>`_.

Quick Start Examples
--------------------

**Basic Orbital Calculations**

.. code-block:: python

   import astrodynx as adx
   import jax.numpy as jnp

   # Orbital period calculation
   a = 1.0  # semimajor axis (AU)
   mu = 1.0  # gravitational parameter
   period = adx.orb_period(a, mu)
   print(f"Orbital period: {period:.4f} time units")

   # Angular momentum from state vectors
   r = jnp.array([1.0, 0.0, 0.0])  # position vector
   v = jnp.array([0.0, 1.0, 0.0])  # velocity vector
   h = adx.angular_momentum(r, v)
   print(f"Angular momentum: {h}")

**Kepler's Equation Solving**

.. code-block:: python

   # Solve Kepler's equation for elliptical orbit
   M = 1.0  # mean anomaly (radians)
   e = 0.1  # eccentricity
   E = adx.solve_kepler_elps(M, e)
   print(f"Eccentric anomaly: {E:.4f} rad")

   # Solve for hyperbolic orbit
   N = 1.0  # hyperbolic mean anomaly
   e_hyp = 1.5  # hyperbolic eccentricity
   H = adx.solve_kepler_hypb(N, e_hyp)
   print(f"Hyperbolic eccentric anomaly: {H:.4f} rad")

**Orbital Propagation**

.. code-block:: python

   # Propagate orbit using Kepler's method
   r0 = jnp.array([1.0, 0.0, 0.0])  # initial position
   v0 = jnp.array([0.0, 1.0, 0.0])  # initial velocity
   dt = jnp.pi  # time step (half orbit)
   mu = 1.0  # gravitational parameter

   r_new, v_new = adx.kepler_prop(dt, r0, v0, mu)
   print(f"New position: {r_new}")
   print(f"New velocity: {v_new}")

**JAX Features in Action**

.. code-block:: python

   import jax

   # Vectorize over multiple orbits
   multiple_a = jnp.array([1.0, 2.0, 3.0])  # multiple semimajor axes
   periods = jax.vmap(adx.orb_period)(multiple_a, mu)
   print(f"Multiple periods: {periods}")

   # Automatic differentiation for sensitivity analysis
   def period_func(a):
       return adx.orb_period(a, mu)

   dP_da = jax.grad(period_func)(1.0)  # derivative of period w.r.t. semimajor axis
   print(f"dP/da = {dP_da:.4f}")

Citation
--------
If you use AstroDynX in your work, please cite our project:

.. code-block:: bibtex

   @misc{astrodynx2025,
     title={AstroDynX: Modern Astrodynamics with JAX},
     author={Peng SHU and contributors},
     year={2025},
     howpublished={\url{https://github.com/adxorg/astrodynx}}
   }

.. toctree::
   :maxdepth: 2
   :hidden:

   basics
   tutorials/index
   examples/index
   api/index
   changelog

   Index <genindex>
   Module Index <modindex>
