.. _astrodynx.prop:

``astrodynx.prop``
==================

.. currentmodule:: astrodynx.prop

.. container:: module-header

   .. automodule:: astrodynx.prop
      :no-index:

.. container:: module-content

   .. rubric:: Orbital Dynamics Configuration

   .. autosummary::
      :toctree: generated/
      :template: class.rst
      :nosignatures:

      OrbDynx

   .. rubric:: Kepler Propagation

   .. autosummary::
      :toctree: generated/
      :template: function.rst
      :nosignatures:

      kepler

   .. rubric:: Cowell's Method Propagation

   .. autosummary::
      :toctree: generated/
      :template: function.rst
      :nosignatures:

      fixed_steps
      adaptive_steps
      custom_steps
      to_final
      cowell_method

.. container:: module-footer

   .. rubric:: Module Information

   **Full name:** ``astrodynx.prop``

   **Source:** `View source on GitHub <https://github.com/adxorg/astrodynx/blob/main/src/astrodynx/prop>`_

.. container:: module-overview

   .. rubric:: Module Overview

   The ``astrodynx.prop`` module provides comprehensive orbital propagation capabilities
   using both analytical and numerical methods. It includes:

   **Analytical Methods:**
      - Kepler propagation for two-body problems using universal variables
      - Exact solutions for elliptical, parabolic, and hyperbolic orbits

   **Numerical Methods (Cowell's Method):**
      - Fixed step size integration for uniform time sampling
      - Adaptive step size integration for optimal accuracy/efficiency balance
      - Custom time point output for mission analysis and observations
      - Final state only computation for optimization and sensitivity analysis

   **Key Features:**
      - JAX-compatible for automatic differentiation and vectorization
      - Event detection for ground impact, apogee/perigee passage, etc.
      - Support for arbitrary perturbation forces (J2, drag, solar radiation pressure)
      - Memory-efficient implementations for large-scale simulations
      - High-precision integration with customizable error tolerances

   **Typical Workflow:**
      1. Define orbital dynamics using :class:`OrbDynx` configuration
      2. Choose appropriate propagation method based on requirements
      3. Execute propagation with desired output format
      4. Analyze results or use for further computations

   The module is designed for both research applications requiring high precision
   and operational scenarios demanding computational efficiency.
