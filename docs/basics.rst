.. _basics:

AstroDynX Basics
================

This section provides an overview of the basic concepts and features of AstroDynX, a modern astrodynamics library powered by JAX. Whether you're new to orbital mechanics or transitioning from other astrodynamics libraries, this guide will help you understand the core concepts and get started with AstroDynX.

What is AstroDynX?
------------------

AstroDynX is a Python library designed for high-performance astrodynamics computations. Built on top of JAX, it provides:

- **Automatic differentiation** for optimization and sensitivity analysis
- **Vectorization** to process multiple orbits simultaneously
- **JIT compilation** for near-C performance
- **GPU/TPU acceleration** for large-scale computations
- **Modern Python design** with full type annotations


Basic Orbital Mechanics
-----------------------

Orbital Elements and Integrals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AstroDynX provides functions to calculate fundamental orbital quantities:

.. code-block:: python

   import astrodynx as adx
   import jax.numpy as jnp

   # Define orbital state
   r = jnp.array([7000.0, 0.0, 0.0])  # km
   v = jnp.array([0.0, 7.5, 0.0])     # km/s
   mu = 398600.4418                    # km³/s²

   # Calculate orbital elements
   h = adx.angular_momentum(r, v)      # Specific angular momentum
   a = adx.semimajor_axis(jnp.linalg.vector_norm(r),
                         jnp.linalg.vector_norm(v), mu)  # Semimajor axis
   e_vec = adx.eccentricity_vector(r, v, mu)  # Eccentricity vector
   e = jnp.linalg.vector_norm(e_vec)   # Eccentricity magnitude

   # Orbital period and mean motion
   period = adx.orb_period(a, mu)      # Orbital period
   n = adx.mean_motion(period)         # Mean motion

Kepler's Equation
~~~~~~~~~~~~~~~~~

Kepler's equation relates time to position in an orbit. AstroDynX provides solvers for different orbit types:

**Elliptical Orbits (e < 1)**

.. code-block:: python

   # Solve Kepler's equation for elliptical orbits
   M = 1.5  # Mean anomaly (radians)
   e = 0.2  # Eccentricity
   E = adx.solve_kepler_elps(M, e)  # Eccentric anomaly

   # Verify the solution
   residual = adx.kepler_equ_elps(E, e, M)  # Should be ~0

**Hyperbolic Orbits (e > 1)**

.. code-block:: python

   # Solve Kepler's equation for hyperbolic orbits
   N = 2.0  # Hyperbolic mean anomaly
   e = 1.5  # Hyperbolic eccentricity
   H = adx.solve_kepler_hypb(N, e)  # Hyperbolic eccentric anomaly

**Universal Formulation**

.. code-block:: python

   # Universal Kepler's equation (works for all orbit types)
   dt = 3600.0  # Time interval (seconds)
   r0 = jnp.linalg.vector_norm(r)  # Initial radius
   alpha = 1.0 / a  # Reciprocal semimajor axis
   sigma0 = jnp.dot(r, v) / jnp.sqrt(mu)  # Initial radial velocity parameter

   chi = adx.solve_kepler_uni(dt, alpha, r0, sigma0, mu)

Orbital Propagation
-------------------

Analytical Propagation (Kepler)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For unperturbed two-body motion, use Kepler propagation:

.. code-block:: python

   # Initial state
   r0 = jnp.array([7000.0, 0.0, 0.0])  # km
   v0 = jnp.array([0.0, 7.5, 0.0])     # km/s
   mu = 398600.4418                     # km³/s²
   dt = 1800.0                          # 30 minutes

   # Propagate using Kepler's method
   r_new, v_new = adx.kepler_prop(dt, r0, v0, mu)

   print(f"New position: {r_new}")
   print(f"New velocity: {v_new}")

Numerical Propagation (Cowell)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For perturbed motion or when high precision is needed:

.. code-block:: python

   # Define acceleration function (two-body + perturbations)
   def acceleration(t, state, mu):
       r = state[:3]
       r_mag = jnp.linalg.vector_norm(r)
       return -mu * r / r_mag**3  # Two-body acceleration

   # Initial state vector [x, y, z, vx, vy, vz]
   state0 = jnp.concatenate([r0, v0])

   # Propagate using Cowell's method
   t_span = jnp.array([0.0, dt])
   result = adx.cowell_method(acceleration, state0, t_span, mu)

Working with JAX Features
-------------------------

Vectorization with vmap
~~~~~~~~~~~~~~~~~~~~~~~

Process multiple orbits simultaneously:

.. code-block:: python

   import jax

   # Multiple semimajor axes
   a_array = jnp.array([7000.0, 8000.0, 9000.0])  # km
   mu = 398600.4418

   # Vectorized period calculation
   periods = jax.vmap(adx.orb_period, in_axes=(0, None))(a_array, mu)
   print(f"Periods: {periods}")

   # Multiple initial states
   r_array = jnp.array([[7000.0, 0.0, 0.0],
                        [8000.0, 0.0, 0.0],
                        [9000.0, 0.0, 0.0]])
   v_array = jnp.array([[0.0, 7.5, 0.0],
                        [0.0, 7.0, 0.0],
                        [0.0, 6.5, 0.0]])

   # Vectorized propagation
   r_new_array, v_new_array = jax.vmap(adx.kepler_prop, in_axes=(None, 0, 0, None))(
       dt, r_array, v_array, mu)

Automatic Differentiation
~~~~~~~~~~~~~~~~~~~~~~~~~

Compute gradients for optimization and sensitivity analysis:

.. code-block:: python

   # Sensitivity of orbital period to semimajor axis
   def period_func(a):
       return adx.orb_period(a, mu)

   # Compute derivative dP/da
   dP_da = jax.grad(period_func)(7000.0)
   print(f"dP/da = {dP_da:.6f} s/km")

   # Sensitivity of final position to initial velocity
   def propagation_func(v0):
       r_final, _ = adx.kepler_prop(dt, r0, v0, mu)
       return r_final

   # Compute Jacobian dr_final/dv0
   jacobian = jax.jacfwd(propagation_func)(v0)
   print(f"Position sensitivity matrix:\n{jacobian}")

JIT Compilation
~~~~~~~~~~~~~~~

Accelerate computations with just-in-time compilation:

.. code-block:: python

   # JIT compile a function for better performance
   @jax.jit
   def fast_propagation(dt, r0, v0, mu):
       return adx.kepler_prop(dt, r0, v0, mu)

   # First call compiles the function
   r_new, v_new = fast_propagation(dt, r0, v0, mu)

   # Subsequent calls are much faster
   r_new2, v_new2 = fast_propagation(2*dt, r0, v0, mu)

Advanced Features
-----------------

Perturbation Modeling
~~~~~~~~~~~~~~~~~~~~~

AstroDynX supports various perturbation models for more realistic orbital dynamics:

.. code-block:: python

   # J2 gravitational perturbation
   from astrodynx.gravity import j2_acceleration

   def perturbed_acceleration(t, state, mu, J2, R_body):
       r = state[:3]

       # Two-body acceleration
       r_mag = jnp.linalg.vector_norm(r)
       a_twobody = -mu * r / r_mag**3

       # J2 perturbation
       a_j2 = j2_acceleration(r, mu, J2, R_body)

       return a_twobody + a_j2

Event Detection
~~~~~~~~~~~~~~~

Detect specific events during orbital propagation:

.. code-block:: python

   from astrodynx import events

   # Define an event (e.g., ground station pass)
   def ground_station_event(t, state, gs_position):
       r = state[:3]
       # Check if satellite is above ground station
       elevation = events.elevation_angle(r, gs_position)
       return elevation - jnp.deg2rad(10.0)  # 10-degree elevation threshold

   # Use event detection during propagation
   # (Implementation depends on specific event detection framework)

Coordinate Transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~

Transform between different reference frames:

.. code-block:: python

   from astrodynx.utils import rotation_matrix

   # Rotation about z-axis
   angle = jnp.pi / 4  # 45 degrees
   R_z = rotation_matrix(angle, axis='z')

   # Transform position vector
   r_rotated = R_z @ r

   # Transform velocity vector
   v_rotated = R_z @ v

Common Patterns and Best Practices
----------------------------------

Error Handling
~~~~~~~~~~~~~~

AstroDynX functions are designed to work with JAX's functional programming paradigm:

.. code-block:: python

   # Check for valid inputs
   def safe_propagation(dt, r0, v0, mu):
       # Ensure positive time step
       dt = jnp.abs(dt)

       # Ensure positive gravitational parameter
       mu = jnp.abs(mu)

       # Check for zero velocity (degenerate case)
       v_mag = jnp.linalg.vector_norm(v0)
       v0 = jnp.where(v_mag > 1e-12, v0, jnp.array([0.0, 1e-6, 0.0]))

       return adx.kepler_prop(dt, r0, v0, mu)

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

Tips for optimal performance:

.. code-block:: python

   # 1. Use JIT compilation for repeated computations
   @jax.jit
   def optimized_function(inputs):
       # Your computation here
       pass

   # 2. Vectorize operations when possible
   vectorized_func = jax.vmap(single_orbit_func)

   # 3. Avoid Python loops in favor of JAX operations
   # Bad: Python loop
   results = []
   for i in range(n):
       result = some_function(data[i])
       results.append(result)

   # Good: Vectorized operation
   results = jax.vmap(some_function)(data)

Memory Management
~~~~~~~~~~~~~~~~~

JAX arrays are immutable, which affects memory usage patterns:

.. code-block:: python

   # Efficient: Reuse arrays when possible
   def efficient_computation(state_array):
       # Process all states at once
       return jax.vmap(process_single_state)(state_array)

   # Less efficient: Creating many intermediate arrays
   def inefficient_computation(state_array):
       results = []
       for state in state_array:
           result = process_single_state(state)
           results.append(result)  # Creates new arrays
       return jnp.array(results)

Common Gotchas
--------------

Unit Consistency
~~~~~~~~~~~~~~~~

Always ensure consistent units throughout your calculations:

.. code-block:: python

   # Good: Consistent units
   r_km = jnp.array([7000.0, 0.0, 0.0])  # km
   v_km_s = jnp.array([0.0, 7.5, 0.0])   # km/s
   mu_km3_s2 = 398600.4418               # km³/s²

   # Bad: Mixed units
   r_m = jnp.array([7000000.0, 0.0, 0.0])  # meters
   v_km_s = jnp.array([0.0, 7.5, 0.0])     # km/s (inconsistent!)
   mu_km3_s2 = 398600.4418                 # km³/s²

Array Shapes
~~~~~~~~~~~~

Be mindful of array broadcasting rules:

.. code-block:: python

   # Single orbit
   r = jnp.array([7000.0, 0.0, 0.0])  # Shape: (3,)
   v = jnp.array([0.0, 7.5, 0.0])     # Shape: (3,)

   # Multiple orbits
   r_multi = jnp.array([[7000.0, 0.0, 0.0],
                        [8000.0, 0.0, 0.0]])  # Shape: (2, 3)
   v_multi = jnp.array([[0.0, 7.5, 0.0],
                        [0.0, 7.0, 0.0]])     # Shape: (2, 3)

Numerical Precision
~~~~~~~~~~~~~~~~~~~

Be aware of floating-point precision limitations:

.. code-block:: python

   # For high-precision applications, consider using float64
   r = jnp.array([7000.0, 0.0, 0.0], dtype=jnp.float64)
   v = jnp.array([0.0, 7.5, 0.0], dtype=jnp.float64)

   # Check for numerical issues
   def check_orbit_validity(r, v, mu):
       energy = 0.5 * jnp.dot(v, v) - mu / jnp.linalg.vector_norm(r)
       h = adx.angular_momentum(r, v)
       h_mag = jnp.linalg.vector_norm(h)

       # Check for reasonable values
       assert jnp.isfinite(energy), "Energy is not finite"
       assert h_mag > 1e-12, "Angular momentum is too small"
       assert jnp.linalg.vector_norm(r) > 1e-6, "Position is too close to origin"

Next Steps
----------

Now that you understand the basics of AstroDynX, you can:

1. **Explore the tutorials** - Work through detailed examples in the :doc:`tutorials/index` section
2. **Check out examples** - See practical applications in the :doc:`examples/index` section
3. **Read the API documentation** - Get detailed information about all functions in :doc:`api/index`
4. **Contribute to the project** - Help improve AstroDynX by following the contribution guidelines

Key Resources
~~~~~~~~~~~~~

- **GitHub Repository**: https://github.com/adxorg/astrodynx
- **Documentation**: https://astrodynx.readthedocs.io/
- **JAX Documentation**: https://jax.readthedocs.io/
- **Issue Tracker**: https://github.com/adxorg/astrodynx/issues

.. tip::
   Start with simple two-body problems and gradually add complexity as you become more familiar with the library. The JAX ecosystem has excellent documentation and community support for advanced features.
