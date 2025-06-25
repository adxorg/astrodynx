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

Features
--------
- JAX support: automatic differentiation, vectorization, JIT compilation
- Modern Python code style and type checking
- Continuous integration and automated testing
- Easy to extend and contribute

Installation
------------
Default installation for CPU usage:

.. code-block:: bash

   pip install astrodynx

.. hint::

   AstroDynX is written in pure Python build with JAX, so it is compatible with any platform that supports JAX, including CPU, GPU, and TPU. By default, it installs the CPU version. If you want to use AstroDynX on GPU/TPU, follow the `instructions <https://jax.readthedocs.io/en/latest/installation.html>`_ to install the appropriate JAX backend for your hardware.


Quickstart
----------
.. code-block:: python

   import astrodynx as adx
   print(adx.__version__)

   # Example: Compute the orbit period of an elliptical orbit
   from astrodynx.twobody.orb_integrals import orb_period
   a = 1.0
   mu = 1.0
   orb_period(a, mu)

Philosophy
----------
AstroDynX aims to provide efficient, composable, and extensible astrodynamics tools for research and engineering users, leveraging the modern numerical computing capabilities of JAX.

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
   :caption: Contents

   tutorials/index
   examples/index
   api
   changelog

.. toctree::
   :caption: Indices and tables

   Index <genindex>
   Module Index <modindex>
