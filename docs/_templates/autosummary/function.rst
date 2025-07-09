{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. container:: function-header

   .. autofunction:: {{ objname }}
      :no-index:

.. container:: function-details

   .. autofunction:: {{ objname }}
      :noindex:

.. container:: function-navigation

   .. rubric:: Related Functions

   {% if module == "astrodynx" %}
   * :func:`astrodynx.rv2coe` - Convert position/velocity to orbital elements
   * :func:`astrodynx.coe2rv` - Convert orbital elements to position/velocity
   * :func:`astrodynx.kepler_prop` - Kepler orbital propagation
   * :func:`astrodynx.lagrange_prop` - Lagrange orbital propagation
   {% elif module == "astrodynx.twobody" %}
   * :func:`astrodynx.twobody.lagrange_F` - Lagrange F coefficient
   * :func:`astrodynx.twobody.lagrange_G` - Lagrange G coefficient
   * :func:`astrodynx.twobody.sigma_fn` - Universal anomaly sigma function
   {% elif module == "astrodynx.gravity" %}
   * :func:`astrodynx.gravity.point_mass_grav` - Point mass gravity acceleration
   * :func:`astrodynx.gravity.j2_acc` - J2 perturbation acceleration
   {% elif module == "astrodynx.utils" %}
   * :func:`astrodynx.utils.rotmat3dx` - Rotation matrix about X-axis
   * :func:`astrodynx.utils.rotmat3dy` - Rotation matrix about Y-axis
   * :func:`astrodynx.utils.rotmat3dz` - Rotation matrix about Z-axis
   {% endif %}

.. container:: function-footer

   .. rubric:: Module Information

   **Module:** :mod:`{{ module }}`

   **Source:** `View source on GitHub <https://github.com/adxorg/astrodynx/blob/main/src/{{ module.replace('.', '/') }}>`_
