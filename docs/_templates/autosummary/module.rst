{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}

{% block attributes %}
{% if attributes %}
.. rubric:: Module Attributes

.. autosummary::
   :toctree:
   :template: attribute.rst
{% for item in attributes %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block functions %}
{% if functions %}
.. rubric:: Functions

.. autosummary::
   :toctree:
   :template: function.rst
   :nosignatures:
{% for item in functions %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block classes %}
{% if classes %}
.. rubric:: Classes

.. autosummary::
   :toctree:
   :template: class.rst
   :nosignatures:
{% for item in classes %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block exceptions %}
{% if exceptions %}
.. rubric:: Exceptions

.. autosummary::
   :toctree:
   :template: class.rst
   :nosignatures:
{% for item in exceptions %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

.. container:: module-navigation

   .. rubric:: Related Modules

   {% if fullname == "astrodynx" %}
   * :mod:`astrodynx.twobody` - Two-body orbital mechanics
   * :mod:`astrodynx.gravity` - Gravitational perturbations
   * :mod:`astrodynx.events` - Event detection functions
   * :mod:`astrodynx.utils` - Utility functions
   {% elif fullname == "astrodynx.twobody" %}
   * :mod:`astrodynx` - Main module functions
   * :mod:`astrodynx.gravity` - Gravitational perturbations
   * :mod:`astrodynx.utils` - Utility functions
   {% elif fullname == "astrodynx.gravity" %}
   * :mod:`astrodynx` - Main module functions
   * :mod:`astrodynx.twobody` - Two-body orbital mechanics
   * :mod:`astrodynx.events` - Event detection functions
   {% elif fullname == "astrodynx.events" %}
   * :mod:`astrodynx` - Main module functions
   * :mod:`astrodynx.gravity` - Gravitational perturbations
   * :mod:`astrodynx.twobody` - Two-body orbital mechanics
   {% elif fullname == "astrodynx.utils" %}
   * :mod:`astrodynx` - Main module functions
   * :mod:`astrodynx.twobody` - Two-body orbital mechanics
   * :mod:`astrodynx.gravity` - Gravitational perturbations
   {% endif %}

.. container:: module-footer

   .. rubric:: Module Information

   **Full name:** ``{{ fullname }}``

   **Source:** `View source on GitHub <https://github.com/adxorg/astrodynx/blob/main/src/{{ fullname.replace('.', '/') }}>`_
