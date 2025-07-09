{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :undoc-members:
   :show-inheritance:

{% block methods %}
{% if methods %}
.. rubric:: Methods

.. autosummary::
   :toctree:
   :template: method.rst
   :nosignatures:
{% for item in methods %}
   ~{{ name }}.{{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block attributes %}
{% if attributes %}
.. rubric:: Attributes

.. autosummary::
   :toctree:
   :template: attribute.rst
{% for item in attributes %}
   ~{{ name }}.{{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

.. container:: class-navigation

   .. rubric:: Related Classes

   {% if module == "astrodynx" %}
   * Browse all :mod:`astrodynx` functions
   * Explore :mod:`astrodynx.twobody` module
   {% elif module == "astrodynx.twobody" %}
   * Browse :mod:`astrodynx.twobody` functions
   * Explore :mod:`astrodynx` main module
   {% endif %}

.. container:: class-footer

   .. rubric:: Class Information

   **Module:** :mod:`{{ module }}`

   **Full name:** ``{{ fullname }}``

   **Source:** `View source on GitHub <https://github.com/adxorg/astrodynx/blob/main/src/{{ module.replace('.', '/') }}>`_
