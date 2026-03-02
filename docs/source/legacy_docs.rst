Documentation Migration
=======================

The markdown-only documentation set has been retired in favor of reST pages under
``docs/source``.

Completed Migration Steps
-------------------------

* Moved high-value scientific and pipeline narrative into dedicated reST pages:
  ``scientific_scope.rst`` and ``methods.rst``.
* Replaced generated markdown API inventory with Sphinx-native autodoc/autosummary pages:
  see ``reference/api_objects.rst`` and ``reference/index.rst``.
* Removed markdown-only docs after confirming no active CI workflow depends on them.
