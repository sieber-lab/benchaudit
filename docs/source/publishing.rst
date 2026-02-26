Publishing to PyPI
==================

Publishing is handled by ``.github/workflows/publish-pypi.yml`` using PyPI Trusted Publishing.

Release flow
------------

1. Configure a PyPI Trusted Publisher for this repository/workflow.
2. Bump ``project.version`` in ``pyproject.toml``.
3. Commit and push the version bump.
4. Create and push a matching tag ``vX.Y.Z``.

Example:

.. code-block:: bash

   git tag v0.1.1
   git push origin v0.1.1

What the workflow does
----------------------

* verifies the git tag matches ``pyproject.toml`` version
* builds distributions
* validates artifacts
* publishes to PyPI via OIDC / Trusted Publishing

Repository visibility behavior
------------------------------

The publish workflow is designed to no-op while the repository is private and publish only when the repository is public.

References
----------

* PyPI package page: https://pypi.org/project/benchaudit/
* uv docs: https://docs.astral.sh/uv/
* PyPI Trusted Publishers: https://docs.pypi.org/trusted-publishers/
