Hosting the Docs on GitHub Pages (Free)
=======================================

This repository includes a GitHub Actions workflow that builds the Sphinx site and deploys it to GitHub Pages.

Cost
----

GitHub Pages hosting for a repository documentation site is free (no extra hosting cost) for standard public-repo docs sites.

How deployment works
--------------------

The workflow in ``.github/workflows/docs-pages.yml``:

* builds the HTML site with Sphinx
* uploads the generated HTML as a Pages artifact
* deploys the artifact to GitHub Pages on pushes to ``main``

Pull requests are validated by ``.github/workflows/benchmark-analysis-docs.yml`` (now a Sphinx docs build check).

One-time repository setup
-------------------------

In GitHub repository settings:

1. Open ``Settings -> Pages``.
2. Under **Build and deployment**, choose **Source: GitHub Actions**.

After that, pushes to ``main`` will publish the site automatically.

Published URL pattern:

* ``https://<github-username>.github.io/<repository-name>/``

Local preview
-------------

Install docs dependencies and build locally:

.. code-block:: bash

   python -m pip install -r docs/requirements.txt
   make -C docs html

Then open ``docs/_build/html/index.html`` in a browser.
