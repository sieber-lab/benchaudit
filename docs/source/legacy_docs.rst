Legacy Markdown Docs
====================

The repository still contains legacy markdown documents under ``docs/`` (including generated artifacts used by CI checks),
but the published documentation site is now Sphinx + reStructuredText.

Why this split exists
---------------------

* It avoids breaking existing CI checks and generated-file workflows immediately.
* It lets the project publish a proper browsable docs website now.
* It supports incremental migration of older markdown files into reST pages.

Current legacy files (not the primary website source)
-----------------------------------------------------

* ``docs/benchmark_and_analysis_class_reference.md`` (generated legacy reference)
* ``docs/pipeline_overview.md``
* ``docs/scientific_documentation.md``
* ``docs/publishing_and_installation.md`` (content now mirrored here in reST form)

Migration strategy
------------------

Preferred next steps:

* move high-value narrative content from legacy markdown into reST pages
* replace generated markdown references with Sphinx autodoc / autosummary pages
* retire markdown-only docs once no workflow depends on them
