Legacy Markdown Docs
====================

The repository still contains legacy markdown documents under ``docs/`` (including generated artifacts used by CI checks).
They are retained for compatibility while scientific and pipeline guidance is maintained in the reST pages.

Why this split exists
---------------------

* It avoids breaking existing CI checks and generated-file workflows immediately.
* It supports incremental migration of older markdown files into focused reST pages.

Current legacy files
--------------------

* ``docs/benchmark_and_analysis_class_reference.md`` (generated legacy reference)
* ``docs/pipeline_overview.md`` (if present in your local branch)
* ``docs/scientific_documentation.md`` (if present in your local branch)

Migration strategy
------------------

Preferred next steps:

* move high-value narrative content from legacy markdown into reST pages
* replace generated markdown references with Sphinx autodoc / autosummary pages
* retire markdown-only docs once no workflow depends on them
