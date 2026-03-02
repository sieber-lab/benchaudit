Methods
=======

This page summarizes the implemented methodology used by the pipeline.
Detailed API signatures are documented under :doc:`reference/index`.

Data Ingestion and Standardization
----------------------------------

* Loaders normalize split inputs into canonical ``train``, ``valid``, ``test`` frames.
* Tabular data supports CSV/TSV/Parquet with configurable column mapping.
* DTI loading additionally normalizes sequence and target-ID columns.
* Optional SMILES cleaning can canonicalize structures, deduplicate, and annotate quality flags.

Similarity Computation
----------------------

BenchAudit computes multiple complementary similarities:

* Molecular similarity using Morgan fingerprints (configurable radius and bit length).
* Scaffold similarity using generic Murcko scaffold fingerprints.
* String-level similarity on SMILES using normalized Levenshtein similarity.

Nearest-neighbor similarity summaries are reported for validation/test against training
and train+valid references.

Conflict and Cliff Detection
----------------------------

* Classification conflicts: identical cleaned SMILES with differing labels.
* Regression conflicts: identical cleaned SMILES with label deltas beyond a 3-sigma threshold.
* Activity cliffs: highly similar molecule pairs with divergent labels under the same task rules.

DTI-Specific Diagnostics
------------------------

DTI mode extends molecule-level auditing with target-level checks:

* target sequence overlap and duplication statistics across splits
* cross-split ligand-target pair reuse checks
* nearest-neighbor sequence alignment diagnostics via EMBOSS ``stretcher``
* optional structure-level leakage diagnostics when Foldseek alignments are provided

Baseline Benchmarking
---------------------

With ``--benchmark``, BenchAudit trains lightweight baseline models and writes
``performance.json``. This is intended as context for dataset difficulty and sanity checks,
not as a deployment pipeline.

Verification
------------

The repository includes unit tests for loaders, pipeline orchestration, baselines, and
result writing. Build docs locally with:

.. code-block:: bash

   sphinx-build -W --keep-going -b html docs/source docs/_build/html
