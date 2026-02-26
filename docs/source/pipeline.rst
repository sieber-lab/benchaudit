Pipeline Overview
=================

BenchAudit runs a configurable audit pipeline over molecular property and DTI benchmarks.

Execution flow
--------------

1. ``run.py`` loads one or more YAML configs.
2. ``utils.build_loader()`` selects a loader (tabular, TDC, Polaris, or DTI).
3. ``utils.build_analyzer()`` selects a SMILES or DTI analyzer.
4. The analyzer computes hygiene, similarity, conflict, and cliff diagnostics.
5. ``utils.ResultWriter`` persists artifacts to a run directory.
6. Optional baseline benchmarking writes ``performance.json``.

Supported modalities
--------------------

* ``tabular``: CSV/TSV/Parquet inputs with configurable column mapping
* ``tdc``: Therapeutics Data Commons via ``pytdc``
* ``polaris``: Polaris benchmarks via ``polaris-lib``
* ``dti``: tabular DTI data with ligand SMILES + target sequences

Core diagnostics
----------------

SMILES / molecular analyses:

* split hygiene and contamination
* nearest-neighbor similarity summaries
* label conflicts on identical cleaned SMILES
* activity cliffs on similar molecules with divergent labels

DTI-specific additions:

* target-sequence overlap and duplication summaries
* sequence-alignment nearest-neighbor diagnostics
* optional Foldseek structure-level leakage checks

Design notes
------------

The code path is intentionally config-driven and dataset-agnostic:

* column inference with explicit override hooks
* optional cleaning / canonicalization
* shared output schema for automated downstream analysis
* deterministic output directory resolution for CI and reproducibility
