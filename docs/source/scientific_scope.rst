Scientific Scope
================

BenchAudit is designed for scientific auditing of molecular property and
drug-target interaction (DTI) benchmarks. The focus is dataset quality,
similarity structure, and leakage risk rather than deployment workflows.

Research Questions
------------------

The implemented analyses target the following questions:

* How much exact overlap exists between training/validation and test data?
* How similar are held-out compounds to training compounds?
* Do identical molecules receive inconsistent labels across splits?
* How often do highly similar molecules show strong label disagreement (activity cliffs)?
* For DTI datasets, do targets leak across splits at sequence or structure level?

Benchmark Families
------------------

BenchAudit supports four modalities:

* ``tabular``: local CSV/TSV/Parquet benchmark files.
* ``tdc``: Therapeutics Data Commons datasets via ``pytdc``.
* ``polaris``: Polaris benchmarks via ``polaris-lib``.
* ``dti``: ligand-target datasets with SMILES and amino-acid sequences.

Primary Outputs
---------------

Each run produces standardized artifacts intended for analysis and reproducibility:

* ``summary.json``: top-level hygiene, similarity, conflict, and cliff statistics.
* ``records.csv``: row-level standardized records used by analysis.
* ``conflicts.jsonl``: detailed conflict events.
* ``cliffs.jsonl``: detailed activity-cliff events.
* ``sequence_alignments.jsonl`` / ``structure_alignments.jsonl`` (DTI when available).
* ``performance.json`` when baseline benchmarking is enabled.

Terminology and Criteria
------------------------

BenchAudit uses explicit criteria:

* Duplicate: repeated cleaned SMILES (or repeated normalized target sequence in DTI).
* Contamination: shared entities across train/valid and test.
* Similar pair: pair that passes the configured consensus similarity threshold.
* Conflict:
  classification labels differ for identical cleaned SMILES (or for DTI cross-split pair checks),
  while regression conflicts use a 3-sigma threshold estimated from train/valid labels.
* Activity cliff: similar molecules with divergent labels under task-specific rules.

The similarity consensus combines molecular fingerprint similarity, scaffold fingerprint
similarity, and normalized SMILES string similarity.
