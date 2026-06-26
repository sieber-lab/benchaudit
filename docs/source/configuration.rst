Configuration and CLI
=====================

CLI entry points
----------------

BenchAudit exposes ``run.py`` as the main CLI and installs console scripts ``benchaudit`` and ``bench``.

Examples:

.. code-block:: bash

   python run.py --config path/to/config.yaml --out-root runs
   python run.py --configs configs --out-root runs --benchmark
   benchaudit --configs configs --out-root runs --force

Important CLI flags
-------------------

* ``--config``: run a single YAML config
* ``--configs``: run all YAML configs under a directory
* ``--out-root``: output root directory (default ``runs``)
* ``--benchmark``: run baseline models and write ``performance.json``
* ``--force``: rerun even if outputs already exist
* ``--log-level``: logging level (``DEBUG``, ``INFO``, ...)

Runtime reporting
-----------------

``summary.json`` contains a ``runtime`` block for newly generated runs. To
summarize recorded runtimes and estimate older runs from artifact timestamps:

.. code-block:: bash

   python experiments/report_runtimes.py --runs-root runs --out-dir experiments/plots

The generated CSV and Markdown report include an ``estimate_type`` column so
exact timings and approximate/lower-bound values are distinguishable.

Config shape (high level)
-------------------------

BenchAudit config files are YAML mappings. Common top-level keys:

* ``type`` (or ``modality``): loader/analyzer routing (e.g. ``tabular``, ``tdc``, ``polaris``, ``dti``)
* ``task``: ``classification`` or ``regression``
* ``name``: dataset identifier
* ``path`` or ``paths``: input file locations for tabular/DTI data
* ``info``: loader/analyzer options (column names, split settings, similarity params)
* ``out``: optional output directory override
* ``seed``: optional random seed used by some components

Tabular single-file example
---------------------------

.. code-block:: yaml

   type: tabular
   name: Tiny Tabular
   task: classification
   path: tests/data/tabular_single.csv
   info:
     split_col: split
     smiles_col: smiles
     label_col: label
     id_col: compound_id
     cleaner: none

Tabular three-path example
--------------------------

.. code-block:: yaml

   type: tabular
   name: Split Tabular
   task: classification
   paths:
     train: train.csv
     valid: valid.csv
     test: test.csv
   info:
     smiles_col: Drug
     label_col: Y
     id_col: ID
     cleaner: none

DTI example
-----------

.. code-block:: yaml

   type: dti
   modality: dti
   name: Example DTI
   task: classification
   paths:
     train: train.csv
     valid: valid.csv
     test: test.csv
   info:
     smiles_col: Ligand
     label_col: classification_label
     sequence_col: Protein
     target_id_col: Target_ID
     sequence_alignment_workers: 4
     cleaner: none
     keep_invalid: true

``info.sequence_alignment_workers`` controls how many independent EMBOSS
``stretcher`` jobs are run at once for DTI nearest-neighbor sequence alignment.
The default is ``1``, which preserves the previous serial behavior.

Opt-in benchmark cleaning
-------------------------

By default, BenchAudit reports invalid molecules, exact contamination, and
label conflicts without changing the loaded benchmark. To curate the in-memory
benchmark used by analysis and optional baselines, enable ``info.clean_benchmark``:

.. code-block:: yaml

   type: tabular
   name: Cleaned Example
   task: classification
   path: data/example.csv
   info:
     split_col: split
     smiles_col: smiles
     label_col: label
     clean_benchmark: true

With ``true``, BenchAudit applies the default policy after loading and before
analysis:

* invalid molecules are removed
* all rows for a molecule with conflicting labels are removed from every split
* exact contaminants are kept only in reference splits

The default reference splits are ``train`` and ``valid``. If ``valid`` is not
present, ``train`` is used. Exact contamination is defined by exact overlap of
the cleaned SMILES string; near-neighbor and scaffold similarity are still audit
signals, not automatic removal criteria. REOS alerts remain annotations unless a
row is otherwise invalid.

The policy can be customized:

.. code-block:: yaml

   info:
     clean_benchmark:
       reference_splits: [train, valid]
       remove_invalid: true
       remove_conflicts: true
       remove_contaminants: true

The source files are never overwritten. Cleaned rows flow into ``records.csv``,
``summary.json``, and ``performance.json`` when ``--benchmark`` is used. The
summary contains a ``benchmark_cleaning`` block with original counts, cleaned
counts, per-split removal counts, and the effective options.

Validation behavior
-------------------

BenchAudit now validates and normalizes config payloads before loaders and analyzers run.

Examples of early validation failures:

* non-mapping YAML root documents
* ``path`` and ``paths`` both present
* malformed ``info`` or ``paths`` sections
* unsupported split labels (must normalize to ``train``, ``valid``/``val``, ``test``)
