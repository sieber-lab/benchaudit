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
     cleaner: none
     keep_invalid: true

Validation behavior
-------------------

BenchAudit now validates and normalizes config payloads before loaders and analyzers run.

Examples of early validation failures:

* non-mapping YAML root documents
* ``path`` and ``paths`` both present
* malformed ``info`` or ``paths`` sections
* unsupported split labels (must normalize to ``train``, ``valid``/``val``, ``test``)
