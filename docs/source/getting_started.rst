Getting Started
===============

Installation
------------

Install from PyPI:

.. code-block:: bash

   pip install benchaudit

or:

.. code-block:: bash

   uv pip install benchaudit

Install from source (``uv``):

.. code-block:: bash

   uv venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   uv sync

Optional sequence alignment support
-----------------------------------

DTI sequence alignment diagnostics require EMBOSS ``stretcher``.

.. code-block:: bash

   sudo apt install emboss

Quick Start
-----------

Run one config:

.. code-block:: bash

   python run.py --config configs/moleculenet/bbbp.yaml --out-root runs

Run a folder of configs:

.. code-block:: bash

   python run.py --configs configs --out-root runs

Run with baseline benchmarking enabled:

.. code-block:: bash

   python run.py --configs configs --out-root runs --benchmark

Outputs
-------

Each run directory typically contains:

* ``summary.json``: high-level audit summary
* ``records.csv``: per-row standardized records
* ``conflicts.jsonl``: label conflicts among identical molecules
* ``cliffs.jsonl``: activity cliffs among similar molecules
* ``sequence_alignments.jsonl``: DTI sequence diagnostics (DTI only)
* ``structure_alignments.jsonl``: Foldseek-based structure diagnostics (DTI only)
* ``performance.json``: baseline model metrics and predictions (when ``--benchmark`` is enabled)
