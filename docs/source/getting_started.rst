Getting Started
===============

Installation
------------

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

* ``summary.json``: high-level audit summary, including runtime metadata for new runs
* ``records.csv``: per-row standardized records
* ``conflicts.jsonl``: label conflicts among identical molecules
* ``cliffs.jsonl``: activity cliffs among similar molecules
* ``sequence_alignments.jsonl``: DTI sequence diagnostics (DTI only)
* ``structure_alignments.jsonl``: Foldseek-based structure diagnostics (DTI only)
* ``performance.json``: baseline model metrics and predictions (when ``--benchmark`` is enabled)

Runtime Reporting
-----------------

Newly generated ``summary.json`` files include a ``runtime`` block with UTC
start/end timestamps, total elapsed seconds/minutes, and stage-level timings.
For older artifacts, generate an approximate runtime table with:

.. code-block:: bash

   python experiments/report_runtimes.py --runs-root runs --out-dir experiments/plots

The report marks each row as exact recorded timing, sequential timestamp
estimate, or artifact-write lower bound.
