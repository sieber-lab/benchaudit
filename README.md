# BenchAudit

[![CI](https://github.com/sieber-lab/benchaudit/actions/workflows/ci.yml/badge.svg)](https://github.com/sieber-lab/benchaudit/actions/workflows/ci.yml)
[![Publish to PyPI](https://github.com/sieber-lab/benchaudit/actions/workflows/publish-pypi.yml/badge.svg)](https://github.com/sieber-lab/benchaudit/actions/workflows/publish-pypi.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://github.com/sieber-lab/benchaudit/blob/main/pyproject.toml)
[![Docs Website](https://img.shields.io/badge/docs-website-2b6cb0)](https://sieber-lab.github.io/benchaudit/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

BenchAudit is a lightweight pipeline for auditing molecular property and drug–target interaction benchmarks. It standardizes SMILES strings, checks split hygiene, surfaces label conflicts and activity cliffs, and can run simple baseline models. Outputs are machine‑readable summaries and drill‑down tables you can inspect or feed into other tools.

## Features
- Config‑driven analysis of tabular, TDC, Polaris, and DTI datasets.
- SMILES standardization with optional REOS alerts and configurable fingerprint settings.
- Split hygiene reports: duplicates, cross‑split contamination, and nearest‑neighbor similarity.
- Conflict and activity‑cliff detection for classification and regression tasks.
- DTI extras: sequence normalization, cross‑split pair conflicts, and EMBOSS `stretcher` alignment summaries.
- Optional simple baselines for quick performance sanity checks.

## Installation

### From source with `uv`
BenchAudit uses a standard `pyproject.toml`. The quickest source setup is with [`uv`](https://docs.astral.sh/uv/):

```bash
# 1) Create a virtual environment
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 2) Install dependencies declared in pyproject.toml
uv sync
```

If you need optional sequence alignment support, install EMBOSS so `stretcher` is available (e.g., `sudo apt install emboss` on Debian/Ubuntu).

## Usage
The main entry point is `run.py`, which consumes one or more YAML configs and writes results under `runs/` by default. After `uv sync`, you can call it via `uv run python run.py ...` or the installed console scripts:
- `uv run benchaudit ...` (primary)
- `uv run bench ...` (legacy alias)

```bash
# Analyze all configs in a folder
uv run python run.py --configs configs --out-root runs
# or: uv run benchaudit --configs configs --out-root runs

# Analyze a single config and train baselines
uv run python run.py --config configs/example.yml --benchmark
# or: uv run benchaudit --config configs/example.yml --benchmark
```

Outputs per config:
- `summary.json`: split sizes, hygiene counts, similarity and conflict statistics.
- `records.csv`: per-row view with cleaned SMILES, labels, and split tags.
- `conflicts.jsonl`: detailed conflict rows.
- `cliffs.jsonl`: detailed activity cliff rows.
- `sequence_alignments.jsonl`: (DTI only) top alignments between splits.
- `performance.json`: (when `--benchmark`) baseline model metrics and predictions.

## Project layout
- `run.py`: CLI runner that loads configs, builds loaders/analyzers, and writes artifacts.
- `utils/`: loaders, analyzers, baseline helpers, and logging utilities.
- `configs/`: example YAML configurations for supported datasets.
- `data/`, `runs/`: expected data and output locations (not tracked).

## Development
- Tests: run `python -m unittest discover -s tests -p "test_*.py"` (or `pytest tests` if pytest is installed).
- Test data: tiny dummy benchmark datasets live under `tests/data/`.
- Docs: build with `sphinx-build -W --keep-going -b html docs/source docs/_build/html` (`docs/source/reference/api_objects.rst` provides autosummary-based API inventory).
- Optional extras: Polaris datasets require `polaris-lib`; sequence alignment requires `pairwise-sequence-alignment` and EMBOSS binaries.

## References
- CI workflow: [`.github/workflows/ci.yml`](.github/workflows/ci.yml)
- `uv` docs: <https://docs.astral.sh/uv/>
