# BenchAudit

[![CI](https://github.com/sieber-lab/bench/actions/workflows/ci.yml/badge.svg)](https://github.com/sieber-lab/bench/actions/workflows/ci.yml)
[![Publish to PyPI](https://github.com/sieber-lab/bench/actions/workflows/publish-pypi.yml/badge.svg)](https://github.com/sieber-lab/bench/actions/workflows/publish-pypi.yml)
[![PyPI version](https://img.shields.io/pypi/v/benchaudit.svg)](https://pypi.org/project/benchaudit/)
[![Python versions](https://img.shields.io/pypi/pyversions/benchaudit.svg)](https://pypi.org/project/benchaudit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

BenchAudit is a lightweight pipeline for auditing molecular property and drug–target interaction benchmarks. It standardizes SMILES strings, checks split hygiene, surfaces label conflicts and activity cliffs, and can run simple baseline models. Outputs are machine‑readable summaries and drill‑down tables you can inspect or feed into other tools.

## Features
- Config‑driven analysis of tabular, TDC, Polaris, and DTI datasets.
- SMILES standardization with optional REOS alerts and configurable fingerprint settings.
- Split hygiene reports: duplicates, cross‑split contamination, and nearest‑neighbor similarity.
- Conflict and activity‑cliff detection for classification and regression tasks.
- DTI extras: sequence normalization, cross‑split pair conflicts, and EMBOSS `stretcher` alignment summaries.
- Optional simple baselines for quick performance sanity checks.

## Installation

### From PyPI
Install the published package:

```bash
pip install benchaudit
```

or with `uv`:

```bash
uv pip install benchaudit
```

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

## Automated PyPI publishing
This repo includes `.github/workflows/publish-pypi.yml` for automated releases.

1. In PyPI, configure a Trusted Publisher for this GitHub repository and workflow file (`.github/workflows/publish-pypi.yml`), using environment `pypi`.
2. Bump `project.version` in `pyproject.toml`.
3. Create and push a tag `vX.Y.Z` matching that version (for example `v0.1.1`).
4. GitHub Actions builds with `uv build` and publishes to PyPI automatically when the repository visibility is `public` (publishing is skipped while private).

Detailed release and install documentation: [`docs/publishing_and_installation.md`](docs/publishing_and_installation.md)

## References
- Package on PyPI: <https://pypi.org/project/benchaudit/>
- Publish workflow: [`.github/workflows/publish-pypi.yml`](.github/workflows/publish-pypi.yml)
- CI workflow: [`.github/workflows/ci.yml`](.github/workflows/ci.yml)
- `uv` docs: <https://docs.astral.sh/uv/>
- PyPI Trusted Publishers: <https://docs.pypi.org/trusted-publishers/>

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
- Code style: keep changes simple, PEP 8-ish. Add short docstrings for public functions.
- Typing: prefer explicit, lightweight type hints when types are clear.
- Tests: run `python -m unittest discover -s tests -p "test_*.py"` (or `pytest tests` if pytest is installed).
- Test data: tiny dummy benchmark datasets live under `tests/data/`.
- Benchmark/analysis docs: run `python scripts/generate_benchmark_analysis_class_docs.py --output docs/benchmark_and_analysis_class_reference.md` to regenerate the class reference; CI enforces freshness via `.github/workflows/benchmark-analysis-docs.yml`.
- Optional extras: Polaris datasets require `polaris-lib`; sequence alignment requires `pairwise-sequence-alignment` and EMBOSS binaries.
