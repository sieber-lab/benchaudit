# BenchAudit

[![CI](https://github.com/sieber-lab/benchaudit/actions/workflows/ci.yml/badge.svg)](https://github.com/sieber-lab/benchaudit/actions/workflows/ci.yml)
[![Publish to PyPI](https://github.com/sieber-lab/benchaudit/actions/workflows/publish-pypi.yml/badge.svg)](https://github.com/sieber-lab/benchaudit/actions/workflows/publish-pypi.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://github.com/sieber-lab/benchaudit/blob/main/pyproject.toml)
[![Docs Website](https://img.shields.io/badge/docs-website-2b6cb0)](https://sieber-lab.github.io/benchaudit/)
[![ChemRxiv preprint](https://img.shields.io/badge/ChemRxiv-preprint-0A7CFF)](https://doi.org/10.26434/chemrxiv.15000559/v1)
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
- `summary.json`: split sizes, hygiene counts, similarity/conflict statistics, and runtime metadata.
- `records.csv`: per-row view with cleaned SMILES, labels, and split tags.
- `conflicts.jsonl`: detailed conflict rows.
- `cliffs.jsonl`: detailed activity cliff rows.
- `sequence_alignments.jsonl`: (DTI only) top alignments between splits.
- `performance.json`: (when `--benchmark`) baseline model metrics and predictions.

New runs include a `runtime` block in `summary.json` with UTC start/end timestamps,
total elapsed seconds/minutes, and stage-level timings. For existing artifacts that
pre-date this metadata, generate a best-effort runtime table with:

```bash
python experiments/report_runtimes.py \
  --runs-root runs \
  --out-dir experiments/plots
```

The report prefers recorded runtimes when available, otherwise uses clearly labeled
timestamp-based estimates or artifact-write lower bounds.

### Generalized noise-injection experiments

To run synthetic noise-injection experiments across non-multitask datasets and
multiple model classes without modifying the original Polaris-only experiment,
see `experiments/noise_generalization_experiment.md`. The generalized driver
writes `results.csv`, `summary_by_fraction.csv`, `dataset_manifest.csv`,
`errors.csv`, and `metadata.json` under `runs/degradation/noise_generalization`.

### Counterfactual benchmark-composition analysis

The `utils.rank_fragility` package implements a post-hoc leaderboard robustness
analysis for molecular property prediction. It starts from an existing
train/valid/test dataset and trained-model prediction CSVs on the original test
set, then asks whether rankings and SOTA-vs-baseline margins remain stable when
the audited chemical composition of the evaluation panel is changed.

Run it with:

```bash
python experiments/run_rank_fragility.py \
  --data data/dataset.csv \
  --pred_dir predictions/ \
  --id_col molecule_id \
  --smiles_col smiles \
  --label_col y \
  --split_col split \
  --task classification \
  --metric auroc \
  --baseline_model ecfp_rf \
  --sota_model auto \
  --near_leak_thresholds 0.85 0.90 \
  --primary_near_leak_threshold 0.85 \
  --regression_conflict_threshold 1.0 \
  --n_panels 1000 \
  --panel_size auto \
  --target_rates 0 0.05 0.10 0.25 observed 0.50 0.75 \
  --random_seed 13 \
  --out-dir runs/rank_fragility/
```

To run across existing BenchAudit artifacts in this repository, use batch mode.
It discovers `runs/*/*/records.csv`, skips multitask datasets with
`--skip-multitask`, skips DTI pair datasets by default, creates prediction
files, and writes one output directory per dataset:

```bash
python experiments/run_rank_fragility.py \
  --from-runs-root runs \
  --batch-out-dir runs/rank_fragility \
  --skip-multitask \
  --train-splits train,valid \
  --batch-models ecfp_linear,ecfp_rf,lgbm_basic,lgbm_advanced,torch_mlp_basic,torch_mlp_advanced \
  --baseline_model ecfp_rf \
  --sota_model auto \
  --n_panels 1000 \
  --panel_size auto \
  --target_rates 0 0.05 0.10 0.25 observed 0.50 0.75 \
  --n-jobs 64 \
  --lgbm-device-type cpu \
  --mlp-accelerator gpu \
  --mlp-devices 1
```

Batch model names are `ecfp_linear`, `ecfp_rf`, `lgbm_basic`,
`lgbm_advanced`, `torch_mlp_basic`, and `torch_mlp_advanced`. The aliases
`lgbm`, `torch_mlp`, and `mlp` map to the basic versions. The Torch MLP uses
the same Lightning-style training path as the generalized noise-injection
study; `--mlp-hidden-size`, `--mlp-max-epochs`, `--mlp-lr`,
`--mlp-weight-decay`, and `--mlp-batch-size` control the basic MLP, while
`--mlp-advanced-hidden-sizes`, `--mlp-advanced-max-epochs`,
`--mlp-advanced-lr`, `--mlp-advanced-weight-decay`, and
`--mlp-advanced-dropout` control the advanced MLP. LightGBM uses
`--lgbm-estimators` and `--lgbm-learning-rate` for the basic model and
`--lgbm-advanced-*` options for the advanced model.

Default rank-fragility outputs are compact CSV summaries intended for manuscript
analysis and version-control safety: audit summaries, original and clean
leaderboards, rank probabilities, SOTA-vs-baseline margins by composition,
Kendall rank correlations by composition, fragility summaries, and aggregate
advantage decompositions. Per-panel manifests, per-panel scores/ranks,
per-molecule annotations, reports, plots, and copied input CSVs are not written
by default. Use `scripts/prune_rank_fragility_outputs.py` to remove legacy
verbose generated artifacts from an existing `runs/rank_fragility` tree.

Dataset CSVs must contain `molecule_id`, `smiles`, `y`, and `split`, where
`split` is one of `train`, `valid`, or `test`. Prediction files are one CSV per
model in `--pred_dir`, with `molecule_id`, `y_true`, and `y_pred`; the file stem
is used as the model name. For binary classification, `y_pred` is the predicted
probability for class 1. For regression, it is the predicted continuous value.

Audit groups are mutually exclusive for test rows:

- `label_conflict`: the same canonical molecule has conflicting labels.
- `exact_leak`: the exact canonical test molecule appears in train.
- `near_train_analogue`: the test molecule has high ECFP/Tanimoto similarity to
  train at the primary threshold.
- `same_scaffold`: the test molecule shares a Murcko scaffold with train but is
  not assigned to a higher-priority group.
- `audit_clean`: none of the above audit conditions applies.

Exact train-test duplicates and duplicate/conflicting labels are treated as
true contamination. Near-train analogues are not treated as erroneous data; they
represent analogue-rich local SAR interpolation and may be valid for lead
optimization, but they are weaker evidence for generalization to novel chemical
space. Same-scaffold overlap is kept as a separate stratum. Activity cliffs are
not treated as contamination in this analysis.

The `leakage_curve` excludes label conflicts and controls the fraction of
examples that are exact train-test leaks or near-train analogues. The
`conflict_curve` controls the fraction of label-conflicting examples while
excluding exact leaks where possible. The `clean_reference` mode samples only
audit-clean molecules, and `observed_composition` resamples panels with the
original audit-group proportions. Random matched controls use the same panel
sizes and target-rate schedule as the leakage/conflict curves but randomize the
composition.

Rank probabilities are empirical probabilities over generated panels, for
example the probability that a model is rank 1 under a specified composition.
SOTA margins are SOTA-vs-baseline metric differences with the sign normalized so
positive always means the SOTA model is better, including lower-is-better
metrics such as RMSE or log loss.

Interpretation examples:

- If the SOTA rank probability remains high on clean-reference panels, the SOTA
  ranking is stable under audit-clean evaluation.
- If the SOTA rank probability drops below 0.5 on clean-reference or zero-leakage
  panels, the original SOTA conclusion is rank-fragile under audit-clean
  evaluation.
- If the SOTA-vs-baseline confidence interval overlaps zero, the SOTA margin is
  statistically indistinguishable from the baseline under that composition.
- If performance increases with analogue-rich fraction, that is consistent with
  local SAR interpolation or leakage-like shortcut effects, not by itself proof
  that a model is intrinsically invalid.

Limitations: this is a post-hoc leaderboard robustness analysis. It does not
replace full retraining on a newly curated benchmark. Clean panels can be
smaller than the original test set. Near-train analogues are not necessarily
erroneous; they represent analogue-rich interpolation. Results should be
interpreted as stability of the benchmark conclusion, not proof that any
architecture is intrinsically invalid.

## Project layout
- `run.py`: CLI runner that loads configs, builds loaders/analyzers, and writes artifacts.
- `utils/`: loaders, analyzers, baseline helpers, and logging utilities.
- `utils/rank_fragility/`: post-hoc counterfactual benchmark-composition analysis.
- `experiments/run_rank_fragility.py`: repo-style entry point for the rank-fragility analysis.
- `configs/`: example YAML configurations for supported datasets.
- `data/`, `runs/`: expected data and output locations (not tracked).

## Development
- Tests: run `python -m unittest discover -s tests -p "test_*.py"` (or `pytest tests` if pytest is installed).
- Test data: tiny dummy benchmark datasets live under `tests/data/`.
- Docs: build with `sphinx-build -W --keep-going -b html docs/source docs/_build/html` (`docs/source/reference/api_objects.rst` provides autosummary-based API inventory).
- Optional extras: Polaris datasets require `polaris-lib`; sequence alignment requires EMBOSS `stretcher`.

## References
- BenchAudit preprint: <https://doi.org/10.26434/chemrxiv.15000559/v1>
- CI workflow: [`.github/workflows/ci.yml`](.github/workflows/ci.yml)
- `uv` docs: <https://docs.astral.sh/uv/>
