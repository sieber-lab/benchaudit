# Test suite

This folder provides a complete pipeline test suite for benchmarking and logging.

## Structure

- `tests/data/`: tiny dummy datasets used by loader and pipeline tests.
- `tests/test_loader.py`: loader behavior for tabular, DTI, TDC-style, and Polaris-style inputs.
- `tests/test_run_pipeline.py`: `run.py` orchestration, output writing, skip behavior, and benchmark error handling.
- `tests/test_utils_logging_writer.py`: logging utilities and result writer persistence.
- `tests/test_baselines.py`: baseline runner task dispatch and generic evaluation contract.
- `tests/test_experiment_similarity_plot.py`: benchmark similarity aggregation and plotting outputs.
- `tests/test_unittest_helpers.py`: unit-level helper tests for config echo and YAML loading.

## Running

Preferred (if pytest is installed):

```bash
pytest tests
```

Fallback with stdlib only:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Core CI suite (experiments excluded):

```bash
python -m unittest \
  tests.test_loader \
  tests.test_run_pipeline \
  tests.test_utils_logging_writer \
  tests.test_baselines \
  tests.test_unittest_helpers
```
