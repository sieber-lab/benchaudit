#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

required_fixtures=(
  "tests/data/tabular_single.csv"
  "tests/data/tabular_paths/train.csv"
  "tests/data/tabular_paths/valid.csv"
  "tests/data/tabular_paths/test.csv"
  "tests/data/dti/train.csv"
  "tests/data/dti/valid.csv"
  "tests/data/dti/test.csv"
)

for fixture in "${required_fixtures[@]}"; do
  if [[ ! -f "$fixture" ]]; then
    echo "[ci] missing required fixture: $fixture"
    echo "[ci] available under tests/data:"
    find tests/data -maxdepth 5 -type f | sort || true
    exit 1
  fi
done

echo "[ci] running core unittest suite (experiments excluded)"
python -m unittest \
  tests.test_loader \
  tests.test_run_pipeline \
  tests.test_utils_logging_writer \
  tests.test_baselines \
  tests.test_unittest_helpers

echo "[ci] all checks passed"
