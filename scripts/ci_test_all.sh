#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

required_fixtures=(
  "tests/data/tabular_single.csv"
  "tests/data/tabular_paths/train.csv"
  "tests/data/tabular_paths/valid.csv"
  "tests/data/tabular_paths/test.csv"
  "tests/data/dti/train.csv"
  "tests/data/dti/valid.csv"
  "tests/data/dti/test.csv"
  "tests/data/alignment_pairs.jsonl"
  "tests/data/runs_smoke/polaris/SmokeA/summary.json"
  "tests/data/runs_smoke/tdc/SmokeB/summary.json"
)

for fixture in "${required_fixtures[@]}"; do
  if [[ ! -f "$fixture" ]]; then
    echo "[ci] missing required fixture: $fixture"
    echo "[ci] available under tests/data:"
    find tests/data -maxdepth 5 -type f | sort || true
    exit 1
  fi
done

echo "[ci] running unittest suite"
python -m unittest discover -s tests -t . -p "test_*.py"

echo "[ci] running benchmark similarity smoke plot"
python experiments/plot_trainvalid_test_similarity_by_benchmark.py \
  --runs-root tests/data/runs_smoke \
  --output "$TMP_DIR/trainvalid_test_similarity.svg" \
  --output-csv "$TMP_DIR/trainvalid_test_similarity.csv" \
  --output-stats-csv "$TMP_DIR/trainvalid_test_similarity_stats.csv"

test -s "$TMP_DIR/trainvalid_test_similarity.svg"
test -s "$TMP_DIR/trainvalid_test_similarity.csv"
test -s "$TMP_DIR/trainvalid_test_similarity_stats.csv"

echo "[ci] running pairwise alignment correlation smoke plot"
python experiments/plot_pairwise_alignment_correlation.py \
  --sequence-file tests/data/alignment_pairs.jsonl \
  --structure-file tests/data/alignment_pairs.jsonl \
  --sequence-metrics "identity_pct,similarity_pct" \
  --structure-metric probability \
  --output "$TMP_DIR/pairwise_alignment_correlation.pdf"

test -s "$TMP_DIR/pairwise_alignment_correlation.pdf"

echo "[ci] all checks passed"
