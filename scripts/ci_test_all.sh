#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "[ci] running unittest suite"
python -m unittest discover -s tests -p "test_*.py"

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
