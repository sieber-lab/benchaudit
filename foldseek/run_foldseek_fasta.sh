#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   run_foldseek_fasta.sh path/to/unique_sequences.jsonl output_dir

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 unique_sequences.jsonl output_dir"
  exit 1
fi

JSONL_PATH="$1"
OUTDIR="$2"

THREADS="${THREADS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

# ProstT5 weights + temp dirs
WEIGHTS_DIR="${FOLDSEEK_WEIGHTS_DIR:-/tmp/foldseek_weights}"
TMP_DOWNLOAD="/tmp/foldseek_dti_download"
SEARCH_TMP="/tmp/foldseek_dti_search_tmp"
CLUSTER_TMP="/tmp/foldseek_dti_cluster_tmp"

mkdir -p "$OUTDIR" "$TMP_DOWNLOAD" "$SEARCH_TMP" "$CLUSTER_TMP"
cd "$OUTDIR"

##############################################
### 1) Export JSONL → FASTA
##############################################
echo "==> Exporting JSONL to FASTA"
FASTA="dti_unique_sequences.fasta"

python - "$JSONL_PATH" "$FASTA" << 'PY'
import json, sys
from pathlib import Path

inp = Path(sys.argv[1])
out = Path(sys.argv[2])

with inp.open() as fin, out.open("w") as fout:
    for line in fin:
        if not line.strip():
            continue
        obj = json.loads(line)
        sid = obj["sequence_id"]
        seq = obj["sequence"].strip()
        fout.write(f">{sid}\n{seq}\n")
PY

echo "   Wrote FASTA to $FASTA"

##############################################
### 2) Download ProstT5 weights if missing
##############################################
if [ ! -d "$WEIGHTS_DIR" ]; then
  echo "==> Downloading ProstT5 weights"
  mkdir -p "$TMP_DOWNLOAD"
  foldseek databases ProstT5 "$WEIGHTS_DIR" "$TMP_DOWNLOAD"
else
  echo "==> ProstT5 weights already present at $WEIGHTS_DIR"
fi

##############################################
### 3) Build ProstT5 DB from FASTA (once)
##############################################
DB_NAME="dti_unique_sequences_db"

if [ ! -d "$DB_NAME" ]; then
  echo "==> Building Foldseek ProstT5 DB: $DB_NAME"
  foldseek createdb \
    "$FASTA" \
    "$DB_NAME" \
    --prostt5-model "$WEIGHTS_DIR" \
    --threads "$THREADS"
else
  echo "==> DB $DB_NAME already exists (skipping createdb)"
fi

##############################################
### 4) All-vs-all search: DB vs DB
##############################################
echo "==> Running all-vs-all search: $DB_NAME vs $DB_NAME"
foldseek search \
  "$DB_NAME" \
  "$DB_NAME" \
  dti_all_vs_all_db \
  "$SEARCH_TMP" \
  --threads "$THREADS"

##############################################
### 5) m8-style TSV via createtsv (safe, no CA needed)
##############################################
echo "==> Creating TSV from search results"
TSV="dti_all_vs_all.m8"

foldseek createtsv \
  "$DB_NAME" \
  "$DB_NAME" \
  dti_all_vs_all_db \
  "$TSV" \
  --threads "$THREADS"

echo "   Wrote: $TSV"

##############################################
### 6) Clustering with easy-cluster (ProstT5 on FASTA)
##############################################
echo "==> Running easy-cluster on FASTA (Foldseek defaults)"
foldseek easy-cluster \
  "$FASTA" \
  dti_clusters \
  "$CLUSTER_TMP" \
  --prostt5-model "$WEIGHTS_DIR" \
  --threads "$THREADS"

##############################################
### 7) Summary
##############################################
echo ""
echo "==> DONE!"
echo "Outputs in $OUTDIR:"
echo "  FASTA:                       $FASTA"
echo "  ProstT5 DB:                  $DB_NAME/"
echo "  Similarity (m8-style TSV):   $TSV"
echo "  Clusters TSV:                dti_clusters_members.tsv"
echo "  Representatives FASTA:       dti_clusters_rep_seq.fasta"
echo "  All cluster seqs FASTA:      dti_clusters_all_seqs.fasta"
echo "Weights dir:                   $WEIGHTS_DIR"
echo "Search tmp dir:                $SEARCH_TMP"
echo "Cluster tmp dir:               $CLUSTER_TMP"
