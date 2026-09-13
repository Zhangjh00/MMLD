#!/usr/bin/env bash
# Run a short prepare + prediction workflow with the unlabelled example data.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

AB="$HERE/example/example2.xlsx"
GRAPH="$HERE/example/example2_graph.xlsx"

MMLD prepare \
  --abundance "$AB" \
  --graph "$GRAPH" \
  --epochs 10 \
  --output-dir "$HERE/outputs/00_prepare_time_series"

MMLD predict \
  --abundance "$AB" \
  --graph "$GRAPH" \
  --split 82 \
  --mode sequence \
  --initial-ratio 0.15 \
  --epochs 10 \
  --output-dir "$HERE/outputs/01_prediction" \
  "$@"
