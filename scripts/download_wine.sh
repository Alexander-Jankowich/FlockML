#!/usr/bin/env bash
# Download the UCI Wine Quality data (red wines, 1,599 rows, ~100 KB) to
# data/wine/winequality-red.csv, for bin/regression.ml. It is a CSV of
# floats with a header row, separated by semicolons: 11 physico-chemical
# measurements, then a quality score from 0 to 10.
set -euo pipefail

out="data/wine"
if [ -f "$out/winequality-red.csv" ]; then
  echo "$out/winequality-red.csv already exists"
  exit 0
fi
mkdir -p "$out"
curl -fsSL "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv" \
  -o "$out/winequality-red.csv"
echo "done: $out/winequality-red.csv"
