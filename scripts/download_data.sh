#!/usr/bin/env bash
# download_data.sh — Download LIBSVM benchmark datasets used by the real-data
# LogSumExp experiments.
#
# Datasets (binary classification, stored in LIBSVM sparse format):
#   mushrooms  8124  samples × 112 features
#   a9a        32561 samples × 123 features
#   w8a        49749 samples × 300 features
#
# The datasets are fetched from the LIBSVM website.  Each is a small text file
# (~1–10 MB).  Re-running this script is safe; existing files are skipped.
#
# Usage:  bash scripts/download_data.sh

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$REPO_ROOT/data"
mkdir -p "$DATA_DIR"

BASE_URL="https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary"

download_if_missing() {
  local name="$1"
  local url="$2"
  local dest="$DATA_DIR/$name"
  if [[ -f "$dest" ]]; then
    echo "  [skip] $name already present."
  else
    echo "  [fetch] $name ..."
    curl -fsSL "$url" -o "$dest"
    echo "  [ok]   saved to $dest"
  fi
}

echo "Downloading LIBSVM datasets to $DATA_DIR/ ..."
echo ""

download_if_missing "mushrooms"  "$BASE_URL/mushrooms"
download_if_missing "a9a"        "$BASE_URL/a9a"
download_if_missing "w8a"        "$BASE_URL/w8a"

echo ""
echo "Done.  Dataset summary:"
echo "  mushrooms : $(wc -l < "$DATA_DIR/mushrooms") samples"
echo "  a9a       : $(wc -l < "$DATA_DIR/a9a") samples"
echo "  w8a       : $(wc -l < "$DATA_DIR/w8a") samples"
