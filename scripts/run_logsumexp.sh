#!/usr/bin/env bash
# run_logsumexp.sh — LogSumExp experiments (synthetic + real data).
# Usage:  bash scripts/run_logsumexp.sh
#         bash scripts/run_logsumexp.sh --no-plot

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EXTRA="${*}"

echo ""
echo "--- LogSumExp: synthetic 1000×1000 ---"
python main.py --config config/logsumexp_synthetic.yaml $EXTRA

echo ""
echo "--- LogSumExp: seed sweep (seeds 1–5) ---"
for SEED in 1 2 3 4 5; do
  python main.py \
    --config config/logsumexp_synthetic_sweep.yaml \
    --seed "$SEED" \
    --output-dir "results/logsumexp_sweep/seed_${SEED}" \
    --no-plot
done
echo "  Sweep complete (no-plot; load pickles for aggregated analysis)."

# Real-data experiments (require downloading data first)
if [[ -f "data/mushrooms" ]]; then
  echo ""
  echo "--- LogSumExp: Mushrooms dataset ---"
  python main.py --config config/logsumexp_real_mushrooms.yaml $EXTRA
else
  echo ""
  echo "  [skip] data/mushrooms not found. Run scripts/download_data.sh first."
fi

if [[ -f "data/a9a" ]]; then
  echo ""
  echo "--- LogSumExp: a9a dataset ---"
  python main.py --config config/logsumexp_real_a9a.yaml $EXTRA
else
  echo "  [skip] data/a9a not found. Run scripts/download_data.sh first."
fi
