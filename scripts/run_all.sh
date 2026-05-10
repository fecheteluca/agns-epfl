#!/usr/bin/env bash
# run_all.sh — run every benchmark experiment sequentially.
# Usage:   bash scripts/run_all.sh
#          bash scripts/run_all.sh --no-plot   (skip plot generation)
#
# All results land in results/<experiment_name>/.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EXTRA="${*}"   # forward any extra flags (e.g. --no-plot) to all runs

echo "=================================================="
echo "  Running ALL benchmark experiments"
echo "  Root: $REPO_ROOT"
echo "=================================================="

bash scripts/run_logsumexp.sh          $EXTRA
bash scripts/run_nonlinear_equations.sh $EXTRA
bash scripts/run_chebyshev.sh          $EXTRA
bash scripts/run_rosenbrock.sh         $EXTRA

echo ""
echo "All experiments complete.  Results in: $REPO_ROOT/results/"
