#!/usr/bin/env bash
# run_chebyshev.sh — Chebyshev polynomial chain benchmarks.
# Covers: main comparison, CHO-only ablation, and a dimension sweep.
# Usage:  bash scripts/run_chebyshev.sh
#         bash scripts/run_chebyshev.sh --no-plot

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EXTRA="${*}"

echo ""
echo "--- Chebyshev: main comparison (n=1000, p=4) ---"
python main.py --config config/chebyshev.yaml $EXTRA

echo ""
echo "--- Chebyshev: CHO ablation (n=1000, p=4) ---"
python main.py --config config/chebyshev_wsm.yaml $EXTRA

echo ""
echo "--- Chebyshev: dimension sweep (n = 200 500 1000 2000) ---"
for DIM in 200 500 1000 2000; do
python - <<PYEOF
import yaml, subprocess, sys, tempfile, os

with open("config/chebyshev.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["problem"]["params"]["n"] = ${DIM}
cfg["output"]["save_dir"] = "results/chebyshev_n${DIM}"
cfg["output"]["title"] = "Chebyshev (n=${DIM}, p=4)"

tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
yaml.dump(cfg, tmp)
tmp.close()
ret = subprocess.call(
    [sys.executable, "main.py", "--config", tmp.name, "--no-plot"],
)
os.unlink(tmp.name)
sys.exit(ret)
PYEOF
  echo "  n=${DIM} done."
done
echo "Dimension sweep complete."
