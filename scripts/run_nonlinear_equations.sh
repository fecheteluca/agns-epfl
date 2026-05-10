#!/usr/bin/env bash
# run_nonlinear_equations.sh — NonlinearEquations benchmarks.
# Covers: CHO-only comparison, WSM vs CHO comparison,
#         and a p-sweep (p = 2, 3, 4, 5, 6).
# Usage:  bash scripts/run_nonlinear_equations.sh
#         bash scripts/run_nonlinear_equations.sh --no-plot

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EXTRA="${*}"

echo ""
echo "--- Nonlinear Equations: CHO backends (n=100, m=200, p=4) ---"
python main.py --config config/nonlinear_equations.yaml $EXTRA

echo ""
echo "--- Nonlinear Equations: WSM vs CHO (n=100, m=200, p=4) ---"
python main.py --config config/nonlinear_equations_wsm.yaml $EXTRA

echo ""
echo "--- Nonlinear Equations: p-sweep (p = 2 3 4 5 6) ---"
for P in 2 3 4 5 6; do
python - <<PYEOF
import yaml, subprocess, sys, tempfile, os

with open("config/nonlinear_equations_wsm.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["problem"]["params"]["p"] = ${P}
cfg["output"]["save_dir"] = "results/nonlinear_equations_p${P}"
cfg["output"]["title"] = "Nonlinear Equations (p=${P})"

tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
yaml.dump(cfg, tmp)
tmp.close()
ret = subprocess.call(
    [sys.executable, "main.py", "--config", tmp.name, "--no-plot"],
)
os.unlink(tmp.name)
sys.exit(ret)
PYEOF
  echo "  p=${P} done."
done
echo "p-sweep complete."
