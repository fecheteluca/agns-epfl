#!/usr/bin/env bash
# run_rosenbrock.sh — Rosenbrock benchmarks (2-D and NLE variants).
# Usage:  bash scripts/run_rosenbrock.sh
#         bash scripts/run_rosenbrock.sh --no-plot

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EXTRA="${*}"

echo ""
echo "--- Rosenbrock 2-D (standard) ---"
python main.py --config config/rosenbrock_2d.yaml $EXTRA

echo ""
echo "--- Rosenbrock-NLE (p=5) ---"
python main.py --config config/rosenbrock_nle.yaml $EXTRA

echo ""
echo "--- Rosenbrock-NLE: p-sweep (p = 2 3 4 5 6 8) ---"
for P in 2 3 4 5 6 8; do
python - <<PYEOF
import yaml, subprocess, sys, tempfile, os

with open("config/rosenbrock_nle.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["problem"]["params"]["p"] = ${P}
cfg["output"]["save_dir"] = "results/rosenbrock_nle_p${P}"
cfg["output"]["title"] = "Rosenbrock-NLE (p=${P})"

tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
yaml.dump(cfg, tmp)
tmp.close()
ret = subprocess.call(
    [sys.executable, "main.py", "--config", tmp.name, "--no-plot"],
)
os.unlink(tmp.name)
sys.exit(ret)
PYEOF
  echo "  p=$P done."
done
echo "p-sweep complete."
