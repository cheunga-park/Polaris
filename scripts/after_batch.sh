#!/usr/bin/env bash
# Queue for the GPU once the pi0.5 sweep is done: release diagnostics (policy server still up),
# then free the server's VRAM, redo the synthetic scan in the coarse (scaled) frame, then run the
# object pipeline (SAM 2 -> TRELLIS).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"; source scripts/env.sh
while pgrep -f "eval_batch.sh" >/dev/null || pgrep -f "batch5_sweep" >/dev/null; do sleep 60; done
echo "=== $(date -Is) sweep done; release diagnostics ==="
.venv/bin/python scripts/diagnose_release.py --env DROID-PanClean --object sponge --container pan --episodes 4 2>&1 | grep -E "^\{|Error|Traceback"
.venv/bin/python scripts/diagnose_release.py --env DROID-FoodBussing --object grapes --container bowl --episodes 4 2>&1 | grep -E "^\{|Error|Traceback"
echo "=== $(date -Is) stopping policy server ==="
../urdf-to-simulater/scripts/openpi/stop.sh || true
sleep 10; nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "=== synth_food re-run in the coarse frame ==="
.venv/bin/python -m polaris_v2s.cli run synth_food --force charuco,splat,mesh,pack 2>&1 | grep -vE "^Warning" | tail -8
echo "=== object pipeline: bowl (SAM 2 -> TRELLIS) ==="
.venv/bin/python - <<'PY' 2>&1 | grep -vE "^Warning"
import json, time
from pathlib import Path
from polaris_v2s import objects
t = time.time()
info = objects.build_object(Path("data/scans/synth_food/upload/objects/bowl.mp4"), Path("data/scans/synth_food/objects/bowl"), "bowl", size_m=0.18)
print(json.dumps(info, indent=1), f"{time.time()-t:.0f}s")
PY
echo "=== $(date -Is) done ==="
