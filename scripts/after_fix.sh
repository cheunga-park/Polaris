#!/usr/bin/env bash
# After the post-batch GPU queue: bring the policy server back and re-evaluate PanClean with the
# corrected gripper mount / splat anchors (30 rollouts, run name pi05_fix), then release diagnostics.
# PanClean only: the reference is progress 0.75 / success 37 % (issue #24); one environment
# answers "did the fix work" in a third of the time.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"; source scripts/env.sh
until grep -q "=== .* done ===" data/after_batch.log 2>/dev/null; do sleep 60; done
echo "=== $(date -Is) starting policy server ==="
(cd ../urdf-to-simulater && XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 scripts/openpi/serve.sh pi05_droid_jointpos_polaris gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris 8100) | tail -1
L=../urdf-to-simulater/outputs/openpi/serve.log; start=$(wc -l < $L)
until tail -n +$((start+1)) $L | grep -qE "Application startup complete|serving|listening"; do sleep 20; done
echo "=== $(date -Is) server up; re-evaluating ==="
scripts/eval_batch.sh pi05_fix 8100 30 DROID-PanClean
.venv/bin/python scripts/report.py --run pi05_fix
.venv/bin/python scripts/diagnose_release.py --env DROID-PanClean --object sponge --container pan --episodes 3 2>&1 | grep -E "^\{|Error|Traceback"
echo "=== $(date -Is) after_fix done ==="
