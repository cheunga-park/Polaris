#!/usr/bin/env bash
# S5.5: the upstream evaluation, unmodified, over MuJoCo, against the openpi policy server.
#   scripts/eval_batch.sh <run-name> <port> <rollouts> <env> [<env> ...]
# e.g. scripts/eval_batch.sh pi05 8100 50 DROID-FoodBussing DROID-TapeIntoContainer
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; source "$ROOT/scripts/env.sh"; cd "$ROOT"
NAME="$1"; PORT="$2"; N="$3"; shift 3
for ENV in "$@"; do
  echo "=== $ENV -> runs/$NAME/$ENV ($N rollouts, policy :$PORT) ==="
  .venv/bin/python -m polaris_mujoco.run third_party/polaris/scripts/eval.py --environment "$ENV" --policy.port "$PORT" --rollouts "$N" --run-folder "runs/$NAME/$ENV" 2>&1 | grep -E "Episode|Starting|Error|Traceback|rollouts" || true
done
