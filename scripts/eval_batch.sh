#!/usr/bin/env bash
# S5.5: the upstream evaluation, unmodified, over MuJoCo, against the openpi policy server.
#   scripts/eval_batch.sh <run-name> <port> <rollouts> <env> [<env> ...]
# e.g. scripts/eval_batch.sh pi05 8100 50 DROID-FoodBussing DROID-PanClean
# Upstream eval.py resumes from eval_results.csv, so an environment is re-launched until its CSV
# holds <rollouts> rows (a policy-server keepalive drop under memory pressure kills one process,
# not the batch). MAX_RETRIES per environment, default 6.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; source "$ROOT/scripts/env.sh"; cd "$ROOT"
NAME="$1"; PORT="$2"; N="$3"; shift 3
MAX_RETRIES="${MAX_RETRIES:-6}"
rows() { local f="runs/$NAME/$1/eval_results.csv"; [ -f "$f" ] && echo $(( $(wc -l < "$f") - 1 )) || echo 0; }
for ENV in "$@"; do
  try=0
  while [ "$(rows "$ENV")" -lt "$N" ] && [ "$try" -lt "$MAX_RETRIES" ]; do
    try=$((try + 1))
    echo "=== $ENV -> runs/$NAME/$ENV ($(rows "$ENV")/$N done, attempt $try, policy :$PORT) $(date -Is) ==="
    .venv/bin/python -m polaris_mujoco.run third_party/polaris/scripts/eval.py --environment "$ENV" --policy.port "$PORT" --rollouts "$N" --run-folder "runs/$NAME/$ENV" 2>&1 | grep -E "Episode|Starting|Error|Traceback|rollouts" | grep -v _ReportErrors || true
    sleep 5
  done
  echo "=== $ENV finished with $(rows "$ENV")/$N episodes ==="
done
