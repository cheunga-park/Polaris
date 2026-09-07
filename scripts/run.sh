#!/usr/bin/env bash
# One line: bring the console up. `--port N` to move it.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"
PORT=8080   # 8000 is the tosim console on this box
while [ $# -gt 0 ]; do case "$1" in --port) PORT="$2"; shift 2;; *) echo "unknown arg $1"; exit 2;; esac; done
[ -x "$ROOT/.venv/bin/uvicorn" ] || { echo "no .venv: run 'uv sync' first"; exit 1; }
[ -d "$ROOT/data/hub/food_bussing" ] || echo "hub not downloaded: uvx --from huggingface_hub hf download owhan/PolaRiS-Hub --repo-type=dataset --local-dir data/hub"
cd "$ROOT"
echo "console: http://localhost:$PORT  (LAN: http://$(hostname -I 2>/dev/null | awk '{print $1}'):$PORT)"
exec "$ROOT/.venv/bin/uvicorn" web.server.app:app --host 0.0.0.0 --port "$PORT"
