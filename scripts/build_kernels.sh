#!/usr/bin/env bash
# Build the two upstream CUDA kernels (unmodified) into .venv:
#   diff-surfel-rasterization (2DGS rasterizer, used by the splat renderer AND by 2DGS training)
#   simple-knn
# Idempotent. Re-run after `uv sync` recreates the venv.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"
UP="$ROOT/third_party/polaris/src"
[ -f "$UP/diff-surfel-rasterization/third_party/glm/glm/glm.hpp" ] || {
  echo "glm missing: fetching over https (upstream .gitmodules pins an ssh url)"
  git -C "$ROOT/third_party/polaris" config submodule.src/diff-surfel-rasterization/third_party/glm.url https://github.com/g-truc/glm.git
  git -C "$ROOT/third_party/polaris" submodule update --init src/diff-surfel-rasterization/third_party/glm
}
# Upstream ships JIT-compiled packages (their setup.py copies nothing to build; a stale
# data_files entry even breaks a regular install). Editable install = no copy, upstream
# tree untouched; the first import compiles with nvcc from CUDA_HOME (2-5 min) into
# TORCH_EXTENSIONS_DIR, which we keep inside the repo so it survives venv rebuilds.
export TORCH_EXTENSIONS_DIR="$ROOT/data/torch_extensions"
uv pip install --python "$ROOT/.venv/bin/python" ninja
for pkg in diff-surfel-rasterization simple-knn; do
  echo "== installing $pkg (editable) =="
  uv pip install --python "$ROOT/.venv/bin/python" --no-build-isolation --no-deps -e "$UP/$pkg"
done
echo "== JIT compiling (first import) =="
"$ROOT/.venv/bin/python" - <<'PY'
import torch, diff_surfel_rasterization, simple_knn
print("kernels import OK | torch", torch.__version__, "| cuda", torch.version.cuda)
PY
