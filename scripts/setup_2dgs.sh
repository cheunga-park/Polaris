#!/usr/bin/env bash
# 2DGS exactly as its authors ship it, in its own venv.
#
# Why a second venv: the evaluation renderer uses upstream-polaris's *fork* of the 2DGS
# rasterizer (extra near/far arguments), 2DGS training uses the original. Both are the
# Python package `diff_surfel_rasterization`, so they cannot share an environment.
# `.venv`      = polaris fork (eval)      `.venv-2dgs` = original (reconstruction)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"
G="$ROOT/third_party/2d-gaussian-splatting"
V="$ROOT/.venv-2dgs"
git -C "$G" submodule update --init --recursive submodules/diff-surfel-rasterization submodules/simple-knn
[ -d "$V" ] || uv venv -q --python 3.12 "$V"
uv pip install -q --python "$V/bin/python" --index-url https://download.pytorch.org/whl/cu128 "torch>=2.8" "torchvision>=0.23"
uv pip install -q --python "$V/bin/python" ninja open3d mediapy lpips scikit-image tqdm trimesh plyfile opencv-python-headless
export TORCH_EXTENSIONS_DIR="$ROOT/data/torch_extensions_2dgs"
# gcc 13 + the 2023 kernel sources: rasterizer_impl.h uses uint32_t without <cstdint>.
# nvcc's --pre-include adds the header on the command line, so the checkout stays untouched.
export NVCC_APPEND_FLAGS="--pre-include cstdint --pre-include cfloat"   # simple_knn.cu: FLT_MAX
for pkg in diff-surfel-rasterization simple-knn; do
  echo "== building original $pkg =="
  uv pip install -q --python "$V/bin/python" --no-build-isolation --no-deps "$G/submodules/$pkg"
done
"$V/bin/python" - <<'PY'
import torch, diff_surfel_rasterization as d, simple_knn
s = d.GaussianRasterizationSettings.__new__.__code__.co_varnames
print("2DGS original kernels OK | torch", torch.__version__, "| settings fields:", len(d.GaussianRasterizationSettings._fields))
PY
