#!/usr/bin/env bash
# Source this: `source scripts/env.sh`. User-space toolchain (no sudo on this box):
#   micromamba env `polaris-tools` = nvcc 12.8, COLMAP (CUDA), ffmpeg, node
#   uv venv `.venv`               = python 3.12, torch cu128, mujoco, usd-core
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
export POLARIS_TOOLS="$MAMBA_ROOT_PREFIX/envs/polaris-tools"
export CUDA_HOME="$POLARIS_TOOLS"                       # nvcc + headers live here
export PATH="$POLARIS_TOOLS/bin:$ROOT/.venv/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"   # RTX 4080 (Ada); add others for a shared box
# MuJoCo offscreen on WSL2: EGL via Mesa's d3d12 driver reaches the real GPU (measured: 5 ms / 1280x720 frame)
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export GALLIUM_DRIVER="${GALLIUM_DRIVER:-d3d12}"
export POLARIS_DATA_PATH="${POLARIS_DATA_PATH:-$ROOT/data/hub}"   # upstream polaris reads this
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$ROOT/data/torch_extensions}"
