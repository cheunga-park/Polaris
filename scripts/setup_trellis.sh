#!/usr/bin/env bash
# TRELLIS (microsoft/TRELLIS, image -> 3D) in its own micromamba env `trellis`.
# Official setup.sh wants conda + torch 2.4/cu118; this box has nvcc 12.8 in user space and a
# driver capped at CUDA 12.7, so: python 3.10, torch 2.5.1+cu124 wheels, xformers attention
# (prebuilt), kaolin/spconv prebuilt wheels, and the three small CUDA extensions built here.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"
MM="$HOME/.local/bin/micromamba"; export MAMBA_ROOT_PREFIX="$HOME/micromamba"
ENV="$MAMBA_ROOT_PREFIX/envs/trellis"; PY="$ENV/bin/python"
[ -x "$PY" ] || "$MM" create -y -q -n trellis -c conda-forge python=3.10 "gxx<13" git
T="$ROOT/third_party/TRELLIS"
[ -d "$T" ] || git clone -q --recursive https://github.com/microsoft/TRELLIS.git "$T"
export TORCH_CUDA_ARCH_LIST="8.9" CUDA_HOME="$POLARIS_TOOLS" PATH="$ENV/bin:$PATH"
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
"$PY" -m pip install -q pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix igraph transformers tensorboard pandas lpips "numpy<2" huggingface_hub
"$PY" -m pip install -q git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
"$PY" -m pip install -q xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu124
"$PY" -m pip install -q kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html
"$PY" -m pip install -q spconv-cu120
export NVCC_APPEND_FLAGS="--pre-include cstdint --pre-include cfloat"
"$PY" -m pip install -q --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git
"$PY" -m pip install -q --no-build-isolation git+https://github.com/JeffreyXiang/diffoctreerast.git
"$PY" -m pip install -q --no-build-isolation "$T/extensions/vox2seq" || echo "vox2seq skipped (optional)"
tmp="$(mktemp -d)"; git clone -q --recursive https://github.com/autonomousvision/mip-splatting.git "$tmp/mip"
"$PY" -m pip install -q --no-build-isolation "$tmp/mip/submodules/diff-gaussian-rasterization/"; rm -rf "$tmp"
cd "$T" && ATTN_BACKEND=xformers SPCONV_ALGO=native "$PY" - <<'PY'
import torch, kaolin, spconv, xformers, nvdiffrast.torch, diffoctreerast
print("TRELLIS env OK | torch", torch.__version__, "cuda", torch.version.cuda, "| kaolin", kaolin.__version__, "| xformers", xformers.__version__)
PY
