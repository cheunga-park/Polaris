"""2D Gaussian Splatting, run exactly as the paper's authors point to: the hbb1 repository's
own ``train.py`` and ``render.py`` (pinned submodule), with the two CUDA kernels we already
built for the evaluation renderer. Nothing here is a reimplementation; this module only
assembles the command lines and reads the outputs.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TWO_DGS = ROOT / "third_party" / "2d-gaussian-splatting"


def _env() -> dict:
    e = dict(os.environ)
    # 2DGS imports its kernels by package name; ours are the upstream-polaris copies, same API
    e["PYTHONPATH"] = str(TWO_DGS) + os.pathsep + e.get("PYTHONPATH", "")
    return e


def _stream(cmd: list[str], log: Path, cwd: Path, progress=None) -> None:
    with log.open("a") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        p = subprocess.Popen(cmd, cwd=cwd, env=_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            f.write(line)
            if progress:
                m = re.search(r"Training progress:\s+(\d+)%", line) or re.search(r"(\d+)/(\d+)", line)
                if m:
                    progress(line.strip()[-120:])
        p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:2])} failed (exit {p.returncode}); see {log}")


@dataclass
class SplatReport:
    model_dir: str
    iterations: int
    n_gaussians: int
    seconds: float
    ply: str


def train(dataset: str | Path, model_dir: str | Path, *, iterations: int = 30_000, lambda_normal: float = 0.05,
          lambda_dist: float = 0.0, depth_ratio: float = 0.0, resolution: int = 1, progress=None) -> SplatReport:
    """``dataset`` = COLMAP layout (images/ + sparse/0/). Writes ``model_dir/point_cloud/iteration_N/point_cloud.ply``."""
    dataset, model_dir = Path(dataset).resolve(), Path(model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cmd = [sys.executable, "train.py", "-s", str(dataset), "-m", str(model_dir), "--iterations", str(iterations),
           "--lambda_normal", str(lambda_normal), "--lambda_dist", str(lambda_dist), "--depth_ratio", str(depth_ratio),
           "-r", str(resolution), "--test_iterations", "-1", "--save_iterations", str(iterations), "--quiet"]
    _stream(cmd, model_dir / "train.log", TWO_DGS, progress)
    ply = model_dir / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    if not ply.exists():
        raise RuntimeError(f"2DGS produced no {ply}")
    from plyfile import PlyData
    n = PlyData.read(str(ply))["vertex"].count
    rep = SplatReport(str(model_dir), iterations, int(n), time.time() - t0, str(ply))
    (model_dir / "splat.json").write_text(json.dumps(asdict(rep), indent=2))
    return rep


def extract_mesh(dataset: str | Path, model_dir: str | Path, *, voxel_size: float = 0.004, depth_trunc: float = 3.0,
                 sdf_trunc: float = 0.02, num_cluster: int = 1, iteration: int = -1, progress=None) -> Path:
    """2DGS ``render.py``: rasterise depth from the splat, TSDF-fuse, marching cubes, keep the
    largest ``num_cluster`` components -> ``model_dir/train/ours_N/fuse_post.ply`` (the file
    upstream's custom_environments.md names)."""
    dataset, model_dir = Path(dataset).resolve(), Path(model_dir).resolve()
    cmd = [sys.executable, "render.py", "-s", str(dataset), "-m", str(model_dir), "--skip_train", "--skip_test",
           "--voxel_size", str(voxel_size), "--depth_trunc", str(depth_trunc), "--sdf_trunc", str(sdf_trunc),
           "--num_cluster", str(num_cluster), "--iteration", str(iteration), "--quiet"]
    _stream(cmd, model_dir / "mesh.log", TWO_DGS, progress)
    cands = sorted(model_dir.glob("train/ours_*/fuse_post.ply"))
    if not cands:
        raise RuntimeError("2DGS render.py produced no fuse_post.ply")
    return cands[-1]
