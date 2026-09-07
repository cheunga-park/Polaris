"""One scan = one directory under data/scans/<id>/, one stage = one subdirectory, and the
stage files on disk are the only state. A job runs stages in order, skips the ones whose
output already exists (re-runnable), and writes ``status.json`` after every stage so the
console can show where it is and what failed.

    data/scans/<id>/
      upload/video.mp4            the phone video (background scan)
      upload/objects/<name>.mp4   object scans (optional)
      recon/images/               frames.extract
      recon/sparse/0, undistorted/ colmap
      recon/charuco/ref_images.txt, sparse/aligned
      splat/                      2DGS model
      env/                        PolaRiS-Hub-shaped environment folder (the product)
      status.json
"""

from __future__ import annotations

import json
import shutil
import threading
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import yaml

from polaris_v2s import charuco, colmap, frames, mesh, splat

ROOT = Path(__file__).resolve().parents[1]
SCANS = ROOT / "data" / "scans"
PIPELINE_CFG = ROOT / "configs" / "pipeline.yaml"

STAGES = ["frames", "colmap", "charuco", "splat", "mesh", "pack"]


@dataclass
class Status:
    scan: str
    stage: str = ""            # running stage
    done: list[str] = field(default_factory=list)
    failed: str | None = None
    error: str | None = None
    progress: str = ""
    reports: dict = field(default_factory=dict)
    started: float = 0.0
    finished: float = 0.0
    warnings: list[str] = field(default_factory=list)


def status_path(scan_dir: Path) -> Path:
    return scan_dir / "status.json"


def read_status(scan_dir: Path) -> Status:
    p = status_path(scan_dir)
    if p.exists():
        d = json.loads(p.read_text())
        return Status(**d)
    return Status(scan=scan_dir.name)


def _write(scan_dir: Path, st: Status) -> None:
    status_path(scan_dir).write_text(json.dumps(asdict(st), indent=2))


def run(scan_id: str, *, cfg_path: Path = PIPELINE_CFG, force: list[str] | None = None,
        require_board: bool = False) -> Status:
    cfg = yaml.safe_load(cfg_path.read_text())
    scan_dir = SCANS / scan_id
    video = scan_dir / "upload" / "video.mp4"
    if not video.exists():
        raise FileNotFoundError(video)
    recon = scan_dir / "recon"
    model_dir = scan_dir / "splat"
    env_dir = scan_dir / "env"
    st = read_status(scan_dir)
    st.failed = st.error = None
    st.started = st.started or time.time()
    force = set(force or [])

    def stage(name: str, exists: bool, fn):
        if exists and name not in force:
            st.done = sorted(set(st.done) | {name}, key=STAGES.index)
            return
        st.stage = name; st.progress = ""; _write(scan_dir, st)
        rep = fn()
        st.reports[name] = rep
        st.done = sorted(set(st.done) | {name}, key=STAGES.index)
        st.stage = ""; _write(scan_dir, st)

    def progress(msg: str):
        st.progress = msg; _write(scan_dir, st)

    try:
        # 1. frames
        stage("frames", (recon / "frames.json").exists(),
              lambda: asdict(frames.extract(video, recon, **cfg["frames"])))
        # 2. colmap (sparse + undistort of the unaligned model; realigned below if a board is found)
        def _colmap():
            r = colmap.reconstruct(recon, camera_model=cfg["colmap"]["camera_model"])
            return asdict(r)
        stage("colmap", (recon / "colmap.json").exists(), _colmap)
        # 3. charuco -> metric alignment (skipped, with a warning, when no board is seen)
        def _charuco():
            spec = charuco.BoardSpec.load()
            K, dist = charuco.colmap_intrinsics(recon / "sparse" / "0")
            poses = charuco.detect(recon / "images", K, dist, spec)
            out = recon / "charuco"; out.mkdir(exist_ok=True)
            n = charuco.write_ref_images(poses, out / "ref_images.txt")
            rep = {"frames_with_board": len(poses), "used": n, "board": asdict(spec)}
            if n >= 3:
                aligned = colmap.align(recon, out / "ref_images.txt")
                rep["aligned_model"] = str(aligned)
                # scale sanity: camera-centre spread before/after
                rep["scale_factor"] = _scale_factor(recon / "sparse" / "0", aligned)
            else:
                msg = f"ChArUco board seen in {len(poses)} frames ({n} usable): scene is NOT metric"
                st.warnings.append(msg)
                if require_board:
                    raise RuntimeError(msg)
            model = Path(rep.get("aligned_model", recon / "sparse" / "0"))
            colmap.undistort(recon, model)
            return rep
        stage("charuco", (recon / "undistorted" / "sparse" / "0").exists(), _charuco)
        # 4. 2DGS
        stage("splat", (model_dir / "splat.json").exists(),
              lambda: asdict(splat.train(recon / "undistorted", model_dir, progress=progress, **cfg["splat"])))
        # 5. mesh
        stage("mesh", any(model_dir.glob("train/ours_*/fuse_post.ply")),
              lambda: {"fuse_post": str(splat.extract_mesh(recon / "undistorted", model_dir, progress=progress, **cfg["mesh"]))})
        # 6. pack -> env/
        def _pack():
            fuse = sorted(model_dir.glob("train/ours_*/fuse_post.ply"))[-1]
            ply = Path(json.loads((model_dir / "splat.json").read_text())["ply"])
            asset = env_dir / "assets" / f"{scan_id}_static"
            info = mesh.package_background(fuse, ply, asset)
            _write_scene(env_dir, scan_id, f"{scan_id}_static")
            return info
        stage("pack", (env_dir / "scene.usda").exists(), _pack)
        st.finished = time.time()
    except Exception as ex:  # noqa: BLE001 - the status file is the report
        st.failed = st.stage or "?"
        st.error = f"{type(ex).__name__}: {ex}\n{traceback.format_exc()[-2000:]}"
        st.stage = ""
    _write(scan_dir, st)
    return st


def _scale_factor(before: Path, after: Path) -> float:
    import pycolmap
    a = pycolmap.Reconstruction(str(before)); b = pycolmap.Reconstruction(str(after))
    ca = np.array([im.projection_center() for im in a.images.values()])
    cb = np.array([im.projection_center() for im in b.images.values()])
    return float(cb.std() / max(ca.std(), 1e-9))


def _write_scene(env_dir: Path, scan_id: str, static_name: str) -> None:
    """A minimal scene.usda in the compose-environments export format: the background only,
    at the origin (the GUI is where objects and the robot offset get placed)."""
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "scene.usda").write_text(f'''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{{
    def Xform "{static_name}" (
        prepend payload = @./assets/{static_name}/mesh.usdz@
    )
    {{
        bool physics:kinematicEnabled = true
        double3 xformOp:translate = (0, 0, 0)
        quatd xformOp:orient = (1, 0, 0, 0)
        float3 xformOp:scale = (1, 1, 1)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
    }}
}}
''')
    ic = env_dir / "initial_conditions.json"
    if not ic.exists():
        ic.write_text(json.dumps({"instruction": "", "poses": []}, indent=2))


# ---- background runner for the console --------------------------------------------------
_threads: dict[str, threading.Thread] = {}


def start(scan_id: str, **kw) -> bool:
    t = _threads.get(scan_id)
    if t and t.is_alive():
        return False
    t = threading.Thread(target=run, args=(scan_id,), kwargs=kw, daemon=True)
    _threads[scan_id] = t
    t.start()
    return True


def list_scans() -> list[dict]:
    out = []
    if SCANS.exists():
        for d in sorted(SCANS.iterdir()):
            if (d / "upload" / "video.mp4").exists():
                s = read_status(d)
                out.append({**asdict(s), "running": bool(_threads.get(d.name) and _threads[d.name].is_alive())})
    return out
