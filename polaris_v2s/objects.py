"""Objects the paper's way (§3, "Object Creation with Generative Models"): a short multi-view
video of one object -> a few sharp frames -> SAM 2 masks -> TRELLIS (image-to-3D, multi-view
conditioning) -> a textured GLB, packed as ``assets/<name>/mesh.{glb,usdz}``.

TRELLIS runs in its own environment (``scripts/setup_trellis.sh``); this module prepares the
masked views, calls ``scripts/trellis_run.py`` there, and converts the result. SAM 2 runs here
with the weights already cached for this box (facebook/sam2.1-hiera-base-plus).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

from polaris_v2s import frames as frames_mod

ROOT = Path(__file__).resolve().parents[1]
TRELLIS_PY = Path.home() / "micromamba" / "envs" / "trellis" / "bin" / "python"
SAM2_CKPT = None  # resolved from the HF cache


def _sam2_paths():
    from huggingface_hub import snapshot_download
    d = Path(snapshot_download("facebook/sam2.1-hiera-base-plus", allow_patterns=["*.pt", "*.yaml"]))
    return d / "sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"


def pick_views(images_dir: Path, n: int = 6) -> list[Path]:
    """n frames spread over the scan (the orbit) -- multi-view conditioning wants coverage."""
    imgs = sorted(images_dir.glob("*.jpg"))
    if len(imgs) <= n:
        return imgs
    idx = np.linspace(0, len(imgs) - 1, n).round().astype(int)
    return [imgs[i] for i in idx]


def segment(views: list[Path], out_dir: Path, *, point: tuple[float, float] | None = None) -> list[Path]:
    """SAM 2 image predictor, prompted with one point (image-relative, default centre): the
    object is what the user put alone on the board in the middle of the frame. Writes RGBA
    PNGs with the mask as alpha (what TRELLIS takes)."""
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    ckpt, cfg = _sam2_paths()
    predictor = SAM2ImagePredictor(build_sam2(cfg, str(ckpt), device="cuda" if torch.cuda.is_available() else "cpu"))
    out_dir.mkdir(parents=True, exist_ok=True)
    outs = []
    for p in views:
        bgr = cv2.imread(str(p)); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        pt = np.array([[w * (point[0] if point else 0.5), h * (point[1] if point else 0.5)]])
        with torch.inference_mode():
            predictor.set_image(rgb)
            masks, scores, _ = predictor.predict(point_coords=pt, point_labels=np.array([1]), multimask_output=True)
        m = masks[int(np.argmax(scores))].astype(np.uint8)
        # keep the connected component under the prompt, fill holes
        n_lbl, lbl = cv2.connectedComponents(m)
        k = lbl[int(pt[0, 1]), int(pt[0, 0])]
        m = (lbl == k).astype(np.uint8) if k > 0 else m
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        rgba = np.dstack([rgb, m * 255])
        ys, xs = np.where(m > 0)
        if len(xs) == 0:
            continue
        pad = 20
        y0, y1, x0, x1 = max(0, ys.min() - pad), min(h, ys.max() + pad), max(0, xs.min() - pad), min(w, xs.max() + pad)
        crop = rgba[y0:y1, x0:x1]
        o = out_dir / f"{p.stem}_rgba.png"
        cv2.imwrite(str(o), cv2.cvtColor(crop, cv2.COLOR_RGBA2BGRA))
        outs.append(o)
    return outs


def trellis(rgba_views: list[Path], out_dir: Path, *, seed: int = 0) -> dict:
    """Run TRELLIS multi-image -> GLB (+ gaussian ply) in the trellis env."""
    if not TRELLIS_PY.exists():
        raise RuntimeError("TRELLIS env missing: run scripts/setup_trellis.sh")
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, ATTN_BACKEND="xformers", SPCONV_ALGO="native", PYTHONPATH=str(ROOT / "third_party" / "TRELLIS"))
    cmd = [str(TRELLIS_PY), str(ROOT / "scripts" / "trellis_run.py"), "--out", str(out_dir), "--seed", str(seed), *map(str, rgba_views)]
    log = out_dir / "trellis.log"
    with log.open("w") as f:
        r = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT / "third_party" / "TRELLIS"))
    if r.returncode != 0:
        raise RuntimeError(f"TRELLIS failed; see {log}")
    return json.loads((out_dir / "result.json").read_text())


def build_object(video: Path, work: Path, name: str, *, n_views: int = 6, size_m: float | None = None) -> dict:
    """video -> assets/<name>/{mesh.glb, mesh.usdz}. ``size_m``: longest side in metres (from the
    ChArUco-scaled scan, or typed by the user); TRELLIS output is unit-normalised."""
    rep = frames_mod.extract(video, work / "recon", fps=3, long_edge=1024, window=2, min_sharpness=20, max_frames=120)
    views = pick_views(work / "recon" / "images", n_views)
    rgba = segment(views, work / "masks")
    res = trellis(rgba, work / "trellis")
    glb = Path(res["glb"])
    import trimesh
    m = trimesh.load(str(glb), force="mesh")
    scale = (size_m / float(m.extents.max())) if size_m else 1.0
    m.apply_scale(scale)
    asset = work / "asset"; asset.mkdir(exist_ok=True)
    m.export(str(asset / "mesh.glb"))
    from polaris_v2s import mesh as mesh_mod
    mesh_mod.write_usdz(m, asset / "mesh.usdz", kinematic=False)
    mesh_mod.write_config(asset, kinematic=False)
    return {"frames": rep.kept, "views": len(rgba), "faces": int(len(m.faces)), "extent_m": [float(x) for x in m.extents], "scale_applied": scale, "asset": str(asset)}
