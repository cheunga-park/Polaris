"""COLMAP as the paper uses it: camera poses for every frame, then metric alignment.

Sequential matching (it is a video) with a loop-closure vocab tree, a single shared
camera model (one phone), then ``image_undistorter`` so 2DGS gets pinhole images —
the same dataset layout 2DGS's own ``convert.py`` produces.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path



def _run(cmd: list[str], log: Path) -> None:
    with log.open("a") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} {cmd[1]} failed (exit {r.returncode}); see {log}")


@dataclass
class ColmapReport:
    images: int
    registered: int
    points3d: int
    mean_track_length: float
    mean_reproj_error: float
    seconds: float
    workspace: str


def model_stats(sparse: Path) -> dict:
    import pycolmap
    rec = pycolmap.Reconstruction(str(sparse))
    return {"registered": int(rec.num_reg_images()), "points3d": int(rec.num_points3D()),
            "mean_track_length": float(rec.compute_mean_track_length()),
            "mean_reproj_error": float(rec.compute_mean_reprojection_error())}


def reconstruct(workspace: str | Path, *, colmap: str = "colmap", use_gpu: bool = True,
                camera_model: str = "OPENCV", cache_dir: Path | None = None,
                loop_detection: bool = True, vocab_tree_path: Path | None = None) -> ColmapReport:
    """``workspace/images`` in -> ``workspace/sparse/0`` (aligned later) and
    ``workspace/undistorted/`` (2DGS input) out."""
    ws = Path(workspace)
    log = ws / "colmap.log"
    t0 = time.time()
    db = ws / "database.db"
    if db.exists():
        db.unlink()
    gpu = "1" if use_gpu else "0"
    _run([colmap, "feature_extractor", "--database_path", str(db), "--image_path", str(ws / "images"),
          "--ImageReader.single_camera", "1", "--ImageReader.camera_model", camera_model,
          "--FeatureExtraction.use_gpu", gpu, "--SiftExtraction.max_image_size", "2000"], log)
    match = [colmap, "sequential_matcher", "--database_path", str(db), "--FeatureMatching.use_gpu", gpu,
             "--SequentialMatching.overlap", "15", "--SequentialMatching.quadratic_overlap", "1"]
    # Loop closure needs a vocab tree in THIS colmap's format (3.12+ switched to a faiss index and
    # cannot read the old demuc.de files). Without one, a long scan still closes loops through the
    # quadratic sequential overlap; with one, revisited views are matched explicitly.
    # COLMAP 4.x downloads and caches its own default faiss tree when no path is given.
    if loop_detection:
        match += ["--SequentialMatching.loop_detection", "1"]
        if vocab_tree_path and Path(vocab_tree_path).exists():
            match += ["--SequentialMatching.vocab_tree_path", str(vocab_tree_path)]
    else:
        match += ["--SequentialMatching.loop_detection", "0"]
    _run(match, log)
    sparse = ws / "sparse"
    shutil.rmtree(sparse, ignore_errors=True)
    sparse.mkdir()
    _run([colmap, "mapper", "--database_path", str(db), "--image_path", str(ws / "images"),
          "--output_path", str(sparse), "--Mapper.ba_global_function_tolerance", "0.000001"], log)
    model = sparse / "0"
    if not model.exists():
        raise RuntimeError("COLMAP mapper produced no model (too few matches?)")
    st = model_stats(model)
    n_images = len(list((ws / "images").glob("*.jpg")))
    rep = ColmapReport(images=n_images, seconds=time.time() - t0, workspace=str(ws), **st)
    (ws / "colmap.json").write_text(json.dumps(asdict(rep), indent=2))
    return rep


def align(workspace: str | Path, ref_images_txt: Path, *, colmap: str = "colmap",
          max_error: float = 0.05) -> Path:
    """Sim(3)-align ``sparse/0`` to metric camera centres (from the ChArUco board) with
    COLMAP's ``model_aligner`` — the paper's method. Writes ``sparse/aligned``."""
    ws = Path(workspace)
    out = ws / "sparse" / "aligned"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    _run([colmap, "model_aligner", "--input_path", str(ws / "sparse" / "0"), "--output_path", str(out),
          "--ref_images_path", str(ref_images_txt), "--ref_is_gps", "0", "--alignment_type", "custom",
          "--alignment_max_error", str(max_error)], ws / "colmap.log")
    return out


def undistort(workspace: str | Path, model: Path, *, colmap: str = "colmap") -> Path:
    """Pinhole images + sparse model in the layout 2DGS's ``train.py -s`` reads."""
    ws = Path(workspace)
    out = ws / "undistorted"
    shutil.rmtree(out, ignore_errors=True)
    _run([colmap, "image_undistorter", "--image_path", str(ws / "images"), "--input_path", str(model),
          "--output_path", str(out), "--output_type", "COLMAP"], ws / "colmap.log")
    # 2DGS/3DGS expect sparse/0/
    (out / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    for f in (out / "sparse").glob("*.bin"):
        shutil.move(str(f), out / "sparse" / "0" / f.name)
    return out
