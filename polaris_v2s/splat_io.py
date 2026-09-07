"""Gaussian-splat PLY conversions.

2DGS writes surfels with two log-scales (``scale_0``, ``scale_1``); every web viewer
expects the 3DGS layout with three. ``to_web_ply`` pads a thin third axis and drops the
view-dependent SH bands so a 110 MB background becomes ~25 MB for the browser. The
simulator-side renderer (upstream ``SplatRenderer``) reads the original file, never this.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

WEB_FIELDS = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
              "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]


def read_splat(path: str | Path) -> dict[str, np.ndarray]:
    v = PlyData.read(str(path))["vertex"]
    return {name: np.asarray(v[name]) for name in v.data.dtype.names}


def to_web_ply(src: str | Path, dst: str | Path, thin: float = 2.0) -> Path:
    """Write a 3DGS-layout PLY (float32, SH degree 0) for browser viewers.

    ``thin``: the padded third log-scale is ``min(scale_0, scale_1) - thin`` (e^-2 ~ 1/7).
    """
    d = read_splat(src)
    n = len(d["x"])
    if "scale_2" not in d:
        d["scale_2"] = np.minimum(d["scale_0"], d["scale_1"]) - thin
    out = np.empty(n, dtype=[(f, "f4") for f in WEB_FIELDS])
    for f in WEB_FIELDS:
        out[f] = d[f].astype(np.float32)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(out, "vertex")], text=False).write(str(dst))
    return dst


def transform_splat(src: str | Path, dst: str | Path, R: np.ndarray, t: np.ndarray, s: float = 1.0) -> Path:
    """Apply a similarity (x' = s R x + t) to a 2DGS/3DGS PLY: positions, rotations, log-scales.
    Normals and SH are left as they are (SH would need a rotation of the bands; degree-0 view is fine)."""
    from scipy.spatial.transform import Rotation
    ply = PlyData.read(str(src)); v = ply["vertex"]
    xyz = np.c_[v["x"], v["y"], v["z"]].astype(np.float64)
    xyz = (s * (xyz @ R.T)) + t
    q = np.c_[v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]].astype(np.float64)          # wxyz
    r = Rotation.from_matrix(R) * Rotation.from_quat(q[:, [1, 2, 3, 0]])                    # scipy is xyzw
    q2 = r.as_quat()[:, [3, 0, 1, 2]]
    names = v.data.dtype.names
    out = np.empty(len(xyz), dtype=v.data.dtype)
    for n in names:
        out[n] = v[n]
    out["x"], out["y"], out["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    for i in range(4):
        out[f"rot_{i}"] = q2[:, i]
    for n in names:
        if n.startswith("scale_"):
            out[n] = v[n] + np.log(s)
    if "nx" in names:
        nrm = np.c_[v["nx"], v["ny"], v["nz"]].astype(np.float64) @ R.T
        out["nx"], out["ny"], out["nz"] = nrm[:, 0], nrm[:, 1], nrm[:, 2]
    dst = Path(dst); dst.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(out, "vertex")], text=False).write(str(dst))
    return dst
