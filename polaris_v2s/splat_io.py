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
