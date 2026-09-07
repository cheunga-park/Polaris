"""scene.usda -> MuJoCo.

Reads the composed environment with usd-core (the same file upstream's ``dynamic_setup``
reads), exports every prim's mesh once into a cache, decomposes it into convex pieces with
CoACD (PhysX uses ``convexDecomposition`` for the same objects), and returns the rows a
MjSpec builder needs. Nothing here touches upstream code.

Cache layout: data/cache/<env>/prims/<name>/{visual.obj, collision_00.obj, ...}
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

from polaris_v2s import usd_io

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("POLARIS_CACHE_PATH", ROOT / "data" / "cache")).resolve()


@dataclass
class BodySpec:
    name: str
    kinematic: bool
    translate: list[float]
    orient_wxyz: list[float]
    scale: list[float]
    visual_obj: Path | None
    collision_objs: list[Path] = field(default_factory=list)
    has_splat: bool = False
    asset_dir: str | None = None
    mass: float | None = None        # None -> density
    extent_m: list[float] = field(default_factory=list)


def _coacd(mesh: trimesh.Trimesh, out_dir: Path, *, threshold: float, max_hulls: int) -> list[Path]:
    import coacd
    coacd.set_log_level("error")
    m = coacd.Mesh(np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int32))
    parts = coacd.run_coacd(m, threshold=threshold, max_convex_hull=max_hulls, preprocess_mode="auto", merge=True)
    paths = []
    for i, (v, f) in enumerate(parts):
        p = out_dir / f"collision_{i:02d}.obj"
        trimesh.Trimesh(vertices=v, faces=f, process=False).export(str(p))
        paths.append(p)
    return paths


def _fingerprint(scene_usda: Path, name: str, params: dict) -> str:
    h = hashlib.sha1()
    h.update(str(scene_usda.stat().st_mtime_ns).encode()); h.update(name.encode()); h.update(json.dumps(params, sort_keys=True).encode())
    return h.hexdigest()[:12]


# The static background is a whole room; only the part the arm can touch needs collision
# fidelity. Static meshes are cropped to this box (world frame, metres) before decomposition
# and decomposed finer than objects. PhysX in upstream collides against the exact triangle
# mesh (``meshSimplification``); this is the honest MuJoCo approximation of that.
WORKSPACE = np.array([[-0.5, -1.0, -0.3], [1.3, 1.0, 0.8]])


def _crop_static(mesh: trimesh.Trimesh, translate, orient_wxyz, scale) -> trimesh.Trimesh:
    """Crop a prim-frame mesh to WORKSPACE expressed in the prim frame."""
    w, x, y, z = orient_wxyz
    R = trimesh.transformations.quaternion_matrix([w, x, y, z])[:3, :3]
    S = np.asarray(scale, dtype=float)
    v_world = (mesh.vertices * S) @ R.T + np.asarray(translate)
    inside = np.all((v_world >= WORKSPACE[0]) & (v_world <= WORKSPACE[1]), axis=1)
    keep_faces = inside[mesh.faces].any(axis=1)
    if keep_faces.sum() == 0:
        return mesh
    out = mesh.submesh([np.where(keep_faces)[0]], append=True)
    return out


def _chunk_hulls(mesh: trimesh.Trimesh, out_dir: Path, *, cell: float = 0.06, thickness: float = 0.004) -> list[Path]:
    """Static geometry as a grid of small convex chunks (prim frame in, prim frame out).

    CoACD on a room-sized mesh is a Hausdorff approximation: with threshold 0.02 the
    tabletop hull sat ~25 mm under the real surface. Splitting the cropped mesh into
    ``cell``-sized boxes and taking the convex hull of each chunk's triangles is exact on
    flat and gently curved surfaces (a tabletop) and only approximates within one cell
    elsewhere. Coplanar chunks get ``thickness`` extruded along -normal so every hull has
    volume (MuJoCo needs it)."""
    tri = mesh.triangles                                   # (F, 3, 3)
    cen = tri.mean(axis=1)
    keys = np.floor(cen / cell).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    fn = mesh.face_normals
    paths = []
    for k in range(inv.max() + 1):
        f = np.where(inv == k)[0]
        v = tri[f].reshape(-1, 3)
        n = fn[f].mean(axis=0)
        n = n / (np.linalg.norm(n) + 1e-12)
        v = np.vstack([v, v - n * thickness])
        try:
            hull = trimesh.Trimesh(vertices=v, process=False).convex_hull
        except Exception:
            continue
        if hull.volume < 1e-9:
            continue
        pth = out_dir / f"collision_{len(paths):03d}.obj"
        hull.export(str(pth))
        paths.append(pth)
    return paths


def prepare(scene_usda: str | Path, *, env_name: str | None = None, coacd_threshold: float = 0.05,
            max_hulls_object: int = 24, static_cell: float = 0.06,
            force: bool = False) -> list[BodySpec]:
    """Every rigid/static prim of the scene as a BodySpec with cached OBJ files."""
    scene_usda = Path(scene_usda)
    env_name = env_name or scene_usda.parent.name
    info = usd_io.read_scene(scene_usda)
    out = []
    for p in info.prims:
        if not p.asset_dir:
            continue  # inline Mesh prims (e.g. a table plane in move_latte_cup) are handled by the builder as boxes if needed
        d = CACHE / env_name / "prims" / p.name
        d.mkdir(parents=True, exist_ok=True)
        params = ({"cell": static_cell, "crop": True, "v": 3} if p.kinematic
                  else {"thr": coacd_threshold, "hulls": max_hulls_object, "v": 1})
        fp = _fingerprint(scene_usda, p.name, params)
        stamp = d / "coacd.json"
        mesh = None
        if force or not stamp.exists() or json.loads(stamp.read_text()).get("fp") != fp:
            mesh = usd_io.prim_to_trimesh(scene_usda, p.name)
            mesh.export(str(d / "visual.obj"))
            for old in d.glob("collision_*.obj"):
                old.unlink()
            if p.kinematic:
                hulls = _chunk_hulls(_crop_static(mesh, p.translate, p.orient_wxyz, p.scale), d, cell=static_cell)
            else:
                hulls = _coacd(mesh, d, threshold=params["thr"], max_hulls=params["hulls"])
            stamp.write_text(json.dumps({"fp": fp, "hulls": len(hulls), "extent": [float(x) for x in mesh.extents],
                                         "faces": int(len(mesh.faces))}))
        meta = json.loads(stamp.read_text())
        out.append(BodySpec(
            name=p.name, kinematic=p.kinematic, translate=p.translate, orient_wxyz=p.orient_wxyz, scale=p.scale,
            visual_obj=d / "visual.obj", collision_objs=sorted(d.glob("collision_*.obj")), has_splat=p.has_splat,
            asset_dir=p.asset_dir, extent_m=meta["extent"]))
    return out
