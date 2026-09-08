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
    hfield: Path | None = None       # static prims: support heightfield (npz), replaces chunk hulls inside the workspace


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


def _chunk_hulls(mesh: trimesh.Trimesh, out_dir: Path, *, cell: float = 0.06, thickness: float = 0.03, z_above: float | None = None) -> list[Path]:
    """Static geometry as a grid of small convex chunks (prim frame in, prim frame out).

    CoACD on a room-sized mesh is a Hausdorff approximation: with threshold 0.02 the
    tabletop hull sat ~25 mm under the real surface. Splitting the cropped mesh into
    ``cell``-sized boxes and taking the convex hull of each chunk's triangles is exact on
    flat and gently curved surfaces (a tabletop) and only approximates within one cell
    elsewhere. Every chunk is extruded ``thickness`` along -normal (into the surface, away from
    the scene): with 4 mm an object dropped from its (hovering) initial condition tunnelled
    through a stovetop in one 1/120 s step; 30 mm is thicker than anything falls per step."""
    tri = mesh.triangles                                   # (F, 3, 3)
    cen = tri.mean(axis=1)
    if z_above is not None:                                # with a support heightfield, hulls only above the band
        keep = cen[:, 2] > z_above; tri = tri[keep]; cen = cen[keep]
        fn_all = mesh.face_normals[keep]
    else:
        fn_all = mesh.face_normals
    keys = np.floor(cen / cell).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    fn = fn_all
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
            max_hulls_object: int = 24, static_cell: float = 0.06, static_thickness: float = 0.03,
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
        # kinematic prims: the room/table-sized background gets the heightfield + chunk hulls; a small
        # static object (a pan or mug pinned in place) keeps its concave shape via CoACD like rigid ones
        big_static = p.kinematic and (p.has_splat or max(usd_io.prim_to_trimesh(scene_usda, p.name).extents * np.asarray(p.scale)) > 0.8)
        params = ({"cell": static_cell, "thick": static_thickness, "crop": True, "hfield": HFIELD_RES, "band": SUPPORT_BAND_Z, "fill": "ic+prims_p90", "v": 8} if big_static
                  else {"thr": coacd_threshold, "hulls": max_hulls_object, "v": 1})
        fp = _fingerprint(scene_usda, p.name, params)
        stamp = d / "coacd.json"
        mesh = None
        if force or not stamp.exists() or json.loads(stamp.read_text()).get("fp") != fp:
            mesh = usd_io.prim_to_trimesh(scene_usda, p.name)
            mesh.export(str(d / "visual.obj"))
            for old in d.glob("collision_*.obj"):
                old.unlink()
            if big_static:
                # chunk hulls in the PRIM frame need the band in world z: crop returns prim-frame mesh; convert the band
                cropped = _crop_static(mesh, p.translate, p.orient_wxyz, p.scale)
                w_, x_, y_, z_ = p.orient_wxyz; Rz = trimesh.transformations.quaternion_matrix([w_, x_, y_, z_])[:3, :3]
                world = (cropped.vertices * np.asarray(p.scale)) @ Rz.T + np.asarray(p.translate)
                high = trimesh.Trimesh(vertices=cropped.vertices, faces=cropped.faces[world[cropped.faces].mean(1)[:, 2] > SUPPORT_BAND_Z - 0.03], process=False)
                hulls = _chunk_hulls(high, d, cell=static_cell, thickness=static_thickness) if len(high.faces) else []
                others = np.array([q.translate[:2] for q in info.prims if q.asset_dir and q.name != p.name])
                support_hfield(mesh, p.translate, p.orient_wxyz, p.scale, d / "hfield.npz", ic_json=scene_usda.parent / "initial_conditions.json", extra_xy=others)
            else:
                hulls = _coacd(mesh, d, threshold=params["thr"], max_hulls=params["hulls"])
            stamp.write_text(json.dumps({"fp": fp, "hulls": len(hulls), "extent": [float(x) for x in mesh.extents],
                                         "faces": int(len(mesh.faces))}))
        meta = json.loads(stamp.read_text())
        out.append(BodySpec(
            name=p.name, kinematic=p.kinematic, translate=p.translate, orient_wxyz=p.orient_wxyz, scale=p.scale,
            visual_obj=d / "visual.obj", collision_objs=sorted(d.glob("collision_*.obj")), has_splat=p.has_splat,
            asset_dir=p.asset_dir, extent_m=meta["extent"], hfield=(d / "hfield.npz") if (big_static and (d / "hfield.npz").exists()) else None))
    return out


# ----------------------------------------------------------------------------- static support as a heightfield
HFIELD_RES = 0.015   # metres per cell (5 mm made hfield-vs-mesh contacts explode)
SUPPORT_BAND_Z = 0.15  # the heightfield takes surfaces up to this height; hulls take what is above (walls, hoods)


def _rasterize_upper_envelope(V: np.ndarray, F: np.ndarray, x0: float, x1: float, y0: float, y1: float, res: float):
    """Max-z of the mesh per grid cell (world frame). Cells no triangle covers are NaN."""
    nx = int(np.ceil((x1 - x0) / res)) + 1; ny = int(np.ceil((y1 - y0) / res)) + 1
    H = np.full((ny, nx), np.nan)
    tri = V[F]                                   # (T, 3, 3)
    keep = (tri[:, :, 0].max(1) >= x0) & (tri[:, :, 0].min(1) <= x1) & (tri[:, :, 1].max(1) >= y0) & (tri[:, :, 1].min(1) <= y1)
    for a, b, c in tri[keep]:
        xs = np.array([a[0], b[0], c[0]]); ys = np.array([a[1], b[1], c[1]])
        i0, i1 = max(0, int((xs.min() - x0) / res)), min(nx - 1, int((xs.max() - x0) / res) + 1)
        j0, j1 = max(0, int((ys.min() - y0) / res)), min(ny - 1, int((ys.max() - y0) / res) + 1)
        if i1 < i0 or j1 < j0:
            continue
        gx = x0 + np.arange(i0, i1 + 1) * res; gy = y0 + np.arange(j0, j1 + 1) * res
        X, Y = np.meshgrid(gx, gy)
        d = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
        if abs(d) < 1e-12:
            continue
        l1 = ((b[0] - X) * (c[1] - Y) - (c[0] - X) * (b[1] - Y)) / d
        l2 = ((c[0] - X) * (a[1] - Y) - (a[0] - X) * (c[1] - Y)) / d
        l3 = 1 - l1 - l2
        inside = (l1 >= -1e-6) & (l2 >= -1e-6) & (l3 >= -1e-6)
        if not inside.any():
            continue
        Z = l1 * a[2] + l2 * b[2] + l3 * c[2]
        sub = H[j0:j1 + 1, i0:i1 + 1]
        cand = np.where(inside, Z, np.nan)
        H[j0:j1 + 1, i0:i1 + 1] = np.fmax(sub, cand)
    return H


def _fill_placement_region(H: np.ndarray, x0: float, y0: float, res: float, ic_json: Path, dilate_m: float = 0.05, pct: float = 90,
                           extra_xy: np.ndarray | None = None) -> dict | None:
    """Raise the support inside the object-placement region to its own p90 height.

    The scans lose thin structure (a stove grate's bars): the TSDF mesh keeps the burner well
    but not the bars over it, so a dropped sponge tips 30-60 deg into the well and no
    top-down pinch can hold it; the real sponge lies flat on the bars. The initial conditions
    say where objects are placed; inside that footprint (+5 cm) the support becomes the
    local p90 height (bar tops), leaving everything else as scanned. Recorded delta (PLAN 8.5)."""
    import json
    from scipy.spatial import ConvexHull, Delaunay
    from scipy.ndimage import binary_dilation
    if not ic_json.exists():
        return None
    poses = json.loads(ic_json.read_text()).get("poses", []) if ic_json.exists() else []
    pts = np.array([p[n][:2] for p in poses for n in p]) if poses else np.zeros((0, 2))
    if extra_xy is not None and len(extra_xy):
        # containers pinned in place (the pan, the mug) sit on the same support; the scan under them
        # is occluded (a hole), and a sponge released over the pan rim fell through to the floor
        pts = np.vstack([pts, np.asarray(extra_xy, float).reshape(-1, 2)]) if len(pts) else np.asarray(extra_xy, float).reshape(-1, 2)
    if len(pts) < 3:
        return None
    hull = ConvexHull(pts); poly = pts[hull.vertices]
    ny, nx = H.shape; gx = x0 + np.arange(nx) * res; gy = y0 + np.arange(ny) * res; X, Y = np.meshgrid(gx, gy)
    inside = (Delaunay(poly).find_simplex(np.c_[X.ravel(), Y.ravel()]) >= 0).reshape(ny, nx)
    inside = binary_dilation(inside, iterations=max(1, int(dilate_m / res)))
    top = float(np.percentile(H[inside], pct)); before = float(np.percentile(H[inside], 50))
    H[inside] = np.maximum(H[inside], top)
    return {"cells": int(inside.sum()), "support_z": top, "median_before": before}


def support_hfield(mesh: trimesh.Trimesh, translate, orient_wxyz, scale, out_path: Path, *, res: float = HFIELD_RES,
                   box=WORKSPACE, zmax: float = 0.6, ic_json: Path | None = None, extra_xy=None) -> dict:
    """The static prim's upper envelope over the workspace as a MuJoCo heightfield (npz + meta).

    Why: a scanned tabletop/stove is not flat -- the sponge in PanClean lies in a groove of the
    grate. Convex chunks bridge such grooves and the fingertips stop ~1 cm too high, which loses
    the grasp; the exact upper envelope keeps them (PhysX in upstream collides with the raw
    triangle mesh). Overhangs become solid, which does not matter for a support surface.
    """
    w, x, y, z = orient_wxyz
    R = trimesh.transformations.quaternion_matrix([w, x, y, z])[:3, :3]
    V = (mesh.vertices * np.asarray(scale, dtype=float)) @ R.T + np.asarray(translate, dtype=float)
    # only the support band: an upper envelope that included the range hood / back panel turned
    # the whole stovetop into a solid block up to the hood; walls and overhangs stay chunk hulls
    F = np.asarray(mesh.faces); F = F[(V[F][:, :, 2].min(1) <= SUPPORT_BAND_Z)]
    x0, y0 = box[0][0], box[0][1]; x1, y1 = box[1][0], box[1][1]
    H = _rasterize_upper_envelope(V, F, x0, x1, y0, y1, res)
    zfloor = float(np.nanmin(H)) if np.isfinite(H).any() else 0.0
    H = np.where(np.isnan(H), zfloor - 0.05, H)           # holes: nothing to stand on there
    fill = _fill_placement_region(H, x0, y0, res, ic_json, extra_xy=extra_xy) if ic_json is not None else None
    np.savez_compressed(out_path, H=H.astype(np.float32), x0=x0, y0=y0, res=res)
    return {"nrow": int(H.shape[0]), "ncol": int(H.shape[1]), "zmin": float(H.min()), "zmax": float(H.max()), "path": str(out_path), "placement_fill": fill}
