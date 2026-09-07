"""From 2DGS's ``fuse_post.ply`` to what the hub folder and the simulators want.

* ``mesh.glb``   — viewer + MuJoCo visual (trimesh, vertex colours kept)
* ``mesh.usdz``  — the hub layout upstream reads (usd-core writes it; no textures, like the
                   hub's own scanned backgrounds, which ship untextured)
* ``config.yaml`` — the upstream MeshConverter sidecar (collision ``meshSimplification``,
                   kinematic) so the folder validates with the upstream uploader
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
import yaml
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt


def clean(src: str | Path, *, max_faces: int = 400_000) -> trimesh.Trimesh:
    m = trimesh.load(str(src), force="mesh", process=True)
    m.remove_unreferenced_vertices()
    if len(m.faces) > max_faces:
        m = m.simplify_quadric_decimation(face_count=max_faces)
    return m


def write_glb(m: trimesh.Trimesh, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    m.export(str(dst), file_type="glb")
    return dst


def write_usdz(m: trimesh.Trimesh, dst: Path, *, kinematic: bool = True) -> Path:
    """Single UsdGeom.Mesh at /root/mesh/mesh (the hub's prim layout), Z-up, metres, with
    displayColor from vertex colours when present, and the physics APIs upstream expects."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".usdc")
    stage = Usd.Stage.CreateNew(str(tmp))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/root")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Xform.Define(stage, "/root/mesh")
    mesh = UsdGeom.Mesh.Define(stage, "/root/mesh/mesh")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(m.vertices, dtype=np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(m.faces)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(m.faces, dtype=np.int32).reshape(-1)))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mn, mx = m.bounds
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*mn.astype(float)), Gf.Vec3f(*mx.astype(float))]))
    if hasattr(m.visual, "vertex_colors") and len(m.visual.vertex_colors) == len(m.vertices):
        col = (np.asarray(m.visual.vertex_colors)[:, :3] / 255.0).astype(np.float32)
        pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec3fArray.FromNumpy(col))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mc = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    mc.CreateApproximationAttr().Set("meshSimplification" if kinematic else "convexDecomposition")
    rb = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    rb.CreateKinematicEnabledAttr().Set(bool(kinematic))
    stage.GetRootLayer().Save()
    # package as usdz (usd-core: UsdUtils.CreateNewUsdzPackage)
    from pxr import UsdUtils
    if dst.exists():
        dst.unlink()
    UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(tmp)), str(dst))
    tmp.unlink()
    return dst


def write_config(dst_dir: Path, *, kinematic: bool = True) -> Path:
    """The sidecar the hub's scanned backgrounds carry (MeshConverter output shape)."""
    cfg = {
        "asset_path": "mesh.glb", "usd_dir": ".", "usd_file_name": "mesh.usdz", "force_usd_conversion": True,
        "make_instanceable": False, "mass_props": {"mass": 1.0, "density": None},
        "rigid_props": {"rigid_body_enabled": None, "kinematic_enabled": bool(kinematic) or None, "disable_gravity": None},
        "collision_props": {"collision_enabled": True},
        "collision_approximation": "meshSimplification" if kinematic else "convexDecomposition",
        "translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0],
    }
    p = dst_dir / "config.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return p


def package_background(fuse_post_ply: Path, splat_ply: Path, asset_dir: Path) -> dict:
    """assets/<name>/{splat.ply, mesh.usdz, mesh.glb, config.yaml} for a scanned background."""
    asset_dir.mkdir(parents=True, exist_ok=True)
    m = clean(fuse_post_ply)
    write_glb(m, asset_dir / "mesh.glb")
    write_usdz(m, asset_dir / "mesh.usdz", kinematic=True)
    write_config(asset_dir, kinematic=True)
    import shutil
    shutil.copy(splat_ply, asset_dir / "splat.ply")
    return {"vertices": int(len(m.vertices)), "faces": int(len(m.faces)), "extent_m": [float(x) for x in m.extents]}
