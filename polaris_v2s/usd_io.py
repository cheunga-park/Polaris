"""Read PolaRiS-Hub USD files with pip ``usd-core`` (no Isaac).

* ``read_scene`` — the composed ``scene.usda``: one row per rigid/static prim with its
  transform and the asset directory it references, plus camera prims.
* ``usdz_to_glb`` — a hub ``mesh.usdz`` (TRELLIS export: one triangle mesh, one texture)
  to a GLB for the browser, and ``usdz_to_obj`` for MuJoCo later.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade


@dataclass
class PrimRow:
    name: str
    asset_dir: str | None          # directory under <env>/assets, if the prim payloads one
    translate: list[float]
    orient_wxyz: list[float]
    scale: list[float]
    kinematic: bool
    rigid_body: bool
    has_splat: bool = False
    has_mesh: bool = False


@dataclass
class CameraRow:
    name: str
    translate: list[float]
    orient_wxyz: list[float]       # USD camera: looks down -Z, +Y up
    focal_length: float
    horizontal_aperture: float
    vertical_aperture: float

    @property
    def fov_deg(self) -> tuple[float, float]:
        fx = 2 * np.degrees(np.arctan(self.horizontal_aperture / (2 * self.focal_length)))
        fy = 2 * np.degrees(np.arctan(self.vertical_aperture / (2 * self.focal_length)))
        return float(fx), float(fy)


@dataclass
class SceneInfo:
    env_dir: str
    prims: list[PrimRow] = field(default_factory=list)
    cameras: list[CameraRow] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        for c, row in zip(d["cameras"], self.cameras):
            c["fov_deg"] = row.fov_deg
        return d


def _quat(v) -> list[float]:
    if v is None:
        return [1.0, 0.0, 0.0, 0.0]
    im = v.GetImaginary()
    return [float(v.GetReal()), float(im[0]), float(im[1]), float(im[2])]


def _vec(v, default) -> list[float]:
    return [float(x) for x in v] if v is not None else list(default)


_PAYLOAD_RE = re.compile(r"assets/([^/@]+)/")


def _asset_dir_of(prim: Usd.Prim) -> str | None:
    """The ``assets/<name>/`` directory the prim's *active* payload or reference points at."""
    stack = prim.GetPrimStack()
    for spec in stack:
        for lst in (spec.payloadList, spec.referenceList):
            for item in list(lst.prependedItems) + list(lst.explicitItems) + list(lst.appendedItems):
                m = _PAYLOAD_RE.search(item.assetPath)
                if m:
                    return m.group(1)
    return None


def read_scene(scene_usda: str | Path) -> SceneInfo:
    scene_usda = Path(scene_usda)
    stage = Usd.Stage.Open(str(scene_usda))
    env_dir = scene_usda.parent
    info = SceneInfo(env_dir=str(env_dir))
    world = stage.GetPrimAtPath("/World")
    for child in world.GetChildren():
        name = child.GetName()
        if child.IsA(UsdGeom.Camera):
            cam = UsdGeom.Camera(child)
            info.cameras.append(CameraRow(
                name=name,
                translate=_vec(child.GetAttribute("xformOp:translate").Get(), (0, 0, 0)),
                orient_wxyz=_quat(child.GetAttribute("xformOp:orient").Get()),
                focal_length=float(cam.GetFocalLengthAttr().Get()),
                horizontal_aperture=float(cam.GetHorizontalApertureAttr().Get()),
                vertical_aperture=float(cam.GetVerticalApertureAttr().Get()),
            ))
            continue
        if child.IsA(UsdGeom.Xformable) and not child.IsA(UsdGeom.Camera) and name not in ("OmniverseKit", "Vars"):
            asset_dir = _asset_dir_of(child)
            if asset_dir is None and not UsdPhysics.RigidBodyAPI(child):
                continue  # lights, render settings
            kin = child.GetAttribute("physics:kinematicEnabled").Get()
            row = PrimRow(
                name=name,
                asset_dir=asset_dir,
                translate=_vec(child.GetAttribute("xformOp:translate").Get(), (0, 0, 0)),
                orient_wxyz=_quat(child.GetAttribute("xformOp:orient").Get()),
                scale=_vec(child.GetAttribute("xformOp:scale").Get(), (1, 1, 1)),
                kinematic=bool(kin) if kin is not None else False,
                rigid_body=bool(UsdPhysics.RigidBodyAPI(child)),
            )
            if asset_dir:
                row.has_splat = (env_dir / "assets" / asset_dir / "splat.ply").exists()
                row.has_mesh = (env_dir / "assets" / asset_dir / "mesh.usdz").exists()
            info.prims.append(row)
    return info


def _texture_from_usdz(usdz: Path, member: str):
    from PIL import Image
    with zipfile.ZipFile(usdz) as z:
        return Image.open(io.BytesIO(z.read(member))).convert("RGB")


def usdz_to_trimesh(usdz: str | Path) -> trimesh.Trimesh:
    """First UsdGeom.Mesh in the file, triangulated, in the file's own units -> metres."""
    usdz = Path(usdz)
    stage = Usd.Stage.Open(str(usdz))
    mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    xf = UsdGeom.XformCache()
    meshes = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        m = UsdGeom.Mesh(prim)
        pts = np.asarray(m.GetPointsAttr().Get(), dtype=np.float64)
        counts = np.asarray(m.GetFaceVertexCountsAttr().Get())
        idx = np.asarray(m.GetFaceVertexIndicesAttr().Get())
        mat = np.asarray(xf.GetLocalToWorldTransform(prim), dtype=np.float64)  # row-vector convention
        pts = (np.c_[pts, np.ones(len(pts))] @ mat)[:, :3] * mpu
        # triangulate (fan) while tracking the corner index for faceVarying primvars
        faces, corners = [], []
        c = 0
        for n in counts:
            for k in range(1, n - 1):
                faces.append((idx[c], idx[c + k], idx[c + k + 1]))
                corners.append((c, c + k, c + k + 1))
            c += n
        faces = np.asarray(faces); corners = np.asarray(corners)
        uvp = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
        visual = None
        if uvp and uvp.HasValue():
            uv = np.asarray(uvp.Get(), dtype=np.float64)
            interp = uvp.GetInterpolation()
            if interp == "faceVarying":
                # unweld: one vertex per corner so each corner carries its own uv
                v = pts[faces.reshape(-1)]
                f = np.arange(len(v)).reshape(-1, 3)
                uv = uv[corners.reshape(-1)]
                pts, faces = v, f
            elif interp == "vertex":
                pass
            else:
                uv = None
            tex = None
            for sh in stage.Traverse():
                if sh.IsA(UsdShade.Shader) and UsdShade.Shader(sh).GetIdAttr().Get() == "UsdUVTexture":
                    ap = UsdShade.Shader(sh).GetInput("file").Get()
                    if ap and "[" in ap.resolvedPath:
                        tex = _texture_from_usdz(usdz, ap.resolvedPath.split("[", 1)[1].rstrip("]"))
                        break
            if uv is not None:
                visual = trimesh.visual.TextureVisuals(uv=uv, image=tex) if tex is not None else None
        tm = trimesh.Trimesh(vertices=pts, faces=faces, visual=visual, process=False)
        if visual is None:
            dc = m.GetDisplayColorAttr().Get()
            if dc:
                col = (np.asarray(dc[0]) * 255).astype(np.uint8)
                tm.visual.face_colors = np.tile(np.r_[col, 255], (len(faces), 1))
        meshes.append(tm)
    if not meshes:
        raise ValueError(f"no UsdGeom.Mesh in {usdz}")
    return meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)


def usdz_to_glb(usdz: str | Path, dst: str | Path) -> Path:
    dst = Path(dst); dst.parent.mkdir(parents=True, exist_ok=True)
    usdz_to_trimesh(usdz).export(str(dst), file_type="glb")
    return dst


def usdz_to_obj(usdz: str | Path, dst: str | Path) -> Path:
    dst = Path(dst); dst.parent.mkdir(parents=True, exist_ok=True)
    usdz_to_trimesh(usdz).export(str(dst), file_type="obj")
    return dst
