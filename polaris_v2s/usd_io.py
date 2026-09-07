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
                adir = env_dir / "assets" / asset_dir
                row.has_splat = (adir / "splat.ply").exists()      # upstream only ever reads splat.ply
                row.has_mesh = any((adir / f).exists() for f in ("mesh.usdz", "mesh.usd", "mesh.usda", "mesh.glb"))
            info.prims.append(row)
    return info


def _texture_from_usdz(usdz: Path, member: str):
    from PIL import Image
    with zipfile.ZipFile(usdz) as z:
        return Image.open(io.BytesIO(z.read(member))).convert("RGB")


def _texture_for(prim: Usd.Prim, usdz: Path | None):
    """The UsdUVTexture bound to the prim's material, as a PIL image (inside a usdz or on disk)."""
    from PIL import Image
    mat = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    if not mat:
        return None
    # UsdPreviewSurface (TRELLIS usdz): UsdUVTexture.inputs:file = "0/texture.jpg" inside the usdz.
    # Omniverse MDL (princeton envs): a shader with inputs:texture = "./textures/x.png" next to mesh.usd.
    for sh in Usd.PrimRange(mat.GetPrim()):
        if not sh.IsA(UsdShade.Shader):
            continue
        for name in ("file", "texture", "diffuse_texture", "base_color_texture"):
            inp = UsdShade.Shader(sh).GetInput(name)
            if not inp:
                continue
            ap = inp.Get()
            if not ap or not getattr(ap, "path", ""):
                continue
            rp = ap.resolvedPath or ""
            if "[" in rp:                       # member of a usdz package
                z, member = rp.split("[", 1)
                return _texture_from_usdz(Path(z), member.rstrip("]"))
            if rp and Path(rp).exists():
                return Image.open(rp).convert("RGB")
            # unresolved relative path: relative to the layer that authored it
            stack = inp.GetAttr().GetPropertyStack(Usd.TimeCode.Default())
            for spec in stack:
                cand = Path(spec.layer.realPath).parent / ap.path
                if cand.exists():
                    return Image.open(cand).convert("RGB")
    return None


def _meshes_under(stage: Usd.Stage, root: Usd.Prim, to_root: np.ndarray | None, mpu: float) -> list[trimesh.Trimesh]:
    """Every UsdGeom.Mesh under ``root`` as trimesh, in ``root``'s frame (or world if to_root is None)."""
    xf = UsdGeom.XformCache()
    meshes = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh) or not UsdGeom.Imageable(prim).ComputeVisibility() == "inherited":
            continue
        m = UsdGeom.Mesh(prim)
        pts_attr = m.GetPointsAttr().Get()
        if not pts_attr:
            continue
        pts = np.asarray(pts_attr, dtype=np.float64)
        counts = np.asarray(m.GetFaceVertexCountsAttr().Get())
        idx = np.asarray(m.GetFaceVertexIndicesAttr().Get())
        mat = np.asarray(xf.GetLocalToWorldTransform(prim), dtype=np.float64)  # row-vector convention
        if to_root is not None:
            mat = mat @ to_root
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
            tex = _texture_for(prim, None)
            if uv is not None and tex is not None:
                # matte PBR: trimesh's default SimpleMaterial exports as metallic, which renders black
                mat_ = trimesh.visual.material.PBRMaterial(baseColorTexture=tex, metallicFactor=0.0, roughnessFactor=0.9)
                visual = trimesh.visual.TextureVisuals(uv=uv, material=mat_)
        tm = trimesh.Trimesh(vertices=pts, faces=faces, visual=visual, process=False)
        if visual is None:
            dc = m.GetDisplayColorAttr().Get()
            if dc:
                col = (np.asarray(dc[0]) * 255).astype(np.uint8)
                tm.visual.face_colors = np.tile(np.r_[col, 255], (len(faces), 1))
        meshes.append(tm)
    return meshes


def _join(meshes: list[trimesh.Trimesh], what: str) -> trimesh.Trimesh:
    if not meshes:
        raise ValueError(f"no UsdGeom.Mesh in {what}")
    return meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)


def usdz_to_trimesh(usdz: str | Path) -> trimesh.Trimesh:
    """A standalone asset file (usdz/usd): all its meshes, in the file's own root frame, metres."""
    usdz = Path(usdz)
    stage = Usd.Stage.Open(str(usdz))
    mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    return _join(_meshes_under(stage, stage.GetPseudoRoot(), None, mpu), str(usdz))


def prim_to_trimesh(scene_usda: str | Path, prim_name: str) -> trimesh.Trimesh:
    """The composed scene's ``/World/<prim_name>`` subtree in the prim's own frame.

    This is what a viewer or a simulator wants: the asset *as placed*, including the
    ``over`` blocks in scene.usda that move a child mesh, but without the prim's own
    translate/orient/scale (those are the pose the initial conditions overwrite).
    """
    stage = Usd.Stage.Open(str(scene_usda))
    prim = stage.GetPrimAtPath(f"/World/{prim_name}")
    if not prim:
        raise KeyError(prim_name)
    mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    xf = UsdGeom.XformCache()
    world = np.asarray(xf.GetLocalToWorldTransform(prim), dtype=np.float64)
    return _join(_meshes_under(stage, prim, np.linalg.inv(world), mpu), f"{scene_usda}:{prim_name}")


def prim_to_glb(scene_usda: str | Path, prim_name: str, dst: str | Path) -> Path:
    dst = Path(dst); dst.parent.mkdir(parents=True, exist_ok=True)
    prim_to_trimesh(scene_usda, prim_name).export(str(dst), file_type="glb")
    return dst


def usdz_to_glb(usdz: str | Path, dst: str | Path) -> Path:
    dst = Path(dst); dst.parent.mkdir(parents=True, exist_ok=True)
    usdz_to_trimesh(usdz).export(str(dst), file_type="glb")
    return dst


def usdz_to_obj(usdz: str | Path, dst: str | Path) -> Path:
    dst = Path(dst); dst.parent.mkdir(parents=True, exist_ok=True)
    usdz_to_trimesh(usdz).export(str(dst), file_type="obj")
    return dst
