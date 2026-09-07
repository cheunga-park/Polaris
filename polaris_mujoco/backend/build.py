"""Assemble one MuJoCo model: robot at the origin + every scene prim from ``stage.prepare``.

Conventions carried over from upstream (droid_cfg.py / robot_cfg.py, read via the stubs):
* world = the hub's scene.usda frame, Z up, robot base at the origin
* rigid objects: freejoint, convex pieces from CoACD, density 1000 kg/m3 unless a mass is given
* static prims (``physics:kinematicEnabled``): no joint, same convex pieces
* sim dt 1/120 (upstream ``sim.dt``) — the control loop decimates to 15 Hz (``decimation``)
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from . import robot as robot_mod
from .stage import BodySpec

ROBOT_GEOM_GROUP = 1     # rendering group for robot geoms (segmentation reads geom -> body)
OBJECT_GEOM_GROUP = 2
COLLISION_GROUP = 3      # convex pieces: rendered off


def build(bodies: list[BodySpec], *, timestep: float = 1 / 120, kp: float = 400.0, kv: float = 80.0,
          gravcomp: bool = True, offwidth: int = 1280, offheight: int = 720) -> tuple[mujoco.MjSpec, dict]:
    spec = robot_mod.build(kp=kp, kv=kv, gravcomp=gravcomp)
    spec.option.timestep = timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.visual.global_.offwidth = offwidth
    spec.visual.global_.offheight = offheight
    spec.visual.map.znear = 0.005
    # upstream lights the scene with a dome light (shadowless, even); approximate with a strong
    # ambient headlight and no shadows so the composited foreground is not lit by a single point
    spec.visual.headlight.ambient[:] = 0.6
    spec.visual.headlight.diffuse[:] = 0.5
    spec.visual.headlight.specular[:] = 0.05
    spec.visual.quality.shadowsize = 0
    # robot geoms: collide with the world (objects, scene) but not with each other
    for g in spec.geoms:
        if g.contype or g.conaffinity:
            g.contype = 1; g.conaffinity = 2
        g.group = ROBOT_GEOM_GROUP if g.group < 3 else g.group
    for b in spec.bodies:
        b.gravcomp = 1.0 if gravcomp else 0.0

    names = {}
    for bs in bodies:
        body = spec.worldbody.add_body()
        body.name = bs.name
        body.pos = np.asarray(bs.translate, dtype=float)
        body.quat = np.asarray(bs.orient_wxyz, dtype=float)
        if not bs.kinematic:
            body.add_freejoint()
        scale = np.asarray(bs.scale, dtype=float)
        if bs.visual_obj is not None:
            m = spec.add_mesh(); m.name = f"{bs.name}_vis"; m.file = str(bs.visual_obj); m.scale = scale
            g = body.add_geom(); g.name = f"{bs.name}_vis"; g.type = mujoco.mjtGeom.mjGEOM_MESH; g.meshname = m.name
            g.contype = 0; g.conaffinity = 0; g.group = OBJECT_GEOM_GROUP
            # baked texture from the usdz/TRELLIS export (trimesh writes material_0.png + vt in the OBJ)
            tex_png = bs.visual_obj.parent / "material_0.png"
            if tex_png.exists():
                t = spec.add_texture(); t.name = f"{bs.name}_tex"; t.file = str(tex_png); t.type = mujoco.mjtTexture.mjTEXTURE_2D
                mat = spec.add_material(); mat.name = f"{bs.name}_mat"; mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = t.name
                mat.specular = 0.05; mat.shininess = 0.05
                g.material = mat.name
        for i, obj in enumerate(bs.collision_objs):
            m = spec.add_mesh(); m.name = f"{bs.name}_col{i}"; m.file = str(obj); m.scale = scale
            g = body.add_geom(); g.name = f"{bs.name}_col{i}"; g.type = mujoco.mjtGeom.mjGEOM_MESH; g.meshname = m.name
            g.contype = 2; g.conaffinity = 3; g.group = COLLISION_GROUP
            g.density = 1000.0
            # sliding 1.0 like the Isaac default material; small torsional/rolling terms so a
            # cylinder dropped from its initial condition stops instead of rolling forever
            # (MuJoCo's default has zero rolling resistance; PhysX has none either, but its
            # faceted contacts damp it) -- a recorded delta, see PLAN §8.5
            g.condim = 6
            g.friction[0] = 1.0; g.friction[1] = 0.005; g.friction[2] = 0.001
        if bs.kinematic:
            for g in body.geoms:
                g.density = 0.0
        names[bs.name] = body.name
    return spec, names


def compile_(spec: mujoco.MjSpec):
    return spec.compile()
