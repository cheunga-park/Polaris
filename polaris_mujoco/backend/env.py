"""The MuJoCo implementation behind the ``isaaclab.envs.ManagerBasedRLEnv`` facade.

Upstream's ``ManagerBasedRLSplatEnv`` (unmodified) inherits from the facade class and runs
its own ``reset``/``step``/``custom_render``/``transform_sim_to_splat``/``render_splat`` on
top of what this module attaches to the instance. The interface it reads is listed in
PLAN.md §8.3 and enforced by tests/test_facade_contract.py.

Control loop (from upstream EnvCfg, read through the stubs at run time):
  sim.dt = 1/120, decimation = 8  ->  one env.step = 8 physics steps = 1/15 s, 450 steps = 30 s.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from . import build as build_mod
from . import robot as robot_mod
from . import stage as stage_mod

ROOT = Path(__file__).resolve().parents[2]


def _t(x):
    return torch.as_tensor(np.asarray(x, dtype=np.float32))


# ----------------------------------------------------------------------------- scene entities
class _RigidData:
    """``scene[name].data`` of an Isaac RigidObject: world pose tensors, shape (1, ...)."""

    def __init__(self, env: "MujocoEnv", body: str):
        self._env, self._body = env, body
        bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, body)
        self._bid = bid
        p, q = env.data.xpos[bid].copy(), env.data.xquat[bid].copy()
        self.default_root_state = _t(np.r_[p, q, np.zeros(6)])[None]

    @property
    def root_pos_w(self):
        return _t(self._env.data.xpos[self._bid])[None]

    @property
    def root_quat_w(self):
        return _t(self._env.data.xquat[self._bid])[None]

    @property
    def root_state_w(self):
        d = self._env.data
        return _t(np.r_[d.xpos[self._bid], d.xquat[self._bid], d.cvel[self._bid][3:], d.cvel[self._bid][:3]])[None]


class RigidObject:
    def __init__(self, env: "MujocoEnv", body: str):
        self._env, self._body = env, body
        self.data = _RigidData(env, body)

    def write_root_pose_to_sim(self, pose):
        """pose: (1, 7) = (x, y, z, qw, qx, qy, qz) -- the hub initial_conditions.json layout."""
        pose = np.asarray(torch.as_tensor(pose).reshape(-1), dtype=np.float64)
        m, d = self._env.model, self._env.data
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, self._body)
        ja = m.body_jntadr[bid]
        if ja < 0 or m.jnt_type[ja] != mujoco.mjtJoint.mjJNT_FREE:
            return  # static prim
        adr = m.jnt_qposadr[ja]
        d.qpos[adr:adr + 3] = pose[:3]
        d.qpos[adr + 3:adr + 7] = pose[3:7]
        va = m.jnt_dofadr[ja]
        d.qvel[va:va + 6] = 0.0
        mujoco.mj_forward(m, d)


class _RobotData:
    joint_names: list[str] = []      # set per instance; declared here so the contract test sees it

    def __init__(self, env: "MujocoEnv"):
        self._env = env
        m = env.model
        self.joint_names = list(robot_mod.ISAAC_ARM_JOINTS) + ["finger_joint"]
        self._arm_adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in robot_mod.ARM_JOINTS]
        self._drv_adr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "gripper/right_driver_joint")]

    @property
    def joint_pos(self):
        q = self._env.data.qpos
        arm = [q[a] for a in self._arm_adr]
        finger = q[self._drv_adr] / robot_mod.DRIVER_CLOSED * robot_mod.FINGER_CLOSED   # 2f85 driver -> Isaac finger_joint
        return _t(arm + [finger])[None]


class Robot:
    def __init__(self, env):
        self.data = _RobotData(env)


class _EEData:
    def __init__(self, env, bid):
        self._env, self._bid = env, bid

    @property
    def target_pos_w(self):
        return _t(self._env.data.xpos[self._bid])[None]


class EEFrame:
    """``scene["ee_frame"].data.target_pos_w``: the Robotiq base_link origin (upstream FrameTransformer target)."""

    def __init__(self, env):
        self.data = _EEData(env, mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "gripper/base"))


# ----------------------------------------------------------------------------- cameras
class _CamData:
    def __init__(self, cam: "CameraSensor"):
        self._cam = cam
        self.output = {}

    @property
    def pos_w(self):
        return _t(self._cam.pos_w)[None]

    @property
    def quat_w_world(self):
        return _t(self._cam.quat_w_world)[None]


def _isaac_camera_class():
    from polaris_mujoco.hooks import facade
    return facade.Camera


class CameraSensor(_isaac_camera_class()):
    """An Isaac ``Camera`` look-alike backed by a MuJoCo camera.

    Intrinsics are the upstream cfg's (focal length + apertures, USD units); the MuJoCo camera
    gets the same vertical FOV and the upstream code reads the apertures back through
    ``_sensor_prims[0]``. ``data.output`` carries ``rgb`` (H, W, 3) and
    ``semantic_segmentation`` (H, W, 1) where >=2 means "rendered by the simulator" (robot +
    objects that have no splat), matching upstream's ``mask >= 2`` rule.
    """

    def __init__(self, env: "MujocoEnv", name: str, *, width: int, height: int, focal_length: float,
                 horizontal_aperture: float, vertical_aperture: float, mj_cam: str):
        self._env, self.name = env, name
        self.image_shape = (height, width)
        self._mj_cam = mj_cam
        self._sensor_prims = [SimpleNamespace(
            GetHorizontalApertureAttr=lambda: SimpleNamespace(Get=lambda: horizontal_aperture),
            GetVerticalApertureAttr=lambda: SimpleNamespace(Get=lambda: vertical_aperture),
            GetFocalLengthAttr=lambda: SimpleNamespace(Get=lambda: focal_length))]
        self.fovy_deg = math.degrees(2 * math.atan(vertical_aperture / (2 * focal_length)))
        self.data = _CamData(self)

    @property
    def pos_w(self):
        cid = mujoco.mj_name2id(self._env.model, mujoco.mjtObj.mjOBJ_CAMERA, self._mj_cam)
        return self._env.data.cam_xpos[cid].copy()

    @property
    def quat_w_world(self):
        """Isaac 'world' convention: +X forward, +Z up. MuJoCo camera frame: -Z forward, +Y up (opengl)."""
        cid = mujoco.mj_name2id(self._env.model, mujoco.mjtObj.mjOBJ_CAMERA, self._mj_cam)
        R_gl = self._env.data.cam_xmat[cid].reshape(3, 3)
        # world-convention axes: forward = -Z_gl, left = -X_gl, up = +Y_gl
        R_w = np.stack([-R_gl[:, 2], -R_gl[:, 0], R_gl[:, 1]], axis=1)
        return _mat_to_quat(R_w)

    def render(self):
        rgb, seg = self._env.render_camera(self._mj_cam, *self.image_shape)
        # Isaac shapes: rgb (num_envs, H, W, 3) uint8, semantic_segmentation (num_envs, H, W, 1)
        self.data.output = {"rgb": torch.as_tensor(rgb)[None], "semantic_segmentation": torch.as_tensor(seg)[None, ..., None]}


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, dtype=np.float64).reshape(9))
    return q


# ----------------------------------------------------------------------------- the env
class Scene(dict):
    """``env.scene``: mapping name -> entity, plus the dicts upstream iterates."""

    def __init__(self):
        super().__init__()
        self.rigid_objects: dict = {}
        self.sensors: dict = {}
        self.num_envs = 1
        self.env_origins = torch.zeros(1, 3)

    def update(self, dt):
        for s in self.sensors.values():
            if isinstance(s, CameraSensor):
                s.render()


class MujocoBase:
    """Everything the upstream class expects from its parent. Constructed by the facade's
    ``ManagerBasedRLEnv.__init__`` with the upstream instance as ``target``; every attribute
    upstream reads (``scene``, ``sim``, ``observation_manager``, ``device``, ``usd_file``,
    ``max_episode_length``, ``cfg``) is set on ``target``."""

    def __init__(self, target, cfg, usd_file: str | None = None, **kw):
        self.target = target
        self.cfg = cfg
        usd_file = usd_file or getattr(cfg, "_usd_file", None) or getattr(target, "usd_file", None)
        self.usd_file = str(usd_file)
        self.device = "cuda"                     # the upstream splat renderer runs on it; our tensors move on demand
        self.decimation = int(cfg.decimation or 8)
        self.physics_dt = float(cfg.sim.dt or 1 / 120)
        self.episode_length_s = float(cfg.episode_length_s or 30)
        self.step_dt = self.physics_dt * self.decimation
        self.max_episode_length = int(round(self.episode_length_s / self.step_dt))
        self._episode_step = 0

        from polaris_v2s import usd_io
        self._scene_info = usd_io.read_scene(self.usd_file)
        self._asset_dirs = {p.name: p.asset_dir for p in self._scene_info.prims if p.asset_dir}

        bodies = stage_mod.prepare(self.usd_file)
        self.spec, _ = build_mod.build(bodies, timestep=self.physics_dt)
        self._link_offsets = robot_mod.isaac_link_offsets()
        self._cam_cfgs, self._cam_meta = {}, {}
        self._add_cameras(cfg)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self._renderer = None
        self._geom_class = None

        from polaris.environments.robot_cfg import NVIDIA_DROID  # upstream, through the stubs
        self._q_init = robot_mod.default_qpos(self.model, NVIDIA_DROID.init_state.joint_pos)
        self.data.qpos[:] = self._q_init
        self.data.ctrl[:7] = self._q_init[:7]
        mujoco.mj_forward(self.model, self.data)

        self.scene = Scene()
        for b in bodies:
            ro = RigidObject(self, b.name)
            self.scene[b.name] = ro
            self.scene.rigid_objects[b.name] = ro
        self.scene["robot"] = Robot(self)
        self.scene["ee_frame"] = EEFrame(self)
        self._cameras = {}
        for name, c in self._cam_cfgs.items():
            fl, ha, va = self._cam_meta[name]
            cam = CameraSensor(self, name, width=int(c.width), height=int(c.height),
                               focal_length=fl, horizontal_aperture=ha, vertical_aperture=va, mj_cam=name)
            self._cameras[name] = cam
            self.scene[name] = cam
            self.scene.sensors[name] = cam
        self.sim = SimpleNamespace(render=lambda: self.scene.update(0), device=self.device)
        self.observation_manager = SimpleNamespace(compute=self._compute_obs)

        from polaris_mujoco.hooks import facade
        facade.open_scene_stage(self.usd_file)

        for k in ("cfg", "usd_file", "device", "max_episode_length", "scene", "sim", "observation_manager", "model", "data"):
            setattr(target, k, getattr(self, k))

    # ---- cameras from the upstream cfg (wrist_cam on the gripper; external_cam from scene.usda or the cfg default)
    def _add_cameras(self, cfg):
        scene_cfg = cfg.scene
        for name in ("wrist_cam", "external_cam"):
            c = getattr(scene_cfg, name, None)
            if c is None:
                continue
            self._cam_cfgs[name] = c
            spawn = getattr(c, "spawn", None)
            if spawn is not None and getattr(spawn, "focal_length", None) is not None:
                meta = (spawn.focal_length, spawn.horizontal_aperture, spawn.vertical_aperture)
            else:   # a camera prim in scene.usda: dynamic_setup keeps no intrinsics (spawn=None); read the file
                row = next((r for r in self._scene_info.cameras if r.name == name), None)
                meta = (row.focal_length, row.horizontal_aperture, row.vertical_aperture) if row else (1.0476, 2.5452, 1.4721)
            self._cam_meta[name] = meta
            pos = np.asarray(c.offset.pos, dtype=float)
            quat = np.asarray(c.offset.rot, dtype=float)       # Isaac "opengl" convention == MuJoCo camera frame
            if name == "wrist_cam":
                # upstream mounts it on the Isaac Robotiq base_link; compose with base_link's offset from Menagerie's base
                o = self._link_offsets["base_link"]
                parent = next(b for b in self.spec.bodies if b.name == o["body"])
                pos, quat = robot_mod._pose_mul(o["pos"], o["quat"], pos, quat)
            else:
                parent = self.spec.worldbody
            cam = parent.add_camera()
            cam.name = name
            cam.pos = pos
            cam.quat = quat
            cam.fovy = math.degrees(2 * math.atan(meta[2] / (2 * meta[0])))

    # ---- poses
    def body_pose(self, body: str):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        if bid < 0:
            raise KeyError(body)
        return self.data.xpos[bid].copy(), self.data.xquat[bid].copy()

    def isaac_link_pose(self, isaac_link: str):
        """World pose of an Isaac USD link frame (what the hub link splats are expressed in)."""
        return robot_mod.isaac_link_pose(self.data, self.model, isaac_link, self._link_offsets)

    # ---- rendering
    def render_camera(self, mj_cam: str, height: int, width: int):
        if self._renderer is None or self._renderer.height != height or self._renderer.width != width:
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, height, width)
            self._opt = mujoco.MjvOption()
            self._opt.geomgroup[:] = 0
            self._opt.geomgroup[0] = 1
            self._opt.geomgroup[build_mod.ROBOT_GEOM_GROUP] = 1
            self._opt.geomgroup[build_mod.OBJECT_GEOM_GROUP] = 1
        r = self._renderer
        r.disable_segmentation_rendering()
        r.update_scene(self.data, mj_cam, self._opt)
        rgb = r.render().copy()
        r.enable_segmentation_rendering()
        r.update_scene(self.data, mj_cam, self._opt)
        seg = r.render()[..., 0].copy()          # geom ids, -1 = background
        r.disable_segmentation_rendering()
        return rgb, self._seg_to_semantic(seg)

    def _seg_to_semantic(self, seg_geom_ids: np.ndarray) -> np.ndarray:
        """0 background, 1 = has a splat (drawn by the splat renderer), >=2 = simulator-rendered
        (robot, objects without a splat) -- upstream composites with ``mask >= 2``."""
        m = self.model
        if self._geom_class is None:
            cls = np.zeros(m.ngeom, dtype=np.int32)
            env_dir = Path(self.usd_file).parent
            # upstream dynamic_setup(robot_splat=True) is the default: the robot is drawn by the splat
            # renderer (its link splats follow the joints) and is NOT in the simulator mask; only with
            # robot_splat=False does it tag the robot "raytraced" (semantic id >= 2).
            robot_spawn = getattr(getattr(self.cfg.scene, "robot", None), "spawn", None)
            robot_raytraced = "semantic_tags" in vars(robot_spawn) if robot_spawn is not None else False
            for g in range(m.ngeom):
                name = m.body(m.geom_bodyid[g]).name
                if name.startswith(("link", "gripper", "attachment")) or name == "world":
                    cls[g] = 3 if robot_raytraced else 1
                else:
                    adir = self._asset_dirs.get(name)
                    cls[g] = 1 if adir and (env_dir / "assets" / adir / "splat.ply").exists() else 2
            self._geom_class = cls
        out = np.zeros(seg_geom_ids.shape, dtype=np.int32)
        valid = seg_geom_ids >= 0
        out[valid] = self._geom_class[seg_geom_ids[valid]]
        return out

    # ---- the ManagerBasedRLEnv surface
    def _compute_obs(self):
        from polaris.environments import droid_cfg  # upstream obs terms
        return {"policy": {"arm_joint_pos": droid_cfg.arm_joint_pos(self.target), "gripper_pos": droid_cfg.gripper_pos(self.target)}}

    def base_reset(self):
        d = self.data
        d.qpos[:] = 0.0; d.qvel[:] = 0.0; d.ctrl[:] = 0.0
        d.qpos[:] = self._q_init
        d.ctrl[:7] = self._q_init[:7]
        self._episode_step = 0
        mujoco.mj_forward(self.model, d)
        for b in self.scene.rigid_objects.values():
            b.data.default_root_state = _t(np.r_[d.xpos[b.data._bid], d.xquat[b.data._bid], np.zeros(6)])[None]
        self.scene.update(0)
        return self._compute_obs(), {}

    def base_step(self, action):
        a = np.asarray(torch.as_tensor(action).reshape(-1), dtype=np.float64)
        self.data.ctrl[:7] = a[:7]
        self.data.ctrl[7] = 255.0 if a[7] > 0.5 else 0.0     # upstream BinaryJointPositionZeroToOneAction: >0.5 = close
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)
        self._episode_step += 1
        self.scene.update(0)
        trunc = torch.tensor([self._episode_step >= self.max_episode_length])
        return self._compute_obs(), torch.zeros(1), torch.tensor([False]), trunc, {}

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
