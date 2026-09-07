"""The Isaac-Lab-shaped objects upstream calls at run time, implemented over MuJoCo.

Stage S5.1 ships the pieces that need no physics: stage accessors, math, the no-op app
launcher, config parsing. The MuJoCo-backed ``ManagerBasedRLEnv`` arrives in S5.2+.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from pxr import Usd

from .stubs import Stub

# ---- USD stage for rubric checkers (checkers.is_within_xy reads /World/envs/env_0/scene/<name>)
_STAGE: dict = {"stage": None}


def open_scene_stage(scene_usda: str | Path) -> Usd.Stage:
    """In-memory stage whose /World/envs/env_0/scene references the environment's scene.usda,
    so upstream prim paths resolve exactly as they do inside Isaac Lab."""
    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/World/envs/env_0/scene", "Xform")
    root.GetReferences().AddReference(str(Path(scene_usda).resolve()), "/World")
    _STAGE["stage"] = stage
    return stage


def get_current_stage():
    return _STAGE["stage"]


class _Context:
    def get_stage(self):
        return _STAGE["stage"]


def get_context():
    return _Context()


# ---- math (isaaclab.utils.math) ------------------------------------------------------------
def matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
    """(..., 4) wxyz -> (..., 3, 3), same as isaaclab.utils.math.matrix_from_quat."""
    q = torch.as_tensor(q, dtype=torch.float32)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], dim=-1).reshape(q.shape[:-1] + (3, 3))


def quat_mul(a, b):
    w1, x1, y1, z1 = torch.as_tensor(a).unbind(-1); w2, x2, y2, z2 = torch.as_tensor(b).unbind(-1)
    return torch.stack([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                        w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2], dim=-1)


# Isaac Lab camera conventions: "opengl" (-Z forward, +Y up), "ros" (+Z forward, -Y up), "world" (+X forward, +Z up)
_Q = {
    ("opengl", "world"): torch.tensor([0.5, -0.5, 0.5, 0.5]),      # from isaaclab (opengl->world: rot(-90 x) then rot(90 y))
}


def convert_camera_frame_orientation_convention(q, origin="opengl", target="world"):
    q = torch.as_tensor(q, dtype=torch.float32)
    if origin == target:
        return q
    key = (origin, target)
    if key == ("opengl", "world"):
        return quat_mul(q, torch.tensor([0.5, -0.5, 0.5, 0.5]).expand_as(q))
    if key == ("world", "opengl"):
        return quat_mul(q, torch.tensor([0.5, 0.5, -0.5, -0.5]).expand_as(q))
    raise NotImplementedError(key)


# ---- app / registry ---------------------------------------------------------------------
class AppLauncher:
    """Upstream scripts/eval.py boots Isaac Sim through this; here there is nothing to boot."""

    def __init__(self, args_cli=None, **kw):
        self.app = Stub(close=lambda: None, update=lambda: None, is_running=lambda: True)


def load_cfg_from_registry(task_name: str, entry_point_key: str):
    import gymnasium as gym
    spec = gym.spec(task_name)
    ep = spec.kwargs[entry_point_key]
    return ep() if isinstance(ep, type) else ep


def parse_env_cfg(task_name: str, device: str = "cuda:0", num_envs: int | None = None, use_fabric: bool | None = None):
    """isaaclab_tasks.utils.parse_env_cfg: the registered cfg class, instantiated, plus the
    usd_file from the registration (upstream's gym.make passes it separately too)."""
    import gymnasium as gym
    cfg = load_cfg_from_registry(task_name.split(":")[-1], "env_cfg_entry_point")
    cfg.sim = getattr(cfg, "sim", Stub())
    cfg.sim.device = device
    if num_envs is not None:
        cfg.scene.num_envs = num_envs
    cfg._usd_file = gym.spec(task_name).kwargs.get("usd_file")
    return cfg


# ---- run-time facades that need the backend (filled in S5.2) -------------------------------
class ManagerBasedRLEnvCfg(Stub):
    pass


class Camera(Stub):
    """isinstance target for upstream's ``isinstance(sensor, Camera)`` checks."""


class GeometryPrim(Stub):
    def __init__(self, prim_paths_expr: str = "", **kw):
        super().__init__(prim_paths_expr=prim_paths_expr, **kw)


class ManagerBasedRLEnv:
    """Parent of upstream ``ManagerBasedRLSplatEnv``. The MuJoCo implementation lands in
    ``polaris_mujoco.backend``; until then constructing one raises so nothing pretends."""

    def __init__(self, cfg=None, *args, **kwargs):
        from polaris_mujoco.backend import env as _env  # noqa: F401  (S5.2)
        _env.attach(self, cfg, *args, **kwargs)
