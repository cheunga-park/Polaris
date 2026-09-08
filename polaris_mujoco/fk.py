"""Forward kinematics service for the web robot page: joint values in, Isaac-link world poses
out (the frames the hub link splats live in). Uses the same Menagerie assembly and the same
derived link offsets as the evaluation backend, so what the page shows is what the policy sees."""

from __future__ import annotations

import threading

import mujoco
import numpy as np

from polaris_mujoco.backend import robot as robot_mod

_lock = threading.Lock()
_state: dict = {}


def _model():
    if "model" not in _state:
        spec = robot_mod.build()
        _state["model"] = spec.compile()
        _state["data"] = mujoco.MjData(_state["model"])
        _state["offsets"] = robot_mod.isaac_link_offsets()
    return _state["model"], _state["data"], _state["offsets"]


def link_poses(arm_q: list[float], gripper: float = 0.0) -> dict[str, dict]:
    """arm_q: 7 joint positions (rad); gripper: 0 open .. 1 closed (Isaac finger_joint / (pi/4))."""
    with _lock:
        m, d, off = _model()
        d.qpos[:] = 0.0
        for j, v in zip(robot_mod.ARM_JOINTS, arm_q):
            d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]] = float(v)
        drv = float(np.clip(gripper, 0, 1)) * robot_mod.DRIVER_CLOSED
        for side in ("right", "left"):
            for jn, k in (("driver_joint", 1.0), ("coupler_joint", -1.0), ("spring_link_joint", 1.0), ("follower_joint", -1.0)):
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"gripper/{side}_{jn}")
                if jid >= 0:
                    d.qpos[m.jnt_qposadr[jid]] = drv * k
        mujoco.mj_forward(m, d)
        out = {}
        for key in off:
            if key.startswith("_"):
                continue
            p, q = robot_mod.isaac_link_pose(d, m, key, off)
            out[key] = {"pos": [float(x) for x in p], "quat_wxyz": [float(x) for x in q]}
        return out
