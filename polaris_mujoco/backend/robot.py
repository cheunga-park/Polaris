"""The DROID arm for MuJoCo: Menagerie's Franka Panda (no hand) + Robotiq 2F-85, assembled
with MjSpec, driven the way upstream's Isaac config drives it (robot_cfg.py, read at run time
through the recording stubs — the numbers are not copied here):

* arm joints: implicit PD, stiffness 400, damping 80  ->  <position kp=400 kv=80>
* gripper: one binary command; Isaac's ``finger_joint`` (0 open .. pi/4 closed) maps to the
  2F-85 driver joints (0 .. 0.8 rad) through the menagerie tendon actuator (ctrl 0..255)
* ``disable_gravity=True`` on the articulation -> gravcomp=1 on every robot body
* ``enabled_self_collisions=False`` -> robot geoms in a contype/conaffinity class that
  collides with the world but not with itself
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MENAGERIE = ROOT / "data" / "robots" / "mujoco_menagerie"
PANDA_XML = MENAGERIE / "franka_emika_panda" / "panda_nohand.xml"
GRIPPER_XML = MENAGERIE / "robotiq_2f85" / "2f85.xml"

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]          # menagerie names
ISAAC_ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
# Isaac link name -> menagerie body name (splat SEGMENTED/*.ply stems start with the Isaac link)
LINK_MAP = {f"panda_link{i}": f"link{i}" for i in range(8)}
LINK_MAP["panda_link0"] = "link0"
# Isaac USD Robotiq link (robotiq_description URDF) -> Menagerie 2f85 body
GRIPPER_MAP = {"base_link": "base", "left_outer_knuckle": "left_driver", "left_outer_finger": "left_coupler",
               "left_inner_knuckle": "left_spring_link", "left_inner_finger": "left_follower",
               "right_outer_knuckle": "right_driver", "right_outer_finger": "right_coupler",
               "right_inner_knuckle": "right_spring_link", "right_inner_finger": "right_follower"}
FINGER_CLOSED = np.pi / 4          # Isaac finger_joint close command (droid_cfg ActionCfg)
DRIVER_CLOSED = 0.8                # 2f85 driver joint range max (menagerie)


def build(kp: float = 400.0, kv: float = 80.0, gravcomp: bool = True) -> mujoco.MjSpec:
    """A spec containing only the robot, base at the world origin (the hub/USD convention)."""
    arm = mujoco.MjSpec.from_file(str(PANDA_XML))
    grip = mujoco.MjSpec.from_file(str(GRIPPER_XML))
    # attach the gripper to the flange (link7 -> attachment site "attachment_site" in menagerie panda_nohand)
    site = next(s for s in arm.sites if s.name == "attachment_site")
    frame = site.parent.add_frame()
    frame.pos = site.pos; frame.quat = site.quat
    frame.attach_body(grip.worldbody.first_body(), "gripper/", "")
    # arm actuators: menagerie ships <general biastype=affine> position actuators (kp 4500/3500/2000,
    # kv = kp/10); re-gain them to the Isaac implicit-PD values so the same joint targets behave alike
    for act in arm.actuators:
        if act.target in ARM_JOINTS:
            act.gainprm[0] = kp; act.biasprm[0] = 0.0; act.biasprm[1] = -kp; act.biasprm[2] = -kv
    arm.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC   # the 2f85 model is tuned for it (attach keeps the parent's)
    if gravcomp:
        for b in arm.bodies:
            b.gravcomp = 1.0
    return arm


def isaac_path_to_body(prim_path: str) -> str:
    """'/World/envs/env_0/robot/panda_link3/geometry/panda_link3' -> 'link3';
    '/World/envs/env_0/robot/Gripper/Robotiq_2F_85/left_inner_finger/...' -> 'gripper/left_follower'."""
    parts = [p for p in prim_path.split("/") if p]
    try:
        i = parts.index("robot")
    except ValueError:
        i = -1
    rest = parts[i + 1:]
    if rest and rest[0] in LINK_MAP:
        return LINK_MAP[rest[0]]
    if rest and rest[0] == "Gripper" and len(rest) >= 3 and rest[2] in GRIPPER_MAP:
        return "gripper/" + GRIPPER_MAP[rest[2]]
    raise KeyError(f"no MuJoCo body for {prim_path}")


def default_qpos(spec_or_model, isaac_init: dict[str, float]) -> np.ndarray:
    """qpos vector with the upstream initial joint positions (keys are Isaac joint names)."""
    m = spec_or_model if isinstance(spec_or_model, mujoco.MjModel) else spec_or_model.compile()
    q = np.zeros(m.nq)
    for isaac, men in zip(ISAAC_ARM_JOINTS, ARM_JOINTS):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, men)
        if jid >= 0 and isaac in isaac_init:
            q[m.jnt_qposadr[jid]] = isaac_init[isaac]
    return q
