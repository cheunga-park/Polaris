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
GRIPPER_YAW = -np.pi / 4           # Robotiq mount yaw vs Menagerie's attachment site (see build())


def _quat_mul(a, b):
    q = np.empty(4); mujoco.mju_mulQuat(q, np.asarray(a, float), np.asarray(b, float)); return q
DRIVER_CLOSED = 0.8                # 2f85 driver joint range max (menagerie)


def build(kp: float = 400.0, kv: float = 80.0, gravcomp: bool = True) -> mujoco.MjSpec:
    """A spec containing only the robot, base at the world origin (the hub/USD convention)."""
    arm = mujoco.MjSpec.from_file(str(PANDA_XML))
    grip = mujoco.MjSpec.from_file(str(GRIPPER_XML))
    # attach the gripper to the flange (link7 -> attachment site "attachment_site" in menagerie panda_nohand)
    site = next(s for s in arm.sites if s.name == "attachment_site")
    frame = site.parent.add_frame()
    frame.pos = site.pos
    # DROID mounts the Robotiq yawed by -45 deg about the flange axis relative to Menagerie's
    # attachment site: measured on the hub's scanned robot -- the finger splats sit at azimuth
    # 221/41 deg in the gripper frame where Menagerie's pads are at 270/90. Without this the
    # fingers open along the wrong diagonal and the wrist camera sits 45 deg off.
    q_yaw = np.array([np.cos(GRIPPER_YAW / 2), 0.0, 0.0, np.sin(GRIPPER_YAW / 2)])
    frame.quat = _quat_mul(np.asarray(site.quat, float), q_yaw)
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


# ---------------------------------------------------------------------------- Isaac <-> Menagerie frames
ISAAC_ROBOT_USD = ROOT / "data" / "hub" / "nvidia_droid" / "noninstanceable.usd"
GRIPPER_ISAAC_LINKS = ["base_link", "left_outer_knuckle", "left_outer_finger", "left_inner_knuckle", "left_inner_finger",
                       "right_outer_knuckle", "right_outer_finger", "right_inner_knuckle", "right_inner_finger"]
GRIPPER_MEN_BODIES = ["gripper/base", "gripper/left_driver", "gripper/left_coupler", "gripper/left_spring_link", "gripper/left_follower",
                      "gripper/right_driver", "gripper/right_coupler", "gripper/right_spring_link", "gripper/right_follower"]


def _pose_mul(p1, q1, p2, q2):
    """(p1,q1) o (p2,q2): first apply 2 then 1. quats wxyz."""
    p = np.empty(3); q = np.empty(4)
    mujoco.mju_rotVecQuat(p, np.asarray(p2, float), np.asarray(q1, float)); p += np.asarray(p1, float)
    mujoco.mju_mulQuat(q, np.asarray(q1, float), np.asarray(q2, float))
    return p, q


def _pose_inv(p, q):
    qi = np.empty(4); mujoco.mju_negQuat(qi, np.asarray(q, float))
    pi = np.empty(3); mujoco.mju_rotVecQuat(pi, -np.asarray(p, float), qi)
    return pi, qi


def _usd_world_poses(stage, paths):
    from pxr import UsdGeom
    xf = UsdGeom.XformCache(); out = {}
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim:
            continue
        M = np.array(xf.GetLocalToWorldTransform(prim)).T
        q = np.zeros(4); mujoco.mju_mat2Quat(q, np.ascontiguousarray(M[:3, :3]).reshape(9))
        out[path] = (M[:3, 3].copy(), q)
    return out


def isaac_body_of_path(prim_path: str) -> str:
    """Menagerie body that carries the Isaac prim: '.../robot/panda_link3/...' -> 'link3';
    '.../robot/Gripper/Robotiq_2F_85/left_inner_finger/...' -> 'gripper/left_follower'."""
    parts = [p for p in prim_path.split("/") if p]
    i = parts.index("robot") if "robot" in parts else (parts.index("panda") if "panda" in parts else -1)
    rest = parts[i + 1:]
    if rest and rest[0] in LINK_MAP:
        return LINK_MAP[rest[0]]
    if rest and rest[0] == "Gripper" and len(rest) >= 3 and rest[2] in GRIPPER_MAP:
        return "gripper/" + GRIPPER_MAP[rest[2]]
    raise KeyError(prim_path)


def isaac_link_offsets(isaac_init: dict[str, float] | None = None, force: bool = False) -> dict[str, dict]:
    """Constant transform from a Menagerie body frame to each Isaac prim frame the hub needs:
    ``T_isaac = T_men(t) * offset``.

    Keys: every ``SEGMENTED/*.ply`` stem (the splat of that file is expressed in the frame of the
    *mesh prim* the stem names -- upstream anchors it with a GeometryPrim on that exact path, and
    the Robotiq mesh prims are rotated and sit at the gripper base, so link frames are NOT
    enough) plus the link names themselves (``base_link`` carries the wrist camera).

    Derived numerically: the hub's ``nvidia_droid/noninstanceable.usd`` stores every prim's
    world transform at the DROID rest pose; Menagerie is put in the same joint configuration
    and compared. Cached in data/cache/robot_link_offsets.json (v2).
    """
    import json
    cache = ROOT / "data" / "cache" / "robot_link_offsets.json"
    if cache.exists() and not force:
        d = json.loads(cache.read_text())
        if d.get("_version") == 3:
            return d
    from pxr import Usd
    if isaac_init is None:
        import sys
        sys.path.insert(0, str(ROOT / "third_party" / "polaris" / "src"))
        from polaris_mujoco import hooks; hooks.install()
        from polaris.environments.robot_cfg import NVIDIA_DROID
        isaac_init = NVIDIA_DROID.init_state.joint_pos
    st = Usd.Stage.Open(str(ISAAC_ROBOT_USD))
    stems = sorted(p.stem for p in (ROOT / "data" / "hub" / "nvidia_droid" / "SEGMENTED").glob("*.ply"))
    paths = {stem: "/panda/" + stem.replace("-", "/") for stem in stems}
    # link prims too (wrist camera hangs off base_link; FK page keys by link)
    for link in list(LINK_MAP) :
        paths[link] = f"/panda/{link}"
    for link in GRIPPER_ISAAC_LINKS:
        paths[link] = f"/panda/Gripper/Robotiq_2F_85/{link}"
    usd = _usd_world_poses(st, paths.values())
    spec = build(); m = spec.compile(); d = mujoco.MjData(m)
    d.qpos[:] = default_qpos(m, isaac_init); mujoco.mj_forward(m, d)
    out = {"_version": 3}
    for key, path in paths.items():
        if path not in usd:
            continue
        try:
            men = isaac_body_of_path(path)
        except KeyError:
            continue
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, men)
        pm, qm = d.xpos[bid], d.xquat[bid]
        pi, qi = usd[path]
        pinv, qinv = _pose_inv(pm, qm)
        po, qo = _pose_mul(pinv, qinv, pi, qi)
        out[key] = {"body": men, "path": path, "pos": po.round(6).tolist(), "quat": qo.round(6).tolist(),
                    "world_gap_mm": float(np.linalg.norm(pi - pm) * 1000)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, indent=1))
    return out


def isaac_link_pose(data: mujoco.MjData, model: mujoco.MjModel, isaac_link: str, offsets: dict) -> tuple[np.ndarray, np.ndarray]:
    """World pose of an Isaac link frame from the current MuJoCo state."""
    o = offsets[isaac_link]
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, o["body"])
    return _pose_mul(data.xpos[bid], data.xquat[bid], o["pos"], o["quat"])


def isaac_link_of_path(prim_path: str) -> str:
    """Offset key for an upstream GeometryPrim path: the ply stem when the path names a mesh prim
    ('/World/envs/env_0/robot/panda_link3/geometry/panda_link3' -> 'panda_link3-geometry-panda_link3'),
    else the link name."""
    parts = [p for p in prim_path.split("/") if p]
    i = parts.index("robot") if "robot" in parts else -1
    rest = parts[i + 1:]
    if not rest:
        raise KeyError(prim_path)
    return "-".join(rest) if len(rest) > 1 else rest[0]
