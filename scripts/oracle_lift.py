"""S5.4 gate: a scripted oracle drives the unmodified upstream env through reach -> grasp -> lift
on one object and the upstream rubric must register progress. Also reports joint tracking.

usage: .venv/bin/python scripts/oracle_lift.py --env DROID-FoodBussing --object ice_cream_ [--ic 0] [--video]
"""
import argparse, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "polaris" / "src")); os.environ.setdefault("POLARIS_DATA_PATH", str(ROOT / "data" / "hub"))
from polaris_mujoco import hooks; hooks.install()
import gymnasium as gym, numpy as np, torch, mujoco
import polaris.environments  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from polaris.utils import load_eval_initial_conditions


def ik_step(m, d, site_body, target, q_idx, iters=30, damping=1e-3, step=0.6):
    """Damped least squares on the 7 arm joints to bring a body origin to `target` (position only)."""
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, site_body)
    for _ in range(iters):
        mujoco.mj_forward(m, d)
        err = target - d.xpos[bid]
        if np.linalg.norm(err) < 2e-3:
            break
        jacp = np.zeros((3, m.nv)); mujoco.mj_jacBody(m, d, jacp, None, bid)
        J = jacp[:, :7]
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(3), err)
        d.qpos[:7] += step * dq
        d.qpos[:7] = np.clip(d.qpos[:7], m.jnt_range[:7, 0], m.jnt_range[:7, 1])
    return d.qpos[:7].copy()


def main(a):
    cfg = parse_env_cfg(a.env, device="cuda", num_envs=1, use_fabric=True)
    env = gym.make(a.env, cfg=cfg).unwrapped
    instr, ics = load_eval_initial_conditions(env.usd_file)
    obs, info = env.reset(object_positions=ics[a.ic])
    mj = env._mj; m, d = mj.model, mj.data
    obj_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, a.object)
    frames, log = [], []

    def go(target_q, grip, n, label):
        nonlocal obs, info
        act = torch.tensor(list(target_q) + [grip], dtype=torch.float32).reshape(1, -1)
        for _ in range(n):
            obs, rew, term, trunc, info = env.step(act, expensive=a.video)
            if a.video:
                frames.append(np.concatenate([obs["splat"]["external_cam"], obs["splat"]["wrist_cam"]], axis=1))
        q = obs["policy"]["arm_joint_pos"][0].numpy()
        log.append((label, float(np.abs(q - np.asarray(target_q)).max()), info["rubric"]["progress"], d.xpos[obj_bid].copy()))
        print(f"{label:12s} max|q-q*|={log[-1][1]:.3f} rad  progress={log[-1][2]:.2f}  obj_z={d.xpos[obj_bid][2]:.3f}")

    # let the objects settle from their (hovering) initial conditions
    q0 = d.qpos[:7].copy(); go(q0, 0.0, 15, "settle")
    p_obj = d.xpos[obj_bid].copy()
    # gripper 'base' origin sits ~0.16 m above the pads; the TCP is ~0.145 m along the gripper's +z (down here)
    tcp_offset = 0.15
    # plan joint targets kinematically on a scratch copy, then track them with the position controllers
    scratch = mujoco.MjData(m); scratch.qpos[:] = d.qpos; scratch.qvel[:] = 0
    above = p_obj + [0, 0, 0.18 + tcp_offset]
    q_above = ik_step(m, scratch, "gripper/base", above, range(7))
    grasp = p_obj + [0, 0, 0.02 + tcp_offset]
    q_grasp = ik_step(m, scratch, "gripper/base", grasp, range(7))
    go(q_above, 0.0, 45, "reach-above")
    go(q_grasp, 0.0, 45, "descend")
    go(q_grasp, 1.0, 20, "close")
    go(q_above, 1.0, 45, "lift")
    z_gain = d.xpos[obj_bid][2] - p_obj[2]
    print(f"\nobject lifted by {z_gain*1000:.0f} mm | rubric progress {info['rubric']['progress']:.2f} | success {info['rubric']['success']}")
    if a.video:
        import mediapy; out = ROOT / "runs" / "oracle" / f"{a.env}_{a.object}.mp4"; out.parent.mkdir(parents=True, exist_ok=True)
        mediapy.write_video(str(out), frames, fps=15); print("video", out)
    env.close()
    return 0 if z_gain > 0.03 else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--env", default="DROID-FoodBussing"); p.add_argument("--object", default="ice_cream_"); p.add_argument("--ic", type=int, default=0); p.add_argument("--video", action="store_true")
    raise SystemExit(main(p.parse_args()))
