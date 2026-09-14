"""Why no successes? Run the policy through the unmodified upstream env for a few episodes and log,
per step: rubric progress, the Isaac-equivalent finger joint, the target object's height and its XY
overlap with the container, plus the first moment the gripper opens after a lift (with a full-res
external frame). Output: runs/diag/<env>/episode_N.json + release_N.png.

usage: .venv/bin/python scripts/diagnose_release.py --env DROID-PanClean --object sponge --container pan --episodes 4 --port 8100"""
import argparse, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "polaris" / "src")); os.environ.setdefault("POLARIS_DATA_PATH", str(ROOT / "data" / "hub"))
from polaris_mujoco import hooks; hooks.install()
import gymnasium as gym, numpy as np, torch, cv2
import polaris.environments  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from polaris.utils import load_eval_initial_conditions
from polaris.policy import InferenceClient
from polaris.config import PolicyArgs
from polaris.environments.rubrics import checkers


def main(a):
    cfg = parse_env_cfg(a.env, device="cuda", num_envs=1, use_fabric=True)
    env = gym.make(a.env, cfg=cfg).unwrapped
    instr, ics = load_eval_initial_conditions(env.usd_file)
    client = InferenceClient.get_client(PolicyArgs(client="DroidJointPos", port=a.port))
    out = ROOT / "runs" / "diag" / a.env; out.mkdir(parents=True, exist_ok=True)
    objects = [o for o in a.object.split(",") if o]
    within_relaxed = {o: checkers.is_within_xy(o, a.container, percent_threshold=0.3, open_finger_threshold=10.0) for o in objects}  # geometry only
    for ep in range(a.episodes):
        obs, info = env.reset(object_positions=ics[ep]); client.reset()
        log, released, lifted, first_close = [], None, False, None
        z0s = {o: float(env.scene[o].data.root_pos_w[0, 2]) for o in objects}
        c0 = env.scene[a.container].data.root_pos_w[0].clone()
        per = {o: {"max_dz": -9.0, "steps_in": 0, "released_in": None} for o in objects}
        for t in range(env.max_episode_length):
            action, _ = client.infer(obs, instr)
            obs, rew, term, trunc, info = env.step(torch.tensor(action).reshape(1, -1), expensive=client.rerender)
            finger = float(env.scene["robot"].data.joint_pos[0, -1])
            ee = env.scene["ee_frame"].data.target_pos_w[0]
            near = min(objects, key=lambda o: float(torch.norm(ee - env.scene[o].data.root_pos_w[0])))
            z = float(env.scene[near].data.root_pos_w[0, 2]); z0 = z0s[near]; obj = env.scene[near].data.root_pos_w[0]
            rec = {"t": t, "progress": info["rubric"]["progress"], "finger": round(finger, 3), "grip_cmd": float(action[-1]), "near": near, "obj_z": round(z, 4),
                   "obj_dz": round(z - z0, 4), "ee_obj_dist": round(float(torch.norm(ee - obj)), 4), "geom_within": bool(within_relaxed[near](env))}
            log.append(rec)
            for o in objects:
                dz = float(env.scene[o].data.root_pos_w[0, 2]) - z0s[o]; per[o]["max_dz"] = max(per[o]["max_dz"], dz)
                if within_relaxed[o](env):
                    per[o]["steps_in"] += 1
                    if finger < 0.1 and per[o]["released_in"] is None and per[o]["max_dz"] > 0.04: per[o]["released_in"] = t
            if z - z0 > 0.04: lifted = True
            if first_close is None and float(action[-1]) > 0.5:
                first_close = t
                img = env.custom_render(expensive=True)
                cv2.imwrite(str(out / f"close_{ep}.png"), cv2.cvtColor(np.hstack([img["external_cam"], img["wrist_cam"]]), cv2.COLOR_RGB2BGR))
            if lifted and released is None and finger < 0.1 and log[-2]["finger"] >= 0.1:
                released = t
                img = env.custom_render(expensive=True)["external_cam"]; cv2.imwrite(str(out / f"release_{ep}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if term[0] or trunc[0]: break
        obj_xy = [round(float(v), 3) for v in env.scene[objects[0]].data.root_pos_w[0, :2]]
        summary = {"episode": ep, "final_progress": log[-1]["progress"], "first_close": first_close, "dist_at_first_close": (log[first_close]["ee_obj_dist"] if first_close is not None else None), "final_obj_xy": obj_xy,
                   "per_object": {o: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in per[o].items()} for o in objects},
                   "container_moved_m": round(float(torch.norm(env.scene[a.container].data.root_pos_w[0] - c0)), 3), "max_dz": max(r["obj_dz"] for r in log), "lifted": lifted, "release_step": released,
                   "steps_open_with_lift": sum(1 for r in log if r["finger"] < 0.1 and r["obj_dz"] > 0.04), "steps_geom_within": sum(1 for r in log if r["geom_within"]),
                   "final_obj_dz": log[-1]["obj_dz"], "final_geom_within": log[-1]["geom_within"], "final_finger": log[-1]["finger"]}
        print(json.dumps(summary)); json.dump({"summary": summary, "log": log}, open(out / f"episode_{ep}.json", "w"))
    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--env", default="DROID-PanClean"); p.add_argument("--object", default="sponge"); p.add_argument("--container", default="pan"); p.add_argument("--episodes", type=int, default=4); p.add_argument("--port", type=int, default=8100)
    main(p.parse_args())
