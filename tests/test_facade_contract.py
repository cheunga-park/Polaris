"""PLAN §8.3: every attribute the unmodified upstream env/rubric code reads from Isaac Lab must
exist on our facades. The list is harvested from the upstream sources so an upstream bump that
starts reading a new attribute fails here first."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / "third_party" / "polaris" / "src" / "polaris"


def _harvest():
    env_src = "\n".join(p.read_text() for p in [UP / "environments/manager_based_rl_splat_environment.py", UP / "environments/rubrics/checkers.py"])
    cfg_src = (UP / "environments/droid_cfg.py").read_text()
    data_attrs = set(re.findall(r"\.data\.([a-zA-Z_]+)", env_src + cfg_src))
    scene_attrs = set(re.findall(r"self\.scene\.([a-zA-Z_]+)", env_src))   # droid_cfg's self.scene is the *cfg*
    return data_attrs, scene_attrs


def test_harvested_attributes_are_provided():
    data_attrs, scene_attrs = _harvest()
    from polaris_mujoco.backend import env as E
    provided_data = set()
    for cls in (E._RigidData, E._RobotData, E._EEData, E._CamData):
        provided_data |= {n for n in dir(cls) if not n.startswith("_")}
    provided_data |= {"output", "default_root_state"}
    missing = sorted(a for a in data_attrs if a not in provided_data)
    assert not missing, f"upstream reads .data.{missing} which no facade provides"
    scene_provided = {n for n in dir(E.Scene) if not n.startswith("_")} | {"rigid_objects", "sensors", "num_envs", "env_origins"}
    missing_scene = sorted(a for a in scene_attrs if a not in scene_provided)
    assert not missing_scene, f"upstream reads scene.{missing_scene}"


@pytest.mark.skipif(not (ROOT / "data/hub/food_bussing/scene.usda").exists(), reason="hub not downloaded")
def test_link_offsets_panda_identity():
    from polaris_mujoco.backend import robot
    off = robot.isaac_link_offsets()
    for i in range(8):
        o = off[f"panda_link{i}"]
        assert abs(o["pos"][0]) < 1e-3 and abs(o["pos"][1]) < 1e-3 and abs(o["pos"][2]) < 1e-3
        assert abs(abs(o["quat"][0]) - 1) < 1e-3
    assert off["base_link"]["body"] == "gripper/base"
