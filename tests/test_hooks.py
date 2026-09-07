"""The unmodified upstream package imports and yields its data under our hooks (PLAN §8.1 S-A..S-D)."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "polaris" / "src"))
os.environ.setdefault("POLARIS_DATA_PATH", str(ROOT / "data" / "hub"))

from polaris_mujoco import hooks  # noqa: E402

hooks.install()

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402


def test_registry_has_six_envs_with_real_rubrics():
    import polaris.environments  # noqa: F401  (upstream)
    ids = ["DROID-BlockStackKitchen", "DROID-FoodBussing", "DROID-PanClean", "DROID-MoveLatteCup", "DROID-OrganizeTools", "DROID-TapeIntoContainer"]
    n = {i: len(gym.spec(i).kwargs["rubric"].criteria) for i in ids}
    assert n == {"DROID-BlockStackKitchen": 7, "DROID-FoodBussing": 6, "DROID-PanClean": 3, "DROID-MoveLatteCup": 3, "DROID-OrganizeTools": 3, "DROID-TapeIntoContainer": 3}


def test_upstream_constants_are_read_not_copied():
    from polaris.environments.droid_cfg import SceneCfg
    from polaris.environments.robot_cfg import NVIDIA_DROID
    assert (SceneCfg.wrist_cam.width, SceneCfg.wrist_cam.height) == (1280, 720)
    assert SceneCfg.wrist_cam.spawn.focal_length == 2.8
    assert NVIDIA_DROID.actuators["panda_shoulder"].stiffness == 400.0
    assert NVIDIA_DROID.actuators["panda_shoulder"].damping == 80.0


@pytest.mark.skipif(not (ROOT / "data/hub/food_bussing/scene.usda").exists(), reason="hub not downloaded")
def test_dynamic_setup_parses_hub_scene_with_usd_core():
    from polaris.environments.droid_cfg import SceneCfg
    cfg = SceneCfg(num_envs=1, env_spacing=7.0)
    cfg.dynamic_setup(str(ROOT / "data/hub/food_bussing/scene.usda"))
    names = {k for k, v in vars(cfg).items() if getattr(getattr(v, "init_state", None), "pos", None) is not None}
    assert {"bowl", "battery1", "battery2", "cup", "ice_cream_", "grapes", "g60_corner_charuco_static"} <= names
    assert hasattr(cfg, "external_cam")


@pytest.mark.skipif(not (ROOT / "data/hub/food_bussing/scene.usda").exists(), reason="hub not downloaded")
def test_checkers_run_on_mujoco_shaped_env():
    from polaris.environments.rubrics import checkers
    from polaris_mujoco.hooks import facade
    facade.open_scene_stage(ROOT / "data/hub/food_bussing/scene.usda")

    class D:
        def __init__(self, pos, quat=(1, 0, 0, 0)):
            self.root_pos_w = torch.tensor([pos], dtype=torch.float32)
            self.root_quat_w = torch.tensor([quat], dtype=torch.float32)
            self.default_root_state = torch.tensor([[*pos, *quat, 0, 0, 0, 0, 0, 0]])

    class Obj:
        def __init__(self, d): self.data = d

    class Robot:
        def __init__(self, finger):
            self.data = type("X", (), {})(); self.data.joint_pos = torch.tensor([[0.0] * 7 + [finger]])
            self.data.joint_names = [f"panda_joint{i}" for i in range(1, 8)] + ["finger_joint"]

    class EE:
        def __init__(self, p): self.data = type("X", (), {})(); self.data.target_pos_w = torch.tensor([p])

    class Env:
        def __init__(self, scene): self.scene = scene

    bowl = (0.276, -0.080, 0.050)
    env = Env({"bowl": Obj(D(bowl)), "ice_cream_": Obj(D((0.28, -0.07, 0.09))), "robot": Robot(0.0), "ee_frame": EE((0.28, -0.07, 0.12))})
    assert checkers.reach("ice_cream_", 0.2)(env)
    assert checkers.lift("ice_cream_", 0.03, default_height=0.05)(env)
    assert checkers.is_within_xy("ice_cream_", "bowl", 0.8)(env)
    far = Env({**env.scene, "ice_cream_": Obj(D((0.55, 0.30, 0.09)))})
    assert not checkers.is_within_xy("ice_cream_", "bowl", 0.8)(far)


def test_matrix_from_quat_matches_upstream_formula():
    from polaris_mujoco.hooks.facade import matrix_from_quat
    q = torch.tensor([0.7071068, 0.7071068, 0.0, 0.0])  # 90 deg about x
    R = matrix_from_quat(q)
    assert torch.allclose(R @ torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0.0, 0.0, 1.0]), atol=1e-5)
