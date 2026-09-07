"""Import hooks that let the unmodified upstream ``polaris`` package run without Isaac Lab.

``install()`` puts one finder on ``sys.meta_path`` that serves every ``isaaclab*``,
``isaaclab_tasks*``, ``isaacsim*`` and ``omni*`` import from this package:

* names upstream only *instantiates at import time* (config classes) resolve to recording
  stubs — constructor kwargs become attributes, so upstream's numbers (camera intrinsics,
  PD gains, initial joints) are read from upstream at runtime, never copied here;
* names upstream *calls into at run time* resolve to real facades over MuJoCo
  (``facade`` module): ``ManagerBasedRLEnv`` (the parent of upstream's env class), the
  camera sensor class, ``GeometryPrim``, the USD stage accessors and a few math helpers.

``pxr`` is the real ``usd-core``; only Isaac's ``Semantics`` schema is attached as a stub.
Upstream files are never edited; see PLAN.md §8.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
import types

from .stubs import Stub, StubModule, configclass

PREFIXES = ("isaaclab", "isaaclab_tasks", "isaacsim", "omni")

# fully-qualified name -> attribute provider (module path in this package, attribute name)
FACADES = {
    "isaaclab.envs.ManagerBasedRLEnv": ("polaris_mujoco.hooks.facade", "ManagerBasedRLEnv"),
    "isaaclab.envs.ManagerBasedRLEnvCfg": ("polaris_mujoco.hooks.facade", "ManagerBasedRLEnvCfg"),
    "isaaclab.sensors.camera.camera.Camera": ("polaris_mujoco.hooks.facade", "Camera"),
    "isaaclab.sensors.Camera": ("polaris_mujoco.hooks.facade", "Camera"),
    "isaaclab.utils.math.matrix_from_quat": ("polaris_mujoco.hooks.facade", "matrix_from_quat"),
    "isaaclab.utils.math.convert_camera_frame_orientation_convention": ("polaris_mujoco.hooks.facade", "convert_camera_frame_orientation_convention"),
    "isaacsim.core.prims.GeometryPrim": ("polaris_mujoco.hooks.facade", "GeometryPrim"),
    "isaacsim.core.utils.stage.get_current_stage": ("polaris_mujoco.hooks.facade", "get_current_stage"),
    "isaaclab.app.AppLauncher": ("polaris_mujoco.hooks.facade", "AppLauncher"),
    "isaaclab_tasks.utils.parse_env_cfg": ("polaris_mujoco.hooks.facade", "parse_env_cfg"),
    "isaaclab_tasks.utils.load_cfg_from_registry": ("polaris_mujoco.hooks.facade", "load_cfg_from_registry"),
}


class HookedModule(StubModule):
    def __getattr__(self, name):
        full = f"{self.__name__}.{name}"
        if full in FACADES:
            modname, attr = FACADES[full]
            import importlib
            return getattr(importlib.import_module(modname), attr)
        return super().__getattr__(name)


class Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname.split(".")[0] in PREFIXES:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        m = HookedModule(spec.name)
        m.__path__ = []
        return m

    def exec_module(self, module):
        pass


_installed = False


def install() -> None:
    global _installed
    if _installed:
        return
    sys.meta_path.insert(0, Finder())
    import pxr
    pxr.Semantics = type("Semantics", (), {"SemanticsAPI": Stub})
    import omni.usd
    from . import facade
    omni.usd.get_context = facade.get_context
    _installed = True
