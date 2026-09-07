"""PolaRiS-Hub on disk: what environments there are and where their files live.

``POLARIS_DATA_PATH`` (the variable upstream reads too) points at the hub root; default
``data/hub``. Environment ids follow upstream's registry names so the same string works
in the web console and in ``scripts/eval.py``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HUB = ROOT / "data" / "hub"

# upstream environments/__init__.py: id -> folder (thumbnail = upstream docs/images)
UPSTREAM_ENVS = {
    "DROID-BlockStackKitchen": ("block_stack_kitchen", "stack.png"),
    "DROID-FoodBussing": ("food_bussing", "foodbus.png"),
    "DROID-PanClean": ("pan_clean", "panclean.png"),
    "DROID-MoveLatteCup": ("move_latte_cup", "latte-cup.png"),
    "DROID-OrganizeTools": ("organize_tools", "organize-tools.png"),
    "DROID-TapeIntoContainer": ("tape_into_container", "tape-into-container.png"),
}
ROBOT_DIR = "nvidia_droid"


def hub_root() -> Path:
    return Path(os.environ.get("POLARIS_DATA_PATH", DEFAULT_HUB)).resolve()


@dataclass
class EnvEntry:
    id: str
    folder: str
    path: str
    instruction: str
    n_poses: int
    assets: list[str]
    thumbnail: str | None
    source: str  # "hub" (upstream) | "scan" (ours)


def _entry(env_id: str, folder: Path, thumb: str | None, source: str) -> EnvEntry | None:
    ic = folder / "initial_conditions.json"
    if not (folder / "scene.usda").exists():
        return None
    instruction, n = "", 0
    if ic.exists():
        d = json.loads(ic.read_text())
        instruction, n = d.get("instruction", ""), len(d.get("poses", []))
    assets = sorted(p.name for p in (folder / "assets").iterdir() if p.is_dir()) if (folder / "assets").exists() else []
    return EnvEntry(env_id, folder.name, str(folder), instruction, n, assets, thumb, source)


def list_envs(root: Path | None = None) -> list[EnvEntry]:
    root = root or hub_root()
    out = []
    for env_id, (folder, thumb) in UPSTREAM_ENVS.items():
        e = _entry(env_id, root / folder, thumb, "hub")
        if e:
            out.append(e)
    known = {f for f, _ in UPSTREAM_ENVS.values()} | {ROBOT_DIR}
    if root.exists():
        for p in sorted(root.iterdir()):
            if p.is_dir() and p.name not in known:
                e = _entry(p.name, p, None, "scan")
                if e:
                    out.append(e)
    return out


def get_env(env_id: str, root: Path | None = None) -> EnvEntry:
    for e in list_envs(root):
        if e.id == env_id or e.folder == env_id:
            return e
    raise KeyError(env_id)


def as_dicts(entries: list[EnvEntry]) -> list[dict]:
    return [asdict(e) for e in entries]
