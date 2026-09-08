"""Run an upstream script (default: scripts/eval.py) with the hooks installed.

    python -m polaris_mujoco.run third_party/polaris/scripts/eval.py --environment DROID-FoodBussing ...
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_SRC = ROOT / "third_party" / "polaris" / "src"


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    script = Path(argv[0]) if argv and argv[0].endswith(".py") else ROOT / "third_party" / "polaris" / "scripts" / "eval.py"
    if argv and argv[0].endswith(".py"):
        argv = argv[1:]
    if str(UPSTREAM_SRC) not in sys.path:
        sys.path.insert(0, str(UPSTREAM_SRC))
    from polaris_mujoco import hooks
    hooks.install()
    # upstream defines FakeClient but never registers it (abstract_client.py); register it here so
    # `--policy.client Fake` works as the docstring implies. A hook, not an edit.
    from polaris.policy.abstract_client import FakeClient, InferenceClient
    InferenceClient.REGISTERED_CLIENTS.setdefault("Fake", FakeClient)
    # Upstream eval.py appends one frame per *policy inference* (every open_loop_horizon = 8 env
    # steps = 0.53 s of sim time) but writes the mp4 at fps=15, so a 30 s episode plays in 3.7 s.
    # Pace the video in real time by default (fps = 15 / horizon); POLARIS_VIDEO_FPS overrides.
    try:
        import os as _os
        import mediapy as _mp
        _orig_write = _mp.write_video

        def _write_video(path, images, *a, fps=15, **k):
            horizon = int(_os.environ.get("POLARIS_OPEN_LOOP_HORIZON", "8"))
            fps = float(_os.environ.get("POLARIS_VIDEO_FPS", fps / horizon))
            return _orig_write(path, images, *a, fps=fps, **k)

        _mp.write_video = _write_video
    except Exception:  # noqa: BLE001
        pass
    # openpi's websocket client keeps the library default ping_timeout (20 s). A stall on the shared
    # GPU longer than that kills the whole run ("keepalive ping timeout"), so widen it here. Hook on
    # the websockets entry point, not on upstream/openpi code.
    try:
        import websockets.sync.client as _wsc
        _orig_connect = _wsc.connect

        def _connect(*a, **k):
            k.setdefault("ping_timeout", 600)
            k.setdefault("ping_interval", 30)
            return _orig_connect(*a, **k)

        _wsc.connect = _connect
    except Exception:  # noqa: BLE001 - only a robustness tweak
        pass
    sys.argv = [str(script)] + argv
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
