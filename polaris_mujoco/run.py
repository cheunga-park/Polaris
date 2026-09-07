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
    sys.argv = [str(script)] + argv
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
