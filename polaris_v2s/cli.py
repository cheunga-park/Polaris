"""polaris-v2s: the pipeline from the shell.

  polaris-v2s run <scan_id> [--force stage,...] [--require-board]   all stages for data/scans/<id>/upload/video.mp4
  polaris-v2s board [--out data/board.png]                          print the ChArUco board (A3)
  polaris-v2s status <scan_id>
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from polaris_v2s import charuco, jobs


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="polaris-v2s")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("scan_id"); r.add_argument("--force", default=""); r.add_argument("--require-board", action="store_true")
    b = sub.add_parser("board"); b.add_argument("--out", default="data/board.png"); b.add_argument("--dpi", type=int, default=300)
    s = sub.add_parser("status"); s.add_argument("scan_id")
    a = p.parse_args(argv)
    if a.cmd == "run":
        st = jobs.run(a.scan_id, force=[x for x in a.force.split(",") if x], require_board=a.require_board)
        print(json.dumps({k: v for k, v in asdict(st).items() if k != "reports"}, indent=2, ensure_ascii=False))
        for k, v in st.reports.items():
            print(f"[{k}]", json.dumps(v, ensure_ascii=False)[:400])
        return 1 if st.failed else 0
    if a.cmd == "board":
        spec = charuco.BoardSpec.load()
        out = charuco.board_png(spec, Path(a.out), a.dpi)
        w, h = spec.size_m
        print(f"{out}: {spec.squares_x}x{spec.squares_y} squares, {w*1000:.0f} x {h*1000:.0f} mm at {a.dpi} dpi -> print at 100% on A3 landscape")
        return 0
    if a.cmd == "status":
        print(json.dumps(asdict(jobs.read_status(jobs.SCANS / a.scan_id)), indent=2, ensure_ascii=False))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
