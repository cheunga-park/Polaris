"""Evaluation runs on disk (``runs/<name>/<env>/eval_results.csv`` + ``episode_N.mp4``) summarised
for the console: success rate with a Wilson 90 % interval, mean progress, and the videos."""

from __future__ import annotations

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"

# Upstream (Isaac Lab) numbers for the same checkpoint, pi05_droid_jointpos_polaris:
# "paper" = project-page progress (50 rollouts); "issue24" = a third party's reproduction with
# the public code, 100 episodes (github.com/arhanjain/polaris/issues/24). The author states the
# Princeton environments' reset states (MoveLatteCup, OrganizeTools, TapeIntoContainer) are broken
# in the public release, so only the UW ones are a fair reference.
REFERENCE = {
    "DROID-BlockStackKitchen": {"paper_progress": 0.544, "issue24_progress": 0.501, "issue24_success": 0.02, "fair": True},
    "DROID-FoodBussing": {"paper_progress": 0.580, "issue24_progress": 0.580, "issue24_success": 0.17, "fair": True},
    "DROID-PanClean": {"paper_progress": 0.550, "issue24_progress": 0.750, "issue24_success": 0.37, "fair": True},
    "DROID-MoveLatteCup": {"paper_progress": 0.333, "issue24_progress": 0.203, "issue24_success": 0.08, "fair": False},
    "DROID-OrganizeTools": {"paper_progress": 0.600, "issue24_progress": 0.333, "issue24_success": 0.00, "fair": False},
    "DROID-TapeIntoContainer": {"paper_progress": 0.800, "issue24_progress": 0.343, "issue24_success": 0.23, "fair": False},
}


def wilson(k: int, n: int, z: float = 1.645) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def summarize(csv_path: Path) -> dict:
    rows = list(csv.DictReader(csv_path.open()))
    n = len(rows)
    succ = sum(1 for r in rows if r["success"].strip().lower() == "true")
    prog = [float(r["progress"]) for r in rows]
    lo, hi = wilson(succ, n)
    return {"episodes": n, "successes": succ, "success_rate": succ / n if n else 0.0, "wilson90": [lo, hi],
            "mean_progress": sum(prog) / n if n else 0.0,
            "progress_hist": [sum(1 for p in prog if abs(p - b) < 1e-6) for b in sorted(set(prog))],
            "progress_bins": sorted(set(prog))}


def list_runs() -> list[dict]:
    out = []
    if not RUNS.exists():
        return out
    for csvf in sorted(RUNS.glob("**/eval_results.csv")):
        d = csvf.parent
        rel = d.relative_to(RUNS)
        videos = sorted(p.name for p in d.glob("episode_*.mp4"))
        out.append({"run": str(rel), "path": str(d), **summarize(csvf), "videos": videos,
                    "mtime": csvf.stat().st_mtime, "reference": REFERENCE.get(d.name)})
    return out
