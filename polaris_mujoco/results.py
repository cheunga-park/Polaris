"""Evaluation runs on disk (``runs/<name>/<env>/eval_results.csv`` + ``episode_N.mp4``) summarised
for the console: success rate with a Wilson 90 % interval, mean progress, and the videos."""

from __future__ import annotations

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


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
                    "mtime": csvf.stat().st_mtime})
    return out
