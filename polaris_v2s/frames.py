"""Video -> frames for COLMAP.

A phone scan is 30 fps for minutes; COLMAP wants a few hundred *sharp* frames with
overlap. So: decode at a fixed rate, score sharpness (variance of the Laplacian), and keep
the sharpest frame of every window. The long edge is capped so 4K phones do not blow up
feature extraction and 2DGS memory.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np


def ffprobe(video: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,nb_frames,duration:format=duration",
                          "-of", "json", str(video)], check=True, capture_output=True, text=True).stdout
    d = json.loads(out)
    st = d["streams"][0]
    num, den = st["r_frame_rate"].split("/")
    dur = float(st.get("duration") or d["format"]["duration"])
    return {"width": st["width"], "height": st["height"], "fps": float(num) / float(den), "duration_s": dur}


def sharpness(img_bgr: np.ndarray) -> float:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


@dataclass
class FramesReport:
    video: str
    probe: dict
    decoded: int
    kept: int
    long_edge: int
    window: int
    min_sharpness: float
    dropped_blurry: int
    out_dir: str


def extract(video: str | Path, out_dir: str | Path, *, fps: float = 4.0, long_edge: int = 1440,
            window: int = 2, min_sharpness: float = 20.0, max_frames: int = 600) -> FramesReport:
    """Decode ``fps`` frames/s scaled to ``long_edge``; keep the sharpest of every ``window``
    consecutive frames; drop anything below ``min_sharpness`` (Laplacian variance); cap at
    ``max_frames`` by uniform thinning. Output ``out_dir/images/%05d.jpg`` (COLMAP layout)."""
    video, out_dir = Path(video), Path(out_dir)
    probe = ffprobe(video)
    tmp = out_dir / "_decoded"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    scale = f"scale='if(gt(iw,ih),min({long_edge},iw),-2)':'if(gt(iw,ih),-2,min({long_edge},ih))'"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vf", f"fps={fps},{scale}",
                    "-q:v", "2", str(tmp / "%05d.jpg")], check=True)
    decoded = sorted(tmp.glob("*.jpg"))
    scores = [sharpness(cv2.imread(str(p))) for p in decoded]
    kept, dropped = [], 0
    for i in range(0, len(decoded), window):
        block = list(range(i, min(i + window, len(decoded))))
        best = max(block, key=lambda k: scores[k])
        if scores[best] < min_sharpness:
            dropped += 1
            continue
        kept.append(best)
    if len(kept) > max_frames:
        idx = np.linspace(0, len(kept) - 1, max_frames).round().astype(int)
        kept = [kept[i] for i in idx]
    images = out_dir / "images"
    if images.exists():
        shutil.rmtree(images)
    images.mkdir(parents=True)
    for n, k in enumerate(kept):
        shutil.copy(decoded[k], images / f"{n:05d}.jpg")
    shutil.rmtree(tmp)
    rep = FramesReport(str(video), probe, len(decoded), len(kept), long_edge, window, min_sharpness, dropped, str(out_dir))
    (out_dir / "frames.json").write_text(json.dumps(asdict(rep), indent=2))
    return rep
