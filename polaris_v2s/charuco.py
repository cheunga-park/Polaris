"""ChArUco board: the ruler and the compass of a scan (paper §3.1).

* ``board_pdf`` prints the board the user tapes to the table.
* ``detect`` finds it in frames and solves each frame's camera pose in the board frame.
* ``write_ref_images`` turns those poses into the ``image_name X Y Z`` file COLMAP's
  ``model_aligner`` consumes, which is how the paper resolves metric scale and gravity.

Board convention (ours, recorded in configs/charuco.yaml): the board lies flat, its origin
is the printed board's lower-left corner, X along the long edge, Y along the short edge,
Z up (out of the table, towards the camera).

OpenCV's own CharucoBoard frame is different and this module converts: measured on this
OpenCV (5.0), chessboard corner 0 sits at the top-left of the printed image, +X runs right,
+Y runs *down* the page and +Z points *into* the board (a frontal camera solves to z<0).
``_to_board_up`` maps OpenCV (x, y, z) -> ours (x, H - y, -z), H = board height.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = ROOT / "configs" / "charuco.yaml"

_DICTS = {"DICT_4X4_50": cv2.aruco.DICT_4X4_50, "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
          "DICT_6X6_250": cv2.aruco.DICT_6X6_250}


@dataclass
class BoardSpec:
    squares_x: int = 7
    squares_y: int = 5
    square_m: float = 0.055
    marker_m: float = 0.041
    dictionary: str = "DICT_5X5_100"

    @classmethod
    def load(cls, path: Path = DEFAULT_CFG) -> "BoardSpec":
        return cls(**yaml.safe_load(path.read_text())["board"])

    def board(self) -> cv2.aruco.CharucoBoard:
        return cv2.aruco.CharucoBoard((self.squares_x, self.squares_y), self.square_m, self.marker_m,
                                      cv2.aruco.getPredefinedDictionary(_DICTS[self.dictionary]))

    @property
    def size_m(self) -> tuple[float, float]:
        return self.squares_x * self.square_m, self.squares_y * self.square_m


def board_png(spec: BoardSpec, dst: Path, dpi: int = 300) -> Path:
    """Board image at true scale for printing (A3 landscape fits 7x5 at 55 mm squares: 385x275 mm)."""
    px_per_m = dpi / 0.0254
    w, h = spec.size_m
    img = spec.board().generateImage((int(round(w * px_per_m)), int(round(h * px_per_m))), marginSize=0)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return dst


@dataclass
class FramePose:
    image: str
    n_corners: int
    rvec: list[float]      # board -> camera (OpenCV)
    tvec: list[float]
    cam_pos_board: list[float]   # camera centre in OUR board frame, metres, Z up (what model_aligner wants)
    reproj_px: float


def _to_board_up(p_cv: np.ndarray, board_h: float) -> np.ndarray:
    return np.array([p_cv[0], board_h - p_cv[1], -p_cv[2]], dtype=np.float64)


def _pose(corners, ids, board, K, dist, board_h: float | None = None):
    # OpenCV >= 4.8 / 5.x: no estimatePoseCharucoBoard; match corners to board points and PnP
    obj, img = board.matchImagePoints(corners, ids)
    if obj is None or len(obj) < 4:
        return None
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    cam_pos = (-R.T @ tvec).ravel()
    if board_h is not None:
        cam_pos = _to_board_up(cam_pos, board_h)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = float(np.linalg.norm(proj.reshape(-1, 2) - corners.reshape(-1, 2), axis=1).mean())
    return rvec.ravel().tolist(), tvec.ravel().tolist(), cam_pos.tolist(), err


def detect(images_dir: Path, K: np.ndarray, dist: np.ndarray, spec: BoardSpec,
           min_corners: int = 8) -> list[FramePose]:
    """Camera pose per frame where the board is seen. ``K``/``dist`` come from COLMAP's
    calibrated camera (so the board never has to be seen while calibrating)."""
    board = spec.board()
    det = cv2.aruco.CharucoDetector(board)
    out = []
    for p in sorted(images_dir.glob("*.jpg")):
        img = cv2.imread(str(p))
        ch_corners, ch_ids, _, _ = det.detectBoard(img)
        if ch_ids is None or len(ch_ids) < min_corners:
            continue
        r = _pose(ch_corners, ch_ids, board, K, dist, board_h=spec.size_m[1])
        if r is None:
            continue
        rvec, tvec, cam_pos, err = r
        out.append(FramePose(p.name, int(len(ch_ids)), rvec, tvec, cam_pos, err))
    return out


def write_ref_images(poses: list[FramePose], dst: Path, max_reproj_px: float = 2.0) -> int:
    """COLMAP model_aligner reference file: ``image_name X Y Z`` per line, metric, board frame."""
    keep = [p for p in poses if p.reproj_px <= max_reproj_px]
    dst.write_text("".join(f"{p.image} {p.cam_pos_board[0]:.6f} {p.cam_pos_board[1]:.6f} {p.cam_pos_board[2]:.6f}\n" for p in keep))
    (dst.with_suffix(".json")).write_text(json.dumps([asdict(p) for p in keep], indent=1))
    return len(keep)


def colmap_intrinsics(cameras_txt_or_bin_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """K and OpenCV distortion from a COLMAP sparse model (OPENCV or PINHOLE camera)."""
    import pycolmap  # optional; fall back to text parsing when absent
    rec = pycolmap.Reconstruction(str(cameras_txt_or_bin_dir))
    cam = next(iter(rec.cameras.values()))
    p = cam.params
    if cam.model.name == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = p
        dist = np.array([k1, k2, p1, p2], dtype=np.float64)
    elif cam.model.name in ("PINHOLE",):
        fx, fy, cx, cy = p; dist = np.zeros(4)
    elif cam.model.name in ("SIMPLE_PINHOLE",):
        fx, cx, cy = p; fy = fx; dist = np.zeros(4)
    elif cam.model.name in ("SIMPLE_RADIAL",):
        fx, cx, cy, k1 = p; fy = fx; dist = np.array([k1, 0, 0, 0.0])
    else:
        raise ValueError(f"unsupported COLMAP camera model {cam.model.name}")
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return K, dist
