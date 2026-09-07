"""Small, CPU-only checks of the reconstruction helpers."""
import numpy as np
import cv2
import pytest
from pathlib import Path
from plyfile import PlyData, PlyElement

from polaris_v2s import charuco, splat_io


def _tiny_2dgs_ply(path: Path, n=50):
    rng = np.random.default_rng(0)
    fields = [("x","f4"),("y","f4"),("z","f4"),("nx","f4"),("ny","f4"),("nz","f4"),("f_dc_0","f4"),("f_dc_1","f4"),("f_dc_2","f4"),
              ("opacity","f4"),("scale_0","f4"),("scale_1","f4"),("rot_0","f4"),("rot_1","f4"),("rot_2","f4"),("rot_3","f4")]
    v = np.zeros(n, dtype=fields)
    for f in ("x","y","z"): v[f] = rng.normal(size=n)
    v["scale_0"] = -4; v["scale_1"] = -5; v["rot_0"] = 1
    PlyData([PlyElement.describe(v, "vertex")]).write(str(path))


def test_to_web_ply_pads_third_scale(tmp_path):
    src = tmp_path / "s.ply"; _tiny_2dgs_ply(src)
    out = splat_io.to_web_ply(src, tmp_path / "w.ply")
    d = splat_io.read_splat(out)
    assert set(splat_io.WEB_FIELDS) <= set(d)
    assert np.allclose(d["scale_2"], np.minimum(d["scale_0"], d["scale_1"]) - 2.0)


def test_transform_splat_rotates_positions(tmp_path):
    src = tmp_path / "s.ply"; _tiny_2dgs_ply(src)
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]]); t = np.array([1.0, 2.0, 3.0])
    out = splat_io.transform_splat(src, tmp_path / "t.ply", R, t, s=2.0)
    a, b = splat_io.read_splat(src), splat_io.read_splat(out)
    xyz = np.c_[a["x"], a["y"], a["z"]]; exp = 2.0 * xyz @ R.T + t
    assert np.allclose(np.c_[b["x"], b["y"], b["z"]], exp, atol=1e-5)
    assert np.allclose(b["scale_0"], a["scale_0"] + np.log(2.0), atol=1e-5)


def test_charuco_round_trip_millimetres():
    spec = charuco.BoardSpec.load(); board = spec.board(); bw, bh = spec.size_m
    img = board.generateImage((1400, 1000), marginSize=0)
    W, H = 1280, 720; K = np.array([[900., 0, W / 2], [0, 900., H / 2], [0, 0, 1]]); dist = np.zeros(4)
    det = cv2.aruco.CharucoDetector(board)
    src = np.float32([[0, 0], [1400, 0], [1400, 1000], [0, 1000]]); obj_cv = np.array([[0, 0, 0], [bw, 0, 0], [bw, bh, 0], [0, bh, 0]], np.float32)
    errs = []
    for k in range(4):
        ang = np.deg2rad(20 + 15 * k); eye = np.array([bw / 2 + 0.6 * np.cos(ang), bh / 2 + 0.6 * np.sin(ang), 0.7])
        eye_cv = np.array([eye[0], bh - eye[1], -eye[2]])
        tgt = np.array([bw / 2, bh / 2, 0.0]); f = tgt - eye_cv; f /= np.linalg.norm(f); r = np.cross(f, [0, 0, -1.0]); r /= np.linalg.norm(r); d = np.cross(f, r)
        R = np.stack([r, d, f], 1).T; t = -R @ eye_cv; rvec, _ = cv2.Rodrigues(R)
        dst = cv2.projectPoints(obj_cv, rvec, t, K, dist)[0].reshape(-1, 2).astype(np.float32)
        view = cv2.cvtColor(cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (W, H), borderValue=200), cv2.COLOR_GRAY2BGR)
        cc, ci, _, _ = det.detectBoard(view)
        res = charuco._pose(cc, ci, board, K, dist, board_h=bh)
        errs.append(np.linalg.norm(np.array(res[2]) - eye))
    assert max(errs) < 0.003, f"camera-centre errors (m): {errs}"
