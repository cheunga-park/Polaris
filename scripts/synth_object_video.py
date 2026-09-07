"""A stand-in object scan: one hub object (textured mesh) on a printed-board plane, orbited by a
MuJoCo camera, encoded as mp4 -- exercises frames -> SAM 2 -> TRELLIS before a phone video exists.
usage: .venv/bin/python scripts/synth_object_video.py --env food_bussing --prim bowl --out data/scans/synth_food/upload/objects/bowl.mp4"""
import argparse, math, subprocess
from pathlib import Path
import cv2, mujoco, numpy as np
ROOT = Path(__file__).resolve().parents[1]
from polaris_v2s import charuco
from polaris_mujoco.backend import stage

def main(a):
    bodies = stage.prepare(ROOT / "data/hub" / a.env / "scene.usda")
    b = next(x for x in bodies if x.name == a.prim)
    board_png = ROOT / "data/cache/board.png"; charuco.board_png(charuco.BoardSpec.load(), board_png)
    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth = 1280; spec.visual.global_.offheight = 720
    spec.visual.headlight.ambient[:] = 0.5; spec.visual.headlight.diffuse[:] = 0.6
    t = spec.add_texture(); t.name = "board"; t.file = str(board_png); t.type = mujoco.mjtTexture.mjTEXTURE_2D
    mt = spec.add_material(); mt.name = "board"; mt.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "board"
    bw, bh = charuco.BoardSpec.load().size_m
    g = spec.worldbody.add_geom(); g.type = mujoco.mjtGeom.mjGEOM_BOX; g.size = [bw/2, bh/2, 0.002]; g.pos = [bw/2, bh/2, -0.002]; g.material = "board"
    floor = spec.worldbody.add_geom(); floor.type = mujoco.mjtGeom.mjGEOM_PLANE; floor.size = [2, 2, 0.1]; floor.pos = [0, 0, -0.004]; floor.rgba = [0.85, 0.82, 0.78, 1]
    body = spec.worldbody.add_body(); body.pos = [bw/2, bh/2, 0.0]
    m = spec.add_mesh(); m.name = "obj"; m.file = str(b.visual_obj); m.scale = np.asarray(b.scale)
    og = body.add_geom(); og.type = mujoco.mjtGeom.mjGEOM_MESH; og.meshname = "obj"
    tex = b.visual_obj.parent / "material_0.png"
    if tex.exists():
        tt = spec.add_texture(); tt.name = "objtex"; tt.file = str(tex); tt.type = mujoco.mjtTexture.mjTEXTURE_2D
        om = spec.add_material(); om.name = "objmat"; om.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "objtex"; og.material = "objmat"
    cam = spec.worldbody.add_camera(); cam.name = "c"; cam.fovy = 50
    model = spec.compile(); data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    r = mujoco.Renderer(model, 720, 1280)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); fr = out.parent / f"{out.stem}_frames"; fr.mkdir(exist_ok=True)
    center = np.array([bw/2, bh/2, 0.05]); cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "c")
    for i in range(a.n):
        u = i / a.n; ang = 2*math.pi*u*2; el = 0.75 if i < a.n//2 else 0.4
        eye = center + np.array([0.55*math.cos(ang), 0.55*math.sin(ang), 0.55*el])
        f = center - eye; f /= np.linalg.norm(f); rgt = np.cross(f, [0,0,1]); rgt /= np.linalg.norm(rgt); up = np.cross(rgt, f)
        R = np.stack([rgt, up, -f], 1)  # mujoco camera: -z forward, +y up
        model.cam_pos[cid] = eye; q = np.empty(4); mujoco.mju_mat2Quat(q, R.reshape(9)); model.cam_quat[cid] = q
        mujoco.mj_forward(model, data); r.update_scene(data, "c"); img = r.render()
        cv2.imwrite(str(fr / f"{i:04d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i", str(fr / "%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)], check=True)
    print("wrote", out)

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--env", default="food_bussing"); p.add_argument("--prim", default="bowl"); p.add_argument("--out", default="data/scans/synth_food/upload/objects/bowl.mp4"); p.add_argument("--n", type=int, default=240)
    main(p.parse_args())
