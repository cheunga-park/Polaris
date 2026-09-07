"""Render an orbit around a hub background splat with the UPSTREAM SplatRenderer and encode
it as an mp4 -- a stand-in for a phone scan so the reconstruction pipeline can be exercised
end to end before a real video exists. Also the first proof that the upstream renderer +
our JIT-built kernels run on this box.

usage: .venv/bin/python scripts/synth_video.py --env food_bussing --out data/scans/synth_food/upload/video.mp4
"""
import argparse, math, subprocess, sys, time
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "polaris" / "src"))
sys.path.insert(0, str(ROOT / "polaris_mujoco"))          # hooks: isaaclab-shaped stubs
from hooks import stubs; stubs.install()                   # polaris.utils imports isaaclab at module top
from polaris.splat_renderer import SplatRenderer          # upstream, unmodified
from polaris_v2s import usd_io

def look_at(eye, target, up=(0, 0, 1)):
    """World->camera rotation with camera +Z forward, -Y up (OpenCV/COLMAP convention), returned as R (3x3, camera axes as columns in world)."""
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, np.asarray(up, float)); r /= np.linalg.norm(r)
    d = np.cross(f, r)                                    # down
    return np.stack([r, d, f], axis=1)

def main(a):
    scene = usd_io.read_scene(ROOT / "data/hub" / a.env / "scene.usda")
    bg = next(p for p in scene.prims if p.has_splat and p.kinematic)
    splat = ROOT / "data/hub" / a.env / "assets" / bg.asset_dir / "splat.ply"
    r = SplatRenderer(splats={bg.name: splat}, device="cuda")
    W, H = a.width, a.height
    fovx = math.radians(a.fov); fovy = 2 * math.atan(math.tan(fovx / 2) * H / W)
    r.init_cameras({"cam": {"res": (H, W), "fovx": fovx, "fovy": fovy}})
    # place the static splat where scene.usda puts it (upstream transform_sim_to_splat with transform_static=True)
    r.transform_many({bg.name: (torch.tensor(bg.translate, dtype=torch.float32), torch.tensor(bg.orient_wxyz, dtype=torch.float32))})
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = out.parent / "synth_frames"; frames_dir.mkdir(exist_ok=True)
    center = np.array(a.center, float)
    t0 = time.time()
    for i in range(a.n):
        u = i / a.n
        # two sweeps (high/wide, low/close) over the half-space the real scan covered: the
        # arc is centred on the policy camera's azimuth, +-70 degrees, phone-scan style
        half = i < a.n // 2
        radius, z = (1.25, 0.8) if half else (0.95, 0.45)
        sweep = (u * 2 % 1.0)
        ang = a.azimuth + math.radians(90) * (sweep - 0.5) * (1 if half else -1)
        eye = center + np.array([radius * math.cos(ang), radius * math.sin(ang), z])
        R = look_at(eye, center)
        # upstream render() expects the Isaac "world" camera convention and applies p_mat itself;
        # render_raw() takes an OpenCV-style R,t directly (R = camera axes in world).
        img = r.render_raw({"cam": {"pos": eye.astype(np.float32), "rot": R.astype(np.float32)}})["cam"]
        img = (img.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
        import cv2; cv2.imwrite(str(frames_dir / f"{i:04d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    dt = time.time() - t0
    print(f"rendered {a.n} frames {W}x{H} in {dt:.1f}s ({1000*dt/a.n:.0f} ms/frame) | VRAM peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i", str(frames_dir / "%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)], check=True)
    print("wrote", out, f"{out.stat().st_size/1e6:.1f} MB")
    import cv2
    pick = [cv2.resize(cv2.imread(str(frames_dir / f"{i:04d}.png")), (320, 180)) for i in np.linspace(0, a.n - 1, 12).astype(int)]
    sheet = np.vstack([np.hstack(pick[r*4:(r+1)*4]) for r in range(3)])
    cv2.imwrite(str(out.parent / "contact_sheet.jpg"), sheet); print("contact sheet", out.parent / "contact_sheet.jpg")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="food_bussing"); p.add_argument("--out", default="data/scans/synth_food/upload/video.mp4")
    p.add_argument("--n", type=int, default=240); p.add_argument("--width", type=int, default=1280); p.add_argument("--height", type=int, default=720)
    p.add_argument("--fov", type=float, default=75.0); p.add_argument("--center", type=float, nargs=3, default=[0.45, -0.05, 0.05])
    p.add_argument("--azimuth", type=float, default=math.atan2(0.507 - (-0.05), 0.134 - 0.45), help="radians; default = food_bussing external_cam direction")
    main(p.parse_args())
