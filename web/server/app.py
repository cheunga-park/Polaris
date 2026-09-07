"""Polaris web console.

Static app under ``web/app`` + a JSON API over the hub and, later, the reconstruction jobs.
Derived files (GLB from usdz, browser PLY from 2DGS PLY) are made on first request and
cached under ``data/cache/<env>/...``; the hub itself is never written to.

Run: ``uvicorn web.server.app:app --host 0.0.0.0 --port 8080``  (``scripts/run.sh`` does this)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import re
import shutil

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from polaris_v2s import hub, jobs, splat_io, usd_io

ROOT = Path(__file__).resolve().parents[2]
APP_DIR = ROOT / "web" / "app"
CACHE = Path(os.environ.get("POLARIS_CACHE_PATH", ROOT / "data" / "cache")).resolve()

app = FastAPI(title="Polaris console")


@app.get("/api/health")
def health():
    return {"ok": True, "hub": str(hub.hub_root()), "cache": str(CACHE)}


@app.get("/api/envs")
def envs():
    return hub.as_dicts(hub.list_envs())


def _env(env_id: str) -> hub.EnvEntry:
    try:
        return hub.get_env(env_id)
    except KeyError:
        raise HTTPException(404, f"unknown environment {env_id}")


@app.get("/api/envs/{env_id}/scene")
def scene(env_id: str):
    e = _env(env_id)
    info = usd_io.read_scene(Path(e.path) / "scene.usda")
    d = info.to_dict()
    d["env"] = hub.as_dicts([e])[0]
    d["robot"] = {"splat": (hub.hub_root() / hub.ROBOT_DIR / "splat.ply").exists()}
    return d


@app.get("/api/envs/{env_id}/initial_conditions")
def initial_conditions(env_id: str):
    e = _env(env_id)
    p = Path(e.path) / "initial_conditions.json"
    if not p.exists():
        raise HTTPException(404, "no initial_conditions.json")
    return json.loads(p.read_text())


# ---- derived assets (cached) -------------------------------------------------------------

@app.get("/assets/envs/{env_id}/prims/{prim}/mesh.glb")
def prim_glb(env_id: str, prim: str):
    """The prim's mesh as placed by scene.usda (child overrides included), in the prim frame."""
    e = _env(env_id)
    scene_usda = Path(e.path) / "scene.usda"
    dst = CACHE / e.folder / "prims" / prim / "mesh.glb"
    if not dst.exists() or dst.stat().st_mtime < scene_usda.stat().st_mtime:
        try:
            usd_io.prim_to_glb(scene_usda, prim, dst)
        except (KeyError, ValueError) as ex:
            raise HTTPException(404, str(ex))
    return FileResponse(dst, media_type="model/gltf-binary")


@app.get("/assets/envs/{env_id}/{asset}/splat_web.ply")
def asset_splat(env_id: str, asset: str):
    e = _env(env_id)
    src = Path(e.path) / "assets" / asset / "splat.ply"
    if not src.exists():
        raise HTTPException(404, "no splat.ply")
    dst = CACHE / e.folder / asset / "splat_web.ply"
    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        splat_io.to_web_ply(src, dst)
    return FileResponse(dst, media_type="application/octet-stream")


@app.get("/assets/robot/splat_web.ply")
def robot_splat():
    src = hub.hub_root() / hub.ROBOT_DIR / "splat.ply"
    if not src.exists():
        raise HTTPException(404, "no robot splat (download the hub)")
    dst = CACHE / hub.ROBOT_DIR / "splat_web.ply"
    if not dst.exists():
        splat_io.to_web_ply(src, dst)
    return FileResponse(dst, media_type="application/octet-stream")


@app.get("/assets/robot/links/{stem}.ply")
def robot_link_splat(stem: str):
    src = hub.hub_root() / hub.ROBOT_DIR / "SEGMENTED" / f"{stem}.ply"
    if not src.exists() or "/" in stem or ".." in stem:
        raise HTTPException(404, "no such link")
    dst = CACHE / hub.ROBOT_DIR / "links" / f"{stem}.ply"
    if not dst.exists():
        splat_io.to_web_ply(src, dst)
    return FileResponse(dst, media_type="application/octet-stream")


@app.get("/api/robot/links")
def robot_links():
    d = hub.hub_root() / hub.ROBOT_DIR / "SEGMENTED"
    return sorted(p.stem for p in d.glob("*.ply")) if d.exists() else []


# ---- scans: upload from a phone, run the pipeline, watch it ------------------------------

_SCAN_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{1,40}$")


@app.get("/api/scans")
def scans():
    return jobs.list_scans()


@app.post("/api/scans")
async def new_scan(name: str = Form(...), video: UploadFile = File(...), start: bool = Form(True)):
    """Multipart: name + the background video. Writes data/scans/<name>/upload/video.mp4 and starts the job."""
    name = name.strip().lower().replace(" ", "_")
    if not _SCAN_ID.match(name):
        raise HTTPException(400, "name: lowercase letters, digits, _ or -, 2-40 chars")
    dst = jobs.SCANS / name / "upload" / "video.mp4"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as f:
        shutil.copyfileobj(video.file, f)
    if dst.stat().st_size < 100_000:
        dst.unlink()
        raise HTTPException(400, "video too small")
    if start:
        jobs.start(name)
    return {"scan": name, "bytes": dst.stat().st_size, "started": start}


@app.post("/api/scans/{scan_id}/objects")
async def add_object(scan_id: str, name: str = Form(...), video: UploadFile = File(...), size_m: float | None = Form(None)):
    """An object scan for an existing scan: upload/objects/<name>.mp4 (+ .json with the longest side in metres)."""
    if not (jobs.SCANS / scan_id / "upload" / "video.mp4").exists():
        raise HTTPException(404, "no such scan")
    name = name.strip().lower().replace(" ", "_")
    if not _SCAN_ID.match(name):
        raise HTTPException(400, "object name: lowercase letters, digits, _ or -")
    d = jobs.SCANS / scan_id / "upload" / "objects"; d.mkdir(parents=True, exist_ok=True)
    with (d / f"{name}.mp4").open("wb") as f:
        shutil.copyfileobj(video.file, f)
    if size_m:
        (d / f"{name}.json").write_text(json.dumps({"size_m": size_m}))
    return {"scan": scan_id, "object": name, "size_m": size_m}


@app.post("/api/scans/{scan_id}/run")
def run_scan(scan_id: str, force: str = ""):
    if not (jobs.SCANS / scan_id / "upload" / "video.mp4").exists():
        raise HTTPException(404, "no such scan")
    ok = jobs.start(scan_id, force=[x for x in force.split(",") if x])
    return {"scan": scan_id, "started": ok}


@app.get("/api/scans/{scan_id}")
def scan_status(scan_id: str):
    d = jobs.SCANS / scan_id
    if not d.exists():
        raise HTTPException(404, "no such scan")
    st = jobs.read_status(d)
    from dataclasses import asdict
    return {**asdict(st), "running": bool(jobs._threads.get(scan_id) and jobs._threads[scan_id].is_alive())}


@app.get("/api/robot/fk")
def robot_fk(q: str = "0,-0.628,0,-2.513,0,1.885,0", gripper: float = 0.0):
    """Isaac-link world poses for 7 arm joints (comma separated radians) and a 0..1 gripper command."""
    from polaris_mujoco import fk
    vals = [float(x) for x in q.split(",")]
    if len(vals) != 7:
        raise HTTPException(400, "q needs 7 values")
    return fk.link_poses(vals, gripper)


@app.get("/api/runs")
def runs():
    from polaris_mujoco import results
    return results.list_runs()


@app.get("/runs/{path:path}")
def run_file(path: str):
    from polaris_mujoco import results
    p = (results.RUNS / path).resolve()
    if not str(p).startswith(str(results.RUNS.resolve())) or not p.exists():
        raise HTTPException(404, "no such file")
    return FileResponse(p)


@app.get("/api/board.png")
def board_png():
    from polaris_v2s import charuco
    p = CACHE / "board.png"
    if not p.exists():
        charuco.board_png(charuco.BoardSpec.load(), p)
    return FileResponse(p, media_type="image/png", filename="polaris_charuco_A3.png")


# upstream scene-composition GUI, built unmodified (vite base = /compose-environments/)
COMPOSE_DIST = ROOT / "third_party" / "compose-environments" / "dist"
if COMPOSE_DIST.exists():
    app.mount("/compose-environments", StaticFiles(directory=COMPOSE_DIST, html=True), name="compose")

app.mount("/", StaticFiles(directory=APP_DIR, html=True), name="app")
