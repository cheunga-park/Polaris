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

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from polaris_v2s import hub, splat_io, usd_io

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


app.mount("/", StaticFiles(directory=APP_DIR, html=True), name="app")
