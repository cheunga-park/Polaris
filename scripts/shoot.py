"""Headless screenshots of the viewer for every environment (visual gate for phase 1).
usage: .venv/bin/python scripts/shoot.py [--port 8080] [--out data/shots]"""
import argparse, asyncio, json, time, urllib.request
from playwright.async_api import async_playwright

async def main(port, out, cdp):
    envs = json.load(urllib.request.urlopen(f"http://localhost:{port}/api/envs"))
    async with async_playwright() as p:
        if cdp:   # e.g. Windows Edge: msedge.exe --headless=new --remote-debugging-port=9222 (WSL mirrored networking)
            b = await p.chromium.connect_over_cdp(cdp)
            ctx = await b.new_context(viewport={"width": 1280, "height": 800})
            pg = await ctx.new_page()
        else:
            b = await p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
            pg = await b.new_page(viewport={"width": 1280, "height": 800})
        logs = []
        pg.on("console", lambda m: logs.append(f"[{m.type}] {m.text}"))
        pg.on("pageerror", lambda e: logs.append(f"[pageerror] {e}"))
        for e in envs:
            t = time.time()
            await pg.goto(f"http://localhost:{port}/viewer.html?env={e['id']}", wait_until="load")
            try:
                await pg.wait_for_function("document.getElementById('status').textContent === ''", timeout=180_000)
            except Exception as ex:
                logs.append(f"[timeout] {e['id']}: {ex}")
            await pg.wait_for_timeout(2500)   # a couple of frames for the splat sort
            await pg.screenshot(path=f"{out}/{e['folder']}.png")
            print(f"{e['id']:28s} {time.time()-t:5.1f}s -> {out}/{e['folder']}.png")
        await b.close()
    bad = [l for l in logs if "[error]" in l or "pageerror" in l or "timeout" in l]
    print("console errors:", len(bad)); [print("  ", l[:300]) for l in bad[:10]]

if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--port", type=int, default=8080); a.add_argument("--out", default="data/shots"); a.add_argument("--cdp", default="")
    ns = a.parse_args(); asyncio.run(main(ns.port, ns.out, ns.cdp))
