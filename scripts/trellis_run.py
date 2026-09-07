"""Run inside the `trellis` env: masked RGBA views -> GLB (+ gaussian ply). Wrapper only."""
import argparse, json, os, sys
from pathlib import Path
os.environ.setdefault("ATTN_BACKEND", "xformers"); os.environ.setdefault("SPCONV_ALGO", "native")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "TRELLIS"))
from PIL import Image
import torch
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils

p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--seed", type=int, default=0); p.add_argument("--model", default="microsoft/TRELLIS-image-large"); p.add_argument("views", nargs="+")
a = p.parse_args(); out = Path(a.out)
pipe = TrellisImageTo3DPipeline.from_pretrained(a.model); pipe.cuda()
imgs = [Image.open(v).convert("RGBA") for v in a.views]
with torch.no_grad():
    if len(imgs) == 1:
        outputs = pipe.run(imgs[0], seed=a.seed)
    else:
        outputs = pipe.run_multi_image(imgs, seed=a.seed, sparse_structure_sampler_params={"steps": 12, "cfg_strength": 7.5}, slat_sampler_params={"steps": 12, "cfg_strength": 3.0})
glb = postprocessing_utils.to_glb(outputs["gaussian"][0], outputs["mesh"][0], simplify=0.95, texture_size=1024)
glb.export(str(out / "mesh.glb"))
outputs["gaussian"][0].save_ply(str(out / "gaussian.ply"))
json.dump({"glb": str(out / "mesh.glb"), "gaussian": str(out / "gaussian.ply"), "views": a.views, "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30}, open(out / "result.json", "w"), indent=1)
print("done", out)
