"""Isolated speed benchmark for backbones (run with nothing else on the GPU).

Measures GPU throughput (bf16, batch 64), GPU latency (batch 1), and CPU latency
(PyTorch fp32, batch 1, 4 threads) per backbone, independent of training runs.

Example:
    python -m species_benchmark.speed --backbones convnext_atto.d2_in1k bioclip2 \
        --out C:/pfspecies/report/speed.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from species_benchmark.model import TaxonClassifier

N_CLASSES = {"family": 16, "genus": 80, "species": 400}


@torch.no_grad()
def bench(backbone: str, img_size: int = 224, cpu: bool = True) -> Dict[str, float]:
    """Benchmark one backbone with hierarchical heads.

    Args:
        backbone: timm name or CLIP key.
        img_size: Input resolution.
        cpu: Also measure CPU latency.

    Returns:
        Dict with params and timings.
    """
    m = TaxonClassifier(backbone, N_CLASSES, pretrained=False, img_size=img_size).eval()
    out = {"params_M": sum(p.numel() for p in m.parameters()) / 1e6, "img_size": img_size}
    m = m.cuda()
    for bs, key in ((64, "gpu_imgs_per_s_bs64"), (1, "gpu_ms_bs1")):
        x = torch.randn(bs, 3, img_size, img_size, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(10):
                m(x)
            torch.cuda.synchronize()
            iters = 50
            t0 = time.perf_counter()
            for _ in range(iters):
                m(x)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
        out[key] = bs * iters / dt if bs > 1 else dt / iters * 1000
    if cpu:
        torch.set_num_threads(4)
        mc = m.cpu().float()
        x = torch.randn(1, 3, img_size, img_size)
        for _ in range(3):
            mc(x)
        ts = []
        for _ in range(15):
            t0 = time.perf_counter()
            mc(x)
            ts.append((time.perf_counter() - t0) * 1000)
        out["cpu_ms_bs1_4threads"] = float(np.median(ts))
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbones", nargs="+", required=True)
    ap.add_argument("--img-sizes", nargs="*", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    sizes = args.img_sizes or [224] * len(args.backbones)
    res = json.loads(args.out.read_text()) if args.out.exists() else {}
    for b, s in zip(args.backbones, sizes):
        res[b] = bench(b, s, cpu=True)
        print(b, {k: round(v, 2) for k, v in res[b].items()}, flush=True)
        torch.cuda.empty_cache()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
