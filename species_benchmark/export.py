"""Calibrate and export a trained run to the PrimateFace API format.

Fits one temperature per level on validation logits (NLL), picks roll-up
thresholds, measures CPU latency, and writes ``primateface-taxon-v1`` checkpoints
(plus optional ONNX).

Example:
    python -m species_benchmark.export --run C:/pfspecies/runs/convnext_atto \
        --out C:/pfspecies/export/taxon_convnext_atto.pt --onnx
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from species_benchmark.model import LEVELS, TaxonClassifier

EXPORT_FORMAT = "primateface-taxon-v1"


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit a scalar temperature minimising NLL on labelled rows.

    Args:
        logits: (n, k) logits.
        labels: (n,) labels, -1 ignored.

    Returns:
        Temperature > 0 (1.0 if no labelled rows).
    """
    m = labels >= 0
    if m.sum() < 10:
        return 1.0
    lg = torch.from_numpy(logits[m]).float()
    y = torch.from_numpy(labels[m]).long()
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lg / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def pick_species_threshold(val: Dict[str, np.ndarray], temps: Dict[str, float], target_acc: float = 0.9) -> float:
    """Lowest species-confidence threshold whose answered-species accuracy ≥ target."""
    y = val["labels"][:, 2]
    m = y >= 0
    if m.sum() == 0:
        return 0.5
    p = torch.softmax(torch.from_numpy(val["species"][m]) / temps["species"], -1).numpy()
    conf, pred = p.max(1), p.argmax(1)
    for t in np.arange(0.2, 0.96, 0.05):
        sel = conf >= t
        if sel.sum() > 20 and (pred[sel] == y[m][sel]).mean() >= target_acc:
            return float(round(t, 2))
    return 0.95


@torch.no_grad()
def cpu_latency(model: TaxonClassifier, img_size: int, threads: int = 4, iters: int = 30) -> float:
    """Median CPU fp32 latency (ms) for a single crop with ``threads`` threads."""
    torch.set_num_threads(threads)
    m = model.cpu().eval()
    x = torch.randn(1, 3, img_size, img_size)
    for _ in range(5):
        m(x)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        m(x)
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--onnx", action="store_true")
    ap.add_argument("--crop-scale", type=float, default=1.25)
    args = ap.parse_args()

    ck = torch.load(args.run / "best.pt", map_location="cpu", weights_only=False)
    space = ck["label_space"]
    val = dict(np.load(args.run / "val_logits.npz", allow_pickle=True))
    temps = {lvl: fit_temperature(val[lvl], val["labels"][:, j]) for j, lvl in enumerate(LEVELS)}
    thr = {"species": pick_species_threshold(val, temps), "genus": 0.3}

    model = TaxonClassifier(ck["backbone"], {lvl: len(space[lvl]) for lvl in LEVELS}, pretrained=False)
    model.load_state_dict(ck["state_dict"])
    lat = cpu_latency(model, ck["img_size"])

    state = {k: v for k, v in ck["state_dict"].items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": EXPORT_FORMAT,
            "backbone": ck["backbone"],
            "img_size": ck["img_size"],
            "mean": list(ck["data_config"]["mean"]),
            "std": list(ck["data_config"]["std"]),
            "cosine": ck["cosine"],
            "crop_scale": args.crop_scale,
            "temperature": temps,
            "thresholds": thr,
            "label_space": space,
            "state_dict": state,
        },
        args.out,
    )
    info = {"temperature": temps, "thresholds": thr, "cpu_latency_ms_4threads": lat, "export": str(args.out)}
    if args.onnx:
        onnx_path = args.out.with_suffix(".onnx")

        class _Flat(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                o = self.m(x)
                return o["family"], o["genus"], o["species"]

        torch.onnx.export(
            _Flat(model.cpu().eval()),
            torch.randn(1, 3, ck["img_size"], ck["img_size"]),
            str(onnx_path),
            input_names=["image"],
            output_names=list(LEVELS),
            dynamic_axes={"image": {0: "batch"}},
            opset_version=17,
        )
        try:
            import onnxruntime as ort

            so = ort.SessionOptions()
            so.intra_op_num_threads = 4
            sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
            x = np.random.randn(1, 3, ck["img_size"], ck["img_size"]).astype(np.float32)
            for _ in range(5):
                sess.run(None, {"image": x})
            ts = []
            for _ in range(30):
                t0 = time.perf_counter()
                sess.run(None, {"image": x})
                ts.append((time.perf_counter() - t0) * 1000)
            info["onnx_cpu_latency_ms_4threads"] = float(np.median(ts))
        except Exception as e:  # noqa: BLE001
            info["onnx_error"] = repr(e)[:300]
    (args.run / "export_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
