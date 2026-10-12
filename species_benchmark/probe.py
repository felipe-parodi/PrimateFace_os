"""Frozen-feature baselines: embedding extraction, linear probes, BioCLIP zero-shot.

Example:
    python -m species_benchmark.probe extract --backbone bioclip2
    python -m species_benchmark.probe fit --backbone bioclip2 --out C:/pfspecies/runs/probe_bioclip2
    python -m species_benchmark.probe zeroshot --backbone bioclip2 --out C:/pfspecies/runs/zs_bioclip2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from species_benchmark.model import CLIP_BACKBONES, LEVELS, TaxonClassifier, build_backbone, taxon_prompts
from species_benchmark.train import DATA_DIR, CropDataset, evaluate, make_transforms


def _safe(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


@torch.no_grad()
def extract(backbone: str, data_dir: Path, img_size: int = 224, workers: int = 12) -> Path:
    """Embed every labeled crop with a frozen backbone (L2-normalised, fp16).

    Args:
        backbone: timm name or CLIP key.
        data_dir: Directory with ``labeled.parquet``.
        img_size: Input size.
        workers: DataLoader workers.

    Returns:
        Path of the saved ``.npz`` (crop_id, feats).
    """
    df = pd.read_parquet(data_dir / "labeled.parquet")
    net, _, cfg = build_backbone(backbone)
    net = net.cuda().eval()
    _, eval_tf = make_transforms(cfg, img_size)
    dl = DataLoader(CropDataset(df, eval_tf), batch_size=256, num_workers=workers, pin_memory=True)
    feats = []
    for b in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = net(b["image"].cuda(non_blocking=True))
        feats.append(F.normalize(f.float(), dim=-1).half().cpu().numpy())
    out = data_dir / "features" / f"{_safe(backbone)}.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez(out, crop_id=df["crop_id"].to_numpy(), feats=np.concatenate(feats))
    return out


def fit_probe(backbone: str, data_dir: Path, out: Path, epochs: int = 60) -> Dict:
    """Train linear heads on frozen features (full-batch-ish AdamW on GPU).

    Args:
        backbone: Backbone whose features were extracted.
        data_dir: Data directory.
        out: Output directory for metrics.
        epochs: Training epochs.

    Returns:
        Metrics dict.
    """
    space = json.loads((data_dir / "label_space.json").read_text())
    df = pd.read_parquet(data_dir / "labeled.parquet")
    z = np.load(data_dir / "features" / f"{_safe(backbone)}.npz", allow_pickle=True)
    assert (z["crop_id"] == df["crop_id"].to_numpy()).all()
    X = torch.from_numpy(z["feats"].astype(np.float32)).cuda()
    Y = torch.from_numpy(np.stack([df[f"{lvl}_idx"].to_numpy() for lvl in LEVELS], 1)).cuda()
    split = df["split"].to_numpy()
    tr = torch.from_numpy(np.where(split == "train")[0]).cuda()
    heads = torch.nn.ModuleDict({lvl: torch.nn.Linear(X.shape[1], len(space[lvl])) for lvl in LEVELS}).cuda()
    opt = torch.optim.AdamW(heads.parameters(), lr=2e-3, weight_decay=1e-4)
    key = df["species"].fillna(df["genus"])
    w = torch.from_numpy((key.map(key.value_counts()).to_numpy(float) ** -0.5)).float().cuda()
    best, best_score, log = None, -1, []
    for ep in range(epochs):
        perm = tr[torch.multinomial(w[tr], len(tr), replacement=True)]
        for i in range(0, len(perm), 4096):
            idx = perm[i : i + 4096]
            x = X[idx] * 20.0
            loss = sum(
                F.cross_entropy(heads[lvl](x), Y[idx, j], ignore_index=-1, label_smoothing=0.1)
                for j, lvl in enumerate(LEVELS)
                if (Y[idx, j] >= 0).any()
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 5 == 4 or ep == epochs - 1:
            vm = _eval_rows(heads, X, Y, np.where(split == "val")[0], space, df)
            s = vm["genus"].get("macro_f1", 0) + vm["species"].get("macro_f1", 0)
            log.append({"epoch": ep, "val_genus_f1": vm["genus"].get("macro_f1"), "val_species_f1": vm["species"].get("macro_f1")})
            if s > best_score:
                best_score, best = s, {k: v.clone() for k, v in heads.state_dict().items()}
    heads.load_state_dict(best)
    res = {"backbone": backbone, "kind": "linear_probe", "log": log}
    for name in ("val", "test", "test_bing"):
        rows = np.where(split == name)[0]
        if len(rows):
            res[name] = _eval_rows(heads, X, Y, rows, space, df)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    return res


@torch.no_grad()
def _eval_rows(heads, X, Y, rows, space, df) -> Dict:
    r = torch.from_numpy(rows).cuda()
    pred = {lvl: heads[lvl](X[r] * 20.0).float().cpu().numpy() for lvl in LEVELS}
    pred["labels"] = Y[r].cpu().numpy()
    tr = df[df["split"] == "train"]
    counts = {lvl: np.bincount(tr.loc[tr[f"{lvl}_idx"] >= 0, f"{lvl}_idx"], minlength=len(space[lvl])) for lvl in LEVELS}
    return evaluate(pred, space, counts)


@torch.no_grad()
def zeroshot(backbone: str, data_dir: Path, out: Path) -> Dict:
    """BioCLIP zero-shot using taxonomic text prompts over the label space."""
    assert backbone in CLIP_BACKBONES
    space = json.loads((data_dir / "label_space.json").read_text())
    df = pd.read_parquet(data_dir / "labeled.parquet")
    z = np.load(data_dir / "features" / f"{_safe(backbone)}.npz", allow_pickle=True)
    model = TaxonClassifier(backbone, {lvl: len(space[lvl]) for lvl in LEVELS}).cuda()
    model.init_heads_from_text({lvl: space[lvl] for lvl in LEVELS}, taxon_prompts(space, space.get("common")))
    W = {lvl: F.normalize(model.heads[lvl].weight.float(), dim=-1) for lvl in LEVELS}
    X = torch.from_numpy(z["feats"].astype(np.float32)).cuda()
    Y = np.stack([df[f"{lvl}_idx"].to_numpy() for lvl in LEVELS], 1)
    split = df["split"].to_numpy()
    tr = df[df["split"] == "train"]
    counts = {lvl: np.bincount(tr.loc[tr[f"{lvl}_idx"] >= 0, f"{lvl}_idx"], minlength=len(space[lvl])) for lvl in LEVELS}
    res = {"backbone": backbone, "kind": "zero_shot"}
    for name in ("val", "test", "test_bing"):
        rows = np.where(split == name)[0]
        if len(rows):
            pred = {lvl: (100.0 * X[rows] @ W[lvl].T).cpu().numpy() for lvl in LEVELS}
            pred["labels"] = Y[rows]
            res[name] = evaluate(pred, space, counts)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    return res


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["extract", "fit", "zeroshot"])
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--img-size", type=int, default=224)
    args = ap.parse_args()
    if args.cmd == "extract":
        print(extract(args.backbone, args.data_dir, args.img_size))
    elif args.cmd == "fit":
        r = fit_probe(args.backbone, args.data_dir, args.out)
        print(json.dumps({k: r.get("test", {}).get(k, {}).get("top1") for k in LEVELS}))
    else:
        r = zeroshot(args.backbone, args.data_dir, args.out)
        print(json.dumps({k: r.get("test", {}).get(k, {}).get("top1") for k in LEVELS}))


if __name__ == "__main__":
    main()
