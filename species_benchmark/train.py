"""Train / evaluate a hierarchical taxon classifier on face crops.

Example:
    python -m species_benchmark.train --backbone convnext_atto.d2_in1k --epochs 15 \
        --out C:/pfspecies/runs/convnext_atto
    python -m species_benchmark.train --backbone bioclip2 --text-init --lr 1e-5 --epochs 8 \
        --out C:/pfspecies/runs/bioclip2_ft
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from species_benchmark.metrics import coverage_curve, level_metrics
from species_benchmark.model import LEVELS, TaxonClassifier, taxon_prompts

DATA_DIR = Path("C:/pfspecies/data")
EXTERNAL_TESTS = {"ext_rare_species": Path("C:/pfspecies/external/rare_species_test.parquet")}


def load_external(path: Path, space: Dict[str, object]) -> pd.DataFrame:
    """Load an external test parquet and index its labels into ``space``.

    Classes outside the label space get -1 at that level (counted as unlabeled).
    """
    df = pd.read_parquet(path)
    for lvl in LEVELS:
        df[f"{lvl}_idx"] = df[lvl].map({c: i for i, c in enumerate(space[lvl])}).fillna(-1).astype(int)
    return df


class CropDataset(Dataset):
    """Face crops with hierarchical integer labels and optional teacher logits."""

    def __init__(self, df: pd.DataFrame, transform, teacher: Optional[Dict[str, np.ndarray]] = None) -> None:
        self.paths = df["crop_path"].tolist()
        self.labels = np.stack([df[f"{lvl}_idx"].to_numpy() for lvl in LEVELS], 1).astype(np.int64)
        self.transform = transform
        self.teacher = teacher

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        item = {"image": self.transform(img), "labels": torch.from_numpy(self.labels[i])}
        if self.teacher is not None:
            for lvl in LEVELS:
                if lvl in self.teacher:
                    item[f"t_{lvl}"] = torch.from_numpy(self.teacher[lvl][i])
        return item


def make_transforms(cfg: Dict[str, object], img_size: int):
    """Train/eval transforms built with timm.

    Args:
        cfg: Data config with mean/std.
        img_size: Square input size.

    Returns:
        (train_transform, eval_transform)
    """
    from timm.data import create_transform

    common = {"input_size": (3, img_size, img_size), "mean": cfg["mean"], "std": cfg["std"]}
    train_tf = create_transform(
        **common,
        is_training=True,
        scale=(0.5, 1.0),
        ratio=(0.85, 1.18),
        hflip=0.5,
        auto_augment="rand-m7-mstd0.5-inc1",
        re_prob=0.1,
        interpolation="bicubic",
    )
    eval_tf = create_transform(**common, is_training=False, crop_pct=1.0, interpolation="bicubic")
    return train_tf, eval_tf


def balanced_sampler(df: pd.DataFrame, power: float = 0.5) -> WeightedRandomSampler:
    """Sample crops with weight count(genus)^-power (dampened class balancing).

    Args:
        df: Training rows.
        power: 0 = natural distribution, 1 = fully balanced.

    Returns:
        Sampler drawing ``len(df)`` crops per epoch.
    """
    key = df["species"].fillna(df["genus"])
    w = key.map(key.value_counts()).to_numpy(dtype=float) ** (-power)
    return WeightedRandomSampler(torch.as_tensor(w), num_samples=len(df), replacement=True)


def hier_loss(
    logits: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    smoothing: float,
    kd_alpha: float,
    kd_t: float,
    level_w: Dict[str, float],
) -> torch.Tensor:
    """Sum of per-level CE (ignore -1) plus optional KL to teacher logits."""
    total = 0.0
    for j, lvl in enumerate(LEVELS):
        y = labels[:, j]
        lg = logits[lvl].float()
        if (y >= 0).any():
            ce = F.cross_entropy(lg, y, ignore_index=-1, label_smoothing=smoothing)
            total = total + level_w[lvl] * (1 - kd_alpha if f"t_{lvl}" in batch else 1.0) * ce
        if f"t_{lvl}" in batch and kd_alpha > 0:
            t = batch[f"t_{lvl}"].float()
            kl = F.kl_div(F.log_softmax(lg / kd_t, -1), F.softmax(t / kd_t, -1), reduction="batchmean")
            total = total + level_w[lvl] * kd_alpha * kd_t**2 * kl
    return total


@torch.no_grad()
def predict(model: TaxonClassifier, loader: DataLoader, device: str) -> Dict[str, np.ndarray]:
    """Run the model over a loader and return stacked logits and labels."""
    model.eval()
    outs = {lvl: [] for lvl in LEVELS}
    labels = []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(x)
        for lvl in LEVELS:
            outs[lvl].append(lg[lvl].float().cpu().numpy())
        labels.append(batch["labels"].numpy())
    res = {lvl: np.concatenate(v) for lvl, v in outs.items()}
    res["labels"] = np.concatenate(labels)
    return res


def evaluate(pred: Dict[str, np.ndarray], space: Dict[str, object], train_counts: Dict[str, np.ndarray]) -> Dict:
    """Per-level metrics plus abstention coverage curve."""
    out = {}
    for j, lvl in enumerate(LEVELS):
        out[lvl] = level_metrics(pred[lvl], pred["labels"][:, j], train_counts.get(lvl))
    g_index = {g: i for i, g in enumerate(space["genus"])}
    s2g = np.array([g_index[space["species_to_genus"][s]] for s in space["species"]])
    if (pred["labels"][:, 2] >= 0).any():
        out["coverage"] = coverage_curve(pred["genus"], pred["species"], pred["labels"][:, 1], pred["labels"][:, 2], s2g)
    return out


@torch.no_grad()
def throughput(model: TaxonClassifier, img_size: int, device: str, bs: int = 64, iters: int = 30) -> Dict[str, float]:
    """GPU images/s (bf16, batch ``bs``) and single-image GPU latency in ms."""
    model.eval()
    x = torch.randn(bs, 3, img_size, img_size, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(5):
            model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        torch.cuda.synchronize()
        ips = bs * iters / (time.perf_counter() - t0)
        x1 = x[:1]
        for _ in range(5):
            model(x1)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x1)
        torch.cuda.synchronize()
        lat = (time.perf_counter() - t0) / iters * 1000
    return {"gpu_imgs_per_s_bs64": ips, "gpu_latency_ms_bs1": lat}


def load_teacher(path: Path, df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Align teacher logits (npz with crop_id + per-level arrays) to ``df`` rows."""
    z = np.load(path, allow_pickle=True)
    idx = pd.Series(np.arange(len(z["crop_id"])), index=z["crop_id"])
    rows = idx.reindex(df["crop_id"].to_numpy()).to_numpy()
    if np.isnan(rows.astype(float)).any():
        raise ValueError("teacher logits missing for some crops")
    rows = rows.astype(int)
    return {lvl: z[lvl][rows].astype(np.float32) for lvl in LEVELS if lvl in z}


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--labeled", default="labeled.parquet")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--head-lr-mult", type=float, default=10.0)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--balance-power", type=float, default=0.5)
    ap.add_argument("--text-init", action="store_true", help="init CLIP heads from BioCLIP text embeddings")
    ap.add_argument("--teacher", type=Path, default=None, help="npz of teacher logits for KD")
    ap.add_argument("--kd-alpha", type=float, default=0.7)
    ap.add_argument("--kd-t", type=float, default=2.0)
    ap.add_argument("--extra-train", type=Path, default=None, help="parquet of pseudo-labelled train rows")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-train", type=int, default=0, help="subsample train (smoke tests)")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    args.out.mkdir(parents=True, exist_ok=True)
    space = json.loads((args.data_dir / "label_space.json").read_text())
    df = pd.read_parquet(args.data_dir / args.labeled)
    train_df = df[df["split"] == "train"]
    if args.extra_train is not None:
        train_df = pd.concat([train_df, pd.read_parquet(args.extra_train)], ignore_index=True)
    if args.max_train:
        train_df = train_df.sample(min(args.max_train, len(train_df)), random_state=0)
    val_df = df[df["split"] == "val"]
    tests = {s: df[df["split"] == s] for s in ("test", "test_bing") if (df["split"] == s).any()}
    tests.update({k: load_external(p, space) for k, p in EXTERNAL_TESTS.items() if p.exists()})
    if (args.data_dir / "yt_consensus_test.parquet").exists():
        tests["test_yt_consensus"] = pd.read_parquet(args.data_dir / "yt_consensus_test.parquet")

    n_classes = {lvl: len(space[lvl]) for lvl in LEVELS}
    model = TaxonClassifier(args.backbone, n_classes).to(device)
    if args.text_init:
        model.init_heads_from_text({lvl: space[lvl] for lvl in LEVELS}, taxon_prompts(space, space.get("common")))
    if args.grad_ckpt and hasattr(model.backbone, "set_grad_checkpointing"):
        model.backbone.set_grad_checkpointing(True)
    elif args.grad_ckpt and hasattr(model.backbone, "visual"):
        model.backbone.visual.set_grad_checkpointing(True)

    train_tf, eval_tf = make_transforms(model.data_config, args.img_size)
    teacher = load_teacher(args.teacher, train_df) if args.teacher else None
    dl_kw = {"num_workers": args.workers, "pin_memory": True, "persistent_workers": args.workers > 0}
    train_dl = DataLoader(
        CropDataset(train_df, train_tf, teacher),
        batch_size=args.bs,
        sampler=balanced_sampler(train_df, args.balance_power),
        drop_last=True,
        **dl_kw,
    )
    val_dl = DataLoader(CropDataset(val_df, eval_tf), batch_size=256, **dl_kw)

    train_counts = {
        lvl: np.bincount(train_df.loc[train_df[f"{lvl}_idx"] >= 0, f"{lvl}_idx"], minlength=n_classes[lvl])
        for lvl in LEVELS
    }
    head_params = list(model.heads.parameters()) + ([model.logit_scale] if model.logit_scale is not None else [])
    opt = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.lr},
            {"params": head_params, "lr": args.lr * args.head_lr_mult, "weight_decay": 0.0},
        ],
        weight_decay=args.wd,
    )
    steps = args.epochs * len(train_dl)
    warm = min(len(train_dl), steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))
    )
    level_w = {"family": 0.5, "genus": 1.0, "species": 1.0}

    log = []
    best, best_score = None, -1.0
    t_start = time.time()
    for ep in range(args.epochs):
        model.train()
        tl, nb, t0 = 0.0, 0, time.time()
        for batch in train_dl:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
            tb = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k.startswith("t_")}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(x)
            loss = hier_loss(lg, y, tb, args.smoothing, args.kd_alpha if teacher else 0.0, args.kd_t, level_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tl += loss.item()
            nb += 1
        vm = evaluate(predict(model, val_dl, device), space, train_counts)
        score = vm["genus"].get("macro_f1", 0) + vm["species"].get("macro_f1", 0)
        rec = {
            "epoch": ep,
            "train_loss": tl / max(nb, 1),
            "val_genus_top1": vm["genus"].get("top1"),
            "val_genus_f1": vm["genus"].get("macro_f1"),
            "val_species_top1": vm["species"].get("top1"),
            "val_species_f1": vm["species"].get("macro_f1"),
            "epoch_s": time.time() - t0,
        }
        log.append(rec)
        print(json.dumps(rec), flush=True)
        if score > best_score:
            best_score = score
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "state_dict": best,
                    "backbone": args.backbone,
                    "img_size": args.img_size,
                    "data_config": model.data_config,
                    "label_space": space,
                    "cosine": model.cosine,
                    "epoch": ep,
                },
                args.out / "best.pt",
            )

    model.load_state_dict(best)
    results = {"args": {k: str(v) for k, v in vars(args).items()}, "train_log": log, "train_minutes": (time.time() - t_start) / 60}
    results["n_params_M"] = sum(p.numel() for p in model.parameters()) / 1e6
    pv = predict(model, val_dl, device)
    results["val"] = evaluate(pv, space, train_counts)
    np.savez_compressed(args.out / "val_logits.npz", crop_id=val_df["crop_id"].to_numpy(), **pv)
    for name, tdf in tests.items():
        p = predict(model, DataLoader(CropDataset(tdf, eval_tf), batch_size=256, **dl_kw), device)
        results[name] = evaluate(p, space, train_counts)
        np.savez_compressed(args.out / f"{name}_logits.npz", crop_id=tdf["crop_id"].to_numpy(), **p)
    results["speed"] = throughput(model, args.img_size, device)
    (args.out / "metrics.json").write_text(json.dumps(results, indent=1))
    print(json.dumps({k: results[k] for k in ("test", "speed") if k in results}, indent=1)[:3000])


if __name__ == "__main__":
    main()
