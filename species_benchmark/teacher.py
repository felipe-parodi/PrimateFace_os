"""Teacher logits, pseudo-label pool, and teacher-consensus test set.

Steps:
    1. ``logits``: run a trained run's best checkpoint (with horizontal-flip TTA) over
       labeled crops and/or the unlabeled YouTube pool; save per-level logits.
    2. ``pool``: split the YouTube pool by video into a KD-training part and a held-out
       part. The held-out part becomes ``test_yt_consensus`` where two independent
       teachers agree on genus with high confidence (no human labels needed).

Example:
    python -m species_benchmark.teacher logits --run C:/pfspecies/runs/bioclip2_ft \
        --data-dir C:/pfspecies/data --out C:/pfspecies/data/teacher_bioclip2.npz --include-yt
    python -m species_benchmark.teacher pool --teachers A.npz B.npz --data-dir C:/pfspecies/data
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from species_benchmark.metrics import softmax
from species_benchmark.model import LEVELS, TaxonClassifier
from species_benchmark.train import CropDataset, make_transforms

YT_MANIFEST = "yt_pool_primates.parquet"  # written by finalize.py (humans removed)


def _with_dummy_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for lvl in LEVELS:
        if f"{lvl}_idx" not in df.columns:
            df[f"{lvl}_idx"] = -1
    return df


@torch.no_grad()
def run_logits(run: Path, df: pd.DataFrame, bs: int = 256, workers: int = 12) -> Dict[str, np.ndarray]:
    """Flip-TTA logits of a trained run over ``df`` crops.

    Args:
        run: Run directory with ``best.pt``.
        df: Rows with ``crop_path``.
        bs: Batch size.
        workers: DataLoader workers.

    Returns:
        Dict level -> (n, k) float16 logits.
    """
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    space = ck["label_space"]
    model = TaxonClassifier(
        ck["backbone"], {lvl: len(space[lvl]) for lvl in LEVELS}, pretrained=False, img_size=ck["img_size"]
    )
    model.load_state_dict(ck["state_dict"])
    model = model.cuda().eval()
    _, eval_tf = make_transforms(ck["data_config"], ck["img_size"])
    dl = DataLoader(CropDataset(_with_dummy_labels(df), eval_tf), batch_size=bs, num_workers=workers, pin_memory=True)
    outs: Dict[str, List[np.ndarray]] = {lvl: [] for lvl in LEVELS}
    for b in dl:
        x = b["image"].cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a = model(x)
            f = model(torch.flip(x, dims=[3]))
        for lvl in LEVELS:
            outs[lvl].append(((a[lvl].float() + f[lvl].float()) / 2).cpu().numpy().astype(np.float16))
    return {lvl: np.concatenate(v) for lvl, v in outs.items()}


def _video_frac(v: str) -> float:
    return int(hashlib.md5(str(v).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def build_pool(teacher_paths: List[Path], data_dir: Path, test_frac: float = 0.15, conf: float = 0.8) -> Dict:
    """Split the YouTube pool and derive teacher-consensus labels.

    Args:
        teacher_paths: Two or more teacher npz files covering the YouTube crops
            (first teacher = primary KD teacher).
        data_dir: Data directory with ``label_space.json`` and the pool manifest.
        test_frac: Fraction of videos held out for the consensus test set.
        conf: Minimum genus probability required from every teacher.

    Returns:
        Summary dict. Writes ``yt_kd_train.parquet`` and ``yt_consensus_test.parquet``.
    """
    space = json.loads((data_dir / "label_space.json").read_text())
    yt = pd.read_parquet(data_dir / YT_MANIFEST)
    yt = yt[yt["crop_path"].map(lambda p: Path(p).exists())].reset_index(drop=True)
    teachers = [np.load(p, allow_pickle=True) for p in teacher_paths]
    probs = []
    for z in teachers:
        idx = pd.Series(np.arange(len(z["crop_id"])), index=z["crop_id"])
        rows = idx.reindex(yt["crop_id"].to_numpy()).to_numpy()
        assert not np.isnan(rows.astype(float)).any(), "teacher missing YouTube crops"
        rows = rows.astype(int)
        probs.append({lvl: softmax(z[lvl][rows].astype(np.float32)) for lvl in LEVELS})
    g_pred = [p["genus"].argmax(1) for p in probs]
    g_conf = [p["genus"].max(1) for p in probs]
    agree = np.all([g == g_pred[0] for g in g_pred], axis=0) & np.all([c >= conf for c in g_conf], axis=0)
    s_pred = [p["species"].argmax(1) for p in probs]
    s_conf = [p["species"].max(1) for p in probs]
    s_agree = agree & np.all([s == s_pred[0] for s in s_pred], axis=0) & np.all([c >= conf for c in s_conf], axis=0)

    yt["genus"] = [space["genus"][i] for i in g_pred[0]]
    yt["family"] = yt["genus"].map(space["genus_to_family"])
    yt["species"] = [space["species"][i] if ok else None for i, ok in zip(s_pred[0], s_agree)]
    is_test = yt["video_id"].map(_video_frac) < test_frac

    test = yt[is_test & agree].copy()
    test["label_source"] = "teacher_consensus"
    test["genus_idx"] = g_pred[0][(is_test & agree).to_numpy()]
    test["family_idx"] = test["family"].map({c: i for i, c in enumerate(space["family"])}).astype(int)
    test["species_idx"] = [
        int(s) if ok else -1 for s, ok in zip(s_pred[0][(is_test & agree).to_numpy()], s_agree[(is_test & agree).to_numpy()])
    ]
    test.to_parquet(data_dir / "yt_consensus_test.parquet", index=False)

    train = yt[~is_test].copy()
    train["label_source"] = "teacher_soft"
    for lvl in LEVELS:
        train[f"{lvl}_idx"] = -1  # KD targets only, no hard labels
    train["species"] = None
    train["split"] = "train"
    train.to_parquet(data_dir / "yt_kd_train.parquet", index=False)
    summary = {
        "yt_crops": len(yt),
        "videos": int(yt["video_id"].nunique()),
        "test_videos": int(yt.loc[is_test, "video_id"].nunique()),
        "consensus_test_crops": len(test),
        "consensus_test_with_species": int((test["species_idx"] >= 0).sum()),
        "consensus_rate_test_videos": float(agree[is_test.to_numpy()].mean()),
        "kd_train_crops": len(train),
        "consensus_test_genera": test["genus"].value_counts().head(25).to_dict(),
    }
    (data_dir / "yt_pool_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["logits", "pool"])
    ap.add_argument("--run", type=Path)
    ap.add_argument("--data-dir", type=Path, default=Path("C:/pfspecies/data"))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--include-yt", action="store_true")
    ap.add_argument("--teachers", type=Path, nargs="+")
    ap.add_argument("--conf", type=float, default=0.8)
    args = ap.parse_args()
    if args.cmd == "logits":
        df = pd.read_parquet(args.data_dir / "labeled.parquet")[["crop_id", "crop_path"]]
        if args.include_yt and (args.data_dir / YT_MANIFEST).exists():
            yt = pd.read_parquet(args.data_dir / YT_MANIFEST)[["crop_id", "crop_path"]]
            yt = yt[yt["crop_path"].map(lambda p: Path(p).exists())]
            df = pd.concat([df, yt], ignore_index=True).drop_duplicates("crop_id")
        lg = run_logits(args.run, df.reset_index(drop=True))
        np.savez(args.out, crop_id=df["crop_id"].to_numpy(), **lg)
        print(f"wrote {args.out} ({len(df)} crops)")
    else:
        print(json.dumps(build_pool(args.teachers, args.data_dir, conf=args.conf), indent=1))


if __name__ == "__main__":
    main()
