"""Label-noise cleaning with out-of-sample model predictions (confident-learning style).

Evidence sources (permissively licensed models only):
    * Out-of-fold linear probes on frozen PE-Core-L features. Folds are grouped by the
      leakage group; probes are fit on verified-label rows only, so Bing / casual iNat
      rows are always scored by models that never saw them.
    * The MVP BioCLIP 2 fine-tune, used only on crops outside its own training set.

Rules (genus level, training split only):
    * Unverified labels (Bing folders, casual / needs-ID iNat): drop when the ensemble
      gives the labelled genus p < ``drop_p`` while another genus has p >= ``drop_conf``,
      or when the labelled genus is implausible (p < ``drop_implausible``).
    * Verified labels: drop from train on a strong genus conflict
      (p < ``verified_p`` and other genus >= ``verified_conf``); species-level conflicts are
      only flagged (sister species are often genuinely ambiguous).
    * Val/test labels are never changed; their flags are reported as a label-noise estimate.

Example:
    python -m species_benchmark.clean --data-dir C:/pfspecies/data \
        --features C:/pfspecies/data/features/vit_pe_core_large_patch14_336.fb.npz \
        --extra-logits C:/pfspecies/data/clean_mvp_bioclip2_logits.npz \
        --extra-space C:/pfspecies/data_mvp/label_space.json \
        --extra-train-ids C:/pfspecies/data_mvp/labeled.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from species_benchmark.gallery import grid
from species_benchmark.labels import EVAL_SOURCES, GENUS_ONLY_SOURCES
from species_benchmark.metrics import softmax


def _fold_of(group: str, k: int) -> int:
    return int(hashlib.md5(str(group).encode()).hexdigest()[:8], 16) % k


def _fit_heads(X: torch.Tensor, y: torch.Tensor, n_cls: int, epochs: int = 30) -> torch.nn.Linear:
    head = torch.nn.Linear(X.shape[1], n_cls).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-4)
    counts = torch.bincount(y, minlength=n_cls).float()
    w = (counts[y] + 1) ** -0.5
    for _ in range(epochs):
        perm = torch.multinomial(w, len(y), replacement=True)
        for i in range(0, len(perm), 4096):
            idx = perm[i : i + 4096]
            loss = F.cross_entropy(head(X[idx] * 20.0), y[idx], label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return head


@torch.no_grad()
def _predict(head: torch.nn.Linear, X: torch.Tensor) -> np.ndarray:
    return torch.softmax(head(X * 20.0), -1).cpu().numpy()


def oof_probs(df: pd.DataFrame, feats: np.ndarray, level: str, n_cls: int, k: int = 5) -> np.ndarray:
    """Out-of-sample class probabilities for every row.

    Verified rows of train/val get out-of-fold predictions; every other row gets the
    mean of the ``k`` fold models (none of which saw it).

    Args:
        df: Labeled frame (row-aligned with ``feats``).
        feats: (n, d) L2-normalised features.
        level: ``"genus"`` or ``"species"``.
        n_cls: Number of classes.
        k: Folds.

    Returns:
        (n, n_cls) probabilities.
    """
    X = torch.from_numpy(feats.astype(np.float32)).cuda()
    y_all = df[f"{level}_idx"].to_numpy()
    fit_mask = df["split"].isin(["train", "val"]).to_numpy() & df["label_source"].isin(EVAL_SOURCES).to_numpy() & (y_all >= 0)
    folds = df["group"].map(lambda g: _fold_of(g, k)).to_numpy()
    out = np.zeros((len(df), n_cls), dtype=np.float32)
    rest = ~fit_mask
    for f in range(k):
        tr = np.where(fit_mask & (folds != f))[0]
        head = _fit_heads(X[tr], torch.from_numpy(y_all[tr]).cuda(), n_cls)
        held = np.where(fit_mask & (folds == f))[0]
        out[held] = _predict(head, X[held])
        out[rest] += _predict(head, X[rest]) / k
    return out


def map_logits(path: Path, space_src: Dict, space_dst: Dict, df: pd.DataFrame, exclude_ids: set, level: str) -> np.ndarray:
    """Probabilities from another model, mapped into ``space_dst`` by class name.

    Rows whose crop is in ``exclude_ids`` (the model's own training set) or classes the
    model lacks get NaN.
    """
    z = np.load(path, allow_pickle=True)
    idx = pd.Series(np.arange(len(z["crop_id"])), index=z["crop_id"]).reindex(df["crop_id"].to_numpy()).to_numpy()
    p_src = softmax(z[level].astype(np.float32))
    col = {c: i for i, c in enumerate(space_src[level])}
    out = np.full((len(df), len(space_dst[level])), np.nan, dtype=np.float32)
    ok = ~np.isnan(idx.astype(float)) & ~df["crop_id"].isin(exclude_ids).to_numpy()
    rows = idx[ok].astype(int)
    for j, c in enumerate(space_dst[level]):
        if c in col:
            out[ok, j] = p_src[rows, col[c]]
    return out


def ensemble(probs: List[np.ndarray]) -> np.ndarray:
    """Average probability arrays, ignoring NaN rows per source; renormalise."""
    stack = np.stack(probs)
    valid = ~np.isnan(stack).any(-1, keepdims=True)
    s = np.where(valid, stack, 0).sum(0) / np.maximum(valid.sum(0), 1)
    return s / np.maximum(s.sum(-1, keepdims=True), 1e-8)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("C:/pfspecies/data"))
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--extra-logits", type=Path, default=None)
    ap.add_argument("--extra-space", type=Path, default=None)
    ap.add_argument("--extra-train-ids", type=Path, default=None, help="labeled.parquet of the extra model's data")
    ap.add_argument("--drop-p", type=float, default=0.10)
    ap.add_argument("--drop-conf", type=float, default=0.60)
    ap.add_argument("--drop-implausible", type=float, default=0.02, help="drop unverified if p(label) below this")
    ap.add_argument("--verified-p", type=float, default=0.03)
    ap.add_argument("--verified-conf", type=float, default=0.85)
    ap.add_argument("--report-dir", type=Path, default=Path("C:/pfspecies/report/cleaning"))
    args = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)

    d = args.data_dir
    raw_path = d / "labeled_raw.parquet"
    if not raw_path.exists():
        shutil.copy2(d / "labeled.parquet", raw_path)
    df = pd.read_parquet(raw_path)
    space = json.loads((d / "label_space.json").read_text())
    z = np.load(args.features, allow_pickle=True)
    assert (z["crop_id"] == df["crop_id"].to_numpy()).all(), "features not aligned with labeled_raw"

    pg = [oof_probs(df, z["feats"], "genus", len(space["genus"]))]
    ps = [oof_probs(df, z["feats"], "species", len(space["species"]))]
    if args.extra_logits:
        sp_src = json.loads(args.extra_space.read_text())
        mvp = pd.read_parquet(args.extra_train_ids, columns=["crop_id", "split"])
        excl = set(mvp.loc[mvp["split"] == "train", "crop_id"])
        pg.append(map_logits(args.extra_logits, sp_src, space, df, excl, "genus"))
        ps.append(map_logits(args.extra_logits, sp_src, space, df, excl, "species"))
    PG, PS = ensemble(pg), ensemble(ps)

    g = df["genus_idx"].to_numpy()
    s = df["species_idx"].to_numpy()
    df["p_genus_label"] = PG[np.arange(len(df)), g]
    df["pred_genus"] = [space["genus"][i] for i in PG.argmax(1)]
    df["pred_genus_conf"] = PG.max(1)
    df["p_species_label"] = np.where(s >= 0, PS[np.arange(len(df)), np.maximum(s, 0)], np.nan)
    df["pred_species"] = [space["species"][i] for i in PS.argmax(1)]
    df["pred_species_conf"] = PS.max(1)

    unverified = df["label_source"].isin(GENUS_ONLY_SOURCES)
    genus_conflict = (df["pred_genus"] != df["genus"])
    drop_unv = unverified & genus_conflict & (
        ((df["p_genus_label"] < args.drop_p) & (df["pred_genus_conf"] >= args.drop_conf))
        | (df["p_genus_label"] < args.drop_implausible)
    )
    drop_ver = ~unverified & genus_conflict & (df["p_genus_label"] < args.verified_p) & (df["pred_genus_conf"] >= args.verified_conf)
    sp_flag = (s >= 0) & (df["pred_species"] != df["species"]) & (df["p_species_label"] < 0.02) & (df["pred_species_conf"] >= 0.8)
    df["noise_flag"] = np.select([drop_unv | drop_ver, sp_flag], ["genus_conflict", "species_conflict"], "")

    is_train = df["split"] == "train"
    dropped = is_train & (drop_unv | drop_ver)
    df.loc[dropped, "split"] = "dropped_noisy"
    df.to_parquet(d / "labeled.parquet", index=False)

    rep = {"thresholds": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}
    by_src = df.groupby("label_source").agg(
        n=("crop_id", "size"),
        genus_conflict=("noise_flag", lambda x: float((x == "genus_conflict").mean())),
        species_conflict=("noise_flag", lambda x: float((x == "species_conflict").mean())),
    )
    rep["by_label_source"] = by_src.to_dict(orient="index")
    rep["dropped_from_train"] = int(dropped.sum())
    rep["dropped_by_source"] = df.loc[dropped, "label_source"].value_counts().to_dict()
    for sp in ("test", "val", "test_bing"):
        m = df["split"] == sp
        rep[f"{sp}_flag_rate"] = df.loc[m, "noise_flag"].value_counts(normalize=True).to_dict()
    bing_m = df["label_source"] == "bing_folder"
    top = (
        df[bing_m & (df["noise_flag"] == "genus_conflict")].groupby(["genus", "pred_genus"]).size().sort_values(ascending=False).head(25)
    )
    rep["bing_top_relabels"] = {f"{g}->{pg}": int(n) for (g, pg), n in top.items()}
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "clean_report.json").write_text(json.dumps(rep, indent=1, default=str))

    rng = np.random.default_rng(0)
    for name, mask in {
        "dropped_bing": dropped & bing_m,
        "dropped_inat_unverified": dropped & df["label_source"].isin(["inat_casual", "inat_needs_id"]),
        "dropped_verified": dropped & ~unverified,
        "flagged_test": (df["split"] == "test") & (df["noise_flag"] == "genus_conflict"),
        "kept_bing_random": (df["split"] == "train") & bing_m,
    }.items():
        sub = df[mask]
        if len(sub) == 0:
            continue
        sub = sub.iloc[rng.choice(len(sub), min(48, len(sub)), replace=False)].copy()
        sub["gt"], sub["pred"], sub["conf"] = sub["genus"], sub["pred_genus"], sub["pred_genus_conf"]
        sub["correct"] = sub["gt"] == sub["pred"]
        grid(sub, f"{name} (label=GT, model=P)", args.report_dir / name)
    print(json.dumps({k: rep[k] for k in ("dropped_from_train", "dropped_by_source", "test_flag_rate")}, indent=1))


if __name__ == "__main__":
    main()
