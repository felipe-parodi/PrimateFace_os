"""Apply the content screen, add a *Homo sapiens* class, and build the KD pool.

Run after ``labels`` and ``clean``:
    * Training crops the screen scores as human / drawing / non-primate with high
      confidence are dropped (``dropped_content``); such held-out crops are excluded
      from evaluation (``excluded_content``) and counted in the report.
    * High-confidence human faces from YouTube frames become a ``Homo sapiens`` class
      (split by pseudo-video group), so humans in primate videos are labelled as
      humans instead of being forced into a primate genus.
    * Remaining YouTube primate faces form the unlabeled KD pool
      (``yt_pool_primates.parquet``), which ``teacher.py pool`` consumes.

Example:
    python -m species_benchmark.finalize --data-dir C:/pfspecies/data
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from species_benchmark.gallery import grid
from species_benchmark.model import LEVELS

HUMAN = {"family": "Hominidae", "genus": "Homo", "species": "Homo sapiens"}
# Per-category screen thresholds, chosen from review grids: "other" is mostly real
# primates in uninformative views and "human" has dark-face false positives, so
# evaluation exclusions are stricter than training drops.
TRAIN_DROP = {"p_human": 0.90, "p_drawing": 0.90, "p_other": 0.95}
EVAL_DROP = {"p_human": 0.98, "p_drawing": 0.95, "p_other": 0.98}


def _frac(key: str) -> float:
    return int(hashlib.md5(str(key).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("C:/pfspecies/data"))
    ap.add_argument("--human-conf", type=float, default=0.95)
    ap.add_argument("--max-humans", type=int, default=8000)
    ap.add_argument("--report-dir", type=Path, default=Path("C:/pfspecies/report/cleaning"))
    args = ap.parse_args()
    d = args.data_dir

    df = pd.read_parquet(d / "labeled.parquet")
    df = df[df["label_source"] != "yt_human_zeroshot"]  # idempotent re-runs
    space = json.loads((d / "label_space.json").read_text())
    scr = pd.read_parquet(d / "screen.parquet").set_index("crop_id")
    s = scr.reindex(df["crop_id"])
    bad_train = np.any([(s[c] >= t).to_numpy() for c, t in TRAIN_DROP.items()], axis=0)
    bad_eval = np.any([(s[c] >= t).to_numpy() for c, t in EVAL_DROP.items()], axis=0)
    df["content"] = s["category"].to_numpy()
    tr = (df["split"] == "train").to_numpy()
    ev = df["split"].isin(["val", "test", "test_bing"]).to_numpy()
    rep = {
        "train_dropped_content": int((tr & bad_train).sum()),
        "eval_excluded_content": int((ev & bad_eval).sum()),
        "train_dropped_by_category": df.loc[tr & bad_train, "content"].value_counts().to_dict(),
        "train_dropped_by_source": df.loc[tr & bad_train, "label_source"].value_counts().to_dict(),
    }
    rng = np.random.default_rng(0)
    for cat in ("human", "drawing", "other"):
        sub = df[tr & bad_train & (df["content"] == cat)]
        if len(sub):
            sub = sub.iloc[rng.choice(len(sub), min(48, len(sub)), replace=False)].copy()
            sub["gt"], sub["pred"], sub["conf"], sub["correct"] = sub["genus"], cat, 1.0, False
            grid(sub, f"train crops dropped as '{cat}' by the content screen", args.report_dir / f"content_dropped_{cat}")
    df.loc[tr & bad_train, "split"] = "dropped_content"
    df.loc[ev & bad_eval, "split"] = "excluded_content"

    # Homo sapiens from YouTube frames
    yt = pd.read_parquet(d / "yt_pool_manifest.parquet")
    ys = scr.reindex(yt["crop_id"])
    yt["p_human"] = ys["p_human"].to_numpy()
    yt["p_primate"] = ys["p_primate"].to_numpy()
    humans = yt[yt["p_human"] >= args.human_conf]
    humans = humans.sample(min(args.max_humans, len(humans)), random_state=0) if len(humans) else humans
    h = pd.DataFrame(
        {
            "crop_id": humans["crop_id"],
            "crop_path": humans["crop_path"],
            "image_path": humans["image_path"],
            "label_source": "yt_human_zeroshot",
            "source": "youtube",
            "license": "unknown_web",
            "group": "yt:" + humans["video_id"].astype(str),
            **{k: v for k, v in HUMAN.items()},
            "label_level": "species",
        }
    )
    frac = h["group"].map(_frac)
    h["split"] = np.where(frac < 0.10, "test", np.where(frac < 0.15, "val", "train"))
    df = pd.concat([df, h], ignore_index=True)

    for lvl in LEVELS:
        if HUMAN[lvl] not in space[lvl]:
            space[lvl] = sorted(space[lvl] + [HUMAN[lvl]])
    space["species_to_genus"][HUMAN["species"]] = HUMAN["genus"]
    space["genus_to_family"][HUMAN["genus"]] = HUMAN["family"]
    space.setdefault("common", {})[HUMAN["species"]] = "Human"
    for lvl in LEVELS:
        df[f"{lvl}_idx"] = df[lvl].map({c: i for i, c in enumerate(space[lvl])}).fillna(-1).astype(int)
    df.to_parquet(d / "labeled.parquet", index=False)
    (d / "label_space.json").write_text(json.dumps(space, indent=1))

    pool = yt[(yt["p_primate"] >= 0.5) & (yt["p_human"] < 0.5)]
    pool.to_parquet(d / "yt_pool_primates.parquet", index=False)
    rep.update(
        {
            "yt_humans_available": int((yt["p_human"] >= args.human_conf).sum()),
            "homo_added": len(h),
            "homo_splits": h["split"].value_counts().to_dict(),
            "yt_primate_pool": len(pool),
            "final_splits": df["split"].value_counts().to_dict(),
            "n_genus": len(space["genus"]),
            "n_species": len(space["species"]),
        }
    )
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "finalize_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
