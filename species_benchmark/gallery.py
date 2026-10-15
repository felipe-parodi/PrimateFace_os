"""Review grids: test crops with ground truth vs prediction.

Example:
    python -m species_benchmark.gallery --run C:/pfspecies/runs/convnext_atto --out C:/pfspecies/report/gallery
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from species_benchmark.metrics import softmax  # noqa: E402


def _short(name: str) -> str:
    parts = name.split()
    return f"{parts[0][0]}. {parts[1]}" if len(parts) == 2 else name


def grid(rows: pd.DataFrame, title: str, out: Path, ncol: int = 8) -> None:
    """Draw a grid of crops annotated with GT and prediction.

    Args:
        rows: Rows with crop_path, gt, pred, conf, correct.
        title: Figure title.
        out: Output path stem (PNG written).
        ncol: Columns.
    """
    n = len(rows)
    if n == 0:
        return
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.9, nrow * 2.25))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, r) in zip(axes.flat, rows.iterrows()):
        ax.imshow(Image.open(r["crop_path"]).convert("RGB"))
        color = "#009E73" if r["correct"] else "#D55E00"
        ax.set_title(f"GT {_short(r['gt'])}\nP {_short(r['pred'])} {r['conf']:.2f}", fontsize=7, color=color)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(color)
            s.set_linewidth(2)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=110)
    plt.close(fig)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("C:/pfspecies/data"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=48)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    space = json.loads((args.data_dir / "label_space.json").read_text())
    df = pd.read_parquet(args.data_dir / "labeled.parquet").set_index("crop_id")
    z = np.load(args.run / "test_logits.npz", allow_pickle=True)
    meta = df.loc[z["crop_id"]].reset_index()
    rng = np.random.default_rng(0)
    for lvl in ("species", "genus"):
        lab = z["labels"][:, 2 if lvl == "species" else 1]
        p = softmax(z[lvl])
        m = lab >= 0
        d = meta[m].copy()
        d["gt"] = [space[lvl][i] for i in lab[m]]
        d["pred"] = [space[lvl][i] for i in p[m].argmax(1)]
        d["conf"] = p[m].max(1)
        d["correct"] = d["gt"] == d["pred"]
        sample = d.iloc[rng.choice(len(d), min(args.n, len(d)), replace=False)]
        grid(sample, f"{args.run.name}: random test {lvl} (acc {d['correct'].mean():.3f})", args.out / f"{args.run.name}_{lvl}_random")
        errs = d[~d["correct"]].sort_values("conf", ascending=False).head(args.n)
        grid(errs, f"{args.run.name}: most confident {lvl} errors", args.out / f"{args.run.name}_{lvl}_errors")
        if lvl == "genus":
            per = d.groupby("gt")["correct"].agg(["mean", "size"]).sort_values("mean")
            per.to_csv(args.out / f"{args.run.name}_per_genus_acc.csv")
    print(f"wrote grids to {args.out}")


if __name__ == "__main__":
    main()
