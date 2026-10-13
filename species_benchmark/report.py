"""Aggregate run metrics into the benchmark table and figures.

Example:
    python -m species_benchmark.report --runs C:/pfspecies/runs --out C:/pfspecies/report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

KIND_COLORS = {"zero_shot": "#999999", "linear_probe": "#E69F00", "finetune": "#0072B2", "distilled": "#009E73"}


def _get(d: Dict, *keys, default=np.nan):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def collect(runs_dir: Path) -> pd.DataFrame:
    """One row per run with headline metrics.

    Args:
        runs_dir: Directory whose subdirectories contain ``metrics.json``.

    Returns:
        DataFrame of run metrics.
    """
    rows: List[Dict] = []
    for mpath in sorted(runs_dir.glob("*/metrics.json")):
        m = json.loads(mpath.read_text())
        run = mpath.parent.name
        exp = mpath.parent / "export_info.json"
        ex = json.loads(exp.read_text()) if exp.exists() else {}
        args = m.get("args", {})
        kind = m.get("kind") or ("distilled" if args.get("teacher") not in (None, "None") else "finetune")
        row = {
            "run": run,
            "kind": kind,
            "backbone": m.get("backbone") or args.get("backbone"),
            "params_M": m.get("n_params_M", np.nan),
            "gpu_img_s": _get(m, "speed", "gpu_imgs_per_s_bs64"),
            "gpu_ms_bs1": _get(m, "speed", "gpu_latency_ms_bs1"),
            "cpu_ms": ex.get("onnx_cpu_latency_ms_4threads", ex.get("cpu_latency_ms_4threads", np.nan)),
            "train_min": m.get("train_minutes", np.nan),
        }
        for lvl in ("family", "genus", "species"):
            for met in ("top1", "top5", "macro_f1", "balanced_acc", "ece", "macro_f1_tail"):
                row[f"{lvl}_{met}"] = _get(m, "test", lvl, met)
        row["bing_genus_top1"] = _get(m, "test_bing", "genus", "top1")
        row["bing_genus_f1"] = _get(m, "test_bing", "genus", "macro_f1")
        for lvl in ("genus", "species"):
            row[f"rare_{lvl}_top1"] = _get(m, "ext_rare_species", lvl, "top1")
            row[f"yt_{lvl}_top1"] = _get(m, "test_yt_consensus", lvl, "top1")
        cov = _get(m, "test", "coverage", default=[])
        if isinstance(cov, list) and cov:
            best = [c for c in cov if c["threshold"] == 0.7] or cov[-1:]
            row["hier_acc@0.7"] = best[0]["hier_accuracy"]
            row["sp_cov@0.7"] = best[0]["species_coverage"]
        rows.append(row)
    return pd.DataFrame(rows)


def pareto_plot(df: pd.DataFrame, out: Path, metric: str = "species_macro_f1", xcol: str = "gpu_img_s") -> None:
    """Accuracy vs throughput scatter, coloured by training kind."""
    d = df.dropna(subset=[metric, xcol])
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for kind, sub in d.groupby("kind"):
        ax.scatter(sub[xcol], sub[metric], s=60, c=KIND_COLORS.get(kind, "#CC79A7"), label=kind, edgecolor="k", lw=0.5)
        for _, r in sub.iterrows():
            ax.annotate(str(r["backbone"]).split(".")[0], (r[xcol], r[metric]), fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("GPU throughput (images/s, bs=64, bf16)")
    ax.set_ylabel(metric.replace("_", " "))
    ax.legend(frameon=False)
    for ext in ("png", "svg"):
        fig.savefig(out / f"pareto_{metric}.{ext}")
    plt.close(fig)


def coverage_plot(runs_dir: Path, runs: List[str], out: Path) -> None:
    """Species coverage vs hierarchical accuracy for selected runs."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for run in runs:
        m = json.loads((runs_dir / run / "metrics.json").read_text())
        cov = _get(m, "test", "coverage", default=[])
        if not isinstance(cov, list) or not cov:
            continue
        ax.plot([c["species_coverage"] for c in cov], [c["hier_accuracy"] for c in cov], marker="o", label=run)
    ax.set_xlabel("Fraction answered at species level")
    ax.set_ylabel("Hierarchical accuracy")
    ax.legend(frameon=False, fontsize=8)
    for ext in ("png", "svg"):
        fig.savefig(out / f"coverage.{ext}")
    plt.close(fig)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=Path("C:/pfspecies/runs"))
    ap.add_argument("--out", type=Path, default=Path("C:/pfspecies/report"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = collect(args.runs).sort_values(["kind", "params_M"])
    df.to_csv(args.out / "benchmark.csv", index=False)
    cols = [
        "run", "kind", "params_M", "gpu_img_s", "cpu_ms", "family_top1", "genus_top1", "genus_macro_f1",
        "species_top1", "species_macro_f1", "species_macro_f1_tail", "species_ece", "hier_acc@0.7", "sp_cov@0.7",
        "bing_genus_top1", "rare_genus_top1", "rare_species_top1", "yt_genus_top1",
    ]
    cols = [c for c in cols if c in df.columns]
    (args.out / "benchmark.md").write_text(df[cols].to_markdown(index=False, floatfmt=".3f"))
    for metric in ("species_macro_f1", "genus_macro_f1", "species_top1"):
        pareto_plot(df, args.out, metric)
    ft = df[df["kind"].isin(["finetune", "distilled"])]["run"].tolist()
    if ft:
        coverage_plot(args.runs, ft, args.out)
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
