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


SHORT = {
    "mobilenetv4_conv_small": "MNv4-S", "convnext_atto": "ConvNeXt-Atto", "tiny_vit_5m_224": "TinyViT-5M",
    "repvit_m1_1": "RepViT-M1.1", "efficientvit_b1": "EffViT-B1", "mobilenetv4_conv_medium": "MNv4-M",
    "convnext_nano": "ConvNeXt-Nano", "vit_small_patch16_224": "ViT-S", "vit_pe_core_small_patch16_384": "PE-S",
    "convnext_tiny": "ConvNeXt-T", "bioclip": "BioCLIP ViT-B", "bioclip2": "BioCLIP 2 ViT-L",
}


def short_name(backbone: str) -> str:
    """Readable label for a backbone identifier."""
    return SHORT.get(str(backbone).split(".")[0], str(backbone).split(".")[0])


def apply_speed(df: pd.DataFrame, speed_path: Path) -> pd.DataFrame:
    """Replace in-training speed numbers with the isolated ``speed.py`` measurements.

    Args:
        df: Output of :func:`collect`.
        speed_path: JSON written by ``species_benchmark.speed``.

    Returns:
        DataFrame with ``gpu_img_s`` and ``gpu_ms_bs1`` from the isolated run where available.
    """
    if not speed_path.exists():
        return df
    sp = json.loads(speed_path.read_text())
    df = df.copy()
    for col, key in (("gpu_img_s", "gpu_imgs_per_s_bs64"), ("gpu_ms_bs1", "gpu_ms_bs1")):
        iso = df["backbone"].map(lambda b, k=key: sp.get(b, {}).get(k, np.nan))
        df[col] = np.where(df["kind"].isin(["finetune", "distilled"]), iso, np.nan)
    return df


def pareto_plot(df: pd.DataFrame, out: Path, metric: str, xcol: str, xlabel: str, log: bool = True) -> None:
    """Accuracy vs speed scatter for trained models, coloured by training kind."""
    d = df[df["kind"].isin(["finetune", "distilled"])].dropna(subset=[metric, xcol])
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for kind, sub in d.groupby("kind"):
        ax.scatter(sub[xcol], sub[metric], s=60, c=KIND_COLORS.get(kind, "#CC79A7"),
                   label="distilled (KD)" if kind == "distilled" else "fine-tuned (CE)", edgecolor="k", lw=0.5)
        dy = 4 if kind == "distilled" else -10  # KD labels above, CE labels below
        for _, r in sub.iterrows():
            ax.annotate(short_name(r["backbone"]), (r[xcol], r[metric]), fontsize=7, xytext=(5, dy), textcoords="offset points")
    if log:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric.replace("_", " ").replace("top1", "top-1 (test)"))
    ax.legend(frameon=False)
    for ext in ("png", "svg"):
        fig.savefig(out / f"pareto_{metric}_{xcol}.{ext}")
    plt.close(fig)


def kd_plot(df: pd.DataFrame, out: Path) -> None:
    """Paired change from distillation (KD minus CE) per backbone."""
    ce = df[df["run"].str.startswith("ce_")].assign(key=lambda x: x["run"].str[3:])
    kd = df[df["run"].str.startswith("kd_")].assign(key=lambda x: x["run"].str[3:])
    ce = pd.concat([ce, df[df["run"] == "convnext_atto"].assign(key="atto")])
    m = kd.merge(ce, on="key", suffixes=("_kd", "_ce"))
    if m.empty:
        return
    m = m.sort_values("params_M_kd")
    mets = [("species_top1", "species top-1"), ("species_macro_f1", "species macro-F1"),
            ("species_macro_f1_tail", "tail species macro-F1"), ("rare_species_top1", "Rare Species top-1")]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = 0.2
    x = np.arange(len(m))
    for i, ((met, lab), col) in enumerate(zip(mets, ["#0072B2", "#009E73", "#D55E00", "#CC79A7"])):
        ax.bar(x + (i - 1.5) * w, 100 * (m[f"{met}_kd"] - m[f"{met}_ce"]), w, label=lab, color=col)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, [short_name(b) for b in m["backbone_kd"]], rotation=35, ha="right")
    ax.set_ylabel("KD minus CE (percentage points)")
    ax.legend(frameon=False, fontsize=9, ncol=2)
    for ext in ("png", "svg"):
        fig.savefig(out / f"kd_vs_ce.{ext}")
    plt.close(fig)


def coverage_plot(runs_dir: Path, runs: List[str], out: Path) -> None:
    """Species coverage vs hierarchical accuracy for selected runs."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
    for run, col in zip(runs, colors):
        m = json.loads((runs_dir / run / "metrics.json").read_text())
        cov = _get(m, "test", "coverage", default=[])
        if not isinstance(cov, list) or not cov:
            continue
        ax.plot([c["species_coverage"] for c in cov], [c["hier_accuracy"] for c in cov], marker="o", ms=3, label=run, color=col)
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
    ap.add_argument("--coverage-runs", nargs="*", default=["convnext_atto", "ce_tinyvit5m", "kd_repvit", "ce_bioclip_b", "bioclip2_ft"])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = apply_speed(collect(args.runs), args.out / "speed.json").sort_values(["kind", "params_M"])
    df.to_csv(args.out / "benchmark.csv", index=False)
    cols = [
        "run", "kind", "params_M", "gpu_img_s", "cpu_ms", "family_top1", "genus_top1", "genus_macro_f1",
        "species_top1", "species_macro_f1", "species_macro_f1_tail", "species_ece", "hier_acc@0.7", "sp_cov@0.7",
        "bing_genus_top1", "rare_genus_top1", "rare_species_top1", "yt_genus_top1",
    ]
    cols = [c for c in cols if c in df.columns and df[c].notna().any()]
    (args.out / "benchmark.md").write_text(df[cols].to_markdown(index=False, floatfmt=".3f"))
    for metric in ("species_macro_f1", "species_top1", "genus_top1"):
        pareto_plot(df, args.out, metric, "gpu_img_s", "GPU throughput (images/s, batch 64, bf16, RTX 4090)")
        pareto_plot(df, args.out, metric, "cpu_ms", "CPU latency (ms/crop, ONNX, 4 threads)")
    kd_plot(df, args.out)
    runs = [r for r in args.coverage_runs if (args.runs / r / "metrics.json").exists()]
    if runs:
        coverage_plot(args.runs, runs, args.out)
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
