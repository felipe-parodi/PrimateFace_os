"""Phase I: Visualization — publication figures and output demo videos."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .au_homology import build_homology_matrix
from .utils import load_config

logger = logging.getLogger("animalfacs.visualize")

# Nature journal style
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _save_fig(fig: plt.Figure, path: Path) -> None:
    """Save figure as both PNG and SVG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path.with_suffix(".png")), dpi=300, bbox_inches="tight")
    fig.savefig(str(path.with_suffix(".svg")), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path.stem)


def plot_au_homology_heatmap(
    output_dir: Path,
    cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """Cross-species AU taxonomy heatmap.

    Rows = AU numbers, cols = species, cells = present/absent.

    Args:
        output_dir: Directory to save figures.
        cfg: Config dict.
    """
    au_list, species_list, matrix = build_homology_matrix()

    fig, ax = plt.subplots(figsize=(8, max(6, len(au_list) * 0.35)))
    data = np.array(matrix, dtype=float)

    # Pretty species labels
    sp_labels = [sp.capitalize() for sp in species_list]

    sns.heatmap(
        data,
        xticklabels=sp_labels,
        yticklabels=[f"AU{au}" for au in au_list],
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Present", "ticks": [0, 1]},
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("Species")
    ax.set_ylabel("Action Unit")
    ax.set_title("Cross-Species AU Homology")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    _save_fig(fig, output_dir / "au_homology_heatmap")


def plot_within_species_f1(
    results: Dict[str, Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Per-species bar chart of per-AU F1 scores.

    Args:
        results: Within-species CV results from evaluate.py.
        output_dir: Directory to save figures.
    """
    if not results:
        logger.warning("No within-species results to plot")
        return

    species_list = sorted(results.keys())
    all_aus = set()
    for sp_results in results.values():
        all_aus.update(sp_results.get("per_au_f1", {}).keys())
    au_list = sorted(all_aus)

    if not au_list:
        return

    n_species = len(species_list)
    n_aus = len(au_list)
    x = np.arange(n_aus)
    width = 0.8 / max(n_species, 1)

    fig, ax = plt.subplots(figsize=(max(8, n_aus * 0.8), 5))
    colors = plt.cm.Set2(np.linspace(0, 1, n_species))

    for i, species in enumerate(species_list):
        f1s = [
            results[species].get("per_au_f1", {}).get(au, 0) for au in au_list
        ]
        ax.bar(
            x + i * width - (n_species - 1) * width / 2,
            f1s,
            width,
            label=species.capitalize(),
            color=colors[i],
        )

    ax.set_xlabel("Action Unit")
    ax.set_ylabel("F1 Score")
    ax.set_title("Within-Species AU Classification (RF Baseline)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"AU{au}" for au in au_list], rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")

    _save_fig(fig, output_dir / "within_species_f1")


def plot_loso_transfer(
    results: Dict[str, Dict[str, Any]],
    output_dir: Path,
) -> None:
    """LOSO transfer matrix heatmap.

    Args:
        results: LOSO results from evaluate.py.
        output_dir: Directory to save figures.
    """
    if not results:
        logger.warning("No LOSO results to plot")
        return

    species = sorted(results.keys())
    metrics = [results[sp].get("f1_macro", 0) for sp in species]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.viridis(np.array(metrics))
    bars = ax.barh(
        range(len(species)),
        metrics,
        color=colors,
    )
    ax.set_yticks(range(len(species)))
    ax.set_yticklabels([sp.capitalize() for sp in species])
    ax.set_xlabel("Macro F1")
    ax.set_title("Leave-One-Species-Out Transfer")
    ax.set_xlim(0, 1.05)

    # Add value labels
    for bar, val in zip(bars, metrics):
        ax.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            fontsize=10,
        )

    _save_fig(fig, output_dir / "loso_transfer")


def plot_feature_importance(
    importances: Dict[int, np.ndarray],
    feature_names: List[str],
    output_dir: Path,
    top_k: int = 20,
) -> None:
    """Feature importance from RF for top AUs.

    Args:
        importances: Dict mapping AU number to importance array.
        feature_names: Names corresponding to feature dimensions.
        output_dir: Directory to save figures.
        top_k: Number of top features to show per AU.
    """
    if not importances:
        return

    for au, imp in importances.items():
        if len(imp) != len(feature_names):
            continue
        top_idx = np.argsort(imp)[-top_k:][::-1]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(
            range(len(top_idx)),
            imp[top_idx],
            color=plt.cm.viridis(imp[top_idx] / max(imp[top_idx].max(), 1e-6)),
        )
        ax.set_yticks(range(len(top_idx)))
        ax.set_yticklabels([feature_names[i] for i in top_idx])
        ax.set_xlabel("Importance")
        ax.set_title(f"AU{au} — Top {top_k} Feature Importances")
        ax.invert_yaxis()

        _save_fig(fig, output_dir / f"importance_AU{au}")


def plot_dataset_summary(
    dataset_df: "pd.DataFrame",
    output_dir: Path,
) -> None:
    """Dataset summary: sample counts and AU distribution.

    Args:
        dataset_df: Dataset DataFrame.
        output_dir: Directory to save figures.
    """
    # Clips per species
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sp_counts = dataset_df["species"].value_counts().sort_index()
    axes[0].bar(
        range(len(sp_counts)),
        sp_counts.values,
        color=plt.cm.Set2(np.linspace(0, 1, len(sp_counts))),
    )
    axes[0].set_xticks(range(len(sp_counts)))
    axes[0].set_xticklabels(
        [s.capitalize() for s in sp_counts.index], rotation=45, ha="right"
    )
    axes[0].set_ylabel("Number of Clips")
    axes[0].set_title("Clips per Species")
    axes[0].yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # AU distribution (across all clips)
    au_counts: Dict[int, int] = {}
    for labels in dataset_df["normalized_labels"]:
        if isinstance(labels, str):
            aus = [int(x) for x in labels.split(",") if x.strip()]
        else:
            aus = labels
        for au in aus:
            au_counts[au] = au_counts.get(au, 0) + 1

    if au_counts:
        sorted_aus = sorted(au_counts.items())
        au_labels = [f"AU{a}" for a, _ in sorted_aus]
        au_vals = [c for _, c in sorted_aus]
        axes[1].bar(range(len(au_labels)), au_vals, color="steelblue")
        axes[1].set_xticks(range(len(au_labels)))
        axes[1].set_xticklabels(au_labels, rotation=45, ha="right")
        axes[1].set_ylabel("Number of Clips")
        axes[1].set_title("AU Distribution")
        axes[1].yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    _save_fig(fig, output_dir / "dataset_summary")


def render_demo_video(
    video_path: Path,
    npz_path: Path,
    species: str,
    gt_aus: List[int],
    pred_aus: Optional[Dict[int, float]],
    output_path: Path,
    max_frames: int = 30,
    fps: int = 10,
) -> None:
    """Render a demo video with landmark overlay and AU predictions.

    Single-panel: original frame with 68-point skeleton + text overlay.

    Args:
        video_path: Path to source video.
        npz_path: Path to .npz with extracted keypoints.
        species: Species name for overlay.
        gt_aus: Ground truth AU list.
        pred_aus: Predicted AU → confidence dict (or None).
        output_path: Where to save the output MP4.
        max_frames: Max frames to render.
        fps: Output video FPS.
    """
    from primateface.analysis.features import SKELETON_EDGES

    if not video_path.exists() or not npz_path.exists():
        logger.warning("Missing files for demo video: %s", video_path)
        return

    data = np.load(npz_path, allow_pickle=True)
    keypoints = data.get("raw_keypoints", np.zeros((0, 68, 3)))
    valid_indices = data.get("valid_frame_indices", np.array([]))
    clip_fps = float(data.get("fps", 25.0))
    total_sampled = int(data.get("total_frames_sampled", 0))

    if keypoints.shape[0] == 0:
        logger.info("Skipping %s: no detections", output_path.stem)
        return

    # Skip very short clips (<1s source duration)
    if total_sampled > 0 and clip_fps > 0:
        duration = total_sampled / clip_fps
        if duration < 1.0:
            logger.info("Skipping %s: too short (%.1fs)", output_path.stem, duration)
            return

    # Build mapping: sampled_frame_index → keypoint_index
    kpt_lookup = {}
    for kpt_i, frame_i in enumerate(valid_indices):
        kpt_lookup[int(frame_i)] = kpt_i

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_sample = min(max_frames, total_frames)
    frame_indices = np.linspace(0, total_frames - 1, n_sample, dtype=int)

    # Setup output — use ffmpeg if available for speed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    use_ffmpeg = True
    try:
        import subprocess

        ffmpeg_cmd = [
            "ffmpeg", "-y", "-f", "rawvideo",
            "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-preset", "fast",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]
        pipe = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        use_ffmpeg = False
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        pipe = None
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for sampled_i, fi in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ret, frame = cap.read()
        if not ret:
            continue

        # Check if this sampled frame has keypoints
        if sampled_i in kpt_lookup:
            kpt_i = kpt_lookup[sampled_i]
            kpts = keypoints[kpt_i]  # (68, 3)

            for si, sj in SKELETON_EDGES:
                if kpts[si, 2] > 0.3 and kpts[sj, 2] > 0.3:
                    pt1 = (int(kpts[si, 0]), int(kpts[si, 1]))
                    pt2 = (int(kpts[sj, 0]), int(kpts[sj, 1]))
                    cv2.line(frame, pt1, pt2, (0, 165, 255), 1)

            for k in range(68):
                if kpts[k, 2] > 0.3:
                    pt = (int(kpts[k, 0]), int(kpts[k, 1]))
                    cv2.circle(frame, pt, 2, (0, 0, 255), -1)
        else:
            # No detection — show indicator
            cv2.putText(
                frame, "No detection", (10, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1,
            )

        # Text overlays
        cv2.putText(
            frame, species.capitalize(), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )

        gt_str = "GT: " + "+".join(f"AU{a}" for a in gt_aus)
        text_size = cv2.getTextSize(gt_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.putText(
            frame, gt_str, (w - text_size[0] - 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1,
        )

        if pred_aus:
            pred_parts = [f"AU{a} ({c:.2f})" for a, c in sorted(pred_aus.items())]
            pred_str = "Pred: " + " ".join(pred_parts)
            cv2.putText(
                frame, pred_str, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1,
            )

        try:
            if use_ffmpeg and pipe is not None:
                pipe.stdin.write(frame.tobytes())
            else:
                writer.write(frame)
        except BrokenPipeError:
            logger.warning("ffmpeg pipe broke for %s, falling back", output_path.stem)
            use_ffmpeg = False
            break

    cap.release()
    if use_ffmpeg and pipe is not None:
        try:
            pipe.stdin.close()
        except BrokenPipeError:
            pass
        pipe.wait()
    elif not use_ffmpeg and "writer" in dir():
        writer.release()

    logger.info("Demo video saved: %s", output_path)


def plot_kinematic_correlations(
    corr_results: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Plot per-species kinematic-AU correlation heatmaps.

    Args:
        corr_results: Results from evaluate.kinematic_au_correlations().
        output_dir: Directory to save figures.
    """
    feature_names = corr_results.get("feature_names", [])
    if not feature_names:
        return

    # Import au_set from the stored data
    species_data = {
        k: v for k, v in corr_results.items()
        if isinstance(v, dict) and "correlations" in v
    }
    if not species_data:
        return

    for species, data in species_data.items():
        corr = data["correlations"]  # (n_features, n_aus)
        n_feat = min(len(feature_names), corr.shape[0])

        # Only show AUs with non-zero correlations
        au_mask = np.any(np.abs(corr[:n_feat, :]) > 0.1, axis=0)
        if not au_mask.any():
            continue
        au_indices = np.where(au_mask)[0]
        corr_subset = corr[:n_feat, au_indices]

        fig, ax = plt.subplots(
            figsize=(max(4, len(au_indices) * 0.6), max(4, n_feat * 0.4))
        )
        sns.heatmap(
            corr_subset,
            xticklabels=[f"AU{i}" for i in au_indices],
            yticklabels=feature_names[:n_feat],
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            ax=ax,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 8},
        )
        ax.set_title(f"{species.capitalize()} — Kinematic-AU Correlations")
        ax.set_xlabel("Action Unit")
        ax.set_ylabel("PrimateFace Feature")

        _save_fig(fig, output_dir / f"correlations_{species}")

    # Cross-species consistency plot
    consistency = corr_results.get("cross_species_consistency", {})
    if consistency:
        fig, ax = plt.subplots(figsize=(5, 3))
        pairs = sorted(consistency.keys())
        vals = [consistency[p] for p in pairs]
        labels = [p.replace("_vs_", " vs\n") for p in pairs]
        colors = ["#2ecc71" if v > 0.3 else "#e74c3c" if v < 0 else "#f39c12"
                  for v in vals]
        bars = ax.bar(range(len(pairs)), vals, color=colors)
        ax.set_xticks(range(len(pairs)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Correlation of Correlations (r)")
        ax.set_title("Cross-Species Feature-AU Consistency")
        ax.set_ylim(-1, 1)
        ax.axhline(0, color="gray", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03,
                f"{val:.2f}",
                ha="center",
                fontsize=9,
            )
        _save_fig(fig, output_dir / "cross_species_consistency")


def generate_all_figures(
    eval_results: Dict[str, Any],
    dataset_df: "pd.DataFrame",
    cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """Generate all publication figures.

    Args:
        eval_results: Results from evaluate.run_all_evaluations().
        dataset_df: Dataset DataFrame.
        cfg: Config dict.
    """
    if cfg is None:
        cfg = load_config()

    fig_dir = Path(cfg["paths"]["results_root"]) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # a. AU homology heatmap
    plot_au_homology_heatmap(fig_dir, cfg)

    # b. Within-species F1
    plot_within_species_f1(
        eval_results.get("within_species", {}), fig_dir
    )

    # c. LOSO transfer
    plot_loso_transfer(eval_results.get("loso", {}), fig_dir)

    # d. Kinematic-AU correlation heatmaps
    corr_results = eval_results.get("correlations", {})
    if corr_results:
        plot_kinematic_correlations(corr_results, fig_dir)

    # f. Dataset summary
    plot_dataset_summary(dataset_df, fig_dir)

    logger.info("All figures saved to %s", fig_dir)
