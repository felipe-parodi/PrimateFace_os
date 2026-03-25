"""Phase E: Video preprocessing via PrimateFace API.

Thin wrapper that calls PrimateFace.process_video() for each clip
and saves results to .npz files.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import load_config

logger = logging.getLogger("animalfacs.preprocess")


def process_dataset(
    dataset_df: pd.DataFrame,
    cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Process all videos: detect, align, smooth via PrimateFace API.

    Saves per-clip .npz files with raw, aligned, and smoothed keypoints.

    Args:
        dataset_df: DataFrame from dataset_builder.build_dataset().
        cfg: Loaded config dict.

    Returns:
        Updated DataFrame with detection columns added.
    """
    if cfg is None:
        cfg = load_config()

    from primateface import PrimateFace

    data_root = Path(cfg["paths"]["data_root"])
    pf_cfg = cfg["preprocessing"]
    max_frames = pf_cfg.get("max_frames_per_clip", 30)
    min_frames = pf_cfg.get("min_frames_per_clip", 5)

    device = cfg["primateface"]["device"]
    pose_model = cfg["primateface"]["pose_model"]

    logger.info(
        "Initializing PrimateFace (device=%s, pose=%s) ...", device, pose_model
    )
    pf = PrimateFace(
        device=device,
        pose_model=pose_model,
        det_threshold=cfg["primateface"]["det_threshold"],
        nms_threshold=cfg["primateface"]["nms_threshold"],
    )

    results = []
    for _, row in tqdm(
        dataset_df.iterrows(), total=len(dataset_df), desc="Processing clips"
    ):
        clip_id = row["clip_id"]
        species = row["species"]
        video_path = Path(row["source_video_path"])

        feat_dir = data_root / "features" / species
        feat_dir.mkdir(parents=True, exist_ok=True)
        npz_path = feat_dir / f"{clip_id}.npz"

        # Skip if already processed
        if npz_path.exists():
            try:
                cached = np.load(npz_path, allow_pickle=True)
                n_valid = len(cached.get("valid_frame_indices", []))
                total_sampled = int(cached.get("total_frames_sampled", max_frames))
                results.append({
                    "clip_id": clip_id,
                    "n_frames_extracted": total_sampled,
                    "n_frames_detected": n_valid,
                    "detection_rate": n_valid / max(total_sampled, 1),
                    "npz_path": str(npz_path),
                })
                continue
            except Exception:
                pass

        if not video_path.exists():
            logger.warning("Video not found: %s", video_path)
            results.append({
                "clip_id": clip_id,
                "n_frames_extracted": 0,
                "n_frames_detected": 0,
                "detection_rate": 0.0,
                "npz_path": "",
            })
            continue

        try:
            vr = pf.process_video(
                video_path,
                max_frames=max_frames,
                min_frames=min_frames,
                align=True,
                smooth=True,
            )
        except (ValueError, FileNotFoundError) as e:
            logger.warning("Skip %s: %s", clip_id, e)
            results.append({
                "clip_id": clip_id,
                "n_frames_extracted": 0,
                "n_frames_detected": 0,
                "detection_rate": 0.0,
                "npz_path": "",
            })
            continue

        n_detected = len(vr.valid_frame_indices)
        det_rate = n_detected / max(vr.total_frames_sampled, 1)

        np.savez_compressed(npz_path, **vr.to_dict())

        results.append({
            "clip_id": clip_id,
            "n_frames_extracted": vr.total_frames_sampled,
            "n_frames_detected": n_detected,
            "detection_rate": det_rate,
            "npz_path": str(npz_path),
        })

        if det_rate < 0.5:
            logger.warning(
                "Low detection rate for %s/%s: %.1f%%",
                species, clip_id, det_rate * 100,
            )

    results_df = pd.DataFrame(results)
    merged = dataset_df.merge(results_df, on="clip_id", how="left")

    for species in sorted(merged["species"].unique()):
        sp = merged[merged["species"] == species]
        mean_det = sp["detection_rate"].mean()
        logger.info(
            "  %s: %d clips, mean detection rate=%.1f%%",
            species, len(sp), mean_det * 100,
        )

    return merged
