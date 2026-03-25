"""Video QA filtering for FACS clips.

Checks that each clip is suitable for AU modeling:
single face, sufficient visibility, minimum duration, adequate face size.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .utils import load_config

logger = logging.getLogger("animalfacs.qa")


def qa_clip(
    npz_path: Path,
    min_single_face_frac: float = 0.7,
    min_visibility: float = 0.5,
    min_detected_frames: int = 5,
    min_face_area_frac: float = 0.05,
) -> Dict[str, Any]:
    """Run QA checks on a single processed clip.

    Args:
        npz_path: Path to .npz file from process_video().
        min_single_face_frac: Fraction of frames needing exactly 1 face.
        min_visibility: Minimum mean keypoint visibility ratio.
        min_detected_frames: Minimum frames with valid detections.
        min_face_area_frac: Minimum face area as fraction of frame.

    Returns:
        Dict with qa_status ("pass", "warn", "fail") and per-check results.
    """
    result: Dict[str, Any] = {"qa_status": "fail", "checks": {}}

    if not npz_path.exists():
        result["checks"]["file_exists"] = False
        return result
    result["checks"]["file_exists"] = True

    data = np.load(npz_path, allow_pickle=True)
    raw_kpts = data.get("raw_keypoints", np.zeros((0, 68, 3)))
    faces_per_frame = data.get("faces_per_frame", np.array([]))
    total_sampled = int(data.get("total_frames_sampled", 0))
    bboxes = data.get("bboxes", np.zeros((0, 4)))

    n_detected = raw_kpts.shape[0]

    # Check 1: Minimum detected frames
    enough_frames = n_detected >= min_detected_frames
    result["checks"]["min_frames"] = {
        "pass": bool(enough_frames),
        "detected": n_detected,
        "required": min_detected_frames,
    }

    # Check 2: Single face prevalence
    if len(faces_per_frame) > 0:
        single_face_frac = float(np.mean(faces_per_frame == 1))
    else:
        single_face_frac = 0.0
    single_face_ok = single_face_frac >= min_single_face_frac
    result["checks"]["single_face"] = {
        "pass": bool(single_face_ok),
        "fraction": round(single_face_frac, 3),
        "required": min_single_face_frac,
    }

    # Check 3: Keypoint visibility
    if n_detected > 0:
        confidences = raw_kpts[:, :, 2]  # (T, 68)
        mean_vis = float(np.mean(confidences > 0.3))
    else:
        mean_vis = 0.0
    vis_ok = mean_vis >= min_visibility
    result["checks"]["visibility"] = {
        "pass": bool(vis_ok),
        "mean_visibility": round(mean_vis, 3),
        "required": min_visibility,
    }

    # Check 4: Face size
    if bboxes.shape[0] > 0 and total_sampled > 0:
        # Estimate frame area from bbox coordinates (rough)
        bbox_areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        # We don't have frame dims in npz, so use bbox-relative check
        # Heuristic: face bbox should be at least some minimum absolute size
        mean_area = float(np.mean(bbox_areas))
        face_size_ok = mean_area > 1000  # at least ~30x30 pixels
    else:
        mean_area = 0.0
        face_size_ok = False
    result["checks"]["face_size"] = {
        "pass": bool(face_size_ok),
        "mean_bbox_area": round(mean_area, 1),
    }

    # Aggregate — single_face failure is a hard fail (multi-individual clips
    # are unsuitable for per-individual AU coding)
    all_pass = enough_frames and single_face_ok and vis_ok and face_size_ok
    any_pass = enough_frames and single_face_ok and (n_detected >= 3)

    if all_pass:
        result["qa_status"] = "pass"
    elif any_pass:
        result["qa_status"] = "warn"
    else:
        result["qa_status"] = "fail"

    return result


def qa_dataset(
    dataset_df: pd.DataFrame,
    cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Run QA on all clips in the dataset.

    Adds qa_status column and filters the dataset.

    Args:
        dataset_df: DataFrame with clip_id, species, npz_path columns.
        cfg: Config dict with QA parameters.

    Returns:
        DataFrame with qa_status and qa_details columns added.
    """
    if cfg is None:
        cfg = load_config()

    data_root = Path(cfg["paths"]["data_root"])

    qa_results = []
    for _, row in dataset_df.iterrows():
        npz_path_str = row.get("npz_path", "")
        if not npz_path_str:
            feat_dir = data_root / "features" / row["species"]
            npz_path = feat_dir / f"{row['clip_id']}.npz"
        else:
            npz_path = Path(npz_path_str)

        result = qa_clip(npz_path)
        qa_results.append({
            "clip_id": row["clip_id"],
            "qa_status": result["qa_status"],
            "qa_details": str(result["checks"]),
        })

    qa_df = pd.DataFrame(qa_results)
    merged = dataset_df.merge(qa_df, on="clip_id", how="left")

    # Summary
    counts = merged["qa_status"].value_counts().to_dict()
    for species in sorted(merged["species"].unique()):
        sp = merged[merged["species"] == species]
        sp_counts = sp["qa_status"].value_counts().to_dict()
        logger.info("  %s QA: %s", species, sp_counts)

    logger.info("QA total: %s (keeping pass+warn)", counts)
    return merged
