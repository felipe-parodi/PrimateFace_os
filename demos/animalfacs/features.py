"""Phase F: FACS-specific feature extraction.

Thin wrapper around PrimateFace API feature functions.
Only contains FACS-specific logic: which features to extract
for AU prediction and how to assemble the feature matrix.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from primateface.analysis.features import (
    DEFAULT_DISTANCE_PAIRS,
    aggregate_timeseries,
    pairwise_distances,
)
from primateface.analysis.kinematics import extract_kinematics

from .utils import load_config

logger = logging.getLogger("animalfacs.features")

# Re-export for backward compat with demo visualize.py
_SKELETON_EDGES = None  # Removed — use primateface.analysis.features.SKELETON_EDGES


def _iod(kpts: np.ndarray) -> float:
    """Interocular distance from 68-point landmarks."""
    left_eye = kpts[42:48, :2].mean(axis=0)
    right_eye = kpts[36:42, :2].mean(axis=0)
    return float(np.linalg.norm(left_eye - right_eye))


def extract_geometric_features(
    keypoints: np.ndarray,
    use_distances: bool = False,
    agg_stats: Optional[List[str]] = None,
) -> np.ndarray:
    """Extract per-clip geometric feature vector.

    Uses PrimateFace kinematics (14 features) and optionally
    pairwise distances. Aggregates across frames.

    Args:
        keypoints: (T, 68, 3) array (uses aligned or smoothed).
        use_distances: Include pairwise distances (adds ~36 features).
        agg_stats: Aggregation stats. Default: ["mean", "std"].

    Returns:
        1D feature vector for this clip.
    """
    if agg_stats is None:
        agg_stats = ["mean", "std"]

    n_frames = keypoints.shape[0]
    if n_frames == 0:
        return np.zeros(0, dtype=np.float32)

    # Per-frame kinematics (14 scalar features)
    kin_per_frame = []
    for t in range(n_frames):
        try:
            kin = extract_kinematics(keypoints[t])
            kin_vec = np.array(
                [v for v in kin.values() if isinstance(v, (int, float))],
                dtype=np.float32,
            )
        except Exception:
            kin_vec = np.zeros(14, dtype=np.float32)
        kin_per_frame.append(kin_vec)

    kin_arr = np.stack(kin_per_frame)  # (T, 14)

    if use_distances:
        # Flatten all default distance pairs
        all_pairs = []
        for pairs in DEFAULT_DISTANCE_PAIRS.values():
            all_pairs.extend(pairs)
        dist_arr = pairwise_distances(
            keypoints, all_pairs, normalize_by_iod=True
        )  # (T, D)
        combined = np.concatenate([kin_arr, dist_arr], axis=1)
    else:
        combined = kin_arr

    return aggregate_timeseries(combined, agg_stats)


def extract_landmark_sequence(
    keypoints: np.ndarray,
    target_length: int = 30,
) -> np.ndarray:
    """Extract normalized landmark sequence for temporal models.

    Normalizes by IOD and centers on nose tip.

    Args:
        keypoints: (T, 68, 2) or (T, 68, 3) aligned keypoints.
        target_length: Pad/truncate to this length.

    Returns:
        (target_length, 68, 2) normalized array.
    """
    t_actual = keypoints.shape[0]
    coords = keypoints[:, :, :2].copy()

    for t in range(t_actual):
        iod = _iod(keypoints[t])
        if iod > 1e-6:
            nose = coords[t, 30, :2]
            coords[t] = (coords[t] - nose) / iod

    if t_actual >= target_length:
        indices = np.linspace(0, t_actual - 1, target_length, dtype=int)
        seq = coords[indices]
    else:
        pad = np.repeat(coords[-1:], target_length - t_actual, axis=0)
        seq = np.concatenate([coords, pad], axis=0)

    return seq.astype(np.float32)


def extract_appearance_embeddings(
    dataset_df: pd.DataFrame,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Extract FMAE-IAT ViT embeddings for all clips.

    For each clip: loads source video frames, crops faces using bboxes,
    passes through frozen ViT-Small, mean-pools across frames.

    Args:
        dataset_df: DataFrame with 'clip_id', 'species', 'source_video_path', 'npz_path'.
        cfg: Loaded config dict.

    Returns:
        Tuple of (N, 384) embedding matrix and clip_id list.
    """
    if cfg is None:
        cfg = load_config()

    from primateface.analysis.face_encoder import FaceEncoder

    device = cfg["primateface"]["device"]
    encoder = FaceEncoder(model_name="vit_small", device=device)

    embeddings = []
    valid_ids = []

    for _, row in tqdm(
        dataset_df.iterrows(), total=len(dataset_df), desc="Extracting ViT embeddings"
    ):
        # Reconstruct npz path if not in DataFrame
        npz_path_str = row.get("npz_path", "")
        if not npz_path_str or not Path(npz_path_str).exists():
            data_root = Path(cfg["paths"]["data_root"])
            npz_path_str = str(
                data_root / "features" / row["species"] / f"{row['clip_id']}.npz"
            )
        video_path = Path(row["source_video_path"])

        if not Path(npz_path_str).exists():
            continue
        if not video_path.exists():
            continue

        data = np.load(npz_path_str, allow_pickle=True)
        bboxes = data.get("bboxes", np.zeros((0, 4)))
        valid_indices = data.get("valid_frame_indices", np.array([]))

        if bboxes.shape[0] == 0:
            continue

        # Read frames corresponding to valid detections
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames = cfg["preprocessing"]["max_frames_per_clip"]
        n_sample = min(max_frames, total)
        sample_indices = np.linspace(0, total - 1, n_sample, dtype=int)

        # Get frames at valid detection positions
        frame_crops = []
        for det_i, frame_i in enumerate(valid_indices):
            if frame_i >= len(sample_indices):
                continue
            source_frame_idx = int(sample_indices[frame_i])
            cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            bbox = bboxes[det_i]
            preprocessed = encoder.preprocess(frame, bbox)
            frame_crops.append(preprocessed)

        cap.release()

        if not frame_crops:
            continue

        # Encode and mean-pool
        frame_embs = encoder.encode(np.stack(frame_crops))  # (T', 384)
        clip_emb = frame_embs.mean(axis=0)  # (384,)
        embeddings.append(clip_emb)
        valid_ids.append(row["clip_id"])

    if not embeddings:
        logger.error("No appearance embeddings extracted!")
        return np.zeros((0, 384)), []

    emb_matrix = np.stack(embeddings)  # (N, 384)
    logger.info("Appearance embeddings: %s, %d clips", emb_matrix.shape, len(valid_ids))
    return emb_matrix, valid_ids


def extract_all_features(
    dataset_df: pd.DataFrame,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """Extract features for all clips.

    Uses smoothed_keypoints from .npz if available, else aligned, else raw.

    Args:
        dataset_df: DataFrame with 'clip_id', 'species', 'npz_path'.
        cfg: Loaded config dict.

    Returns:
        Tuple of (geo_features, seq_features, clip_ids, updated_df).
    """
    if cfg is None:
        cfg = load_config()

    max_frames = cfg["preprocessing"]["max_frames_per_clip"]
    use_distances = cfg.get("features", {}).get("use_distances", False)
    agg_stats = cfg.get("features", {}).get(
        "aggregation_stats", ["mean", "std"]
    )

    geo_list = []
    seq_list = []
    valid_ids = []

    for _, row in tqdm(
        dataset_df.iterrows(), total=len(dataset_df), desc="Extracting features"
    ):
        npz_path = row.get("npz_path", "")
        if not npz_path or not Path(npz_path).exists():
            continue

        data = np.load(npz_path, allow_pickle=True)

        # Prefer smoothed > aligned > raw
        if "smoothed_keypoints" in data and data["smoothed_keypoints"].shape[0] > 0:
            kpts = data["smoothed_keypoints"]
            # Add dummy confidence channel for kinematics
            if kpts.ndim == 3 and kpts.shape[2] == 2:
                ones = np.ones((*kpts.shape[:2], 1), dtype=np.float32)
                kpts = np.concatenate([kpts, ones], axis=2)
        elif "aligned_keypoints" in data and data["aligned_keypoints"].shape[0] > 0:
            kpts = data["aligned_keypoints"]
            if kpts.ndim == 3 and kpts.shape[2] == 2:
                ones = np.ones((*kpts.shape[:2], 1), dtype=np.float32)
                kpts = np.concatenate([kpts, ones], axis=2)
        elif "raw_keypoints" in data:
            kpts = data["raw_keypoints"]
        else:
            continue

        if kpts.shape[0] < 3:
            continue

        geo_vec = extract_geometric_features(
            kpts, use_distances=use_distances, agg_stats=agg_stats
        )
        if geo_vec.size == 0:
            continue

        seq = extract_landmark_sequence(kpts, target_length=max_frames)

        geo_list.append(geo_vec)
        seq_list.append(seq)
        valid_ids.append(row["clip_id"])

    if not geo_list:
        logger.error("No features extracted!")
        return np.zeros((0, 0)), np.zeros((0, 0, 0, 0)), [], dataset_df

    geo_features = np.stack(geo_list)
    seq_features = np.stack(seq_list)

    np.savez_compressed(
        Path(cfg["paths"]["features_matrix"]),
        geo_features=geo_features,
        seq_features=seq_features,
        clip_ids=np.array(valid_ids),
    )
    logger.info(
        "Features: geo=%s, seq=%s, %d clips",
        geo_features.shape, seq_features.shape, len(valid_ids),
    )

    return geo_features, seq_features, valid_ids, dataset_df
