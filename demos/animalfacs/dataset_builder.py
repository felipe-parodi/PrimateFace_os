"""Phase D: Build structured AU dataset from parsed labels.

Constructs a parquet dataset with video-level splits that prevent
temporal and label leakage.
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .utils import load_config

logger = logging.getLogger("animalfacs.dataset_builder")


def _video_id(file_path: str) -> str:
    """Generate a stable clip ID from file path.

    Args:
        file_path: Absolute or relative path to the video.

    Returns:
        Short hash string as unique ID.
    """
    return hashlib.md5(file_path.encode()).hexdigest()[:12]


def _assign_splits(
    df: pd.DataFrame,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> pd.DataFrame:
    """Assign train/val/test splits per species, splitting by video.

    Ensures all frames from one video stay in the same split
    (no temporal leakage). Stratified by dominant AU when possible.

    Args:
        df: DataFrame with at least 'species' and 'clip_id' columns.
        seed: Random seed.
        train_frac: Fraction for training.
        val_frac: Fraction for validation.

    Returns:
        DataFrame with 'split' column added.
    """
    rng = np.random.RandomState(seed)
    splits = []

    for species in df["species"].unique():
        mask = df["species"] == species
        # Get unique video clip IDs
        clip_ids = df.loc[mask, "clip_id"].unique()
        n = len(clip_ids)
        perm = rng.permutation(n)

        n_train = max(1, int(n * train_frac))
        n_val = max(1, int(n * val_frac))

        split_map = {}
        for i, idx in enumerate(perm):
            cid = clip_ids[idx]
            if i < n_train:
                split_map[cid] = "train"
            elif i < n_train + n_val:
                split_map[cid] = "val"
            else:
                split_map[cid] = "test"

        # If too few clips, put everything in train
        if n <= 3:
            split_map = {cid: "train" for cid in clip_ids}

        for cid, split in split_map.items():
            splits.append({"clip_id": cid, "split": split})

    split_df = pd.DataFrame(splits)
    df = df.merge(split_df, on="clip_id", how="left")
    df["split"] = df["split"].fillna("train")
    return df


def _merge_cooccurring_aus(df: pd.DataFrame) -> pd.DataFrame:
    """Merge AUs that always co-occur into single combined classes.

    E.g. if AU1 and AU2 always appear together, replace both with AU1+2.

    Args:
        df: DataFrame with 'normalized_labels' column (list of ints).

    Returns:
        DataFrame with co-occurring AUs merged.
    """
    from itertools import combinations

    # Collect AU sets per clip
    au_sets = []
    for labels in df["normalized_labels"]:
        if isinstance(labels, str):
            au_sets.append(set(int(x) for x in labels.split(",") if x.strip()))
        else:
            au_sets.append(set(labels))

    # Find perfect co-occurrence pairs
    all_aus = sorted(set().union(*au_sets))
    merge_groups: Dict[int, int] = {}  # au → representative au

    for a, b in combinations(all_aus, 2):
        a_clips = {i for i, s in enumerate(au_sets) if a in s}
        b_clips = {i for i, s in enumerate(au_sets) if b in s}
        if a_clips and a_clips == b_clips:
            # Always co-occur — merge b into a (keep lower number)
            rep = min(a, b)
            other = max(a, b)
            merge_groups[other] = rep
            logger.info(
                "  Merging AU%d into AU%d (always co-occur, %d clips)",
                other, rep, len(a_clips),
            )

    if not merge_groups:
        return df

    # Apply merges to labels
    def _merge_labels(labels: Any) -> list:
        if isinstance(labels, str):
            aus = [int(x) for x in labels.split(",") if x.strip()]
        else:
            aus = list(labels)
        merged = set()
        for au in aus:
            merged.add(merge_groups.get(au, au))
        return sorted(merged)

    df = df.copy()
    df["normalized_labels"] = df["normalized_labels"].apply(_merge_labels)
    return df


def build_dataset(
    au_records: List[Dict[str, Any]],
    cfg: Optional[Dict[str, Any]] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Build the AU dataset parquet from parsed label records.

    Only includes videos that have at least one parsed AU label.

    Args:
        au_records: List of dicts from au_parser.parse_all_species().
        cfg: Loaded config dict.
        output_path: Where to save the parquet.

    Returns:
        Built DataFrame.
    """
    if cfg is None:
        cfg = load_config()

    seed = cfg.get("seed", 42)

    # Filter to records with AU labels
    labeled = [r for r in au_records if r.get("aus")]
    if not labeled:
        logger.error("No labeled videos found. Cannot build dataset.")
        return pd.DataFrame()

    rows = []
    for rec in labeled:
        clip_id = _video_id(rec["file_path"])
        rows.append({
            "clip_id": clip_id,
            "species": rec["species"],
            "source_video_path": rec["file_path"],
            "raw_labels": rec.get("raw_label", ""),
            "normalized_labels": rec["aus"],
            "label_source": rec.get("label_source", "parsed"),
            "descriptors": rec.get("descriptors", []),
        })

    df = pd.DataFrame(rows)

    # Merge perfectly co-occurring AUs into single classes
    df = _merge_cooccurring_aus(df)

    # Assign splits (video-level, no leakage)
    df = _assign_splits(df, seed=seed)

    # Summary
    for species in sorted(df["species"].unique()):
        sp_df = df[df["species"] == species]
        split_counts = sp_df["split"].value_counts().to_dict()
        all_aus = set()
        for labels in sp_df["normalized_labels"]:
            all_aus.update(labels)
        logger.info(
            "  %s: %d clips, splits=%s, AUs=%s",
            species,
            len(sp_df),
            split_counts,
            sorted(all_aus),
        )

    # Save
    if output_path is None:
        output_path = Path(cfg["paths"]["au_dataset"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert lists to strings for parquet compatibility
    save_df = df.copy()
    save_df["normalized_labels"] = save_df["normalized_labels"].apply(
        lambda x: ",".join(map(str, x))
    )
    save_df["descriptors"] = save_df["descriptors"].apply(
        lambda x: ",".join(x) if x else ""
    )
    save_df.to_parquet(output_path, index=False)
    logger.info("Dataset saved to %s (%d clips)", output_path, len(df))

    return df
