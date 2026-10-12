"""Attach MDD labels to the face manifest and build leakage-free splits.

Groups: crops sharing an image hash, an observer, or a source group id are merged
(union-find) so that near-identical photos (GBIF mirrors of iNat, burst shots by the
same observer) never straddle train and test.

Usage:
    python -m species_benchmark.labels \
        --manifest C:/pfspecies/data/faces_manifest.parquet \
        --taxonomy C:/pfspecies/data/taxonomy_mdd_v2.5_primates.json \
        --out-dir C:/pfspecies/data
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from species_benchmark.taxonomy import PrimateTaxonomy

MIN_SPECIES_IMGS = 20
GENUS_ONLY_SOURCES = {"bing_folder"}


class _UnionFind:
    """Minimal union-find over hashable keys."""

    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        """Return the root of ``x``, creating it if needed."""
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        """Merge the sets containing ``a`` and ``b``."""
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_groups(df: pd.DataFrame) -> pd.Series:
    """Compute leakage groups by merging shared image/observer/group keys.

    Args:
        df: Manifest with columns ``crop_id`` and optionally ``image_sha256``,
            ``observer_id``, ``group_id``, ``image_path``.

    Returns:
        Series of group root ids aligned with ``df.index``.
    """
    uf = _UnionFind()
    key_cols = [c for c in ("image_sha256", "observer_id", "group_id", "image_path") if c in df.columns]
    for row in df[["crop_id", *key_cols]].itertuples(index=False):
        node = f"crop:{row[0]}"
        uf.find(node)
        for col, val in zip(key_cols, row[1:]):
            if val is None or (isinstance(val, float) and np.isnan(val)) or str(val) in ("", "nan", "None"):
                continue
            uf.union(node, f"{col}:{val}")
    return pd.Series([uf.find(f"crop:{c}") for c in df["crop_id"]], index=df.index)


def _hash_frac(key: str) -> float:
    """Deterministic uniform [0, 1) value for a string key."""
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def assign_splits(
    df: pd.DataFrame,
    strat_col: str,
    test_frac: float = 0.10,
    val_frac: float = 0.05,
) -> pd.Series:
    """Assign train/val/test per group, stratified by label.

    Groups are ordered per stratum by a deterministic hash and greedily filled so each
    stratum receives roughly ``test_frac`` / ``val_frac`` of its crops. Strata with fewer
    than three groups stay in train; a single group larger than half its stratum is never
    held out (it would leave too little to train on).

    Args:
        df: Frame with ``group`` and ``strat_col`` columns.
        strat_col: Column used for stratification (dominant label per group).
        test_frac: Target test fraction of crops.
        val_frac: Target validation fraction of crops.

    Returns:
        Series of split names aligned with ``df.index``.
    """
    g = df.groupby("group").agg(n=("crop_id", "size"), lab=(strat_col, lambda s: s.mode().iat[0]))
    split_of: Dict[str, str] = {}
    for _, sub in g.groupby("lab"):
        order = sorted(sub.index, key=_hash_frac)
        total = sub["n"].sum()
        n_groups = len(order)
        acc_test = acc_val = 0
        for grp in order:
            n = sub.at[grp, "n"]
            if n_groups < 3 or n > 0.5 * total:
                split_of[grp] = "train"
            elif acc_test < test_frac * total:
                split_of[grp] = "test"
                acc_test += n
            elif n_groups >= 5 and acc_val < val_frac * total:
                split_of[grp] = "val"
                acc_val += n
            else:
                split_of[grp] = "train"
    return df["group"].map(split_of)


def attach_labels(df: pd.DataFrame, tax: PrimateTaxonomy) -> pd.DataFrame:
    """Resolve raw names to MDD family/genus/species.

    Args:
        df: Manifest with ``scientific_name`` and/or ``genus_raw`` and ``label_source``.
        tax: Loaded taxonomy.

    Returns:
        Copy of ``df`` with ``family``, ``genus``, ``species``, ``label_level`` columns.
    """
    out = df.copy()
    cache: Dict[str, object] = {}
    fam: List[Optional[str]] = []
    gen: List[Optional[str]] = []
    spp: List[Optional[str]] = []
    lvl: List[str] = []
    for name, graw, src in zip(
        out.get("scientific_name", pd.Series([None] * len(out))),
        out.get("genus_raw", pd.Series([None] * len(out))),
        out["label_source"],
    ):
        query = name if isinstance(name, str) and name.strip() else graw
        if not isinstance(query, str) or not query.strip():
            fam.append(None), gen.append(None), spp.append(None), lvl.append("unresolved")
            continue
        if query not in cache:
            cache[query] = tax.resolve(query)
        r = cache[query]
        species = r.species if src not in GENUS_ONLY_SOURCES else None
        level = r.level if species or r.level != "species" else "genus"
        fam.append(r.family), gen.append(r.genus), spp.append(species), lvl.append(level)
    out["family"], out["genus"], out["species"], out["label_level"] = fam, gen, spp, lvl
    return out


def build_label_space(train: pd.DataFrame, min_species: int = MIN_SPECIES_IMGS) -> Dict[str, List[str]]:
    """Choose class vocabularies from the training split.

    Args:
        train: Training rows with family/genus/species.
        min_species: Minimum training crops for a species to get its own class.

    Returns:
        Dict with sorted ``family``, ``genus``, ``species`` class lists and parent maps.
    """
    sp_counts = Counter(train["species"].dropna())
    species = sorted(s for s, c in sp_counts.items() if c >= min_species)
    genus = sorted(train["genus"].dropna().unique())
    family = sorted(train["family"].dropna().unique())
    sp2g = train.dropna(subset=["species"]).drop_duplicates("species").set_index("species")["genus"].to_dict()
    g2f = train.dropna(subset=["genus"]).drop_duplicates("genus").set_index("genus")["family"].to_dict()
    return {
        "family": family,
        "genus": genus,
        "species": species,
        "species_to_genus": {s: sp2g[s] for s in species},
        "genus_to_family": {g: g2f[g] for g in genus},
        "min_species_imgs": min_species,
    }


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--taxonomy", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--min-species", type=int, default=MIN_SPECIES_IMGS)
    args = ap.parse_args()

    tax = PrimateTaxonomy.from_json(args.taxonomy)
    df = pd.read_parquet(args.manifest)
    if "crop_exists" in df.columns:
        df = df[df["crop_exists"]]
    df = attach_labels(df, tax)
    df = df[df["genus"].notna() & (df["genus"] != "Homo")].reset_index(drop=True)
    df["group"] = build_groups(df)
    df["strat"] = df["species"].fillna(df["genus"])

    hard = ~df["label_source"].isin(GENUS_ONLY_SOURCES)
    df["split"] = "train"
    df.loc[hard, "split"] = assign_splits(df[hard], "strat").values
    # Bing genus-only: a held-out genus test of its own, never mixed into the main test.
    bing = df[~hard]
    if len(bing):
        df.loc[~hard, "split"] = assign_splits(bing, "genus", test_frac=0.10, val_frac=0.0).replace(
            {"test": "test_bing"}
        ).values

    # Leakage check: no group spans splits.
    spans = df.groupby("group")["split"].nunique()
    assert (spans == 1).all(), f"{(spans > 1).sum()} groups span splits"

    space = build_label_space(df[df["split"] == "train"], args.min_species)
    common = tax.common_names if isinstance(tax.common_names, dict) else {}
    space["common"] = {s: common[s] for s in space["species"] if s in common}
    df["family_idx"] = df["family"].map({c: i for i, c in enumerate(space["family"])}).fillna(-1).astype(int)
    df["genus_idx"] = df["genus"].map({c: i for i, c in enumerate(space["genus"])}).fillna(-1).astype(int)
    df["species_idx"] = df["species"].map({c: i for i, c in enumerate(space["species"])}).fillna(-1).astype(int)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_dir / "labeled.parquet", index=False)
    (args.out_dir / "label_space.json").write_text(json.dumps(space, indent=1))
    summary = {
        "n_crops": len(df),
        "splits": df["split"].value_counts().to_dict(),
        "label_level": df["label_level"].value_counts().to_dict(),
        "n_family": len(space["family"]),
        "n_genus": len(space["genus"]),
        "n_species_head": len(space["species"]),
        "n_groups": int(df["group"].nunique()),
        "test_species_with_head": int(
            df[(df["split"] == "test") & (df["species_idx"] >= 0)]["species"].nunique()
        ),
    }
    (args.out_dir / "labels_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
