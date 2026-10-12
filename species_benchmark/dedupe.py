"""Near-duplicate detection on face crops via 64-bit perceptual hashes.

Duplicates (same photo re-uploaded, mirrored across iNat/GBIF/EOL, re-encoded) are
linked so that ``labels.build_groups`` keeps them in one split, and external test rows
that duplicate a training crop can be flagged.

Candidate pairs are found with multi-index hashing: the hash is split into
``max_dist + 1`` blocks, so any pair within ``max_dist`` bits shares at least one
identical block (pigeonhole), and only pairs sharing a block are compared.

Example:
    python -m species_benchmark.dedupe hash --crops-from C:/pfspecies/data/faces_manifest.parquet \
        --out C:/pfspecies/data/phash.parquet
    python -m species_benchmark.dedupe pairs --hashes C:/pfspecies/data/phash.parquet \
        --out C:/pfspecies/data/dup_pairs.parquet
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

MAX_BUCKET = 4000  # buckets larger than this are degenerate (flat/dark crops); skipped


def _phash(path: str) -> Optional[int]:
    import imagehash
    from PIL import Image

    try:
        h = imagehash.phash(Image.open(path).convert("L"), hash_size=8)
        return int(str(h), 16)
    except Exception:  # noqa: BLE001
        return None


def compute_hashes(paths: List[str], workers: int = 24) -> np.ndarray:
    """Perceptual hashes (uint64) for image paths; 0 where unreadable.

    Args:
        paths: Image paths.
        workers: Process count.

    Returns:
        Array of uint64 hashes.
    """
    with ProcessPoolExecutor(workers) as ex:
        hs = list(ex.map(_phash, paths, chunksize=512))
    return np.array([h if h is not None else 0 for h in hs], dtype=np.uint64)


_BYTE_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)


def _popcount(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.uint64)
    return _BYTE_POP[x.view(np.uint8).reshape(*x.shape, 8)].sum(-1)


def near_pairs(hashes: np.ndarray, max_dist: int = 4) -> pd.DataFrame:
    """All index pairs (i < j) with Hamming distance ≤ ``max_dist``.

    Args:
        hashes: (n,) uint64 hashes (0 = invalid, ignored).
        max_dist: Maximum Hamming distance.

    Returns:
        DataFrame with columns ``i``, ``j``, ``dist``.
    """
    n_blocks = max_dist + 1
    edges = np.linspace(0, 64, n_blocks + 1).astype(int)
    valid = np.where(hashes != 0)[0]
    found = set()
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = np.uint64((1 << (hi - lo)) - 1)
        block = (hashes[valid] >> np.uint64(lo)) & mask
        order = np.argsort(block, kind="stable")
        sb = block[order]
        cuts = np.flatnonzero(np.diff(sb)) + 1
        for grp in np.split(order, cuts):
            if len(grp) < 2 or len(grp) > MAX_BUCKET:
                continue
            idx = valid[grp]
            h = hashes[idx]
            d = _popcount(h[:, None] ^ h[None, :])
            a, b = np.where(np.triu(d <= max_dist, k=1))
            for x, y in zip(idx[a], idx[b]):
                key = (min(x, y), max(x, y))
                if key not in found:
                    found.add(key)
                    rows.append((key[0], key[1], 0))  # distance filled in below
    df = pd.DataFrame(rows, columns=["i", "j", "dist"])
    if len(df):
        df["dist"] = _popcount(hashes[df["i"].to_numpy()] ^ hashes[df["j"].to_numpy()])
    return df


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["hash", "pairs"])
    ap.add_argument("--crops-from", type=Path, nargs="+", help="parquet(s) with crop_id, crop_path")
    ap.add_argument("--hashes", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-dist", type=int, default=4)
    args = ap.parse_args()
    if args.cmd == "hash":
        frames = []
        for p in args.crops_from:
            d = pd.read_parquet(p, columns=["crop_id", "crop_path"])
            d["origin"] = p.stem
            frames.append(d)
        df = pd.concat(frames, ignore_index=True).drop_duplicates("crop_id")
        df = df[df["crop_path"].map(lambda q: Path(q).exists())].reset_index(drop=True)
        df["phash"] = compute_hashes(df["crop_path"].tolist()).astype(np.int64)
        df.to_parquet(args.out, index=False)
        print(f"hashed {len(df)} crops -> {args.out}")
    else:
        df = pd.read_parquet(args.hashes)
        pairs = near_pairs(df["phash"].to_numpy().astype(np.uint64), args.max_dist)
        pairs["crop_a"] = df["crop_id"].to_numpy()[pairs["i"]]
        pairs["crop_b"] = df["crop_id"].to_numpy()[pairs["j"]]
        pairs["origin_a"] = df["origin"].to_numpy()[pairs["i"]]
        pairs["origin_b"] = df["origin"].to_numpy()[pairs["j"]]
        pairs.to_parquet(args.out, index=False)
        print(f"{len(pairs)} near-duplicate pairs (<={args.max_dist} bits) -> {args.out}")
        print(pd.crosstab(pairs["origin_a"], pairs["origin_b"]))


if __name__ == "__main__":
    main()
