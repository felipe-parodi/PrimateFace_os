"""Build an unlabeled face-crop pool from PrimateFace YouTube video frames.

The pool is intended for teacher pseudo-labeling (knowledge distillation) of the
genus/species classifier. Faces come from the cleaned PrimateFace v1 COCO file
(``instances_labels_f68_v1_clean.json``, ~173k frames).

Video provenance note:
    The YouTube frames are stored as ``yt_<random int>.jpg``; the integer is not
    ordered by video (consecutive ids and annotation ids show chance-level bbox
    overlap) and no frame-to-video map exists in the repository. ``video_id`` is
    therefore parsed from the file name only for the Budongo clips
    (``Budongo...Clip..._frame_N.jpg``). For every other frame, ``video_id`` is a
    *pseudo-video* key (``pv_XXXXX``) obtained by linking frames whose global
    16x16 colour thumbnails are near-identical (shared background / scene) and
    taking connected components. ``video_id_source`` records which applies.

Stages (all run by default, each resumable):
    1. select : filter faces (min(w, h) >= 32 px) and draw a deterministic
       uniform candidate set (seed 0).
    2. crop   : write ``square_crop(rgb, bbox, 1.25, 256)`` JPEGs (quality 92)
       with a moderate thread pool; also log a frame thumbnail per crop.
    3. group  : build pseudo-video groups from thumbnails, cap each group at
       ``--per-video-cap`` faces and fill uniformly up to ``--max-faces``.
    4. grid   : 6x8 contact sheet of random crops.

Usage:
    C:/pfspecies/.venv/Scripts/python.exe species_benchmark/build_yt_pool.py
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from primateface.taxon import square_crop

PF = Path(r"A:\NonEnclosureProjects\inprep\PrimateFace")
DEF_JSON = PF / "data" / "inventory_251228" / "instances_labels_f68_v1_clean.json"
DEF_IMG_ROOT = PF / "data" / "primatefacev1_173k" / "imgs"
OUT = Path(r"C:\pfspecies\data")
THUMB = 16
BUDONGO_RE = re.compile(r"^(Budongo.*)_frame_\d+\.\w+$")


def crop_id_for(image_path: str, bbox: tuple[float, float, float, float]) -> str:
    """Return a 16-hex sha1 id of image path plus bbox.

    Args:
        image_path: Absolute image path (A:\\ form).
        bbox: ``(x1, y1, x2, y2)`` in pixels.

    Returns:
        First 16 hex characters of the sha1 digest.
    """
    key = f"{image_path}|{bbox[0]:.2f},{bbox[1]:.2f},{bbox[2]:.2f},{bbox[3]:.2f}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def name_pattern(file_name: str) -> str:
    """Classify a frame file name into its naming family.

    Args:
        file_name: Bare file name.

    Returns:
        One of ``yt``, ``ytnew``, ``budongo``, ``numeric``, ``other``.
    """
    if file_name.startswith("ytnew_"):
        return "ytnew"
    if file_name.startswith("yt_"):
        return "yt"
    if file_name.startswith("Budongo"):
        return "budongo"
    if re.fullmatch(r"\d+\.\w+", file_name):
        return "numeric"
    return "other"


def load_faces(json_path: Path, img_root: Path, min_side: float) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load all face annotations and apply the size filter.

    Args:
        json_path: COCO json with images and face annotations.
        img_root: Directory holding the frames.
        min_side: Minimum of bbox width/height in pixels.

    Returns:
        Filtered face table and a dict of source statistics.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    images = {im["id"]: im for im in coco["images"]}
    rows = []
    n_missing_img = 0
    for a in coco["annotations"]:
        im = images.get(a["image_id"])
        if im is None:
            n_missing_img += 1
            continue
        x, y, w, h = [float(v) for v in a["bbox"][:4]]
        fn = im["file_name"]
        m = BUDONGO_RE.match(fn)
        rows.append(
            {
                "ann_id": int(a["id"]),
                "image_id": int(a["image_id"]),
                "file_name": fn,
                "image_path": str(img_root / fn),
                "x1": x, "y1": y, "x2": x + w, "y2": y + h, "w": w, "h": h,
                "name_pattern": name_pattern(fn),
                "parsed_video": m.group(1) if m else None,
            }
        )
    df = pd.DataFrame(rows)
    on_disk = {e.name for e in os.scandir(img_root)}
    exists = df["file_name"].isin(on_disk)
    stats = {
        "n_faces_file_missing": int((~exists).sum()),
        "n_images": len(images),
        "n_annotations": len(coco["annotations"]),
        "n_videos_parsed": int(df["parsed_video"].nunique()),
        "n_ann_missing_image": n_missing_img,
    }
    del coco
    df = df[exists].reset_index(drop=True)
    keep = np.minimum(df["w"], df["h"]) >= min_side
    stats["n_faces_ge_min_side"] = int(keep.sum())
    df = df[keep].reset_index(drop=True)
    df["crop_id"] = [
        crop_id_for(p, (a, b, c, d)) for p, a, b, c, d in zip(df.image_path, df.x1, df.y1, df.x2, df.y2)
    ]
    df = df.drop_duplicates("crop_id").reset_index(drop=True)
    stats["n_faces_unique"] = len(df)
    return df, stats


def _read_rgb(path: str) -> np.ndarray | None:
    """Read an image as RGB uint8, or None on failure."""
    buf = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def crop_one(row: dict, crop_dir: Path, need_thumb: bool) -> dict:
    """Write one crop and optionally compute the frame thumbnail.

    Args:
        row: Face row with ``crop_id``, ``image_path`` and bbox columns.
        crop_dir: Output directory for crops.
        need_thumb: Whether to compute the frame thumbnail.

    Returns:
        Log record with status and (optionally) base64 thumbnail.
    """
    out = crop_dir / f"{row['crop_id']}.jpg"
    rec = {"crop_id": row["crop_id"]}
    try:
        rgb = _read_rgb(row["image_path"])
        if rgb is None:
            rec["status"] = "read_fail"
            return rec
        if not out.exists():
            crop = square_crop(rgb, [row["x1"], row["y1"], row["x2"], row["y2"]], scale=1.25, size=256)
            tmp = out.with_suffix(".tmp.jpg")
            Image.fromarray(crop).save(tmp, format="JPEG", quality=92)
            tmp.replace(out)
        if need_thumb:
            th = cv2.resize(rgb, (THUMB, THUMB), interpolation=cv2.INTER_AREA)
            rec["thumb"] = base64.b64encode(th.tobytes()).decode("ascii")
        rec["status"] = "ok"
    except Exception as e:  # noqa: BLE001
        rec["status"] = f"error:{type(e).__name__}:{e}"[:300]
    return rec


def load_log(log_path: Path) -> dict[str, dict]:
    """Load the latest record per crop_id from the jsonl log."""
    recs: dict[str, dict] = {}
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                recs[r["crop_id"]] = r
    return recs


def run_crops(df: pd.DataFrame, crop_dir: Path, log_path: Path, threads: int) -> dict[str, dict]:
    """Extract crops for all rows of ``df`` (resumable).

    Args:
        df: Face rows to crop.
        crop_dir: Crop output directory.
        log_path: jsonl log of statuses and thumbnails.
        threads: Thread pool size.

    Returns:
        Mapping crop_id -> latest log record.
    """
    crop_dir.mkdir(parents=True, exist_ok=True)
    recs = load_log(log_path)
    todo = [
        r for r in df.to_dict("records")
        if not (recs.get(r["crop_id"], {}).get("status") == "ok"
                and "thumb" in recs[r["crop_id"]]
                and (crop_dir / f"{r['crop_id']}.jpg").exists())
    ]
    print(f"[crop] {len(df)} rows, {len(todo)} to do", flush=True)
    lock = threading.Lock()
    t0 = time.time()
    with open(log_path, "a", encoding="utf-8") as logf, ThreadPoolExecutor(threads) as ex:
        futs = [ex.submit(crop_one, r, crop_dir, True) for r in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            rec = fu.result()
            with lock:
                logf.write(json.dumps(rec) + "\n")
                recs[rec["crop_id"]] = rec
            if i % 2000 == 0:
                logf.flush()
                rate = i / (time.time() - t0)
                print(f"[crop] {i}/{len(todo)} {rate:.1f}/s eta {(len(todo) - i) / rate / 60:.1f} min", flush=True)
    return recs


def pseudo_video_groups(crop_ids: list[str], recs: dict[str, dict], thr: float) -> dict[str, str]:
    """Group frames into pseudo-videos by near-identical frame thumbnails.

    Thumbnails are mean-centred, L2-normalised colour vectors; pairs with cosine
    similarity >= ``thr`` among each frame's 10 nearest neighbours are linked and
    connected components become groups.

    Args:
        crop_ids: Crop ids with a thumbnail available.
        recs: Log records holding base64 thumbnails.
        thr: Cosine similarity threshold.

    Returns:
        Mapping crop_id -> ``pv_XXXXX`` label.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from sklearn.neighbors import NearestNeighbors

    X = np.stack([
        np.frombuffer(base64.b64decode(recs[c]["thumb"]), dtype=np.uint8).astype(np.float32) for c in crop_ids
    ])
    X -= X.mean(1, keepdims=True)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-6
    nn = NearestNeighbors(n_neighbors=11, n_jobs=8).fit(X)
    dist, idx = nn.kneighbors(X)
    sim = 1 - dist ** 2 / 2
    r, c = np.nonzero(sim[:, 1:] >= thr)
    cols = idx[:, 1:][r, c]
    g = coo_matrix((np.ones(len(r)), (r, cols)), shape=(len(X), len(X)))
    _, lab = connected_components(g, directed=False)
    return {cid: f"pv_{g_id:05d}" for cid, g_id in zip(crop_ids, lab)}


def cap_and_fill(df: pd.DataFrame, cap: int, k: int, seed: int) -> pd.DataFrame:
    """Cap faces per video_id, then fill uniformly from the rest up to ``k``.

    Args:
        df: Candidate rows with ``video_id``.
        cap: Max faces per video in the first pass.
        k: Target number of faces.
        seed: RNG seed.

    Returns:
        Selected rows.
    """
    rng = np.random.default_rng(seed)
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    rank = df.groupby("video_id").cumcount()
    first = df[rank < cap]
    if len(first) >= k:
        return first.iloc[:k]
    rest = df[rank >= cap]
    return pd.concat([first, rest.iloc[: k - len(first)]])


def contact_sheet(paths: list[Path], out: Path, rows: int = 6, cols: int = 8, tile: int = 128) -> None:
    """Write a rows x cols contact sheet PNG."""
    sheet = Image.new("RGB", (cols * tile, rows * tile), (255, 255, 255))
    for i, p in enumerate(paths[: rows * cols]):
        im = Image.open(p).convert("RGB").resize((tile, tile))
        sheet.paste(im, ((i % cols) * tile, (i // cols) * tile))
    sheet.save(out)


def main() -> None:
    """Run select, crop, group, and grid stages."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path, default=DEF_JSON)
    ap.add_argument("--img-root", type=Path, default=DEF_IMG_ROOT)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--min-side", type=float, default=32)
    ap.add_argument("--max-faces", type=int, default=60000)
    ap.add_argument("--per-video-cap", type=int, default=150)
    ap.add_argument("--candidates", type=int, default=64000,
                    help="uniform candidates cropped before pseudo-video capping")
    ap.add_argument("--sim-thr", type=float, default=0.97)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = args.out_dir
    crop_dir = out / "yt_crops"
    log_path = out / "yt_crop_log.jsonl"
    faces, stats = load_faces(args.json, args.img_root, args.min_side)
    print("[select]", stats, flush=True)

    rng = np.random.default_rng(args.seed)
    cand = faces.iloc[rng.permutation(len(faces))[: args.candidates]].reset_index(drop=True)
    recs = run_crops(cand, crop_dir, log_path, args.threads)

    ok = cand[[recs.get(c, {}).get("status") == "ok" and "thumb" in recs.get(c, {}) for c in cand.crop_id]].copy()
    fails = {c: recs.get(c, {}).get("status", "missing") for c in cand.crop_id if c not in set(ok.crop_id)}
    pv = pseudo_video_groups(ok.crop_id.tolist(), recs, args.sim_thr)
    ok["video_id"] = [pv[c] if pd.isna(p) else p for p, c in zip(ok.parsed_video, ok.crop_id)]
    ok["video_id_source"] = np.where(ok.parsed_video.notna(), "filename", "pseudo_thumb_cluster")
    sel = cap_and_fill(ok, args.per_video_cap, args.max_faces, args.seed)

    sel = sel.assign(crop_path=[str(crop_dir / f"{c}.jpg") for c in sel.crop_id])
    cols = ["crop_id", "image_path", "x1", "y1", "x2", "y2", "ann_id", "image_id", "video_id",
            "video_id_source", "name_pattern", "crop_path"]
    sel = sel[cols].sort_values("crop_id").reset_index(drop=True)
    sel.to_parquet(out / "yt_pool_manifest.parquet", index=False)

    gsz = ok.groupby("video_id").size()
    summary = dict(
        **stats,
        n_candidates=len(cand),
        n_crops_ok=len(ok),
        n_failures=len(fails),
        failures_sample=dict(list(fails.items())[:10]),
        n_groups_candidates=int(gsz.size),
        n_groups_over_cap=int((gsz > args.per_video_cap).sum()),
        largest_groups=gsz.sort_values(ascending=False).head(10).to_dict(),
        n_kept=len(sel),
        n_groups_kept=int(sel.video_id.nunique()),
        name_pattern_kept=sel.name_pattern.value_counts().to_dict(),
        sim_thr=args.sim_thr,
    )
    with open(out / "yt_pool_stats.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[done]", json.dumps(summary, indent=1), flush=True)

    pick = np.random.default_rng(args.seed).choice(len(sel), size=min(48, len(sel)), replace=False)
    contact_sheet([Path(sel.crop_path[i]) for i in pick], out / "yt_pool_grid.png")


if __name__ == "__main__":
    main()
