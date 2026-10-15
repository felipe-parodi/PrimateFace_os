"""Extract square face crops listed in ``faces_manifest.parquet``.

Each crop is centered on the detection box with side ``1.25 * max(w, h)``,
zero (black) padded where it leaves the image, resized to 256x256 with
INTER_AREA (identical to ``primateface.taxon.square_crop``) and saved as JPEG
(quality 92). Work is grouped per image so each source image is read
once. The run is resumable: crops that already exist are skipped. Results are
appended to ``crop_log.jsonl`` (one line per crop: ok / error).

Example:
    python extract_face_crops.py --manifest C:/pfspecies/data/faces_manifest.parquet \
        --workers 24 --sources api_inat,api_gbif
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None

OUT_SIZE = 256
SCALE = 1.25
QUALITY = 92


def square_crop(image: np.ndarray, bbox: Sequence[float], scale: float = SCALE, size: int = OUT_SIZE) -> np.ndarray:
    """Square face crop centred on ``bbox``, zero padded at borders.

    Verbatim copy of ``primateface.taxon.square_crop`` (kept local to avoid
    importing the full package in every worker); outputs must stay identical.

    Args:
        image: HxWx3 image (any channel order; returned unchanged).
        bbox: ``[x1, y1, x2, y2]`` in pixels.
        scale: Side length as a multiple of ``max(w, h)``.
        size: Output side length.

    Returns:
        ``size x size x 3`` uint8 crop.
    """
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * scale
    half = side / 2
    sx1, sy1 = int(round(cx - half)), int(round(cy - half))
    sx2, sy2 = int(round(cx + half)), int(round(cy + half))
    h, w = image.shape[:2]
    pad = max(0, -sx1, -sy1, sx2 - w, sy2 - h)
    if pad:
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        sx1, sy1, sx2, sy2 = sx1 + pad, sy1 + pad, sx2 + pad, sy2 + pad
    crop = image[sy1:sy2, sx1:sx2]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def process_image(image_path: str, orientation: str, faces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Read one image and write all of its face crops.

    Args:
        image_path: Source image path.
        orientation: ``exif`` to apply EXIF transpose, ``raw`` to keep stored pixels.
        faces: Dicts with crop_id, crop_path, x1, y1, x2, y2.

    Returns:
        One result dict per face.
    """
    results = []
    try:
        with Image.open(image_path) as im:
            im.load()
            raw_size = im.size
            if orientation == "exif":
                im = ImageOps.exif_transpose(im)
            arr = np.asarray(im.convert("RGB"))
    except Exception as exc:  # noqa: BLE001
        return [dict(crop_id=f["crop_id"], status="error", error=f"read: {type(exc).__name__}: {exc}"[:300])
                for f in faces]
    for f in faces:
        try:
            sub = square_crop(arr, (f["x1"], f["y1"], f["x2"], f["y2"]))
            out = Image.fromarray(np.ascontiguousarray(sub))
            tmp = f["crop_path"] + ".tmp"
            out.save(tmp, format="JPEG", quality=QUALITY)
            os.replace(tmp, f["crop_path"])
            results.append(dict(crop_id=f["crop_id"], status="ok", img_w=arr.shape[1], img_h=arr.shape[0],
                                raw_w=raw_size[0], raw_h=raw_size[1]))
        except Exception as exc:  # noqa: BLE001
            results.append(dict(crop_id=f["crop_id"], status="error", error=f"crop: {type(exc).__name__}: {exc}"[:300]))
    return results


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("C:/pfspecies/data/faces_manifest.parquet"))
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--sources", type=str, default="", help="Comma list; empty = all, in manifest priority order")
    ap.add_argument("--limit-images", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_parquet(args.manifest)
    if args.sources:
        order = [s.strip() for s in args.sources.split(",")]
        df = df[df["source"].isin(order)].copy()
        df["_o"] = df["source"].map({s: i for i, s in enumerate(order)})
        df = df.sort_values(["_o", "priority"])
    else:
        df = df.sort_values("priority")
    crops_dir = Path(df["crop_path"].iloc[0]).parent
    crops_dir.mkdir(parents=True, exist_ok=True)
    existing = {p[:-4] for p in os.listdir(crops_dir) if p.endswith(".jpg")}
    todo = df[~df["crop_id"].isin(existing)]
    print(f"faces={len(df)} existing={len(df) - len(todo)} todo={len(todo)}", flush=True)

    groups = []
    for (img, orient), g in todo.groupby(["image_path", "orientation"], sort=False):
        groups.append((img, orient, g[["crop_id", "crop_path", "x1", "y1", "x2", "y2"]].to_dict("records")))
    if args.limit_images:
        groups = groups[: args.limit_images]
    log_path = args.manifest.parent / "crop_log.jsonl"
    n_ok = n_err = 0
    t0 = time.time()
    with open(log_path, "a", encoding="utf-8") as log, ProcessPoolExecutor(args.workers) as ex:
        pending = set()
        it = iter(groups)
        done_imgs = 0

        def submit_more() -> None:
            while len(pending) < args.workers * 8:
                try:
                    img, orient, faces = next(it)
                except StopIteration:
                    return
                pending.add(ex.submit(process_image, img, orient, faces))

        submit_more()
        while pending:
            fut = next(as_completed(pending))
            pending.discard(fut)
            for r in fut.result():
                r["t"] = round(time.time(), 1)
                log.write(json.dumps(r) + "\n")
                if r["status"] == "ok":
                    n_ok += 1
                else:
                    n_err += 1
            done_imgs += 1
            if done_imgs % 1000 == 0:
                rate = done_imgs / (time.time() - t0)
                print(f"images {done_imgs}/{len(groups)} ok={n_ok} err={n_err} {rate:.1f} img/s", flush=True)
                log.flush()
            submit_more()
    print(f"DONE images={len(groups)} ok={n_ok} err={n_err} elapsed={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
