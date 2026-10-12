"""Full-pipeline demo: detection + landmarks + taxon labels on images and video frames.

Runs in an environment with MMDetection/MMPose (e.g. the ``primateface`` conda env).

Example:
    python -m species_benchmark.demo --taxon-model C:/pfspecies/export/taxon_convnext_atto.pt \
        --videos A:/NonEnclosureProjects/inprep/PrimateFace/demos --out C:/pfspecies/report/demos
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import cv2
import numpy as np

from primateface import PrimateFace

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mts"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
COLORS = {"species": (115, 158, 0), "genus": (0, 159, 230), "family": (0, 94, 213)}  # BGR, Okabe-Ito


def annotate(pf: PrimateFace, bgr: np.ndarray) -> tuple[np.ndarray, List[dict]]:
    """Run the pipeline and draw landmarks plus a taxon label per face.

    Args:
        pf: Initialised :class:`PrimateFace` with a taxon model.
        bgr: Input image.

    Returns:
        (annotated image, list of per-face dicts)
    """
    faces = pf.analyze(bgr)
    canvas = PrimateFace.draw(faces, bgr, draw_bbox=False) if faces else bgr.copy()
    scale = max(canvas.shape[:2]) / 1000
    records = []
    for f in faces:
        t = f.taxon
        x1, y1, x2, y2 = f.bbox.astype(int)
        color = COLORS[t.level]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, max(2, int(3 * scale)))
        common = pf._taxon_clf.common_names.get(t.label, "") if t.level == "species" else ""
        text = f"{t.label} {t.confidence:.2f}" + (f" ({common})" if common else "")
        fs = 0.6 * max(scale, 0.6)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        ty = max(th + 6, y1 - 6)
        cv2.rectangle(canvas, (x1, ty - th - 6), (x1 + tw + 6, ty + 4), color, -1)
        cv2.putText(canvas, text, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)
        records.append({"bbox": f.bbox.tolist(), "det_score": f.score, **t.to_dict()})
    return canvas, records


def sample_frames(path: Path, n: int) -> List[np.ndarray]:
    """Evenly sample ``n`` frames from a video (skipping the first/last 10%)."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for idx in np.linspace(total * 0.1, total * 0.9, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    cap.release()
    return frames


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--taxon-model", required=True)
    ap.add_argument("--videos", type=Path, default=None, help="directory of demo videos")
    ap.add_argument("--images", type=Path, default=None, help="directory of images")
    ap.add_argument("--frames-per-video", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pf = PrimateFace(device=args.device, taxon_model=args.taxon_model)
    summary = {}
    if args.videos:
        for v in sorted(p for p in args.videos.iterdir() if p.suffix.lower() in VIDEO_EXTS):
            for k, fr in enumerate(sample_frames(v, args.frames_per_video)):
                img, recs = annotate(pf, fr)
                cv2.imwrite(str(args.out / f"{v.stem}_f{k}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                summary[f"{v.stem}_f{k}"] = recs
                print(v.stem, k, [(r["label"], round(r["confidence"], 2)) for r in recs], flush=True)
    if args.images:
        for p in sorted(q for q in args.images.iterdir() if q.suffix.lower() in IMAGE_EXTS):
            img, recs = annotate(pf, cv2.imread(str(p)))
            cv2.imwrite(str(args.out / f"{p.stem}_taxon.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            summary[p.stem] = recs
            print(p.stem, [(r["label"], round(r["confidence"], 2)) for r in recs], flush=True)
    (args.out / "demo_predictions.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
