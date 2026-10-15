"""Write ``manifest_report.md`` summarizing the face manifest and crop run.

Example:
    python manifest_report.py --data-dir C:/pfspecies/data
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict

import pandas as pd

S3_NOTE = """## iNat S3 detections

- `data/acquisition/manifests/candidates/inat_s3_primates.jsonl` lists 256,335 S3 candidates. Of these,
  {s3_dl:,} have a file in `data/acquisition/raw/images/inaturalist_s3/`. No detection manifest references
  those files or their candidate ids (no hits in `manifests/detections/` or `results/acquisition/`).
  The image-level table is `inat_s3_image_manifest.parquet`.
- The refresh pipeline did detect faces on a re-download of S3 candidates. `data/acquisition/refresh_260924/qc/faces.jsonl`
  holds 50,906 screened images (46,155 boxes, Cascade R-CNN `casc-rcnn_FaceDet_12k_250602`, score>=0.3, min side 32,
  raw EXIF orientation). The images are in `refresh_260924/raw/inaturalist_s3/`. Each refresh candidate maps back to an
  `inat_s3_primates` row through `original_candidate_id` in `refresh_260924/events.jsonl`.
  These faces are included as `source=refresh_inat_s3`. The remaining ~205k S3 candidates still lack detections.
- Other refresh runs with face QC and iNat or GBIF taxon metadata are also included: `inaturalist_live`, `inat_historical`,
  `inat_historical_expanded`, `historical_s3_next_260925`, `inat_august_snapshot`, `inat_all_grades`, `gbif_refresh`, and
  `treeoflife`. Refresh runs that were not ingested (no Linnaean label or out of scope) are commons*, ccr*, gorillawatch*,
  nkhotakota*, panaf20k, privi, safari2024, swg, wcs and youtube.
"""


def md_table(counts: Dict, header=("key", "n")) -> str:
    """Render a mapping as a markdown table."""
    lines = [f"| {header[0]} | {header[1]} |", "|---|---:|"]
    lines += [f"| {k} | {v:,} |" for k, v in counts.items()]
    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("C:/pfspecies/data"))
    args = ap.parse_args()
    d = args.data_dir
    df = pd.read_parquet(d / "faces_manifest.parquet")
    stats = json.loads((d / "manifest_build_stats.json").read_text())
    s3 = pd.read_parquet(d / "inat_s3_image_manifest.parquet")

    status: Dict[str, str] = {}
    errors: Dict[str, str] = {}
    log = d / "crop_log.jsonl"
    if log.exists():
        for line in open(log, encoding="utf-8"):
            r = json.loads(line)
            status[r["crop_id"]] = r["status"]
            if r["status"] != "ok":
                errors[r["crop_id"]] = r.get("error", "")
            else:
                errors.pop(r["crop_id"], None)
    on_disk = {p.name[:-4] for p in (d / "crops").iterdir() if p.name.endswith(".jpg")}
    df["crop_written"] = df["crop_id"].isin(on_disk)
    df["crop_failed"] = df["crop_id"].isin(errors) & ~df["crop_written"]

    per_source = df.groupby("source").agg(
        faces=("crop_id", "size"), crops=("crop_written", "sum"), failed=("crop_failed", "sum"),
        images=("image_path", "nunique"), genera_raw=("genus_raw", "nunique"),
        species_raw=("scientific_name", "nunique")).sort_values("faces", ascending=False)
    err_kinds = Counter(e.split(":")[0] + ": " + e.split(":")[1].strip() for e in errors.values() if ":" in e)
    genus = df.groupby("genus_raw").agg(faces=("crop_id", "size"), crops=("crop_written", "sum"),
                                        api_inat_gbif=("source", lambda s: int(s.str.startswith("api").sum())),
                                        refresh=("source", lambda s: int(s.str.startswith("refresh").sum())),
                                        bing=("source", lambda s: int((s == "bing").sum())))
    genus = genus.sort_values("faces", ascending=False)

    out = ["# Species face-crop manifest report", "",
           f"- Manifest is `{d / 'faces_manifest.parquet'}`. It has {len(df):,} faces from {df['image_path'].nunique():,} images.",
           f"- Crops are in `{d / 'crops'}`. {int(df['crop_written'].sum()):,} are written and {int(df['crop_failed'].sum()):,} failed.",
           "- Each crop is a square of side 1.25*max(w,h) centred on the box, zero-padded, resized to 256 with INTER_AREA, and saved as JPEG at q92.",
           "  This matches `primateface.taxon.square_crop`.",
           "- Faces were kept when min(w,h) >= 24 px and det_score >= 0.5. Images duplicated across sources (same sha256 or iNat photo id)",
           "  were kept only in the highest-priority source, in the order API > refresh > Bing.",
           "- API and Bing boxes use EXIF-transposed pixels, which matches cv2.imread in the original detector. Refresh boxes use raw stored pixels.", "",
           "## Filtering", "",
           f"- Raw faces per source: {stats.get('raw_faces_per_source')}",
           f"- Dropped as small: {stats.get('dropped_small')}",
           f"- Dropped for low score: {stats.get('dropped_lowscore')}",
           f"- Dropped as duplicate images: {stats.get('dropped_duplicate_image')}. Dropped as duplicate boxes: {stats.get('dropped_duplicate_box')}", "",
           "## Per source", "", per_source.to_markdown(), "",
           "## Label source", "", md_table(df["label_source"].value_counts().to_dict(), ("label_source", "faces")), "",
           "## License", "", md_table(df["license"].fillna("missing").value_counts().to_dict(), ("license", "faces")), "",
           "## Crop failures", "", md_table(dict(err_kinds.most_common(10)), ("error", "n")) if err_kinds else "None.", "",
           S3_NOTE.format(s3_dl=int(s3["downloaded"].sum())), "",
           f"## Per genus (raw, {genus.shape[0]} values, including some higher-rank names)", "", genus.to_markdown(), ""]
    (d / "manifest_report.md").write_text("\n".join(out), encoding="utf-8")
    print(per_source.to_string())


if __name__ == "__main__":
    main()
