"""Build the face-level manifest for the species/genus face-crop classifier.

Joins existing acquisition manifests and face detections into a single
parquet table with one row per detected face. Sources, in priority order:

1. GBIF / iNaturalist API acquisition (mega_81taxa + scale_28taxa trios).
2. Refresh-260924 face QC runs (iNat S3 / live / historical / all-grades /
   August snapshot, Tree of Life, GBIF refresh), joined to their event ledgers.
3. Bing genus folders with boxes from ``pose_labels_final.json``.

Images that appear in several sources (same sha256, or same iNat photo id) are
kept only for the highest-priority source.

Also writes an image-level manifest for the iNat S3 candidate file.

Example:
    python build_faces_manifest.py --out-dir C:/pfspecies/data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd

ROOT = Path(r"A:\NonEnclosureProjects\inprep\PrimateFace")
ACQ = ROOT / "data" / "acquisition"
MAN = ACQ / "manifests"
REFRESH = ACQ / "refresh_260924"
BING_COCO = ROOT / "results" / "pl_bingimgs_250512" / "annotations" / "pose_labels_final.json"

UNC_PREFIX = "\\\\10.0.0.4\\fparodi\\"
INAT_GBIF_DATASET = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"

MIN_SIDE = 24
MIN_SCORE = 0.5

# (subdir relative to REFRESH ('' = root), source label), in dedupe priority order.
REFRESH_RUNS = [
    ("", "refresh_inat_s3"),
    ("inaturalist_live", "refresh_inat_live"),
    ("inat_historical", "refresh_inat_historical"),
    ("inat_historical_expanded", "refresh_inat_historical_expanded"),
    ("historical_s3_next_260925", "refresh_inat_historical_s3_next"),
    ("inat_august_snapshot", "refresh_inat_august_snapshot"),
    ("inat_all_grades", "refresh_inat_all_grades"),
    ("gbif_refresh", "refresh_gbif"),
    ("treeoflife", "refresh_treeoflife"),
]

BING_SKIP_FOLDERS = {"adobe_infant", "labels", "cercopithecus.cache"}


def to_a_path(p: Optional[str]) -> Optional[str]:
    """Normalize UNC or repo-relative paths to the ``A:\\`` form.

    Args:
        p: Path string (UNC, absolute, or relative to the repo root).

    Returns:
        Normalized Windows path string, or None for empty input.
    """
    if not p:
        return None
    p = str(p).replace("/", "\\")
    if p.lower().startswith(UNC_PREFIX.lower()):
        return "A:\\" + p[len(UNC_PREFIX):]
    if re.match(r"^[A-Za-z]:\\", p):
        return p
    return str(ROOT) + "\\" + p.lstrip("\\")


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield parsed rows from a JSONL file, skipping malformed lines.

    Args:
        path: JSONL file path.

    Yields:
        Parsed JSON objects.
    """
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def norm_license(raw: Optional[str]) -> Optional[str]:
    """Map heterogeneous license strings/URLs to short codes.

    Args:
        raw: License short name or URL.

    Returns:
        Short code such as ``cc-by-nc``, ``cc0``, ``all-rights-reserved``; or
        the lowercased raw value when unrecognized; None if missing.
    """
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in ("", "none", "null", "nan"):
        return None
    if "publicdomain/zero" in s or s in ("cc0", "cc-0") or "cc0" in s:
        return "cc0"
    if "publicdomain/mark" in s or s == "pd":
        return "public-domain"
    if "all rights reserved" in s or s == "all-rights-reserved":
        return "all-rights-reserved"
    m = re.search(r"licenses/([a-z\-]+)/", s)
    if m:
        return "cc-" + m.group(1)
    s = s.replace("_", "-").replace(" ", "-")
    if s.startswith("cc-"):
        return s
    if s.startswith("by"):
        return "cc-" + s
    if "copyright" in s or len(s) > 40:
        return "other-restrictive"
    return s


def genus_of(name: Optional[str]) -> Optional[str]:
    """Return the first token of a scientific name, capitalized."""
    if not name:
        return None
    tok = str(name).strip().split()
    return tok[0].capitalize() if tok else None


def rank_from_name(name: Optional[str]) -> Optional[str]:
    """Infer taxon rank from token count of a canonical name."""
    if not name:
        return None
    n = len(str(name).split())
    return {1: "genus", 2: "species"}.get(n, "subspecies" if n >= 3 else None)


def canonical(name: Optional[str]) -> Optional[str]:
    """Strip authorship from a scientific name (keep Latin tokens)."""
    if not name:
        return None
    toks = []
    for t in str(name).split():
        if not re.match(r"^[A-Za-z][a-z\-]*$", t) or (toks and t[0].isupper()):
            break
        toks.append(t)
    return " ".join(toks) if toks else str(name)


def make_crop_id(image_path: str, x1: float, y1: float, x2: float, y2: float) -> str:
    """Stable 16-hex crop id from image path and integer box."""
    key = f"{image_path.lower()}|{int(round(x1))},{int(round(y1))},{int(round(x2))},{int(round(y2))}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def nz(*vals: Any) -> Any:
    """Return the first value that is not None/empty/'None'."""
    for v in vals:
        if v is None:
            continue
        if isinstance(v, str) and v.strip() in ("", "None", "null", "nan"):
            continue
        return v
    return None


def inat_label_source(grade: Optional[str]) -> str:
    """Map an iNat quality grade to a label_source tag."""
    g = (grade or "unknown").lower()
    return {"research": "inat_research", "needs_id": "inat_needs_id", "casual": "inat_casual"}.get(g, f"inat_{g}")


# --------------------------------------------------------------------------- API


def load_api(prefix: str) -> List[Dict[str, Any]]:
    """Load faces for one GBIF/iNat API acquisition trio.

    Args:
        prefix: Manifest stem, e.g. ``gbif_inat_mega_81taxa_species_260317``.

    Returns:
        List of face row dicts (unfiltered).
    """
    cands = {d["candidate_id"]: d for d in iter_jsonl(MAN / "candidates" / f"{prefix}_merged.jsonl")}
    dls = {d["candidate_id"]: d for d in iter_jsonl(MAN / "downloads" / f"{prefix}_downloads.jsonl")}
    rows = []
    for f in iter_jsonl(MAN / "detections" / f"{prefix}_faces.jsonl"):
        if f.get("status") not in (None, "detected"):
            continue
        cid = f["candidate_id"]
        c = cands.get(cid, {})
        dl = dls.get(cid, {})
        src = c.get("source_name") or f.get("source_name")
        if src == "inaturalist":
            sci = nz(c.get("taxon_name"), c.get("scientific_name"))
            genus = genus_of(sci)
            rank = nz(c.get("taxon_rank"), rank_from_name(sci))
            label_source = inat_label_source(c.get("quality_grade"))
            group_id = nz(c.get("observation_id"))
            group_id = f"inat_obs:{group_id}" if group_id else None
            observer = nz(c.get("user_login"), c.get("artist"))
            observer = f"inat_user:{observer}" if observer else None
            lic = norm_license(nz(c.get("license_short_name"), c.get("license_url")))
            photo_id = nz(c.get("photo_id"))
            source = "api_inat"
        else:
            sci = nz(c.get("gbif_species"), c.get("scientific_name"))
            genus = nz(c.get("gbif_genus"), genus_of(sci))
            rank = rank_from_name(sci)
            is_inat = c.get("dataset_key") == INAT_GBIF_DATASET
            label_source = "gbif_inat_research" if is_inat else "gbif"
            occ = nz(c.get("occurrence_key"), c.get("gbif_id"))
            group_id = f"gbif_occ:{occ}" if occ else None
            observer = nz(c.get("rights_holder"), c.get("artist"))
            observer = f"gbif_person:{observer}" if observer else None
            lic = norm_license(nz(c.get("license_url"), c.get("license_short_name")))
            photo_id = None
            source = "api_gbif"
        img = to_a_path(nz(f.get("image_path"), dl.get("local_path")))
        rows.append(dict(
            source=source, source_run=prefix, candidate_id=cid, image_path=img,
            image_sha256=nz(dl.get("sha256"), f.get("asset_id")),
            photo_id=str(photo_id) if photo_id else None,
            img_w=f.get("image_width"), img_h=f.get("image_height"),
            x1=float(f["x1"]), y1=float(f["y1"]), x2=float(f["x2"]), y2=float(f["y2"]),
            det_score=float(f["score"]), scientific_name=sci, genus_raw=genus,
            taxon_rank=rank, label_source=label_source, quality_grade=c.get("quality_grade"),
            license=lic, license_raw=nz(c.get("license_url"), c.get("license_short_name")),
            artist=c.get("artist"), group_id=group_id, observer_id=observer,
            page_url=c.get("page_url"), query_raw=c.get("query"),
            orientation="exif",
        ))
    return rows


# ----------------------------------------------------------------------- refresh


def _maybe_dict(v: Any) -> Dict[str, Any]:
    return v if isinstance(v, dict) else {}


def load_refresh_ledger(path: Path) -> Dict[str, Dict[str, Any]]:
    """Collapse a refresh events.jsonl into per-candidate metadata.

    Args:
        path: events.jsonl path.

    Returns:
        Mapping candidate_id -> compact metadata dict.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for e in iter_jsonl(path):
        cid = e.get("candidate_id")
        if not cid:
            continue
        st = _maybe_dict(e.get("source_taxon"))
        stx = _maybe_dict(e.get("source_taxonomy"))
        cat = _maybe_dict(e.get("source_catalog_row"))
        sci = nz(e.get("scientific_name"), stx.get("name"), st.get("name"), st.get("species"),
                 canonical(st.get("scientificName")), cat.get("scientific_name"))
        rank = nz(e.get("taxon_rank"), stx.get("rank"), st.get("rank"),
                  (st.get("taxonRank") or "").lower() or None, rank_from_name(sci))
        rec = dict(
            source_name=e.get("source_name"),
            scientific_name=sci,
            genus_raw=nz(e.get("genus"), e.get("genus_proposal"), st.get("genus"), cat.get("genus"), genus_of(sci)),
            taxon_rank=rank,
            quality_grade=e.get("quality_grade"),
            license_raw=nz(e.get("photo_license"), e.get("license_short_name"), e.get("occurrence_license")),
            artist=nz(e.get("photo_attribution"), e.get("artist")),
            observation=nz(e.get("observation_id"), e.get("observation_uuid")),
            occurrence=nz(e.get("occurrence_key")),
            catalog_source_id=nz(cat.get("source_id")),
            catalog_data_source=nz(cat.get("data_source")),
            catalog_publisher=nz(cat.get("publisher")),
            observer=nz(e.get("observer"), e.get("observer_login"), e.get("observer_id"),
                        e.get("photo_observer_id"), e.get("observation_observer_id")),
            photo_id=nz(e.get("photo_id")),
            page_url=nz(e.get("page_url"), e.get("media_url")),
            original_candidate_id=nz(e.get("original_candidate_id")),
        )
        prev = out.get(cid)
        if prev is None:
            out[cid] = rec
        else:
            for k, v in rec.items():
                if v is not None:
                    prev[k] = v
    return out


def load_refresh_run(subdir: str, source: str) -> List[Dict[str, Any]]:
    """Load faces from one refresh QC run joined to its ledger.

    Args:
        subdir: Run directory under refresh_260924 ('' for the root S3 run).
        source: Source label written into the manifest.

    Returns:
        List of face row dicts (unfiltered).
    """
    base = REFRESH / subdir if subdir else REFRESH
    faces_path = base / "qc" / "faces.jsonl"
    ledger_path = base / "events.jsonl"
    if not faces_path.exists():
        return []
    ledger = load_refresh_ledger(ledger_path) if ledger_path.exists() else {}
    rows = []
    for q in iter_jsonl(faces_path):
        boxes = q.get("boxes") or []
        if not boxes:
            continue
        cid = q["candidate_id"]
        m = ledger.get(cid, {})
        is_inat = source.startswith("refresh_inat")
        if is_inat:
            label_source = inat_label_source(m.get("quality_grade"))
            group_id = f"inat_obs:{m['observation']}" if m.get("observation") else None
            observer = f"inat_user:{m['observer']}" if m.get("observer") else None
        elif source == "refresh_gbif":
            label_source = "gbif"
            group_id = f"gbif_occ:{m['occurrence']}" if m.get("occurrence") else None
            observer = None
        else:  # treeoflife
            ds = m.get("catalog_data_source") or "unknown"
            label_source = f"treeoflife_{ds}"
            group_id = f"tol_{ds}:{m['catalog_source_id']}" if m.get("catalog_source_id") else None
            observer = f"tol_publisher:{m['catalog_publisher']}" if m.get("catalog_publisher") else None
        img = to_a_path(q.get("local_path"))
        for b in boxes:
            x1, y1, x2, y2 = b["bbox_xyxy"]
            rows.append(dict(
                source=source, source_run=f"refresh_260924/{subdir or 'root'}", candidate_id=cid,
                image_path=img, image_sha256=q.get("sha256"),
                photo_id=str(m["photo_id"]) if m.get("photo_id") else None,
                img_w=q.get("width"), img_h=q.get("height"),
                x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2), det_score=float(b["score"]),
                scientific_name=m.get("scientific_name"), genus_raw=m.get("genus_raw"),
                taxon_rank=m.get("taxon_rank"), label_source=label_source,
                quality_grade=m.get("quality_grade"),
                license=norm_license(m.get("license_raw")) or (
                    "all-rights-reserved" if "all rights reserved" in str(m.get("artist") or "").lower() else None),
                license_raw=m.get("license_raw"), artist=m.get("artist"),
                group_id=group_id, observer_id=observer, page_url=m.get("page_url"),
                query_raw=m.get("original_candidate_id"), orientation="raw",
            ))
    return rows


# -------------------------------------------------------------------------- Bing


def load_bing() -> List[Dict[str, Any]]:
    """Load Bing faces from the pose pseudo-label COCO file.

    Returns:
        List of face row dicts (unfiltered).
    """
    with open(BING_COCO, encoding="utf-8") as fh:
        coco = json.load(fh)
    imgs = {im["id"]: im for im in coco["images"]}
    rows = []
    for a in coco["annotations"]:
        im = imgs.get(a["image_id"])
        if im is None:
            continue
        path = to_a_path(im["file_name"])
        parts = path.split("\\")
        try:
            folder = parts[parts.index("bing_images") + 1]
        except ValueError:
            continue
        if folder in BING_SKIP_FOLDERS:
            continue
        x, y, w, h = a["bbox"]
        stem = Path(parts[-1]).stem
        q = re.sub(r"_\d+$", "", stem)
        q = q[len(folder) + 1:] if q.lower().startswith(folder.lower() + "_") else ""
        rows.append(dict(
            source="bing", source_run="pl_bingimgs_250512", candidate_id=None, image_path=path,
            image_sha256=None, photo_id=None, img_w=im.get("width"), img_h=im.get("height"),
            x1=float(x), y1=float(y), x2=float(x + w), y2=float(y + h),
            det_score=float(a.get("det_confidence", a.get("score", 1.0))),
            scientific_name=None, genus_raw=folder.capitalize(), taxon_rank="genus",
            label_source="bing_folder", quality_grade=None, license="unknown_web",
            license_raw=None, artist=None, group_id=f"bing_img:{stem}", observer_id=None,
            page_url=None, query_raw=q.replace("_", " ") or None, orientation="exif",
        ))
    return rows


# --------------------------------------------------------------------------- S3


def build_s3_image_manifest(out_path: Path) -> Dict[str, Any]:
    """Write the image-level manifest for iNat S3 candidates.

    Args:
        out_path: Destination parquet path.

    Returns:
        Summary statistics dict.
    """
    img_dir = ACQ / "raw" / "images" / "inaturalist_s3"
    files = {}
    for p in img_dir.iterdir():
        files.setdefault(p.stem, p.name)
    rows = []
    for c in iter_jsonl(MAN / "candidates" / "inat_s3_primates.jsonl"):
        cid = c["candidate_id"]
        fn = files.get(cid)
        rows.append(dict(
            candidate_id=cid, source="inat_s3", photo_id=c.get("photo_id"),
            observation_id=c.get("observation_id"), observer_id=c.get("observer_login"),
            scientific_name=c.get("scientific_name"), taxon_id_raw=c.get("taxon_id"),
            inat_taxon_id=c.get("inat_taxon_id"), taxon_rank=c.get("taxon_rank"),
            quality_grade=c.get("quality_grade"), license=norm_license(c.get("license_short_name")),
            license_raw=c.get("license_short_name"), observed_on=c.get("observed_on"),
            latitude=c.get("decimal_latitude"), longitude=c.get("decimal_longitude"),
            media_url=c.get("media_url"),
            image_path=str(img_dir / fn) if fn else None, downloaded=fn is not None,
        ))
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    return dict(n_candidates=len(df), n_downloaded=int(df["downloaded"].sum()),
                n_files_in_dir=len(files),
                quality=df["quality_grade"].value_counts().to_dict(),
                license=df["license"].value_counts(dropna=False).to_dict())


# -------------------------------------------------------------------------- main


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Keep each image only in its highest-priority source.

    Duplicate images are identified by sha256 or by iNat photo id. Rows are
    assumed to be ordered by source priority.

    Args:
        df: Face rows in priority order.

    Returns:
        Deduplicated frame.
    """
    seen_sha: Dict[str, str] = {}
    seen_photo: Dict[str, str] = {}
    keep = []
    for img, sha, pid in zip(df["image_path"], df["image_sha256"], df["photo_id"]):
        sha = sha if isinstance(sha, str) and sha else None
        pid = pid if isinstance(pid, str) and pid else None
        owner = None
        if sha and sha in seen_sha:
            owner = seen_sha[sha]
        elif pid and pid in seen_photo:
            owner = seen_photo[pid]
        if owner is not None and owner != img:
            keep.append(False)
            continue
        if sha:
            seen_sha.setdefault(sha, img)
        if pid:
            seen_photo.setdefault(pid, img)
        keep.append(True)
    return df[pd.Series(keep, index=df.index)]


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("C:/pfspecies/data"))
    ap.add_argument("--skip-s3", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = args.out_dir / "crops"
    stats: Dict[str, Any] = {}

    parts: List[pd.DataFrame] = []
    for prefix in ("gbif_inat_mega_81taxa_species_260317", "gbif_inat_scale_28taxa_260317"):
        r = load_api(prefix)
        print(prefix, len(r), flush=True)
        parts.append(pd.DataFrame(r))
    for subdir, source in REFRESH_RUNS:
        r = load_refresh_run(subdir, source)
        print(source, len(r), flush=True)
        if r:
            parts.append(pd.DataFrame(r))
    r = load_bing()
    print("bing", len(r), flush=True)
    parts.append(pd.DataFrame(r))

    df = pd.concat(parts, ignore_index=True)
    stats["raw_faces_per_source"] = df["source"].value_counts().to_dict()
    w = df["x2"] - df["x1"]
    h = df["y2"] - df["y1"]
    small = pd.concat([w, h], axis=1).min(axis=1) < MIN_SIDE
    low = df["det_score"] < MIN_SCORE
    stats["dropped_small"] = df[small].groupby("source").size().to_dict()
    stats["dropped_lowscore"] = df[low & ~small].groupby("source").size().to_dict()
    df = df[~small & ~low].copy()
    n_before = len(df)
    df = dedupe(df)
    stats["dropped_duplicate_image"] = n_before - len(df)
    df["crop_id"] = [make_crop_id(p, a, b, c, d) for p, a, b, c, d in
                     zip(df["image_path"], df["x1"], df["y1"], df["x2"], df["y2"])]
    n_before = len(df)
    df = df.drop_duplicates("crop_id")
    stats["dropped_duplicate_box"] = n_before - len(df)
    df["crop_path"] = [str(crops_dir / f"{c}.jpg") for c in df["crop_id"]]
    df["priority"] = range(len(df))
    for col in ("img_w", "img_h"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    cols = ["crop_id", "source", "image_path", "x1", "y1", "x2", "y2", "det_score",
            "scientific_name", "genus_raw", "taxon_rank", "label_source", "license",
            "group_id", "observer_id", "crop_path"]
    rest = [c for c in df.columns if c not in cols]
    df = df[cols + rest]
    df.to_parquet(args.out_dir / "faces_manifest.parquet", index=False)
    stats["final_faces_per_source"] = df["source"].value_counts().to_dict()
    print(json.dumps(stats, indent=1, default=str), flush=True)

    if not args.skip_s3:
        stats["s3"] = build_s3_image_manifest(args.out_dir / "inat_s3_image_manifest.parquet")
        print(json.dumps(stats["s3"], indent=1, default=str), flush=True)
    (args.out_dir / "manifest_build_stats.json").write_text(json.dumps(stats, indent=1, default=str))


if __name__ == "__main__":
    main()
