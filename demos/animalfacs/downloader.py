"""Phase B: Download AnimalFACS videos and manuals from Google Drive.

Uses gdown for folder and file downloads with retry logic and
graceful degradation when rate-limited.
"""

import csv
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import gdown
import requests

from .utils import find_videos, load_config

logger = logging.getLogger("animalfacs.downloader")

# Regex to extract file IDs from Google Drive folder HTML
_GDRIVE_FILE_ID_RE = re.compile(r'"([a-zA-Z0-9_-]{20,})"')


def _parse_folder_file_ids(folder_id: str) -> List[Dict[str, str]]:
    """Parse individual file IDs from a Google Drive folder page.

    Fetches the folder HTML and extracts file IDs + names.

    Args:
        folder_id: Google Drive folder ID.

    Returns:
        List of dicts with 'id' and 'name' keys.
    """
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning("Failed to fetch folder page %s: %s", folder_id, e)
        return []

    # gdown provides a utility for this
    try:
        from gdown.download_folder import _parse_google_drive_file

        return_code, gdrive_file = _parse_google_drive_file(
            url, content=resp.text
        )
        if return_code and gdrive_file:
            files = []
            for child in getattr(gdrive_file, "children", []):
                if not child.is_folder:
                    files.append({"id": child.id, "name": child.name})
                else:
                    # Recurse into subfolders
                    sub_files = _parse_folder_file_ids(child.id)
                    for sf in sub_files:
                        sf["name"] = f"{child.name}/{sf['name']}"
                    files.extend(sub_files)
            return files
    except Exception as e:
        logger.debug("gdown folder parse failed: %s, trying regex", e)

    return []


def _download_folder_robust(gdrive_id: str, output_dir: Path) -> bool:
    """Download a Google Drive folder with robust fallback.

    1. Try gdown.download_folder() (works for <=50 files)
    2. On failure: parse folder HTML for file IDs, download individually

    Args:
        gdrive_id: Google Drive folder ID.
        output_dir: Local directory to save files into.

    Returns:
        True if any files were downloaded.
    """
    url = f"https://drive.google.com/drive/folders/{gdrive_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Attempt 1: gdown folder download with remaining_ok
    # This downloads up to ~50 files and continues without error
    try:
        gdown.download_folder(
            url, output=str(output_dir), quiet=False, remaining_ok=True
        )
        if any(output_dir.rglob("*")):
            n_files = sum(1 for _ in output_dir.rglob("*") if _.is_file())
            logger.info("Downloaded %d files from folder (may be partial)", n_files)
            return True
    except Exception as e:
        logger.info("gdown folder download failed: %s. Trying individual files.", e)

    # Attempt 2: parse folder HTML → download files individually
    logger.info("Parsing folder %s for individual file IDs...", gdrive_id)
    files = _parse_folder_file_ids(gdrive_id)
    if not files:
        logger.warning("Could not parse any files from folder %s", gdrive_id)
        return False

    logger.info("Found %d files in folder. Downloading individually...", len(files))
    n_ok = 0
    for finfo in files:
        fid = finfo["id"]
        fname = finfo["name"]
        # Create subdirectory structure if name contains /
        out_path = output_dir / fname
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            dl_url = f"https://drive.google.com/uc?id={fid}"
            gdown.download(dl_url, str(out_path), quiet=True, fuzzy=True)
            if out_path.exists():
                n_ok += 1
        except Exception as e:
            logger.warning("Failed to download %s: %s", fname, e)

    logger.info("Downloaded %d/%d files from folder %s", n_ok, len(files), gdrive_id)
    return n_ok > 0


def _download_file(gdrive_id: str, output_path: Path) -> bool:
    """Download a single Google Drive file.

    Args:
        gdrive_id: Google Drive file ID.
        output_path: Local path to save to.

    Returns:
        True if download succeeded.
    """
    url = f"https://drive.google.com/uc?id={gdrive_id}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        gdown.download(url, str(output_path), quiet=False, fuzzy=True)
        return output_path.exists()
    except Exception as e:
        logger.warning("File download failed for %s: %s", gdrive_id, e)
        return False


def download_species(
    species_id: str,
    sp_manifest: Dict[str, Any],
    data_root: Path,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Download all resources for one species.

    Args:
        species_id: Species identifier.
        sp_manifest: Manifest entry for this species.
        data_root: Root data directory.
        dry_run: If True, only log what would be downloaded.

    Returns:
        Download status dict.
    """
    status = {
        "species_id": species_id,
        "manual": "skipped",
        "training_videos": "skipped",
        "test_materials": "skipped",
    }
    raw_dir = data_root / "raw" / species_id

    # Manual
    manual_id = sp_manifest.get("manual_gdrive_id")
    if manual_id:
        manual_dir = raw_dir / "manual"
        if dry_run:
            logger.info("[DRY RUN] Would download manual for %s", species_id)
            status["manual"] = "dry_run"
        else:
            manual_type = sp_manifest.get("manual_type", "file")
            if manual_type == "folder":
                ok = _download_folder_robust(manual_id, manual_dir)
            else:
                manual_dir.mkdir(parents=True, exist_ok=True)
                ok = _download_file(
                    manual_id, manual_dir / f"{species_id}_manual.pdf"
                )
            status["manual"] = "success" if ok else "failed"

    # Training videos
    tv_id = sp_manifest.get("training_videos_gdrive_id")
    if tv_id:
        tv_dir = raw_dir / "training_videos"
        if dry_run:
            logger.info(
                "[DRY RUN] Would download training videos for %s", species_id
            )
            status["training_videos"] = "dry_run"
        else:
            ok = _download_folder_robust(tv_id, tv_dir)
            status["training_videos"] = "success" if ok else "failed"

    # Test materials
    tm_id = sp_manifest.get("test_materials_gdrive_id")
    if tm_id:
        tm_dir = raw_dir / "test_videos"
        if dry_run:
            logger.info(
                "[DRY RUN] Would download test materials for %s", species_id
            )
            status["test_materials"] = "dry_run"
        else:
            ok = _download_folder_robust(tm_id, tm_dir)
            status["test_materials"] = "success" if ok else "failed"

    return status


def download_all(
    cfg: Optional[Dict[str, Any]] = None,
    species_filter: Optional[List[str]] = None,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """Download resources for all species in the manifest.

    Args:
        cfg: Loaded config dict.
        species_filter: If set, only download these species.
        dry_run: If True, only log what would be downloaded.

    Returns:
        List of download status dicts.
    """
    if cfg is None:
        cfg = load_config()

    data_root = Path(cfg["paths"]["data_root"])
    manifest_path = Path(cfg["paths"]["manifest"])

    if not manifest_path.exists():
        logger.error("Manifest not found at %s. Run scraper first.", manifest_path)
        return []

    with open(manifest_path) as f:
        manifest = json.load(f)

    statuses = []
    for entry in manifest:
        sid = entry["species_id"]
        if species_filter and sid not in species_filter:
            logger.info("Skipping %s (not in filter)", sid)
            continue
        logger.info("Downloading %s ...", sid)
        status = download_species(sid, entry, data_root, dry_run=dry_run)
        statuses.append(status)
        logger.info("  %s: %s", sid, status)

    return statuses


def build_video_inventory(
    cfg: Optional[Dict[str, Any]] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Walk the raw data tree and inventory all video files.

    Args:
        cfg: Loaded config dict.
        output_path: Where to save video_inventory.csv.

    Returns:
        Path to the inventory CSV.
    """
    if cfg is None:
        cfg = load_config()

    data_root = Path(cfg["paths"]["data_root"])
    raw_dir = data_root / "raw"

    if output_path is None:
        output_path = Path(cfg["paths"]["video_inventory"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for species_dir in sorted(raw_dir.iterdir()):
        if not species_dir.is_dir():
            continue
        species_id = species_dir.name
        for source_type in ["training_videos", "test_videos"]:
            source_dir = species_dir / source_type
            if not source_dir.exists():
                continue
            for vid in find_videos(source_dir):
                size_mb = vid.stat().st_size / (1024 * 1024)
                # Relative path from data root for portability
                rel_path = vid.relative_to(data_root)
                rows.append({
                    "file_path": str(rel_path),
                    "species": species_id,
                    "source_type": source_type.replace("_videos", ""),
                    "filename": vid.name,
                    "file_size_mb": round(size_mb, 2),
                    "parent_folder": vid.parent.name,
                })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file_path",
                "species",
                "source_type",
                "filename",
                "file_size_mb",
                "parent_folder",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        "Video inventory: %d files across %d species → %s",
        len(rows),
        len({r["species"] for r in rows}),
        output_path,
    )
    return output_path
