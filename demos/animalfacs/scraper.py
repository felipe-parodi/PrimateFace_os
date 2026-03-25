"""Phase A: Scrape AnimalFACS website for primate FACS resource links.

Fetches each species page, extracts Google Drive links for manuals,
training videos, and test materials, and saves a structured manifest.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup

from .utils import load_config

logger = logging.getLogger("animalfacs.scraper")

# Google Drive URL patterns
_GDRIVE_FOLDER_RE = re.compile(
    r"https?://drive\.google\.com/drive/folders/([a-zA-Z0-9_-]+)"
)
_GDRIVE_FILE_RE = re.compile(
    r"https?://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)"
)


def _extract_gdrive_ids(html: str) -> Dict[str, List[str]]:
    """Extract Google Drive folder and file IDs from HTML content.

    Args:
        html: Raw HTML string.

    Returns:
        Dict with "folders" and "files" keys, each a list of IDs.
    """
    folders = list(set(_GDRIVE_FOLDER_RE.findall(html)))
    files = list(set(_GDRIVE_FILE_RE.findall(html)))
    return {"folders": folders, "files": files}


def scrape_species_page(url: str) -> Optional[Dict[str, Any]]:
    """Fetch a single AnimalFACS species page and extract links.

    Args:
        url: URL of the species FACS page.

    Returns:
        Dict of extracted links, or None if fetch fails.
    """
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning("Failed to fetch %s: %s", url, e)
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    gdrive = _extract_gdrive_ids(resp.text)

    # Try to find section-specific links by scanning anchor tags
    links_by_context: Dict[str, List[str]] = {
        "manual": [],
        "training": [],
        "test": [],
        "other": [],
    }
    for tag in soup.find_all("a", href=True):
        href = str(tag["href"])
        text = tag.get_text(strip=True).lower()
        parent_text = tag.parent.get_text(strip=True).lower() if tag.parent else ""
        context = text + " " + parent_text

        folder_match = _GDRIVE_FOLDER_RE.search(href)
        file_match = _GDRIVE_FILE_RE.search(href)
        drive_id = None
        link_type = "folder" if folder_match else "file" if file_match else None
        if folder_match:
            drive_id = folder_match.group(1)
        elif file_match:
            drive_id = file_match.group(1)

        if drive_id is None:
            continue

        if any(kw in context for kw in ["manual", "pdf", "coding system"]):
            links_by_context["manual"].append(drive_id)
        elif any(kw in context for kw in ["training", "example", "instructional"]):
            links_by_context["training"].append(drive_id)
        elif any(kw in context for kw in ["test", "certification", "assessment"]):
            links_by_context["test"].append(drive_id)
        else:
            links_by_context["other"].append(drive_id)

    return {
        "all_gdrive_folders": gdrive["folders"],
        "all_gdrive_files": gdrive["files"],
        "links_by_context": links_by_context,
        "link_type": link_type,
    }


def build_manifest(
    cfg: Optional[Dict[str, Any]] = None,
    output_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Build the full manifest from config + optional live scraping.

    Uses pre-scraped Google Drive IDs from config.yaml as primary source,
    with live scraping as supplementary enrichment.

    Args:
        cfg: Loaded config dict. Loads default if None.
        output_path: Where to save manifest.json.

    Returns:
        List of per-species manifest entries.
    """
    if cfg is None:
        cfg = load_config()

    manifest = []
    for species_id, sp_cfg in cfg["species"].items():
        entry = {
            "species_id": species_id,
            "common_name": sp_cfg["common_name"],
            "latin_name": sp_cfg["latin_name"],
            "facs_system": sp_cfg["facs_system"],
            "source_url": sp_cfg["source_url"],
            "manual_gdrive_id": sp_cfg.get("manual_gdrive_id"),
            "manual_type": sp_cfg.get("manual_type", "file"),
            "training_videos_gdrive_id": sp_cfg.get("training_videos_gdrive_id"),
            "test_materials_gdrive_id": sp_cfg.get("test_materials_gdrive_id"),
            "permission_status": "explicit_written",
            "permission_holder": "Bridget Waller",
            "has_training_videos": sp_cfg.get("training_videos_gdrive_id") is not None,
            "has_test_materials": sp_cfg.get("test_materials_gdrive_id") is not None,
        }

        # Optionally enrich with live scrape
        logger.info("Scraping %s: %s", species_id, sp_cfg["source_url"])
        scraped = scrape_species_page(sp_cfg["source_url"])
        if scraped:
            entry["scraped_folders"] = scraped["all_gdrive_folders"]
            entry["scraped_files"] = scraped["all_gdrive_files"]
        else:
            entry["scraped_folders"] = []
            entry["scraped_files"] = []

        manifest.append(entry)

    if output_path is None:
        output_path = Path(cfg["paths"]["manifest"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest saved to %s (%d species)", output_path, len(manifest))

    return manifest
