"""Phase C: Parse AU labels from downloaded folder/file structure.

AnimalFACS training videos are typically organized by AU code —
folders named AU1, AU5, AU10+25, etc. This module parses those
naming conventions and builds structured AU label records.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .au_homology import SPECIES_AU_CATALOGUES
from .utils import load_config

logger = logging.getLogger("animalfacs.au_parser")

# Regex patterns for AU codes in filenames/folder names
_AU_SINGLE = re.compile(r"AU\s*(\d+)", re.IGNORECASE)
_AU_COMBO = re.compile(r"AU\s*(\d+(?:\s*\+\s*\d+)+)", re.IGNORECASE)
_AD_SINGLE = re.compile(r"AD\s*(\d+)", re.IGNORECASE)
# Also handle "EAD" (ear action descriptor) patterns
_EAD_SINGLE = re.compile(r"EAD\s*(\d+)", re.IGNORECASE)


def parse_au_string(text: str) -> List[int]:
    """Extract AU numbers from a text string.

    Handles formats: AU1, AU10, AU10+25, AU 1, au1, etc.

    Args:
        text: Text that may contain AU codes.

    Returns:
        Sorted list of unique AU integers found.
    """
    aus: Set[int] = set()

    # First try combo pattern (AU10+25)
    for match in _AU_COMBO.finditer(text):
        combo = match.group(1)
        for num_str in re.findall(r"\d+", combo):
            aus.add(int(num_str))

    # Then single AU patterns
    for match in _AU_SINGLE.finditer(text):
        aus.add(int(match.group(1)))

    return sorted(aus)


def parse_ad_string(text: str) -> List[str]:
    """Extract Action Descriptor codes from text.

    Args:
        text: Text that may contain AD/EAD codes.

    Returns:
        List of descriptor strings like ["AD160", "EAD101"].
    """
    descriptors = []
    for match in _AD_SINGLE.finditer(text):
        descriptors.append(f"AD{match.group(1)}")
    for match in _EAD_SINGLE.finditer(text):
        descriptors.append(f"EAD{match.group(1)}")
    return descriptors


def parse_folder_structure(
    species_dir: Path,
    species_id: str,
) -> List[Dict[str, Any]]:
    """Parse AU labels from the folder structure of downloaded videos.

    Walks the training_videos directory looking for AU-named folders
    and files. Returns one record per video file found.

    Args:
        species_dir: Path to raw/{species_id}/ directory.
        species_id: Species identifier.

    Returns:
        List of label records with keys: file_path, species, aus,
        raw_label, label_source, descriptors.
    """
    records = []
    training_dir = species_dir / "training_videos"
    if not training_dir.exists():
        logger.warning("No training_videos dir for %s", species_id)
        return records

    valid_aus = set(SPECIES_AU_CATALOGUES.get(species_id, {}).keys())

    # Walk all subdirectories
    for path in sorted(training_dir.rglob("*")):
        if not path.is_file():
            continue
        # Skip non-video files (but don't skip — we want all media)
        suffix = path.suffix.lower()
        if suffix not in {
            ".mp4", ".mov", ".avi", ".wmv", ".webm", ".mkv",
            ".m4v", ".mpg", ".mpeg", ".mts",
        }:
            continue

        # Try to extract AU from parent folders and filename
        # Check each path component
        aus_found: Set[int] = set()
        raw_labels: List[str] = []
        descriptors: List[str] = []

        parts_to_check = list(path.relative_to(training_dir).parts)
        for part in parts_to_check:
            part_aus = parse_au_string(part)
            if part_aus:
                aus_found.update(part_aus)
                raw_labels.append(part)
            part_ads = parse_ad_string(part)
            if part_ads:
                descriptors.extend(part_ads)

        # Determine label source
        if aus_found:
            label_source = "parsed"
        else:
            label_source = "none"

        # Filter to valid AUs for this species
        normalized_aus = sorted(aus_found & valid_aus) if valid_aus else sorted(aus_found)
        unknown_aus = aus_found - valid_aus if valid_aus else set()
        if unknown_aus:
            logger.debug(
                "  %s: AU%s not in %s catalogue, kept anyway",
                path.name,
                unknown_aus,
                species_id,
            )
            # Keep them — the catalogue may be incomplete
            normalized_aus = sorted(aus_found)

        records.append({
            "file_path": str(path),
            "species": species_id,
            "aus": normalized_aus,
            "raw_label": " | ".join(raw_labels) if raw_labels else "",
            "label_source": label_source,
            "descriptors": descriptors,
        })

    logger.info(
        "  %s: %d videos, %d with AU labels",
        species_id,
        len(records),
        sum(1 for r in records if r["aus"]),
    )
    return records


def parse_all_species(
    cfg: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Parse AU labels for all downloaded species.

    Args:
        cfg: Loaded config dict.

    Returns:
        Combined list of label records across all species.
    """
    if cfg is None:
        cfg = load_config()

    data_root = Path(cfg["paths"]["data_root"])
    raw_dir = data_root / "raw"
    all_records = []

    for species_dir in sorted(raw_dir.iterdir()):
        if not species_dir.is_dir():
            continue
        species_id = species_dir.name
        records = parse_folder_structure(species_dir, species_id)
        all_records.extend(records)

    # Save AU definitions per species
    au_defs = {}
    for species_id, catalogue in SPECIES_AU_CATALOGUES.items():
        au_defs[species_id] = {
            str(k): v for k, v in catalogue.items()
        }

    au_defs_path = Path(cfg["paths"]["au_definitions"])
    au_defs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(au_defs_path, "w") as f:
        json.dump(au_defs, f, indent=2)
    logger.info("AU definitions saved to %s", au_defs_path)

    logger.info(
        "Total: %d videos with AU labels across %d species",
        sum(1 for r in all_records if r["aus"]),
        len({r["species"] for r in all_records if r["aus"]}),
    )
    return all_records
