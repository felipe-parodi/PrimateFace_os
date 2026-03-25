"""Shared utilities: logging, seeds, config loading, environment checks."""

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

logger = logging.getLogger("animalfacs")

# Resolve paths relative to this file
_THIS_DIR = Path(__file__).resolve().parent


def load_config(config_path: Path | None = None) -> Dict[str, Any]:
    """Load YAML config, resolving relative paths against config file location.

    Args:
        config_path: Path to config.yaml. Defaults to config.yaml in this dir.

    Returns:
        Parsed config dictionary with resolved absolute paths.
    """
    if config_path is None:
        config_path = _THIS_DIR / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Resolve relative paths against the config file's directory
    base = config_path.resolve().parent
    for key, val in cfg.get("paths", {}).items():
        if isinstance(val, str):
            resolved = (base / val).resolve()
            cfg["paths"][key] = str(resolved)
    return cfg


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Random seed set to %d", seed)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging with consistent format.

    Args:
        level: Logging level.
    """
    fmt = "[%(asctime)s %(name)s %(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, force=True)


def check_environment() -> Dict[str, str]:
    """Check required packages and GPU availability.

    Returns:
        Dictionary of package versions and GPU info.
    """
    info: Dict[str, str] = {}
    info["python"] = sys.version.split()[0]
    info["torch"] = torch.__version__
    info["cuda_available"] = str(torch.cuda.is_available())
    if torch.cuda.is_available():
        info["gpu_count"] = str(torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            info[f"gpu_{i}"] = torch.cuda.get_device_name(i)

    try:
        import mmdet

        info["mmdet"] = mmdet.__version__
    except ImportError:
        info["mmdet"] = "NOT INSTALLED"

    try:
        import mmpose

        info["mmpose"] = mmpose.__version__
    except ImportError:
        info["mmpose"] = "NOT INSTALLED"

    try:
        import primateface  # noqa: F401

        info["primateface"] = "installed"
    except ImportError:
        info["primateface"] = "NOT INSTALLED"

    for key, val in info.items():
        logger.info("  %s: %s", key, val)
    return info


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".wmv", ".webm", ".mkv", ".m4v", ".mpg", ".mpeg"}


def find_videos(root: Path) -> list[Path]:
    """Recursively find all video files under root.

    Args:
        root: Directory to search.

    Returns:
        List of video file paths sorted by name.
    """
    videos = []
    if not root.exists():
        return videos
    for p in root.rglob("*"):
        if p.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(p)
    return sorted(videos)
