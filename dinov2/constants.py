"""Constants and configuration for DINOv2 feature extraction.

This module contains shared constants, default configurations, and model
specifications used across all DINOv2-related scripts.
"""

from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

# DINOv2 Model Configurations
DINOV2_MODELS: Dict[Literal["small", "base", "large", "giant"], str] = {
    "small": "facebook/dinov2-small",
    "base": "facebook/dinov2-base",
    "large": "facebook/dinov2-large",
    "giant": "facebook/dinov2-giant",
}
DEFAULT_MODEL: Literal["facebook/dinov2-base"] = "facebook/dinov2-base"

# Processing Configuration
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 4
DEFAULT_DEVICE = "cuda"  # Will auto-fallback to CPU if not available
DEFAULT_IMAGE_SIZE = 224  # DINOv2 expected input size

# UMAP Configuration
DEFAULT_UMAP_N_NEIGHBORS = 15
DEFAULT_UMAP_MIN_DIST = 0.1
DEFAULT_UMAP_METRIC = "cosine"

# Clustering Configuration
DEFAULT_NUM_CLUSTERS = 100
DEFAULT_RANDOM_STATE = 42

# Visualization Configuration
DEFAULT_THUMBNAIL_SIZE: Tuple[int, int] = (64, 64)
DEFAULT_FIGURE_DPI = 300
DEFAULT_SAMPLES_PER_CLUSTER = 5
DEFAULT_COLORMAP = "tab20"  # For cluster colors
DEFAULT_DOT_SIZE = 50  # For scatter plots (matplotlib s parameter)

# Patch Visualization Configuration
PATCH_VIZ_FIGSIZE: Tuple[int, int] = (15, 8)
PATCH_VIZ_DPI = 150
NUM_ATTENTION_HEADS = 12  # DINOv2 has 12 attention heads

# Active Learning Selection
DEFAULT_TARGET_SUBSET_SIZE = 1000
DEFAULT_FPS_BATCH_SIZE = 1000  # For farthest point sampling

# File Extensions
SUPPORTED_IMAGE_EXTENSIONS: List[str] = [
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"
]

# Output Formats
EMBEDDING_FILE_EXTENSION = ".pt"
PLOT_FORMATS: List[str] = ["png", "pdf", "svg"]
DEFAULT_PLOT_FORMAT = "svg"

# Data Column Names (for CSV/DataFrame inputs)
IMAGE_PATH_COLUMN = "image_path"
IMAGE_ID_COLUMN = "image_id"
DATASET_COLUMN = "dataset"
LABEL_COLUMN = "label"

# Logging Configuration
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"