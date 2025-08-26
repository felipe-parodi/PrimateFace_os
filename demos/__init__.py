"""PrimateFace demonstration and example scripts.

This package contains demo scripts and utilities for PrimateFace:
- Unified processing for videos and images
- Primate genus classification 
- COCO annotation visualization
- Utility modules for smoothing and visualization
"""

from .constants import (
    DEFAULT_BBOX_THR,
    DEFAULT_KPT_THR,
    DEFAULT_NMS_THR,
    IMAGE_EXTENSIONS,
    PRIMATE_GENERA,
    VIDEO_EXTENSIONS,
)
from .process import PrimateFaceProcessor
from .smooth_utils import MedianSavgolSmoother
from .viz_utils import FastPoseVisualizer

__all__ = [
    # Main processor
    'PrimateFaceProcessor',
    # Utilities
    'MedianSavgolSmoother',
    'FastPoseVisualizer',
    # Constants
    'DEFAULT_BBOX_THR',
    'DEFAULT_KPT_THR',
    'DEFAULT_NMS_THR',
    'IMAGE_EXTENSIONS',
    'VIDEO_EXTENSIONS',
    'PRIMATE_GENERA',
]

__version__ = '0.1.0'