"""GUI module for PrimateFace pseudo-labeling and annotation refinement.

This module provides tools for:
- Detection and pose estimation on images/videos
- COCO format conversion
- Interactive annotation refinement GUI
- Parallel processing across multiple GPUs
"""

from .constants import *
from .converters import COCOConverter, ImageCOCOConverter, VideoCOCOConverter
from .core import Detector, FrameworkType, ModelManager, PoseEstimator, SAMMasker
from .refine_boxes import COCORefinementGUI

__version__ = "0.1.0"
__all__ = [
    "Detector",
    "PoseEstimator",
    "SAMMasker",
    "ModelManager",
    "FrameworkType",
    "COCOConverter",
    "ImageCOCOConverter",
    "VideoCOCOConverter",
    "COCORefinementGUI",
]