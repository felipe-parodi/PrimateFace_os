"""PrimateFace demonstration and example scripts.

Backward compatibility: core modules have moved to the ``primateface``
package. This shim re-exports them so existing code continues to work.
"""

from primateface._constants import (
    DEFAULT_BBOX_THR,
    DEFAULT_KPT_THR,
    DEFAULT_NMS_THR,
    IMAGE_EXTENSIONS,
    PRIMATE_GENERA,
    VIDEO_EXTENSIONS,
)
from primateface._model_registry import (
    HF_REPO_ID,
    MODEL_ENTRIES,
    MODELS,
)
from primateface._processor import PrimateFaceProcessor
from primateface._smooth import MedianSavgolSmoother
from primateface._viz import FastPoseVisualizer

__all__ = [
    'PrimateFaceProcessor',
    'MedianSavgolSmoother',
    'FastPoseVisualizer',
    'DEFAULT_BBOX_THR',
    'DEFAULT_KPT_THR',
    'DEFAULT_NMS_THR',
    'IMAGE_EXTENSIONS',
    'VIDEO_EXTENSIONS',
    'PRIMATE_GENERA',
    'HF_REPO_ID',
    'MODEL_ENTRIES',
    'MODELS',
]

__version__ = '0.1.0'
