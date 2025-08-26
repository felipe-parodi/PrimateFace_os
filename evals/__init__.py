"""Evaluation utilities for PrimateFace.

This module provides comprehensive evaluation tools for pose estimation models
across multiple frameworks including MMPose, DeepLabCut, and SLEAP.
"""

from .constants import (
    DEFAULT_CONFIGS,
    METRICS_CONFIG,
    VIZ_CONFIG,
    DATA_CONFIG,
    MODEL_ZOO,
    TRAINING_CONFIG,
    get_default_config,
    get_model_config,
)

from .core import (
    MetricsCalculator,
    NMECalculator,
    PCKCalculator,
    OKSCalculator,
    ModelManager,
    ModelConfig,
    EvalVisualizer,
    plot_training_curves,
    plot_predictions,
)

__version__ = '0.1.0'

__all__ = [
    # Constants and configs
    'DEFAULT_CONFIGS',
    'METRICS_CONFIG',
    'VIZ_CONFIG',
    'DATA_CONFIG',
    'MODEL_ZOO',
    'TRAINING_CONFIG',
    'get_default_config',
    'get_model_config',
    # Core components
    'MetricsCalculator',
    'NMECalculator',
    'PCKCalculator',
    'OKSCalculator',
    'ModelManager',
    'ModelConfig',
    'EvalVisualizer',
    'plot_training_curves',
    'plot_predictions',
]