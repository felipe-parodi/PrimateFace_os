"""Constants and configuration for evaluation modules.

This module centralizes all configuration settings for the evaluation
framework, removing hardcoded paths and providing a unified configuration
system for all evaluation scripts.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Framework-specific default configurations
DEFAULT_CONFIGS: Dict[str, Dict[str, Any]] = {
    'mmpose': {
        'device': 'cuda:0',
        'batch_size': 32,
        'bbox_thr': 0.3,
        'nms_thr': 0.3,
        'visualization': {
            'bbox_color': '#66B2FF',        # Sky Blue
            'keypoint_color': '#FFFF00',    # Yellow
            'skeleton_color': '#FF00FF',    # Magenta
            'keypoint_size': 5,
            'line_thickness': 1.0,
            'bbox_line_thickness': 1.0,
            'fig_dpi': 300,
        }
    },
    'dlc': {
        'batch_size': 64,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'device': 'cuda:0',                 # If torch is available, will be updated dynamically
        'kpt_threshold': 0.05,
        'snapshot_frequency': 10,
        'display_iters': 500,
    },
    'sleap': {
        'preload': False,
        'input_scale': 1.0,
        'augmentation': True,
        'batch_size': 32,
        'prefetch': True,
        'max_instances': 6,
    },
    'grounding_dino': {
        'ontology': {
            'face': 'face, head',
        },
        'color_map': {
            'face': (0, 0, 255),            # Red in BGR
        },
        'box_threshold': 0.35,
        'text_threshold': 0.25,
    }
}

# Evaluation metrics configuration
METRICS_CONFIG: Dict[str, Any] = {
    'nme': {
        'normalize_by': 'bbox',         # Options: 'bbox', 'torso', 'head', 'interocular'
        'threshold': 0.05,
    },
    'pck': {
        'threshold': 0.2,               # Fraction of reference distance
        'reference': 'bbox_diagonal',   # Options: 'bbox_diagonal', 'head_size', 'torso_size'
    },
    'oks': {
        'sigmas': None,                 # If None, use COCO defaults, otherwise provide custom sigmas
        'area_normalize': True,
    },
    'map': {
        'iou_thresholds': [0.5, 0.75],  # IoU thresholds for detection mAP
        'recall_thresholds': None,      # Use default if None
    }
}

# Visualization settings
VIZ_CONFIG: Dict[str, Any] = {
    'colors': {
        'gt_bbox': '#00FF00',           # Green for ground truth
        'pred_bbox': '#FF0000',         # Red for predictions
        'keypoint_left': (0, 255, 0),   # Green in RGB
        'keypoint_right': (255, 0, 0),  # Red in RGB
        'keypoint_center': (0, 0, 255), # Blue in RGB
    },
    'output_formats': ['png', 'pdf', 'svg'],
    'grid_layout': (2, 3),              # Default grid for comparison plots
    'figure_size': (12, 8),             # Default figure size in inches
    'font_size': 10,
    'save_individual_frames': False,
}

# Data processing settings
# TODO: Add data processing settings
DATA_CONFIG: Dict[str, Any] = {
    'image_extensions': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
    'video_extensions': ['.mp4', '.avi', '.mov', '.mkv', '.webm'],
    'coco_keypoint_names': [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        # Add your specific keypoint names here
    ],
    'max_instances_per_image': 10,
    'min_keypoints_visible': 3,
    'image_size_threshold': (100, 100), 
}

# Model zoo references (no hardcoded paths)
# TODO: Add model zoo references
MODEL_ZOO: Dict[str, Dict[str, str]] = {
    'mmpose': {
        # Users should provide their own model paths via CLI or config files
        'hrnet_w32': 'configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/wflw/hrnet_w18_wflw_256x256.py',
        'hrnet_w48': 'configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/wflw/hrnet_w18_wflw_256x256.py',
        'vitpose_base': 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitpose_base_coco_256x192.py',
    },
    'dlc': {
        'resnet_50': 'top_down_resnet_50',
        'resnet_101': 'top_down_resnet_101',
        'hrnet_w32': 'top_down_hrnet_w32',
        'hrnet_w48': 'top_down_hrnet_w48',
        'efficientnet_b0': 'efficientnet_b0',
    },
    'sleap': {
        'baseline_small': 'baseline.centroid.json',
        'baseline_medium': 'baseline_medium_rf.topdown.json',
        'baseline_large': 'baseline_large_rf.topdown.json',
    }
}

# Training hyperparameters
TRAINING_CONFIG: Dict[str, Dict[str, Any]] = {
    'optimizers': {
        'adam': {'lr': 1e-4, 'betas': (0.9, 0.999), 'eps': 1e-8},
        'sgd': {'lr': 0.01, 'momentum': 0.9, 'weight_decay': 1e-4},
        'adamw': {'lr': 1e-4, 'betas': (0.9, 0.999), 'weight_decay': 0.01},
    },
    'schedulers': {
        'step': {'step_size': 30, 'gamma': 0.1},
        'cosine': {'T_max': 100, 'eta_min': 1e-6},
        'plateau': {'mode': 'min', 'factor': 0.5, 'patience': 10},
    },
    'augmentation': {
        'random_flip': {'prob': 0.5, 'direction': 'horizontal'},
        'random_rotate': {'max_angle': 30, 'prob': 0.5},
        'random_scale': {'scale_range': (0.8, 1.2), 'prob': 0.5},
        'color_jitter': {'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2},
    }
}

# Output directory structure
OUTPUT_STRUCTURE: Dict[str, str] = {
    'models': 'models',
    'logs': 'logs',
    'visualizations': 'visualizations',
    'metrics': 'metrics',
    'checkpoints': 'checkpoints',
    'configs': 'configs',
}


def get_default_config(framework: str) -> Dict[str, Any]:
    """Get default configuration for a specific framework.
    
    Args:
        framework: Name of the framework ('mmpose', 'dlc', 'sleap', etc.)
        
    Returns:
        Default configuration dictionary for the framework.
        
    Raises:
        ValueError: If framework is not recognized.
    """
    if framework not in DEFAULT_CONFIGS:
        raise ValueError(
            f"Unknown framework: {framework}. "
            f"Available frameworks: {list(DEFAULT_CONFIGS.keys())}"
        )
    return DEFAULT_CONFIGS[framework].copy()


def get_model_config(framework: str, model_name: str) -> Optional[str]:
    """Get model configuration path from model zoo.
    
    Args:
        framework: Name of the framework
        model_name: Name of the model
        
    Returns:
        Model configuration path if found, None otherwise.
    """
    if framework in MODEL_ZOO and model_name in MODEL_ZOO[framework]:
        return MODEL_ZOO[framework][model_name]
    return None


def validate_image_extensions(extensions: list) -> list:
    """Validate and normalize image file extensions.
    
    Args:
        extensions: List of file extensions
        
    Returns:
        Normalized list of extensions (with leading dots).
    """
    normalized = []
    for ext in extensions:
        if not ext.startswith('.'):
            ext = '.' + ext
        normalized.append(ext.lower())
    return normalized


def get_color_rgb(color_key: str) -> Tuple[int, int, int]:
    """Get RGB color tuple from color key.
    
    Args:
        color_key: Key for color in VIZ_CONFIG
        
    Returns:
        RGB color tuple.
    """
    color = VIZ_CONFIG['colors'].get(color_key, (128, 128, 128))
    if isinstance(color, str):
        # Convert hex to RGB
        color = color.lstrip('#')
        return tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
    return color


try:
    import torch
    if torch.cuda.is_available():
        DEFAULT_CONFIGS['dlc']['device'] = 'cuda:0'
    else:
        DEFAULT_CONFIGS['dlc']['device'] = 'cpu'
except ImportError:
    DEFAULT_CONFIGS['dlc']['device'] = 'cpu'