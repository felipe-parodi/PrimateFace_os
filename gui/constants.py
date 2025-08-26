"""Constants and configuration for GUI module.

This module centralizes all configuration parameters and constants
used across the GUI components.
"""

from typing import Dict, List, Tuple

# Detection parameters
DEFAULT_BBOX_THR: float = 0.3
DEFAULT_NMS_THR: float = 0.9
DEFAULT_MAX_MONKEYS: int = 3
DEFAULT_DEVICE: str = "cuda:0"

# Pose estimation parameters
DEFAULT_KPT_THR: float = 0.05
DEFAULT_MIN_KEYPOINTS: int = 10
DEFAULT_MASK_KPT_THR: float = 0.3

# Processing parameters
DEFAULT_FRAME_INTERVAL_SEC: float = 1.0
DEFAULT_RESIZE_MAX: int = 1280
DEFAULT_BATCH_SIZE: int = 1

# Visualization parameters
KEYPOINT_WIDTH_DEFAULT: int = 3
SKELETON_WIDTH_DEFAULT: int = 2

# Color constants for visualization (BGR format for OpenCV)
KEYPOINT_COLOR_LEFT: Tuple[int, int, int] = (0, 255, 0)      # Green
KEYPOINT_COLOR_RIGHT: Tuple[int, int, int] = (255, 0, 0)     # Blue  
KEYPOINT_COLOR_CENTER: Tuple[int, int, int] = (0, 255, 255)  # Yellow

SKELETON_COLOR_LEFT: Tuple[int, int, int] = (0, 255, 0)      # Green
SKELETON_COLOR_RIGHT: Tuple[int, int, int] = (255, 0, 0)     # Blue
SKELETON_COLOR_CENTER: Tuple[int, int, int] = (0, 255, 255)  # Yellow
SKELETON_COLOR_MIXED: Tuple[int, int, int] = (255, 0, 255)   # Magenta

# Instance colors for multi-instance visualization
INSTANCE_COLORS: List[str] = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",
    "#800000", "#008000", "#000080", "#808000", "#800080", "#008080",
    "#FFA500", "#A52A2A", "#8B008B", "#556B2F", "#FF1493", "#00CED1",
]

# COCO-17 keypoint names (fallback when dataset_meta is empty)
COCO17_KEYPOINTS: List[str] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# COCO-17 skeleton links (fallback)
COCO17_SKELETON: List[List[int]] = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 7],
    [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], 
    [2, 4], [3, 5]
]

# COCO-68 face keypoint names
COCO68_FACE_KEYPOINTS: List[str] = [
    f"face-{i}" for i in range(68)
]

# File extensions
IMAGE_EXTENSIONS: List[str] = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
VIDEO_EXTENSIONS: List[str] = ['.mp4', '.avi', '.mov', '.mkv', '.webm']

# GUI parameters
GUI_DEFAULT_WIDTH: int = 1000
GUI_DEFAULT_HEIGHT: int = 750
GUI_CANVAS_PADDING: int = 50

# Sampling parameters
IMAGES_TO_SAMPLE_PER_SUBDIR: int = 2000

# Model configurations (removed hardcoded paths)
MODEL_CONFIGS: Dict[str, Dict] = {
    'ssd300': {
        'config': None,  # To be specified by user
        'checkpoint': None,  # To be specified by user
        'description': 'SSD300 detection model'
    },
    'mobilenet_pose': {
        'config': None,  # To be specified by user
        'checkpoint': None,  # To be specified by user
        'description': 'MobileNet pose estimation model'
    }
}