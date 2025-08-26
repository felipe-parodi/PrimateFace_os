"""Constants and configuration for PrimateFace demos.

This module contains shared constants, default configurations, and color schemes
used across all demo scripts for consistency.
"""

from typing import Dict, Tuple, List

# Detection and tracking constants
DET_CAT_ID = 0                  # Category ID for primate detection; corresponds to "face" in COCO
DEFAULT_BBOX_THR = 0.5          # Default detection confidence threshold
DEFAULT_KPT_THR = 0.7           # Default keypoint confidence threshold
DEFAULT_NMS_THR = 0.3           # Default NMS threshold

# Visualization colors (BGR format for OpenCV)
ID_COLORS_BGR: Dict[int, Tuple[int, int, int]] = {
    1: (0, 255, 0),             # Green
    2: (255, 0, 0),             # Blue
    3: (0, 0, 255),             # Red
    4: (255, 255, 0),           # Cyan
    5: (255, 0, 255),           # Magenta
    6: (0, 255, 255),           # Yellow
    7: (128, 0, 128),           # Purple
    8: (0, 128, 255),           # Orange
}
DEFAULT_COLOR_BGR: Tuple[int, int, int] = (128, 128, 128)  # Gray
TEXT_COLOR_BGR: Tuple[int, int, int] = (255, 255, 255)  # White

# Matplotlib colors (for visualization scripts)
BBOX_COLOR_HEX = '#66B2FF'      # Sky Blue
KEYPOINT_COLOR_HEX = '#FFFF00'  # Yellow
SKELETON_COLOR_HEX = '#FF00FF'  # Magenta

# Drawing parameters
KEYPOINT_RADIUS = 3
LINE_THICKNESS = 1
BBOX_THICKNESS = 2
KEYPOINT_SIZE_MPL = 1           # Size for matplotlib scatter plot (in points)
LINE_THICKNESS_MPL = 1.5
BBOX_LINE_THICKNESS_MPL = 1.5
DEFAULT_FIG_DPI = 300

# Smoothing parameters
DEFAULT_MEDIAN_WINDOW = 5       # Window size for median filter (must be odd)
DEFAULT_SAVGOL_WINDOW = 7       # Window size for Savitzky-Golay filter (must be odd)
DEFAULT_SAVGOL_ORDER = 3        # Polynomial order for Savitzky-Golay filter

# File extensions
IMAGE_EXTENSIONS: List[str] = [
    "*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"
]
VIDEO_EXTENSIONS: List[str] = [
    "*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"
]

# Primate genera for classification (subject to change)
PRIMATE_GENERA: List[str] = [
    "Allenopithecus", "Allocebus", "Alouatta", "Aotus", "Arctocebus", 
    "Ateles", "Avahi", "Brachyteles", "Cacajao", "Callicebus", "Callimico", 
    "Callithrix", "Carlito", "Cebuella", "Cebus", "Cephalopachus", 
    "Cercocebus", "Cercopithecus", "Cheirogaleus", "Chiropotes", "Chlorocebus", 
    "Daubentonia", "Erythrocebus", "Eulemur", "Euoticus", "Galago", 
    "Galagoides", "Gorilla", "Hapalemur", "Hoolock", "Hylobates", "Indri", 
    "Lagothrix", "Lemur", "Leontopithecus", "Lepilemur", "Loris", "Macaca", 
    "Mandrillus", "Mico", "Microcebus", "Miopithecus", "Mirza", "Nasalis", 
    "Nomascus", "Nycticebus", "Otolemur", "Pan", "Papio", "Paragalago", 
    "Perodicticus", "Phaner", "Piliocolobus", "Pithecia", "Pongo", "Presbytis", 
    "Procolobus", "Propithecus", "Pseudopotto", "Rhinopithecus", "Saguinus", 
    "Saimiri", 
    "Sciurocheirus", # Debateable
    "Semnopithecus", "Symphalangus", "Tarsius", 
    "Theropithecus", "Trachypithecus", "Varecia"
]

# Model configurations for VLMs
VLM_CONFIGS: Dict[str, Dict[str, any]] = {
    "SmolVLM": {
        "model_id": "HuggingFaceTB/SmolVLM-Instruct",
        "prompt_template": """<|im_start|>system
You are a primate classification expert. From the provided list, identify the single most likely genus for the primate in the image. Respond with only the genus name.

Genus list: {genera_list}<|im_end|>
<|im_start|>user
<image>

What is the genus of this primate?<|im_end|>
<|im_start|>assistant
""",
    },
    "InternVL2-2B": {
        "model_id": "OpenGVLab/InternVL-Chat-V1-5",
        "trust_remote_code": True,
        "prompt_template": "From the provided list, identify the single most likely genus for the primate in the image. Respond with only the genus name.\n\nGenus list: {genera_list}",
    }
}