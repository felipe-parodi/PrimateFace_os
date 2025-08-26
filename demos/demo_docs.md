# PrimateFace Demo API - Technical Documentation

This document provides detailed technical documentation for the PrimateFace demo modules, including API reference, implementation details, and advanced usage patterns.

## Module Architecture

```
demos/
├── primateface_demo.py     # CLI entry point
├── process.py              # Core processing pipeline
├── classify_genus.py       # VLM-based classification
├── visualize_coco_annotations.py  # Publication visualizations
├── viz_utils.py           # Real-time visualization
├── smooth_utils.py        # Temporal smoothing
├── constants.py           # Configuration constants
├── mmdet_config.py        # Detection model config
└── mmpose_config.py       # Pose model config
```

## Core API Reference

### `process.py` - Main Processing Pipeline

#### Class: `PrimateFaceProcessor`

The primary interface for face detection and pose estimation on primate images and videos.

```python
class PrimateFaceProcessor:
    def __init__(
        self,
        det_config: str,
        det_checkpoint: str,
        pose_config: Optional[str] = None,
        pose_checkpoint: Optional[str] = None,
        device: str = 'cuda:0',
        bbox_thr: float = 0.5,
        kpt_thr: float = 0.7,
        nms_thr: float = 0.3,
        use_oks_nms: bool = False
    )
```

**Parameters:**
- `det_config`: Path to MMDetection config file
- `det_checkpoint`: Path to detection model weights
- `pose_config`: Path to MMPose config file (optional)
- `pose_checkpoint`: Path to pose model weights (optional)
- `device`: CUDA device or 'cpu'
- `bbox_thr`: Detection confidence threshold (0-1)
- `kpt_thr`: Keypoint confidence threshold (0-1)
- `nms_thr`: Non-maximum suppression threshold
- `use_oks_nms`: Use Object Keypoint Similarity for NMS

**Methods:**

##### `process_image(image_path, save_viz=False, viz_pose=True) -> dict`
Process a single image for face detection and optional pose estimation.

**Returns:**
```python
{
    "image_path": str,
    "detections": [
        {
            "bbox": [x1, y1, x2, y2],
            "score": float,
            "keypoints": [[x, y, conf], ...] # If pose model provided
        }
    ],
    "visualization": np.ndarray  # If save_viz=True
}
```

##### `process_video(video_path, output_dir, save_viz=False, save_predictions=False, smooth=False, viz_pose=True) -> dict`
Process video frame-by-frame with optional temporal smoothing.

**Returns:**
```python
{
    "video_path": str,
    "fps": float,
    "total_frames": int,
    "predictions": {...},  # COCO format if save_predictions=True
    "output_video": str    # Path if save_viz=True
}
```

##### `process_image_directory(img_dir, output_dir, save_viz=False, save_predictions=False, viz_pose=True) -> dict`
Batch process all images in a directory.

### `classify_genus.py` - Genus Classification

#### Class: `PrimateClassifierVLM`

Vision-Language Model based classification of primate genus.

```python
class PrimateClassifierVLM:
    def __init__(
        self,
        model_name: str = "SmolVLM",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16
    )
```

**Supported Models:**
- `SmolVLM`: Lightweight, fast inference
- `InternVL2-2B`: Higher accuracy, more compute

**Methods:**

##### `classify_image(image_path: str, return_probabilities: bool = False) -> Union[str, dict]`
Classify a single image.

##### `classify_directory(directory: str, output_path: str, batch_size: int = 1)`
Batch classify all images in directory, save COCO-formatted results.

### Utility Modules

#### `viz_utils.py` - Fast Visualization

##### Class: `FastPoseVisualizer`

Real-time visualization optimized for video processing.

```python
class FastPoseVisualizer:
    def __init__(
        self,
        kpt_thr: float = 0.7,
        line_width: int = 3,
        circle_radius: int = 5
    )
    
    def draw_instances(
        self,
        image: np.ndarray,
        instances: List[dict],
        draw_bbox: bool = True,
        draw_pose: bool = True
    ) -> np.ndarray
```

**Features:**
- ID-based color coding for multi-instance tracking
- Optimized OpenCV drawing operations
- Configurable visualization parameters
- Support for both bbox and pose visualization

#### `smooth_utils.py` - Temporal Smoothing

##### Class: `MedianSavgolSmoother`

Hybrid smoothing for temporal keypoint stabilization.

```python
class MedianSavgolSmoother:
    def __init__(
        self,
        window_size: int = 7,
        savgol_window: int = 11,
        savgol_polyorder: int = 2
    )
    
    def smooth_keypoints(
        self,
        keypoints: np.ndarray,
        instance_id: int
    ) -> np.ndarray
```

**Algorithm:**
1. Median filter for outlier removal
2. Savitzky-Golay filter for trajectory smoothing
3. Per-instance history tracking
4. Automatic confidence weighting

#### `constants.py` - Configuration

Central configuration for all demo modules:

```python
# Detection parameters
DEFAULT_BBOX_THR = 0.5
DEFAULT_KPT_THR = 0.7
DEFAULT_NMS_THR = 0.3

# Visualization
POSE_PALETTE = [(255, 128, 0), ...]  # BGR colors
SKELETON_INFO = [(0, 1), (1, 2), ...]  # Keypoint connections

# Primate genera for classification
PRIMATE_GENERA = [
    "Homo", "Pan", "Gorilla", "Pongo", "Hylobates",
    "Macaca", "Papio", "Chlorocebus", "Cercopithecus",
    "Lemur", "Eulemur", "Varecia", ...
]

# VLM configurations
VLM_CONFIGS = {
    "SmolVLM": {
        "model_id": "HuggingFaceTB/SmolVLM-Instruct",
        "max_new_tokens": 500
    },
    "InternVL2-2B": {
        "model_id": "OpenGVLab/InternVL2-2B",
        "max_new_tokens": 1000
    }
}
```

## Advanced Usage Patterns

### Custom Processing Pipeline

```python
from demos import PrimateFaceProcessor
from demos.smooth_utils import MedianSavgolSmoother
from demos.viz_utils import FastPoseVisualizer

# Initialize components
processor = PrimateFaceProcessor(
    det_config="mmdet_config.py",
    det_checkpoint="mmdet_checkpoint.pth",
    pose_config="mmpose_config.py",
    pose_checkpoint="mmpose_checkpoint.pth"
)

smoother = MedianSavgolSmoother(window_size=5)
visualizer = FastPoseVisualizer(kpt_thr=0.8)

# Custom video processing
import cv2
cap = cv2.VideoCapture("video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect and estimate pose
    results = processor.process_frame(frame)
    
    # Apply smoothing
    for idx, instance in enumerate(results['instances']):
        if 'keypoints' in instance:
            instance['keypoints'] = smoother.smooth_keypoints(
                instance['keypoints'], 
                instance_id=idx
            )
    
    # Visualize
    viz_frame = visualizer.draw_instances(
        frame, 
        results['instances'],
        draw_bbox=True,
        draw_pose=True
    )
    
    cv2.imshow('Processed', viz_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### Batch Processing with Progress Tracking

```python
from demos.process import PrimateFaceProcessor
from pathlib import Path
from tqdm import tqdm

processor = PrimateFaceProcessor(
    det_config="mmdet_config.py",
    det_checkpoint="mmdet_checkpoint.pth"
)

# Process large dataset
image_dir = Path("dataset/images")
results = []

for img_path in tqdm(image_dir.glob("*.jpg"), desc="Processing"):
    result = processor.process_image(
        str(img_path),
        save_viz=False  # Skip viz for speed
    )
    results.append(result)

# Save aggregated results
import json
with open("results.json", "w") as f:
    json.dump(results, f)
```

### Multi-Model Ensemble

```python
# Run multiple detection models and combine results
models = [
    {"config": "yolo_config.py", "checkpoint": "yolo.pth"},
    {"config": "cascade_config.py", "checkpoint": "cascade.pth"}
]

all_detections = []
for model in models:
    processor = PrimateFaceProcessor(
        det_config=model["config"],
        det_checkpoint=model["checkpoint"]
    )
    detections = processor.process_image("image.jpg")
    all_detections.extend(detections["detections"])

# Apply ensemble NMS
from demos.process import apply_nms
final_detections = apply_nms(all_detections, threshold=0.5)
```

## Performance Optimization

### Memory Management

```python
# Process video in chunks for large files
def process_large_video(video_path, chunk_size=1000):
    processor = PrimateFaceProcessor(...)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for start in range(0, total_frames, chunk_size):
        end = min(start + chunk_size, total_frames)
        chunk_results = process_chunk(cap, start, end, processor)
        save_chunk_results(chunk_results, start)
        
        # Clear GPU memory
        torch.cuda.empty_cache()
```

### Inference Speed Optimization

```python
# Optimize for speed
processor = PrimateFaceProcessor(
    det_config="lightweight_config.py",
    det_checkpoint="lightweight.pth",
    bbox_thr=0.7,  # Higher threshold = fewer detections
    nms_thr=0.5    # More aggressive NMS
)

# Batch processing for images
batch_size = 8
images = load_images_batch(image_paths, batch_size)
results = processor.process_batch(images)
```

## Error Handling

### Common Issues and Solutions

```python
from demos.process import PrimateFaceProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    processor = PrimateFaceProcessor(
        det_config="config.py",
        det_checkpoint="model.pth"
    )
except FileNotFoundError as e:
    logger.error(f"Model files not found: {e}")
    logger.info("Run 'python download_models.py' to download models")
except RuntimeError as e:
    if "CUDA" in str(e):
        logger.warning("CUDA not available, falling back to CPU")
        processor = PrimateFaceProcessor(
            det_config="config.py",
            det_checkpoint="model.pth",
            device="cpu"
        )
```

## Testing

### Unit Tests

```python
# test_processor.py
import unittest
from demos.process import PrimateFaceProcessor

class TestProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = PrimateFaceProcessor(
            det_config="test_config.py",
            det_checkpoint="test_model.pth",
            device="cpu"
        )
    
    def test_single_image(self):
        result = self.processor.process_image("test.jpg")
        self.assertIn("detections", result)
        self.assertIsInstance(result["detections"], list)
    
    def test_threshold_filtering(self):
        processor_high = PrimateFaceProcessor(
            det_config="test_config.py",
            det_checkpoint="test_model.pth",
            bbox_thr=0.9
        )
        result = processor_high.process_image("test.jpg")
        # High threshold should produce fewer detections
        self.assertLessEqual(len(result["detections"]), 5)
```

## Framework Integration

### MMDetection/MMPose Backend

The current implementation uses MMDetection and MMPose as backends:

```python
# Internal implementation (simplified)
from mmdet.apis import init_detector, inference_detector
from mmpose.apis import init_pose_estimator, inference_topdown

class PrimateFaceProcessor:
    def _init_models(self):
        self.det_model = init_detector(
            self.det_config,
            self.det_checkpoint,
            device=self.device
        )
        
        if self.pose_config and self.pose_checkpoint:
            self.pose_model = init_pose_estimator(
                self.pose_config,
                self.pose_checkpoint,
                device=self.device
            )
```

### Adding New Frameworks

To add support for new frameworks (e.g., YOLO, MediaPipe):

```python
# Abstract base class for framework adapters
from abc import ABC, abstractmethod

class DetectorAdapter(ABC):
    @abstractmethod
    def detect(self, image):
        pass

class YOLOAdapter(DetectorAdapter):
    def __init__(self, model_path):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
    
    def detect(self, image):
        results = self.model(image)
        return self._convert_to_standard_format(results)
```

## Output Format Specification

### COCO Format Output

All detection and pose outputs follow COCO format:

```json
{
    "images": [
        {
            "id": 1,
            "file_name": "image.jpg",
            "width": 1920,
            "height": 1080
        }
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [x, y, width, height],
            "keypoints": [x1, y1, v1, x2, y2, v2, ...],
            "score": 0.95
        }
    ],
    "categories": [
        {
            "id": 1,
            "name": "primate_face",
            "keypoints": ["nose", "left_eye", ...],
            "skeleton": [[0, 1], [1, 2], ...]
        }
    ]
}
```

## Contributing

When adding new functionality to the demos module:

1. Follow existing patterns for consistency
2. Add comprehensive docstrings
3. Include unit tests in `test_demos.py`
4. Update this documentation
5. Ensure backward compatibility

## See Also

- [Main README](README.md) - Quick start guide
- [API Documentation](../docs/api/index.md) - Full API reference
- [Notebooks](notebooks/) - Interactive tutorials
- [Evaluation Tools](../evals/) - Model evaluation scripts