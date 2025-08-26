# Interactive Pseudo-Labeling with GUI

**🚧 Note: GUI module has been consolidated. See new unified CLI below.**

## Overview

Generate and refine annotations using PrimateFace's interactive GUI for streamlined detection review, pose adjustment, and COCO export.

**Key features:**
- Multi-framework support (MMDetection, Ultralytics)
- Flexible skeleton handling (no hardcoded keypoints)
- SAM integration for precise boundaries
- Parallel GPU processing
- Unified CLI interface

## Installation

```bash
# Core dependencies
pip install -e ".[gui]"

# Framework support
mim install mmdet mmpose  # For MMDetection/MMPose
pip install ultralytics    # For YOLO models

# Optional: SAM support
pip install segment-anything
```

## Quick Start

### 1. Generate Annotations

```bash
# Basic detection + pose
python -m gui.pseudolabel generate \
    --input ./images --type image \
    --det-checkpoint det.pth \
    --pose-checkpoint pose.pth \
    --output-dir ./output

# With SAM refinement
python -m gui.pseudolabel generate \
    --input ./images --type image \
    --det-checkpoint det.pth \
    --sam-checkpoint sam_vit_h.pth \
    --output-dir ./output
```

### 2. Process Videos

```bash
# Single video processing
python -m gui.pseudolabel generate \
    --input ./videos --type video \
    --det-checkpoint det.pth \
    --frame-interval 30 \
    --output-dir ./output

# Parallel GPU processing
python -m gui.pseudolabel generate \
    --input ./videos --type video \
    --det-checkpoint det.pth \
    --gpus 0 1 2 3 \
    --jobs-per-gpu 2 \
    --output-dir ./output
```

### 3. Refine with GUI

```bash
python -m gui.pseudolabel refine \
    --coco annotations.json \
    --images ./images \
    --output refined.json \
    --enable-keypoints
```

## CLI Commands

### `generate` - Auto-annotate with models

```bash
python -m gui.pseudolabel generate [OPTIONS]

Required:
  --input PATH         Input directory (images/videos)
  --type {image,video} Input type
  --output-dir PATH    Output directory
  --det-checkpoint     Detection model checkpoint

Optional:
  --framework          Detection framework (mmdet/ultralytics)
  --det-config         Detection model config
  --pose-checkpoint    Pose model checkpoint
  --pose-config        Pose model config
  --sam-checkpoint     SAM checkpoint for masks
  --coco-template      COCO JSON to copy skeleton from
  --device             CUDA device (default: cuda:0)
  --bbox-thr           Detection threshold (default: 0.3)
  --nms-thr           NMS threshold (default: 0.9)
  --max-instances     Max faces per image (default: 3)
  --kpt-thr           Keypoint threshold (default: 0.05)
  --frame-interval    Video frame sampling (default: 30)
  --save-viz          Save visualizations
  --gpus              GPU IDs for parallel processing
  --jobs-per-gpu      Parallel jobs per GPU (default: 1)
```

### `detect` - Detection only

```bash
python -m gui.pseudolabel detect [OPTIONS]

Required:
  --input PATH         Input directory
  --type {image,video} Input type
  --output PATH        Output COCO JSON
  --checkpoint         Model checkpoint

Optional:
  --framework         Detection framework (default: mmdet)
  --config           Model config file
  --device           Device (default: cuda:0)
  --bbox-thr         Detection threshold (default: 0.3)
```

### `pose` - Add poses to detections

```bash
python -m gui.pseudolabel pose [OPTIONS]

Required:
  --coco PATH         COCO JSON with detections
  --images PATH       Image directory
  --output PATH       Output COCO JSON
  --checkpoint        Pose model checkpoint

Optional:
  --framework        Pose framework (default: mmpose)
  --config          Model config file
  --device          Device (default: cuda:0)
  --kpt-thr         Keypoint threshold (default: 0.05)
```

### `refine` - Interactive GUI refinement

```bash
python -m gui.pseudolabel refine [OPTIONS]

Required:
  --coco PATH         COCO JSON to refine
  --images PATH       Image directory

Optional:
  --output PATH       Output path
  --enable-keypoints  Enable keypoint editing
  --enable-sam       Enable SAM refinement
  --sam-checkpoint   SAM model path
```

## Python API

### Basic Usage

```python
from gui import Detector, PoseEstimator, ImageCOCOConverter

# Initialize models
detector = Detector(
    framework="mmdet",
    config_path="det.py",
    checkpoint_path="det.pth",
    device="cuda:0",
    bbox_thr=0.3
)

pose = PoseEstimator(
    framework="mmpose",
    config_path="pose.py",
    checkpoint_path="pose.pth",
    device="cuda:0",
    kpt_thr=0.05
)

# Process images
converter = ImageCOCOConverter(
    detector,
    pose,
    output_dir="./output",
    save_visualizations=True
)

json_path = converter.process_directory("./images")
```

### Video Processing

```python
from gui import VideoCOCOConverter

converter = VideoCOCOConverter(
    detector,
    pose,
    frame_interval=30,
    save_frames=True
)

# Single video
json_path = converter.process_video("video.mp4")

# Directory of videos
json_path = converter.process_directory(
    "./videos",
    max_videos=10,
    max_frames_per_video=1000
)
```

### Flexible Skeleton Handling

```python
# Use skeleton from existing COCO JSON
converter = ImageCOCOConverter(
    detector,
    pose,
    coco_template="existing_annotations.json"
)

# Or from model metadata
pose = PoseEstimator(
    framework="mmpose",
    config_path="pose.py",
    checkpoint_path="pose.pth",
    coco_metadata={"keypoints": [...], "skeleton": [...]}
)
```

### SAM Integration

```python
from gui import SAMMasker

# Initialize SAM
sam = SAMMasker(
    checkpoint_path="sam_vit_h.pth",
    model_type="vit_h",
    device="cuda:0"
)

# Process with masks
converter = ImageCOCOConverter(
    detector,
    pose,
    sam_masker=sam
)
```

## GUI Controls

### Detection Review

| Key | Action |
|-----|--------|
| `Space` | Accept and next |
| `Delete` | Remove bbox |
| `a` | Add new bbox |
| `r` | Reset |
| `s` | Save progress |
| `←/→` | Navigate |
| `Esc` | Exit |

### Pose Review

| Key | Action |
|-----|--------|
| `Space` | Accept and next |
| `v` | Toggle visibility |
| `1-9` | Set confidence |
| `h` | Toggle skeleton |
| `z/x` | Point size |
| `←/→` | Navigate |

## Advanced Features

### Multi-Framework Support

```python
# Ultralytics detection + MMPose
detector = Detector("ultralytics", checkpoint_path="yolov8x.pt")
pose = PoseEstimator("mmpose", config_path="hrnet.py", checkpoint_path="hrnet.pth")

# Both Ultralytics
detector = Detector("ultralytics", checkpoint_path="yolov8x.pt")
pose = PoseEstimator("ultralytics", checkpoint_path="yolov8x-pose.pt")
```

### Parallel Processing

```python
from gui.utils import parallel_process_videos

# Advanced users: custom parallel processing
json_path = parallel_process_videos(
    video_dir="./videos",
    output_dir="./output",
    process_func=custom_process,
    args=args,
    gpus=[0, 1, 2, 3],
    jobs_per_gpu=2
)
```

### Model Manager

```python
from gui.core import ModelManager

# Efficient model caching
manager = ModelManager()

# Load once, use multiple times
model1, meta1 = manager.load_model("mmdet", config_path="det.py", checkpoint_path="det.pth")
model2, meta2 = manager.load_model("mmpose", config_path="pose.py", checkpoint_path="pose.pth")

# Clear cache when done
manager.clear_cache()
```

## Output Format

Standard COCO JSON with flexible skeleton:

```json
{
    "images": [...],
    "annotations": [
        {
            "id": 0,
            "image_id": 0,
            "category_id": 1,
            "bbox": [x, y, w, h],
            "keypoints": [x1, y1, v1, ...],
            "num_keypoints": 48,
            "area": 39600.0,
            "iscrowd": 0
        }
    ],
    "categories": [
        {
            "id": 1,
            "name": "primate",
            "keypoints": ["nose", "left_eye", ...],  // From model or template
            "skeleton": [[0, 1], [1, 2], ...]       // Flexible connectivity
        }
    ]
}
```

## Quality Validation

```python
import json
import numpy as np

def validate_annotations(json_path):
    with open(json_path) as f:
        data = json.load(f)
    
    issues = []
    for ann in data['annotations']:
        # Check bbox validity
        bbox = ann['bbox']
        if bbox[2] <= 0 or bbox[3] <= 0:
            issues.append(f"Invalid bbox in annotation {ann['id']}")
        
        # Check keypoint count
        kpts = np.array(ann['keypoints']).reshape(-1, 3)
        visible = (kpts[:, 2] > 0).sum()
        
        if visible < 10:
            issues.append(f"Too few keypoints ({visible}) in {ann['id']}")
    
    return issues

# Validate
issues = validate_annotations("annotations.json")
print(f"Found {len(issues)} issues" if issues else "✓ All valid!")
```

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
--max-instances 1

# Use CPU
--device cpu
```

### Import Errors
```bash
# Verify installation
pip install -e ".[gui]"

# Install frameworks
mim install mmdet mmpose
pip install ultralytics
```

### Detection Issues
```bash
# Adjust thresholds
--bbox-thr 0.1  # Lower for more detections
--nms-thr 0.5   # Lower for less overlap
```

## Migration from Old Scripts

| Old Script | New Command |
|------------|-------------|
| `pseudolabel_gui.py` | `python -m gui.pseudolabel generate` |
| `pseudolabel_gui_fm.py` | `python -m gui.pseudolabel generate` |
| `imgdir2coco_facedet.py` | `python -m gui.pseudolabel detect --type image` |
| `viddir2coco_facedet.py` | `python -m gui.pseudolabel detect --type video` |
| `refine_boxes.py` | `python -m gui.pseudolabel refine` |

## Next Steps

- **Train models**: Use annotations with [Framework Integration](../framework-integration/)
- **Evaluate quality**: Check [Evaluation Metrics](../utilities/evaluation.md)
- **Convert formats**: Use [Landmark Converter](./landmark-converter.md)

## API Reference

Core classes:
- `gui.core.Detector` - Multi-framework detection
- `gui.core.PoseEstimator` - Flexible pose estimation
- `gui.core.SAMMasker` - SAM integration
- `gui.core.ModelManager` - Model caching
- `gui.converters.ImageCOCOConverter` - Image processing
- `gui.converters.VideoCOCOConverter` - Video processing

See [GUI API Reference](../../api/gui.md) for complete documentation.