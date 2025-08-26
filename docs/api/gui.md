# GUI API - Interactive Annotation Tools

The `gui` module provides unified tools for generating, reviewing, and refining annotations with multi-framework support.

## Quick Reference

```python
from gui import Detector, PoseEstimator, SAMMasker
from gui import ImageCOCOConverter, VideoCOCOConverter
from gui.core import ModelManager, FrameworkType
```

## Core Classes

### Detector

Multi-framework object detection supporting MMDetection and Ultralytics.

```python
from gui import Detector

detector = Detector(
    framework="mmdet",  # or "ultralytics"
    config_path="det.py",  # Optional for Ultralytics
    checkpoint_path="det.pth",
    device="cuda:0",
    bbox_thr=0.3,
    nms_thr=0.9,
    max_instances=3
)

# Single frame detection
bboxes, scores = detector.detect_frame(image, return_scores=True)

# Batch detection
results = detector.detect_batch(frames)

# Process directory
detections = detector.detect_directory("./images/")
```

**Methods:**
- `detect_frame(frame, return_scores=False)` - Detect in single image
- `detect_batch(frames, return_scores=False)` - Batch processing
- `detect_directory(img_dir, extensions=None)` - Process directory
- `process_video(video_path, frame_interval=1)` - Process video
- `update_thresholds(bbox_thr, nms_thr, max_instances)` - Update params
- `warmup(size=(640,480))` - Warmup model

### PoseEstimator

Flexible pose estimation with dynamic skeleton handling.

```python
from gui import PoseEstimator

pose = PoseEstimator(
    framework="mmpose",  # or "ultralytics"
    config_path="pose.py",
    checkpoint_path="pose.pth",
    device="cuda:0",
    kpt_thr=0.05,
    min_keypoints=10,
    coco_metadata={"keypoints": [...], "skeleton": [...]}  # Optional
)

# Estimate poses
poses = pose.estimate_pose(
    frame,
    bboxes,
    return_named=True,  # Return named keypoint dict
    composite_mask=sam_mask  # Optional SAM mask
)

# Process with COCO format
annotations, next_id = pose.process_frame(
    frame, bboxes, img_id=0, ann_id_start=0
)
```

**Methods:**
- `estimate_pose(frame, bboxes, return_named, composite_mask, mask_kpt_thr)` - Estimate poses
- `process_frame(frame, bboxes, img_id, ann_id_start)` - COCO format processing
- `process_directory(img_dir, detections, return_named)` - Batch processing
- `update_thresholds(kpt_thr, min_keypoints)` - Update parameters
- `get_model_info()` - Get model metadata

### SAMMasker

Segment Anything Model integration for mask refinement.

```python
from gui import SAMMasker

sam = SAMMasker(
    checkpoint_path="sam_vit_h.pth",
    model_type="vit_h",  # vit_b, vit_l, vit_h
    device="cuda:0"
)

# Generate masks for detections
refined_bboxes, composite_mask, masks = sam.process_detections(
    image,
    bboxes,
    refine_bboxes=True,
    generate_composite=True
)

# Project keypoints to mask
projected_kpts = SAMMasker.project_keypoints_to_mask(
    keypoints, scores, mask, threshold=0.3
)
```

**Methods:**
- `generate_masks(image, bboxes)` - Generate masks for multiple boxes
- `generate_mask(image, bbox, point_prompts)` - Single mask generation
- `process_detections(image, bboxes, refine_bboxes, generate_composite)` - Full pipeline
- `composite_mask(masks, img_hw)` - Create composite mask
- `project_keypoints_to_mask(keypoints, scores, mask)` - Project keypoints

### ModelManager

Efficient model loading and caching across frameworks.

```python
from gui.core import ModelManager, FrameworkType

manager = ModelManager()

# Load models with caching
det_model, det_meta = manager.load_model(
    FrameworkType.MMDET,
    config_path="det.py",
    checkpoint_path="det.pth",
    device="cuda:0"
)

pose_model, pose_meta = manager.load_model(
    FrameworkType.MMPOSE,
    config_path="pose.py",
    checkpoint_path="pose.pth"
)

# Distribute GPUs for parallel processing
devices = manager.distribute_gpus(num_workers=4)  # ['cuda:0', 'cuda:1', ...]

# Clear cache
manager.clear_cache()
```

## Converter Classes

### ImageCOCOConverter

Convert image directories to COCO format.

```python
from gui import ImageCOCOConverter

converter = ImageCOCOConverter(
    detector,
    pose_estimator=pose,  # Optional
    sam_masker=sam,  # Optional
    output_dir="./output",
    coco_template="existing.json",  # Copy skeleton from existing
    save_visualizations=True
)

# Process directory
json_path = converter.process_directory(
    "./images",
    max_images=1000,
    save_images=True,
    images_output_dir="./output/images"
)

# Process specific images
json_path = converter.process_image_list(["img1.jpg", "img2.jpg"])
```

### VideoCOCOConverter

Process videos with frame extraction.

```python
from gui import VideoCOCOConverter

converter = VideoCOCOConverter(
    detector,
    pose_estimator=pose,
    frame_interval=30,  # Every 30 frames
    save_frames=True
)

# Single video
json_path = converter.process_video(
    "video.mp4",
    max_frames=1000,
    frames_output_dir="./frames"
)

# Directory of videos
json_path = converter.process_directory(
    "./videos",
    max_videos=10,
    max_frames_per_video=1000
)

# Parallel processing
json_path = converter.process_parallel(
    "./videos",
    gpus=[0, 1, 2, 3],
    jobs_per_gpu=2
)
```

## CLI Interface

### Main Entry Point

```bash
python -m gui.pseudolabel [COMMAND] [OPTIONS]
```

### Commands

#### generate
```bash
python -m gui.pseudolabel generate \
    --input ./images --type image \
    --det-checkpoint det.pth \
    --pose-checkpoint pose.pth \
    --output-dir ./output \
    --device cuda:0 \
    --bbox-thr 0.3 \
    --save-viz
```

#### detect
```bash
python -m gui.pseudolabel detect \
    --input ./images --type image \
    --checkpoint det.pth \
    --output detections.json \
    --framework mmdet
```

#### pose
```bash
python -m gui.pseudolabel pose \
    --coco detections.json \
    --images ./images \
    --checkpoint pose.pth \
    --output poses.json
```

#### refine
```bash
python -m gui.pseudolabel refine \
    --coco annotations.json \
    --images ./images \
    --output refined.json \
    --enable-keypoints \
    --enable-sam
```

## Utility Functions

### Parallel Processing

```python
from gui.utils import parallel_process_videos

json_path = parallel_process_videos(
    video_dir="./videos",
    output_dir="./output",
    process_func=custom_processor,
    args=args,
    gpus=[0, 1, 2, 3],
    jobs_per_gpu=2
)
```

### Visualization

```python
from gui.utils import visualize_skeleton

vis_image = visualize_skeleton(
    image,
    keypoints,
    skeleton_links,
    keypoint_names,
    thickness=2
)
```

### COCO Utilities

```python
from gui.utils import load_coco_categories

keypoint_names, skeleton_links = load_coco_categories("annotations.json")
```

## Framework Types

```python
from gui.core import FrameworkType

# Available frameworks
FrameworkType.MMDET       # MMDetection
FrameworkType.MMPOSE      # MMPose
FrameworkType.ULTRALYTICS # Ultralytics YOLO
FrameworkType.DEEPLABCUT  # DeepLabCut (coming soon)
FrameworkType.SLEAP       # SLEAP (coming soon)
FrameworkType.SAM         # Segment Anything
```

## Configuration Constants

```python
from gui.constants import (
    DEFAULT_DEVICE,
    DEFAULT_BBOX_THR,
    DEFAULT_NMS_THR,
    DEFAULT_MAX_MONKEYS,
    DEFAULT_KPT_THR,
    DEFAULT_MIN_KEYPOINTS,
    DEFAULT_MASK_KPT_THR
)
```

## Error Handling

```python
try:
    detector = Detector("mmdet", checkpoint_path="det.pth")
except ImportError as e:
    print(f"Framework not installed: {e}")
except ValueError as e:
    print(f"Invalid configuration: {e}")
```

## Performance Tips

1. **Model Caching**: Use single ModelManager instance
```python
manager = ModelManager()
detector = Detector(..., model_manager=manager)
pose = PoseEstimator(..., model_manager=manager)
```

2. **Batch Processing**: Process multiple images together
```python
results = detector.detect_batch(frames)
```

3. **GPU Distribution**: Use parallel processing for videos
```python
converter.process_parallel(video_dir, gpus=[0,1,2,3])
```

4. **Memory Management**: Clear cache when done
```python
manager.clear_cache()
```

## Complete Example

```python
from gui import Detector, PoseEstimator, SAMMasker, ImageCOCOConverter
from gui.core import ModelManager

# Initialize manager for caching
manager = ModelManager()

# Setup models
detector = Detector(
    "mmdet",
    config_path="cascade_rcnn.py",
    checkpoint_path="cascade_rcnn.pth",
    bbox_thr=0.3,
    model_manager=manager
)

pose = PoseEstimator(
    "mmpose",
    config_path="hrnet.py",
    checkpoint_path="hrnet.pth",
    kpt_thr=0.05,
    model_manager=manager
)

sam = SAMMasker(
    checkpoint_path="sam_vit_h.pth",
    model_manager=manager
)

# Create converter
converter = ImageCOCOConverter(
    detector,
    pose,
    sam,
    output_dir="./output",
    save_visualizations=True
)

# Process images
json_path = converter.process_directory(
    "./images",
    max_images=1000
)

print(f"Annotations saved to: {json_path}")

# Cleanup
manager.clear_cache()
```