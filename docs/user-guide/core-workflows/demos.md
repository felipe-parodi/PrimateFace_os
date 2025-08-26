# Using Pretrained PrimateFace Models for Inference

## Overview

This guide walks through using PrimateFace's pretrained models for face detection and landmark estimation on primate images and videos. The models are based on **MMDetection** (Cascade R-CNN) and **MMPose** (HRNet-W32) frameworks.

**Key capabilities:**
- Face detection with ~95% mAP on primate faces
- 48-point facial landmark estimation with <5% NME
- Batch processing for directories
- Temporal smoothing for videos
- GPU acceleration with CPU fallback

## Prerequisites

### 1. Environment Setup

Ensure you have the PrimateFace conda environment activated:

```bash
conda activate primateface
```

### 2. Install MMDetection/MMPose Dependencies

The demo scripts require OpenMMLab frameworks:

```bash
# Install OpenMIM package manager
uv pip install -U openmim "numpy<2.0"

# Install MMDetection and MMPose (with SSL workaround if needed)
mim install mmengine==0.10.3 --trusted-host download.openmmlab.com --trusted-host pypi.org
mim install "mmcv==2.1.0" --trusted-host download.openmmlab.com --trusted-host pypi.org
mim install "mmdet==3.2.0" --trusted-host download.openmmlab.com --trusted-host pypi.org
mim install "mmpose==1.3.2" --trusted-host download.openmmlab.com --trusted-host pypi.org
```

**Note:** GPU is highly recommended. The models work on CPU but will be significantly slower.

## Download Pretrained Models

PrimateFace provides pretrained models via Google Drive. Use the download script:

```bash
cd demos
python download_models.py  # Downloads to current directory
```

Or specify a custom directory:

```bash
python download_models.py ./models
```

This downloads 4 files (~410MB total):
- `mmdet_config.py` - Cascade R-CNN configuration
- `mmdet_checkpoint.pth` - Detection weights (~300MB)
- `mmpose_config.py` - HRNet-W32 configuration
- `mmpose_checkpoint.pth` - Pose estimation weights (~110MB)

## CLI Usage

The main entry point is `demos/primateface_demo.py` which provides a unified interface.

### Single Image Processing

#### Detection Only
```bash
python primateface_demo.py process \
    --input ateles_000003.jpeg \
    --input-type image \
    --det-config mmdet_config.py \
    --det-checkpoint mmdet_checkpoint.pth \
    --output-dir results/ \
    --save-viz
```

#### Detection + Pose Estimation
```bash
python primateface_demo.py process \
    --input ateles_000003.jpeg \
    --input-type image \
    --det-config mmdet_config.py \
    --det-checkpoint mmdet_checkpoint.pth \
    --pose-config mmpose_config.py \
    --pose-checkpoint mmpose_checkpoint.pth \
    --output-dir results/ \
    --save-viz --viz-pose
```

### Video Processing

Process videos with optional temporal smoothing:

```bash
python demos/primateface_demo.py process \
    --input path/to/primate_video.mp4 \
    --input-type video \
    --det-config demos/mmdet_config.py \
    --det-checkpoint demos/mmdet_checkpoint.pth \
    --pose-config demos/mmpose_config.py \
    --pose-checkpoint demos/mmpose_checkpoint.pth \
    --output-dir results/ \
    --save-viz --save-predictions \
    --smooth  # Enable temporal smoothing
```

Smoothing parameters (optional):
- `--smooth-median-window 5` - Median filter window (default: 5)
- `--smooth-savgol-window 7` - Savitzky-Golay window (default: 7)
- `--smooth-savgol-order 3` - Polynomial order (default: 3)

### Batch Directory Processing

Process all images in a directory:

```bash
python primateface_demo.py process \
    --input ./primate_images/ \
    --input-type images \
    --det-config mmdet_config.py \
    --det-checkpoint mmdet_checkpoint.pth \
    --pose-config mmpose_config.py \
    --pose-checkpoint mmpose_checkpoint.pth \
    --output-dir batch_results/ \
    --save-predictions --save-viz
```

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`

## Configuration Options

### Confidence Thresholds

| Parameter | Default | Description | Recommended Range |
|-----------|---------|-------------|-------------------|
| `--bbox-thr` | 0.5 | Detection confidence threshold | 0.3-0.7 |
| `--kpt-thr` | 0.7 | Keypoint visibility threshold | 0.5-0.9 |
| `--nms-thr` | 0.3 | NMS IoU threshold | 0.3-0.5 |

### Output Options

| Flag | Description |
|------|-------------|
| `--save-viz` | Save visualization images/videos |
| `--save-predictions` | Save COCO JSON annotations |
| `--viz-pose` | Draw keypoints (not just boxes) |
| `--output-dir` | Output directory (default: `./output`) |

### Device Selection

```bash
# Use specific GPU
--device cuda:0

# Use CPU (slower)
--device cpu
```

## Python API

For integration into your own code, use the `PrimateFaceProcessor` class:

### Basic Usage

```python
from demos.process import PrimateFaceProcessor

# Initialize processor
processor = PrimateFaceProcessor(
    det_config="demos/mmdet_config.py",
    det_checkpoint="demos/mmdet_checkpoint.pth",
    pose_config="demos/mmpose_config.py",  # Optional
    pose_checkpoint="demos/mmpose_checkpoint.pth",  # Optional
    device="cuda:0"
)
```

### Process Single Image

```python
import cv2

# Load image
image = cv2.imread("primate.jpg")

# Detect faces
bboxes, scores = processor.detect_primates(
    image, 
    bbox_thr=0.5,
    nms_thr=0.3
)

# Get pose if model loaded
if processor.pose_model:
    keypoints, kpt_scores = processor.estimate_pose(
        image, 
        bboxes,
        kpt_thr=0.7
    )
```

### Process Video

```python
# Process entire video
results = processor.process_video(
    video_path="primate_video.mp4",
    output_dir="results/",
    save_viz=True,
    save_predictions=True,
    smooth=True,  # Apply temporal smoothing
    viz_pose=True  # Visualize keypoints
)

# Results include frame-by-frame detections and keypoints
for frame_id, frame_results in results.items():
    print(f"Frame {frame_id}: {len(frame_results['bboxes'])} faces detected")
```

### Batch Processing

```python
# Process directory of images
results = processor.process_image_directory(
    img_dir="./primate_images/",
    output_dir="batch_results/",
    save_predictions=True,
    save_viz=True
)

# Export as COCO format
processor.export_coco_json(results, "annotations.json")
```

## Output Formats

### COCO JSON Structure

The `--save-predictions` flag generates COCO-format JSON (`predictions.json`):

```json
{
    "images": [
        {
            "id": 0,
            "file_name": "image001.jpg",
            "width": 640,
            "height": 480
        }
    ],
    "annotations": [
        {
            "id": 0,
            "image_id": 0,
            "category_id": 1,
            "bbox": [100, 150, 200, 250],  // [x, y, width, height]
            "area": 50000,
            "keypoints": [x1, y1, v1, x2, y2, v2, ...],  // 48 points × 3
            "num_keypoints": 48,
            "score": 0.95
        }
    ],
    "categories": [
        {
            "id": 1,
            "name": "primate_face",
            "keypoints": ["point_0", "point_1", ...],  // 48 landmark names
            "skeleton": [[0, 1], [1, 2], ...]  // Connectivity
        }
    ]
}
```

### Visualization Output

- **Images**: Saved as `{filename}_viz.jpg` with bboxes and keypoints drawn
- **Videos**: Saved as `{filename}_viz.mp4` with annotations overlaid

## Advanced Features

### Temporal Smoothing (Videos)

The smoothing pipeline (`demos/smooth_utils.py:MedianSavgolSmoother`) applies:
1. Median filter to remove outliers
2. Savitzky-Golay filter for smooth trajectories

```python
from demos.smooth_utils import MedianSavgolSmoother

smoother = MedianSavgolSmoother(
    median_window=5,
    savgol_window=7,
    savgol_order=3
)

# Apply to keypoint sequence
smoothed_keypoints = smoother.smooth_keypoints(keypoint_sequence)
```

### Custom Visualization

The visualizer (`demos/viz_utils.py:FastPoseVisualizer`) supports customization:

```python
from demos.viz_utils import FastPoseVisualizer

visualizer = FastPoseVisualizer(
    draw_keypoints=True,
    draw_skeleton=True,
    draw_bbox=True,
    keypoint_color=(0, 255, 0),  # Green keypoints
    skeleton_color=(255, 0, 0),   # Red skeleton
    bbox_color=(0, 0, 255),       # Blue boxes
    thickness=2
)

# Draw on image
viz_image = visualizer.draw_pose(
    image,
    keypoints,
    scores,
    bboxes
)
```

## Performance Optimization

### GPU Memory Management

For large batches or high-resolution videos:

```python
# Process in chunks
processor = PrimateFaceProcessor(
    det_config="mmdet_config.py",
    det_checkpoint="mmdet_checkpoint.pth",
    device="cuda:0"
)

# Set smaller batch size for video processing
processor.process_video(
    video_path="large_video.mp4",
    batch_size=4,  # Process 4 frames at a time
    output_dir="results/"
)
```

### Speed vs Accuracy Trade-offs

1. **Detection only** (~10 FPS on V100): Fastest, provides face locations
2. **Detection + Pose** (~5 FPS on V100): Full pipeline, all landmarks
3. **With smoothing** (minimal overhead): Best for videos, reduces jitter

## Troubleshooting

### CUDA Out of Memory

```bash
# Solution 1: Use CPU
python primateface_demo.py process --device cpu ...

# Solution 2: Reduce input resolution (resize images before processing)
```

### Import Errors

```bash
# Verify installation
python -c "import mmdet; print(mmdet.__version__)"  # Should be 3.2.0
python -c "import mmpose; print(mmpose.__version__)"  # Should be 1.3.2

# Reinstall if needed
mim uninstall mmdet mmpose
mim install "mmdet==3.2.0" "mmpose==1.3.2"
```

### Model Loading Issues

```bash
# Verify model files exist
ls -la demos/*.pth demos/*.py

# Re-download if corrupted
rm demos/*.pth demos/*.py
python demos/download_models.py
```

## Testing Your Setup

Run the test suite to verify installation:

```bash
cd demos
python test_demos.py

# Or with pytest for detailed output
pytest test_demos.py -v
```

Expected output:
```
test_detection_only ... ok
test_detection_and_pose ... ok
test_video_processing ... ok
test_batch_processing ... ok
```

## Next Steps

- **Evaluate on your data**: See [Evaluation Metrics](evaluation-metrics.md) to compute mAP and NME
- **Generate training data**: Use [Pseudo-labeling](pseudo-labeling.md) to annotate your images
- **Fine-tune models**: Train on your data with [MMPose Training](mmpose-training.md)
- **Add species classification**: Integrate [Genus Classification](genus-classification.md)

## References

- Main processing script: `demos/process.py:PrimateFaceProcessor`
- CLI interface: `demos/primateface_demo.py`
- Visualization utilities: `demos/viz_utils.py:FastPoseVisualizer`
- Smoothing utilities: `demos/smooth_utils.py:MedianSavgolSmoother`
- Model configs: `demos/mmdet_config.py`, `demos/mmpose_config.py`

## Detailed API Documentation

For comprehensive API reference, advanced usage patterns, and detailed parameter documentation, see the [Demos API Reference](../../api/demos.md).

This includes:
- Complete class and method documentation
- All CLI options and parameters
- Advanced configuration examples
- Performance optimization details
- Troubleshooting guides