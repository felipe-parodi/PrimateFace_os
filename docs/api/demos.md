# Demos API - Core Processing Pipeline

The `demos` module provides the main Python API for PrimateFace, offering face detection, pose estimation, and visualization capabilities.

## Quick Reference

```python
from demos import PrimateFaceProcessor, FastPoseVisualizer
from demos.classify_genus import PrimateClassifierVLM
from demos.smooth_utils import MedianSavgolSmoother
```

## Main Classes

### PrimateFaceProcessor

Primary interface for detection and pose estimation.

```python
processor = PrimateFaceProcessor(
    det_config="mmdet_config.py",
    det_checkpoint="mmdet_checkpoint.pth",
    pose_config="mmpose_config.py",  # Optional
    pose_checkpoint="mmpose_checkpoint.pth",  # Optional
    device="cuda:0",
    bbox_thr=0.5,
    kpt_thr=0.7
)
```

**Key Methods:**
- `process_image(image_path, save_viz=False)` - Single image processing
- `process_video(video_path, output_dir, smooth=False)` - Video processing
- `process_image_directory(img_dir, output_dir)` - Batch processing

### FastPoseVisualizer

Real-time visualization for detection and pose results.

```python
visualizer = FastPoseVisualizer(
    kpt_thr=0.7,
    line_width=3,
    circle_radius=5
)

# Draw on image
viz_image = visualizer.draw_instances(
    image, 
    instances,
    draw_bbox=True,
    draw_pose=True
)
```

### PrimateClassifierVLM

Vision-Language Model based genus classification.

```python
classifier = PrimateClassifierVLM(
    model_name="SmolVLM",  # or "InternVL2-2B"
    device="cuda"
)

genus = classifier.classify_image("primate.jpg")
```

## Common Usage Patterns

### Detection Only Mode

```python
# No pose models provided = detection only
processor = PrimateFaceProcessor(
    det_config="mmdet_config.py",
    det_checkpoint="mmdet_checkpoint.pth"
)

result = processor.process_image("image.jpg")
print(f"Found {len(result['detections'])} faces")
```

### Detection + Pose Estimation

```python
# Both detection and pose models
processor = PrimateFaceProcessor(
    det_config="mmdet_config.py",
    det_checkpoint="mmdet_checkpoint.pth",
    pose_config="mmpose_config.py",
    pose_checkpoint="mmpose_checkpoint.pth"
)

result = processor.process_image("image.jpg")
for det in result["detections"]:
    keypoints = det.get("keypoints", [])
    print(f"Detected {len(keypoints)} keypoints")
```

### Video Processing with Smoothing

```python
# Temporal smoothing for stable keypoints
results = processor.process_video(
    "video.mp4",
    output_dir="results/",
    smooth=True,  # Enable smoothing
    save_viz=True
)
```

## Output Format

All methods return COCO-compatible format:

```python
{
    "image_path": "path/to/image.jpg",
    "detections": [
        {
            "bbox": [x1, y1, x2, y2],
            "score": 0.95,
            "keypoints": [  # If pose enabled
                [x, y, confidence],
                ...
            ]
        }
    ]
}
```

## Advanced Features

### Custom Visualization

```python
from demos.viz_utils import FastPoseVisualizer
import cv2

visualizer = FastPoseVisualizer()

# Process and visualize frame-by-frame
cap = cv2.VideoCapture("video.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = processor.process_frame(frame)
    viz_frame = visualizer.draw_instances(
        frame, 
        results["instances"]
    )
    
    cv2.imshow("PrimateFace", viz_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### Temporal Smoothing

```python
from demos.smooth_utils import MedianSavgolSmoother

smoother = MedianSavgolSmoother(
    window_size=7,
    savgol_window=11
)

# Apply to keypoints
smoothed = smoother.smooth_keypoints(
    keypoints, 
    instance_id=0
)
```

## Configuration

### Detection Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `bbox_thr` | Detection confidence threshold | 0.5 | 0.0-1.0 |
| `nms_thr` | Non-maximum suppression threshold | 0.3 | 0.0-1.0 |
| `max_per_img` | Maximum detections per image | 100 | 1-1000 |

### Pose Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `kpt_thr` | Keypoint confidence threshold | 0.7 | 0.0-1.0 |
| `use_oks_nms` | Use OKS for NMS | False | True/False |
| `soft_nms` | Use soft NMS | False | True/False |

## Performance Tips

1. **GPU Acceleration**: Always use CUDA when available
2. **Batch Processing**: Process multiple images together
3. **Model Selection**: Use lighter models for real-time processing
4. **Threshold Tuning**: Adjust thresholds based on your data
5. **Temporal Smoothing**: Essential for video stability

## Detailed Documentation

For comprehensive API documentation including:
- Full parameter descriptions
- Implementation details
- Custom pipeline examples
- Testing guidelines
- Framework integration

See: **[demos/demo_docs.md](../../demos/demo_docs.md)**

## Related Modules

- [GUI Tools](gui.md) - Interactive annotation
- [DINOv2](dinov2.md) - Feature extraction
- [Evaluation](evaluation.md) - Model comparison
- [Converter](converter.md) - Landmark conversion

## Examples

### Process Dataset

```python
from pathlib import Path
import json

# Process all images in dataset
dataset_path = Path("dataset/images")
results = []

for img_path in dataset_path.glob("*.jpg"):
    result = processor.process_image(str(img_path))
    results.append({
        "filename": img_path.name,
        "detections": result["detections"]
    })

# Save results
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Generate Publication Figures

```python
from demos.visualize_coco_annotations import COCOVisualizer

visualizer = COCOVisualizer()
visualizer.visualize_dataset(
    coco_path="annotations.json",
    img_dir="images/",
    output_dir="figures/",
    format="pdf",  # For publication
    num_samples=10
)
```

## Troubleshooting

### Common Issues

1. **Module not found**: Ensure correct installation and environment
2. **CUDA errors**: Check GPU availability and CUDA version
3. **Memory errors**: Reduce batch size or use CPU
4. **Import errors**: Verify all dependencies installed

### Getting Help

- Check [Technical Documentation](../../demos/demo_docs.md)
- See [GitHub Issues](https://github.com/KordingLab/PrimateFace/issues)
- Review [Notebooks](../../demos/notebooks/) for examples

## Next Steps

- Explore [Interactive Tutorials](../tutorials/index.md)
- Try [GUI Tools](gui.md) for annotation
- Use [DINOv2](dinov2.md) for feature analysis
- Run [Evaluations](evaluation.md) on your models