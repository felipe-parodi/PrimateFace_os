# Detection API

Face detection interfaces and utilities.

## Core Classes

### PrimateFaceProcessor

Main detection and pose estimation pipeline.

```python
from demos.process import PrimateFaceProcessor

processor = PrimateFaceProcessor(
    det_config="path/to/det_config.py",
    det_checkpoint="path/to/det_model.pth",
    pose_config="path/to/pose_config.py",
    pose_checkpoint="path/to/pose_model.pth",
    device="cuda:0"
)
```

#### Methods

##### process()
```python
result = processor.process(
    image="path/to/image.jpg",
    det_threshold=0.5,
    return_visualization=True
)
```

##### process_batch()
```python
results = processor.process_batch(
    images=["img1.jpg", "img2.jpg"],
    batch_size=8,
    show_progress=True
)
```

### Detection Models

#### MMDetection Integration

```python
from mmdet.apis import init_detector, inference_detector

# Initialize detector
det_model = init_detector(
    config="configs/cascade_rcnn_r50.py",
    checkpoint="checkpoints/cascade_rcnn.pth",
    device="cuda:0"
)

# Run inference
result = inference_detector(det_model, image)
```

#### YOLO Detection

```python
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n-face.pt")

# Detect faces
results = model(image, conf=0.5)
boxes = results[0].boxes.xyxy.cpu().numpy()
```

## Utility Functions

### Bounding Box Operations

```python
from demos.utils import (
    expand_bbox,
    crop_face,
    nms_boxes
)

# Expand bounding box by 20%
expanded = expand_bbox(bbox, scale=1.2)

# Crop face region
face_img = crop_face(image, bbox, padding=10)

# Non-maximum suppression
filtered_boxes = nms_boxes(boxes, scores, threshold=0.5)
```

### Batch Processing

```python
# TODO: Verify if BatchProcessor class exists in demos.utils
# If not, use PrimateFaceProcessor.process_image_directory instead

from demos.process import PrimateFaceProcessor

processor = PrimateFaceProcessor(
    det_config="demos/mmdet_config.py",
    det_checkpoint="demos/mmdet_checkpoint.pth"
)

# Process directory
results = processor.process_image_directory(
    img_dir="images/",
    output_dir="results/",
    save_viz=True,
    save_predictions=True
)
```

## Configuration

### Detection Config Structure

```python
det_config = dict(
    model=dict(
        type='CascadeRCNN',
        backbone=dict(
            type='ResNet',
            depth=50
        ),
        neck=dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048]
        ),
        rpn_head=dict(
            type='RPNHead',
            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[8],
                ratios=[0.5, 1.0, 2.0]
            )
        )
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7)
        )
    )
)
```

## Performance Optimization

### GPU Optimization

```python
# TODO: Verify if ParallelProcessor exists in demos.parallel
# Multi-GPU inference may require custom implementation
# For now, single GPU is recommended:

processor = PrimateFaceProcessor(
    det_config="demos/mmdet_config.py",
    det_checkpoint="demos/mmdet_checkpoint.pth",
    device="cuda:0"  # Specify GPU device
)
```

### Memory Management

```python
import torch

# Process large batches
with torch.cuda.amp.autocast():
    results = processor.process_large_batch(
        images=large_image_list,
        chunk_size=100
    )
```

## Error Handling

```python
from demos.exceptions import (
    ModelNotFoundError,
    InvalidImageError,
    DetectionFailedError
)

try:
    result = processor.process(image)
except ModelNotFoundError:
    print("Please download models first")
except InvalidImageError:
    print("Image format not supported")
except DetectionFailedError:
    print("No faces detected")
```

## Integration Examples

### With Pose Estimation

```python
# Detect then estimate pose
detections = detector.detect(image)
for box in detections:
    face_crop = crop_face(image, box)
    landmarks = pose_model.predict(face_crop)
```

### With Tracking

```python
# TODO: Verify if FaceTracker exists in demos.tracking
# Tracking may be implemented via smoothing in process_video

processor = PrimateFaceProcessor(
    det_config="demos/mmdet_config.py",
    det_checkpoint="demos/mmdet_checkpoint.pth",
    use_smoothing=True  # Enable temporal smoothing for videos
)

# Process video with tracking/smoothing
results = processor.process_video(
    video_path="video.mp4",
    output_dir="results/"
)
```

## See Also

- [Pose API](./pose.md)
- [Demos Module](./demos.md)
- [User Guide](../user-guide/core-workflows/demos.md)