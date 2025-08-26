# Ultralytics (YOLO) Integration

Integration guide for using PrimateFace with Ultralytics YOLO for real-time detection.

## Overview

Ultralytics YOLO provides fast, real-time detection and pose estimation. PrimateFace models can be exported to YOLO format for edge deployment.

## Quick Start

```python
from ultralytics import YOLO

# Load PrimateFace-trained YOLO model
model = YOLO("path/to/primateface_yolo.pt")

# Run inference
results = model("primate_image.jpg")
```

## Integration Points

### Model Formats

PrimateFace provides:
- Pre-trained YOLO models for primate faces
- Conversion scripts from COCO format
- Export utilities for deployment

### Detection Pipeline

1. **Face Detection**
   ```python
   # YOLOv8 detection
   detector = YOLO("yolov8_primate_faces.pt")
   boxes = detector.predict(source="image.jpg")
   ```

2. **Pose Estimation**
   ```python
   # YOLOv8-pose for landmarks
   pose_model = YOLO("yolov8_primate_pose.pt")
   keypoints = pose_model.predict(source="image.jpg")
   ```

## Training Custom Models

Convert COCO annotations to YOLO format:
```bash
# Conversion utility
python scripts/coco_to_yolo.py \
  --coco-json annotations.json \
  --output-dir yolo_dataset
```

Train with Ultralytics:
```python
model = YOLO("yolov8n.yaml")
model.train(
    data="primateface.yaml",
    epochs=100,
    imgsz=640
)
```

## Deployment

### Export Options

```python
# Export for different platforms
model.export(format="onnx")  # ONNX for cross-platform
model.export(format="tflite")  # TensorFlow Lite for mobile
model.export(format="coreml")  # CoreML for iOS
```

### Edge Deployment

- Raspberry Pi: Use TFLite export
- NVIDIA Jetson: Use TensorRT export
- Mobile: Use CoreML (iOS) or TFLite (Android)

## Performance Optimization

- **Input Size**: 640x640 for best speed/accuracy trade-off
- **Model Size**: YOLOv8n for edge, YOLOv8x for accuracy
- **Batch Processing**: Increase batch size for throughput

## Troubleshooting

### Common Issues

1. **Low FPS**
   - Use smaller model (nano/small variants)
   - Reduce input resolution

2. **Export errors**
   - Ensure compatible PyTorch version
   - Check export requirements for target platform

## See Also

- [Ultralytics Documentation](https://docs.ultralytics.com/)
- [YOLO Export Guide](https://docs.ultralytics.com/modes/export/)
- [Demos Workflow](../core-workflows/demos.md)