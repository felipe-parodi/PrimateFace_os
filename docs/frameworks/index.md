# Framework Integration

PrimateFace works with multiple popular computer vision and pose estimation frameworks. Choose the framework that best fits your workflow and requirements.

## Supported Frameworks

### 🎯 [Ultralytics (YOLO)](ultralytics.md)
Fast, accurate object detection using YOLO models.

**Best for**: Real-time face detection, quick inference
```python
from ultralytics import YOLO
model = YOLO('primateface_yolo.pt')
results = model('primate_image.jpg')
```

### 🔍 [MMDetection & MMPose](mmpose.md)
Comprehensive detection and pose estimation framework.

**Best for**: High-accuracy landmark detection, research applications
```python
from mmpose.apis import inference_top_down_pose_model
results = inference_top_down_pose_model(model, img)
```

### 🐾 [DeepLabCut](deeplabcut.md)
Markerless pose estimation with transfer learning.

**Best for**: Custom landmark definitions, fine-tuning on specific species
```python
import deeplabcut
deeplabcut.analyze_videos(config_path, videos)
```

### 🦎 [SLEAP](sleap.md)
Multi-animal pose tracking with deep learning.

**Best for**: Multi-individual tracking, social behavior analysis
```python
import sleap
model = sleap.load_model("primateface_sleap")
predictions = model.predict(video)
```

## Framework Comparison

| Framework | Speed | Accuracy | Multi-Animal | Custom Training | GPU Required |
|-----------|-------|----------|--------------|-----------------|--------------|
| Ultralytics | ⚡⚡⚡ | ⭐⭐⭐ | ✅ | ✅ | Optional |
| MMPose | ⚡⚡ | ⭐⭐⭐⭐ | ✅ | ✅ | Yes |
| DeepLabCut | ⚡⚡ | ⭐⭐⭐⭐ | ✅ | ✅ | Yes |
| SLEAP | ⚡⚡ | ⭐⭐⭐⭐ | ✅ | ✅ | Yes |

## Quick Start by Framework

### Detection Task
```bash
# Ultralytics (fastest)
python demos/primateface_demo.py --framework ultralytics --task detection

# MMDetection (most accurate)
python demos/primateface_demo.py --framework mmdet --task detection
```

### Landmark Detection
```bash
# DeepLabCut (most flexible)
python demos/primateface_demo.py --framework dlc --task landmarks

# SLEAP (best for multiple individuals)
python demos/primateface_demo.py --framework sleap --task landmarks
```

## Model Zoo

Pre-trained models are available for each framework:

| Framework | Model | Task | Download |
|-----------|-------|------|----------|
| Ultralytics | YOLOv8-PrimateFace | Detection | [Link](https://github.com/KordingLab/PrimateFace/releases) |
| MMPose | HRNet-PrimateFace | Landmarks | [Link](https://github.com/KordingLab/PrimateFace/releases) |
| DeepLabCut | DLC-PrimateFace | Full Pipeline | [Link](https://github.com/KordingLab/PrimateFace/releases) |
| SLEAP | SLEAP-PrimateFace | Multi-Animal | [Link](https://github.com/KordingLab/PrimateFace/releases) |

## Choosing a Framework

### For Beginners
Start with **Ultralytics** - easiest setup, good documentation, fast results.

### For Research
Use **DeepLabCut** or **SLEAP** - more control, better for custom applications.

### For Production
Consider **MMDetection/MMPose** - robust, scalable, well-maintained.

### For Real-time Applications
Choose **Ultralytics YOLO** - optimized for speed.

## Converting Between Frameworks

Use our conversion tools to move annotations between frameworks:

```python
from landmark_converter import convert_annotations

# Convert DeepLabCut to SLEAP format
convert_annotations(
    source_format="dlc",
    target_format="sleap",
    input_file="dlc_annotations.h5",
    output_file="sleap_annotations.slp"
)
```

## Custom Training

Each framework supports training on custom data:

- [Training with Ultralytics](ultralytics.md#custom-training)
- [Training with DeepLabCut](deeplabcut.md#custom-training)
- [Training with SLEAP](sleap.md#custom-training)
- [Training with MMPose](mmpose.md#custom-training)

## Support

- Framework-specific issues: Check respective documentation
- PrimateFace integration: [GitHub Issues](https://github.com/KordingLab/PrimateFace/issues)
- General questions: [primateface@gmail.com](mailto:primateface@gmail.com)