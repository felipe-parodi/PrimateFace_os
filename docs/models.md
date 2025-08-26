# Models

## Pretrained Models

Access our pretrained models optimized for primate facial detection and landmark analysis.

## Currently Available Models

Download our models using the provided script:

```bash
python demos/download_models.py
```

This downloads:
- **Cascade R-CNN** - Face detection model (~300MB)
- **HRNet-W32** - 68-point landmark detection (~110MB)
- Configuration files for both models

## Model Performance

| Model | Task | Performance | Framework |
|-------|------|-------------|-----------|
| Cascade R-CNN | Face Detection | ~95% mAP | MMDetection |
| HRNet-W32 | Landmark Detection | <5% NME | MMPose |

*Performance measured on primate face validation set.*

## Quick Start

### Download Models

```bash
# Download to demos directory
cd demos
python download_models.py

# Or specify custom directory
python download_models.py /path/to/models/
```

### Basic Usage

```python
from demos.process import PrimateFaceProcessor

# Initialize with downloaded models
processor = PrimateFaceProcessor(
    det_config="demos/mmdet_config.py",
    det_checkpoint="demos/mmdet_checkpoint.pth",
    pose_config="demos/mmpose_config.py",
    pose_checkpoint="demos/mmpose_checkpoint.pth",
    device="cuda:0"
)

# Process image
import cv2
image = cv2.imread("primate.jpg")
bboxes, scores = processor.detect_primates(image)
```

## Coming Soon

### Additional Models
- **YOLOv8-Face** - Ultralytics real-time detection
- **Landmark Converters** - 68→48 point conversion models
- **Species Classifiers** - Genus/species identification

### Model Zoo Features
- HuggingFace integration
- Direct download links
- Model cards with detailed metrics
- ONNX export for deployment

## Training Custom Models

### Using Your Data

Train on your own annotations:

```bash
# Detection model
python evals/train_detection.py \
    --config configs/custom_detection.py \
    --data your_coco_annotations.json

# Pose model
python evals/train_pose.py \
    --config configs/custom_pose.py \
    --data your_coco_annotations.json
```

### Framework Options

We support training with multiple frameworks:
- **MMDetection/MMPose** - Primary framework
- **DeepLabCut** - Direct COCO training
- **SLEAP** - Multi-animal support
- **Ultralytics** - Real-time models

See [Framework Integration](user-guide/framework-integration/index.md) for details.

## Model Architecture

### Detection Model
- **Architecture**: Cascade R-CNN
- **Backbone**: ResNet-50-FPN
- **Input Size**: 800×800
- **Classes**: 1 (primate_face)

### Pose Model
- **Architecture**: HRNet-W32
- **Input Size**: 256×192
- **Output**: 68 facial landmarks
- **Heatmap Resolution**: 64×48

## License

Models and code are released under the MIT License for research purposes.

## Citation

If you use our models, please cite:

```bibtex
@article{parodi2025primateface,
  title={PrimateFace: A Machine Learning Resource for Automated Face Analysis in Human and Non-human Primates},
  author={Parodi, Felipe and Matelsky, Jordan and Lamacchia, Alessandro and others},
  journal={bioRxiv},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```

## Contact

For questions about models or early access to additional models:
- Email: [primateface@gmail.com](mailto:primateface@gmail.com)
- GitHub: [Issues](https://github.com/KordingLab/PrimateFace/issues)