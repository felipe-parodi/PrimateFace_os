# Tools & Utilities

PrimateFace provides a collection of tools and utilities for primate facial analysis. Each tool is designed for specific tasks in the analysis pipeline.

## Available Tools

### 🎨 [GUI for Pseudo-Labeling](gui.md)
Interactive interface for creating and refining annotations on your primate images.

```bash
python gui/main.py --input your_images/
```

### 🔄 [Landmark Converter](landmark_converter.md)
Convert between different landmark annotation formats (e.g., 68-point to 48-point).

```bash
python landmark-converter/convert.py --from 68 --to 48 --input annotations.json
```

### 🧠 [DINOv2 Feature Extraction](dinov2.md)
Extract and visualize self-supervised features using DINOv2.

```bash
python dinov2/dinov2_cli.py --input image.jpg --visualize
```

### 📊 [Evaluation Tools](evaluation.md)
Compare performance across different models and frameworks.

```bash
python evals/compare_det_models.py --models yolo,mmdet --dataset primateface
```

### 🎯 [Demo Scripts](demos.md)
Ready-to-run scripts for common tasks:

- Face detection on images/videos
- Landmark prediction
- Species classification
- Visualization utilities

## Installation

Most tools come with the base installation:

```bash
git clone https://github.com/KordingLab/PrimateFace.git
cd PrimateFace
pip install -e .
```

Some tools may require additional dependencies:

```bash
# For GUI tools
pip install -r gui/requirements.txt

# For evaluation tools
pip install -r evals/requirements.txt
```

## Usage Patterns

### Processing Single Images

```python
from demos import primateface_demo

# Run detection on a single image
results = primateface_demo.process_image("path/to/image.jpg")
```

### Batch Processing

```python
from demos import process

# Process multiple images
for image_path in image_paths:
    results = process.batch_process(image_path)
```

### Visualization

```python
from demos import viz_utils

# Visualize results
viz_utils.draw_landmarks(image, landmarks)
viz_utils.draw_bboxes(image, detections)
```

## Framework Integration

These tools work with various pose estimation frameworks:

- **[DeepLabCut](../frameworks/deeplabcut.md)** - For markerless tracking
- **[SLEAP](../frameworks/sleap.md)** - For multi-animal tracking
- **[MMPose](../frameworks/mmpose.md)** - For pose estimation
- **[Ultralytics](../frameworks/ultralytics.md)** - For YOLO-based detection

## Contributing New Tools

We welcome contributions! See our [Contributing Guide](../contribute.md) for:

- Code style guidelines
- Testing requirements
- Pull request process

## Support

- **Documentation**: Browse tool-specific docs
- **Examples**: Check `demos/` directory
- **Issues**: [GitHub Issues](https://github.com/KordingLab/PrimateFace/issues)
- **Contact**: [primateface@gmail.com](mailto:primateface@gmail.com)