# PrimateFace Evaluation

Comprehensive evaluation tools for pose estimation models across multiple frameworks.

To evaluate PrimateFace, we trained face detection models in Ultralytics and mmdetection, and face pose estimation models in MMPose, DeepLabCut, and SLEAP.

We evaluated the detection models using mean average precision (mAP) and the pose estimation models using normalized mean error (NME).

This directory contains the code for evaluating the models as well as for training [DeepLabCut](dlc/README.md) and [SLEAP](sleap/README.md) models from COCO-formatted data.

## Overview

The `evals/` directory provides minimal code for:
- Training pose estimation models (DeepLabCut, SLEAP)
- Evaluating model performance with standard metrics (NME, mAP)

    
## Module Structure

```
evals/
├── core/                  # Core utilities
│   ├── metrics.py        # Evaluation metrics (NME, PCK, OKS)
│   ├── models.py         # Model loading and management
│   └── visualization.py  # Plotting and visualization
├── constants.py          # Configuration and defaults
├── dlc/                  # DeepLabCut-specific scripts
├── sleap/                # SLEAP-specific scripts
├── compare_det_models.py # Detection model comparison
├── compare_pose_models.py # Pose model comparison
└── eval_genera.py        # Per-genus evaluation
```

## Quick Start

### Installation

```bash
# Core dependencies (already in environment.yml)
conda env create -f ../environment.yml
conda activate primateface

# Framework-specific (install as needed)
# - MMPose/MMDetection: See demos/README.md
# - DeepLabCut: https://github.com/DeepLabCut/DeepLabCut
# - SLEAP: https://sleap.ai/installation.html
```

### Basic Usage

```bash
# Qualitative Comparison
# Face detection evaluation
python compare_det_models.py \
    --coco path/to/test_annotations.json \
    --model-config path/to/config.py \
    --model-checkpoint path/to/model.pth \
    --output results/detection_eval.json

# Face pose estimation evaluation  
python compare_pose_models.py \
    --coco path/to/test_annotations.json \
    --det-config path/to/det_config.py \
    --det-checkpoint path/to/det_model.pth \
    --pose-config path/to/pose_config.py \
    --pose-checkpoint path/to/pose_model.pth \
    --output results/pose_eval.json

# Per-Genus Quantitative Evaluation
python eval_genera.py \
    --predictions path/to/predictions.json \
    --annotations path/to/annotations.json \
    --output results/genus_metrics.json

# Genus Distribution
python plot_genus_distribution.py \
    --annotations path/to/annotations.json \
    --output genus_distribution.png
```


### Human Dataset Benchmarks

We also benchmarked PrimateFace (non-human primates) models against human benchmarks, including WIDERFACE, a popular human face detection dataset, and COCO-WholeBody-Face, a whole-body dataset containing human facial landmark data.

For these benchmarks, we trained and evaluated all models in mmdetection (WIDERFace) and mmpose (COCO-WholeBody-Face).



## Evaluation Metrics

### Mean Average Precision (mAP) for face detection
For detection evaluation:

```python
from evals import MetricsCalculator

calculator = MetricsCalculator()
# Note: mAP calculation is performed via framework-specific scripts
```

### Normalized Mean Error (NME) for facial landmark estimation
Measures average keypoint localization error normalized by a reference distance.

```python
from evals import NMECalculator

calculator = NMECalculator(normalize_by='bbox')  # or 'interocular'
nme = calculator.calculate(predictions, ground_truth, bboxes)
```


<!-- ## Configuration

All models are configured via YAML/JSON files to avoid hardcoded paths:

```yaml
# model_config.yaml
name: pf_hrnet_68kpt
framework: mmpose
config_path: configs/hrnet_w32_primateface.py
checkpoint_path: checkpoints/pf_hrnet_best.pth
device: cuda:0
additional_params:
  bbox_thr: 0.3
  kpt_thr: 0.3
``` -->

<!-- ## Model Zoo

Pre-trained models are available on [Hugging Face](https://huggingface.co/primateface):

| Model | Framework | Dataset | mAP/NME | Download |
|-------|-----------|---------|---------|----------|
| PF-Cascade-RCNN | MMDetection | PrimateFace | TBD | [Link]() |
| PF-HRNet | MMPose | PrimateFace | TBD | [Link]() |
| PF-ViTPose | MMPose | PrimateFace | TBD | [Link]() | -->

## Testing

Run the test suite:

```bash
# All tests
python -m pytest evals/ -v

# Specific test modules
python -m pytest evals/test_metrics.py -v
python -m pytest evals/test_visualization.py -v

# With coverage
python -m pytest evals/ --cov=evals --cov-report=html
```

## Best Practices

- Use configuration files instead of hardcoded paths
- Store model configs in YAML/JSON for reproducibility
<!-- - Use the unified `ModelConfig` and `ModelManager` classes -->
- Leverage batch processing for efficient evaluation