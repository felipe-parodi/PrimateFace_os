# SLEAP Training & Evaluation

Train and evaluate [SLEAP](https://sleap.ai/) pose estimation models using COCO-formatted annotations.

## Features

- Automatic COCO to SLEAP format conversion
- Support for multiple training strategies
- Cached preprocessing for faster iterations
- NME evaluation with customizable normalization

## Quick Start

```bash
# Train a model
python train_sleap_with_coco.py --profile baseline_large_rf.topdown.json --output_dir ./sleap_model

# Evaluate model performance
python evaluate_sleap_nme.py --model_dir ./sleap_model --test_json test.json

# List available training profiles
python train_sleap_with_coco.py --list-profiles

# Train with custom settings
python train_sleap_with_coco.py \
    --profile baseline_medium_rf.topdown.json \
    --epochs 50 \
    --input-scale 0.5
```

## Training Profiles

SLEAP offers various pre-configured training strategies:

- **Receptive Fields**: `large_rf`, `medium_rf`, `small_rf` - Control context size
- **Approaches**: 
  - `topdown` - Instance detection followed by pose estimation
  - `bottomup` - Keypoint detection followed by grouping
  - `single` - Single instance scenarios
  - `centroid` - Center-based instance detection

Example profiles: `baseline_large_rf.topdown.json`, `baseline.centroid.json`

## Requirements

Install [SLEAP](https://sleap.ai/installation.html) following their installation guide.

## Key Features

- **Automatic Conversion**: Seamlessly converts COCO annotations to SLEAP format
- **Data Caching**: Preprocessed data cached for efficient re-training
- **Flexible Architecture**: Choose from multiple model architectures and training strategies
- **GPU Support**: Optimized for CUDA-enabled TensorFlow