# SLEAP Integration

Integration guide for using PrimateFace with SLEAP for multi-animal tracking.

## Overview

SLEAP (Social LEAP) excels at multi-animal pose estimation and tracking. PrimateFace provides utilities to convert COCO annotations to SLEAP format.

## Quick Start

```bash
# Train SLEAP model with COCO format
python evals/sleap/train_sleap_with_coco.py \
    --profile baseline_large_rf.topdown.json \
    --output_dir ./my_sleap_model
```

**Note:** Update data paths in the script before running.

## Integration Points

### Data Preparation

PrimateFace provides a script for training SLEAP models with COCO format:
- Script: `evals/sleap/train_sleap_with_coco.py`
- Converts COCO to SLEAP's .slp format automatically
- Validates images and filters invalid data
- Caches validation results for faster reruns

### Training Pipeline

1. **Configure Data Paths**
   Edit `evals/sleap/train_sleap_with_coco.py` to set:
   ```python
   TRAIN_JSON_PATH = "path/to/train.json"
   VAL_JSON_PATH = "path/to/val.json"
   TEST_JSON_PATH = "path/to/test.json"
   IMAGE_DIR = "path/to/images/"
   ```

2. **List Available Profiles**
   ```bash
   # See all built-in training profiles
   python evals/sleap/train_sleap_with_coco.py
   ```

3. **Train Model**
   ```bash
   # Basic training
   python evals/sleap/train_sleap_with_coco.py \
     --profile baseline_large_rf.topdown.json \
     --output_dir ./sleap_model
   
   # With custom parameters
   python evals/sleap/train_sleap_with_coco.py \
     --profile baseline_large_rf.topdown.json \
     --output_dir ./sleap_model \
     --epochs 50 \
     --input-scale 0.5 \
     --preload  # Load data into RAM
   ```

4. **Evaluation**
   ```bash
   python evals/sleap/evaluate_sleap_nme.py \
     --model-path sleap_model \
     --test-data test.json
   ```

## Project Structure

After conversion:
```
sleap_project/
├── labels.v001.slp
├── models/
│   ├── centroid/
│   └── instance/
└── predictions/
```

## Multi-Animal Tracking

SLEAP excels at scenarios with multiple primates:
- Automatic identity tracking
- Occlusion handling
- Social interaction analysis

## Performance Tips

- **Model Architecture**: Use UNet for best accuracy
- **Augmentation**: Enable rotation and scale augmentation
- **Tracking**: Configure Kalman filters for smooth tracks

## Troubleshooting

### Common Issues

1. **Instance confusion**
   - Adjust tracking parameters
   - Increase centroid model accuracy

2. **Memory issues**
   - Reduce batch size
   - Use mixed precision training

## See Also

- [SLEAP Documentation](https://sleap.ai/)
- [Training Script](https://github.com/KordingLab/PrimateFace/blob/main/evals/sleap/train_sleap_with_coco.py)
- [GUI Workflow](../core-workflows/gui.md)