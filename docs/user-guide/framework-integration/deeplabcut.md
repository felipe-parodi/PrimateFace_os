# DeepLabCut Integration

Integration guide for using PrimateFace with DeepLabCut for markerless tracking.

## Overview

DeepLabCut is a popular framework for markerless pose estimation. PrimateFace provides utilities to convert COCO annotations to DeepLabCut format and train models.

## Quick Start

```bash
# Train DeepLabCut model directly with COCO format
python evals/dlc/train_with_coco.py \
    --model_type hrnet_w32 \
    --output_dir ./my_trained_model \
    --num_epochs 100 \
    --batch_size 64
```

**Note:** DeepLabCut now supports COCO format directly via the PyTorch engine.

## Integration Points

### Direct COCO Training

PrimateFace provides a script for training DeepLabCut models directly with COCO format:
- Script: `evals/dlc/train_with_coco.py`
- Supports multiple model architectures (ResNet, HRNet, RTMPose)
- No conversion needed - uses COCO format directly

### Training Pipeline

1. **Configure Data Paths**
   Edit `evals/dlc/train_with_coco.py` to set your dataset paths:
   ```python
   TRAIN_JSON_PATH = "path/to/train.json"
   VAL_JSON_PATH = "path/to/val.json"
   TEST_JSON_PATH = "path/to/test.json"
   ```

2. **Train Model**
   ```bash
   # List available models
   python evals/dlc/train_with_coco.py
   
   # Train with specific architecture
   python evals/dlc/train_with_coco.py \
     --model_type hrnet_w32 \
     --output_dir ./dlc_model \
     --num_epochs 100
   ```

3. **Resume Training**
   ```bash
   python evals/dlc/train_with_coco.py \
     --resume path/to/snapshot.pt \
     --output_dir ./continued_training
   ```

4. **Evaluation**
   ```bash
   python evals/dlc/evaluate_nme.py \
     --model-path dlc_project \
     --test-data test.json
   ```

## Project Structure

After conversion:
```
dlc_project/
├── config.yaml
├── labeled-data/
│   └── dataset/
│       ├── CollectedData.csv
│       └── images/
└── training-datasets/
```

## Performance Tips

- **Data Augmentation**: Enable in DeepLabCut config
- **Multi-animal**: Supported via maDLC
- **Transfer Learning**: Use PrimateFace pretrained models as starting point

## Troubleshooting

### Common Issues

1. **Keypoint mismatch**
   - Ensure using 68-point annotations
   - Check keypoint ordering in config.yaml

2. **Training convergence**
   - Adjust learning rate
   - Increase training iterations

## See Also

- [DeepLabCut Documentation](http://www.mackenziemathislab.org/deeplabcut)
- [Training Script](https://github.com/KordingLab/PrimateFace/blob/main/evals/dlc/train_with_coco.py)
- [GUI Workflow](../core-workflows/gui.md)