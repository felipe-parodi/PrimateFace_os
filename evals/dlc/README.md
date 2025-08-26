# DeepLabCut Training & Evaluation

Train and evaluate [DeepLabCut](http://www.mackenziemathislab.org/deeplabcut) pose estimation models using COCO-formatted annotations.

## Features

- Direct training from COCO JSON format
- Support for multiple backbone architectures
- Automatic best model tracking
- NME evaluation with interocular normalization

## Quick Start

```bash
# Train a model
python train_with_coco.py --model_type hrnet_w32 --output_dir ./dlc_model

# Evaluate model performance
python evaluate_nme.py --model_dir ./dlc_model

# Resume training from checkpoint
python train_with_coco.py --resume ./dlc_model/snapshot_epoch_50.pt

# List available architectures
python train_with_coco.py --list-models
```

## Supported Architectures

- **HRNet**: `hrnet_w18`, `hrnet_w32`, `hrnet_w48` - High-resolution networks
- **ResNet**: `resnet_50`, `resnet_101`, `resnet_152` - Classic architectures  
- **RTMPose**: `rtmpose_s`, `rtmpose_m`, `rtmpose_x` - Real-time models

## Requirements

Install [DeepLabCut PyTorch](https://github.com/DeepLabCut/DeepLabCut) following their installation guide.

## Key Features

- **Direct COCO Support**: Works directly with COCO JSON annotations
- **Progress Tracking**: Real-time training metrics with progress bars
- **Automatic Validation**: Best model saved based on validation performance
- **GPU Acceleration**: Full CUDA support for efficient training