# Installation Guide

This guide covers installation of PrimateFace on different platforms and configurations.

## Quick Install

```bash
# Create conda environment
conda env create -f environment.yml
conda activate primateface

# Install PyTorch (check your CUDA version first)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install PrimateFace
pip install -e .
```

For detailed instructions, see the sections below.

## Prerequisites

- Python 3.8 or higher
- Conda (recommended) or pip
- Git
- 10GB+ free disk space
- CUDA-capable GPU (optional but recommended)

## Platform-Specific Installation

### Linux

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-dev python3-pip git

# Clone repository
git clone https://github.com/KordingLab/PrimateFace.git
cd PrimateFace

# Create environment
conda env create -f environment.yml
conda activate primateface

# Install PyTorch with CUDA support
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install PrimateFace
pip install -e .
```

### macOS

```bash
# Install Xcode command line tools
xcode-select --install

# Install Homebrew if needed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Clone repository
git clone https://github.com/KordingLab/PrimateFace.git
cd PrimateFace

# Create environment
conda env create -f environment.yml
conda activate primateface

# Install PyTorch (CPU only for Mac)
pip install torch==2.1.0 torchvision==0.16.0

# Install PrimateFace
pip install -e .
```

### Windows

```powershell
# Clone repository
git clone https://github.com/KordingLab/PrimateFace.git
cd PrimateFace

# Create environment
conda env create -f environment.yml
conda activate primateface

# Install PyTorch with CUDA support
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install PrimateFace
pip install -e .
```

## GPU Setup

### Check CUDA Version

```bash
# Check if CUDA is available
nvcc --version

# Or check with nvidia-smi
nvidia-smi
```

### Install Correct PyTorch Version

Based on your CUDA version:

```bash
# CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
```

## Optional Dependencies

### Full Installation

Install all optional components:

```bash
pip install -e ".[all]"
```

### Individual Components

```bash
# DINOv2 features
pip install -e ".[dinov2]"

# GUI tools
pip install -e ".[gui]"

# Development tools
pip install -e ".[dev]"

# Evaluation tools
pip install -e ".[evals]"
```

## Framework-Specific Setup

### MMDetection & MMPose

```bash
# Install using mim
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmdet>=3.0.0"
mim install "mmpose>=1.0.0"
```

### DeepLabCut

```bash
pip install deeplabcut
```

### SLEAP

```bash
conda install -c conda-forge -c nvidia -c sleap -c anaconda sleap=1.3.3
```

### Ultralytics YOLO

```bash
pip install ultralytics
```

## Verify Installation

### Test Basic Import

```python
# Test PrimateFace import
import demos
import dinov2
import gui
import evals
print("PrimateFace imported successfully!")
```

### Test GPU Support

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Download Models

```bash
# Download pretrained models
python demos/download_models.py
```

### Run Test Detection

```bash
# Test detection on sample image
python demos/primateface_demo.py process \
    --input samples/test.jpg \
    --det-config demos/mmdet_config.py \
    --det-checkpoint demos/mmdet_checkpoint.pth
```

## Troubleshooting

### Common Issues

#### ImportError: No module named 'xxx'
```bash
# Reinstall with all dependencies
pip install -e ".[all]"
```

#### CUDA out of memory
```python
# Use smaller batch size or CPU
processor = PrimateFaceProcessor(device="cpu")
```

#### Model files not found
```bash
# Re-download models
python demos/download_models.py --force
```

See [Troubleshooting Guide](../troubleshooting/) for more solutions.

## Docker Installation

For containerized deployment:

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app
COPY . /app

RUN pip install -e .
RUN python demos/download_models.py

CMD ["python", "demos/primateface_demo.py"]
```

Build and run:

```bash
docker build -t primateface .
docker run --gpus all -v /path/to/data:/data primateface
```

## Next Steps

Installation complete! Now:

1. [Try the Quickstart](../getting-started/quickstart/) - 5-minute tutorial
2. [Choose your workflow](../getting-started/decision-tree/) - Find the right tools
3. [Explore tutorials](../tutorials/) - Learn through examples

## Getting Help

- Check [FAQ](../faq/)
- Visit [GitHub Issues](https://github.com/KordingLab/PrimateFace/issues)
- Email: primateface@gmail.com