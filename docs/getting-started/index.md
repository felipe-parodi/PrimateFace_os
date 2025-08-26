# Getting Started

Welcome to PrimateFace! This guide will help you get up and running quickly.

## Overview

PrimateFace is a comprehensive toolkit for primate facial analysis. Whether you're studying behavior, conducting research, or building applications, we'll help you get started.

## Quick Navigation

### 🚀 [Installation](../installation/)
Complete setup guide for all platforms and configurations.

### ⚡ [Quickstart](./quickstart/)
Get running in 5 minutes with our quickstart tutorial.

### 🗺️ [Which Workflow Should I Use?](decision-tree.md)
Decision guide to help you choose the right tools for your task.

## Typical First Steps

### 1. Install PrimateFace

```bash
# Quick install
conda env create -f environment.yml
conda activate primateface
pip install -e .
```

See [Installation Guide](../installation/) for detailed instructions.

### 2. Download Models

```bash
python demos/download_models.py
```

### 3. Run Your First Detection

```bash
python demos/primateface_demo.py process \
    --input sample_image.jpg \
    --det-config demos/mmdet_config.py \
    --det-checkpoint demos/mmdet_checkpoint.pth
```

### 4. Explore Further

Based on your needs:

- **Have images to analyze?** → [Demos Workflow](../user-guide/core-workflows/demos.md)
- **Need to annotate data?** → [GUI Workflow](../user-guide/core-workflows/gui.md)
- **Want to understand features?** → [DINOv2 Workflow](../user-guide/core-workflows/dinov2.md)
- **Training models?** → [Framework Integration](../user-guide/framework-integration/index.md)

## Learning Path

### For Researchers
1. Start with [Tutorials](../tutorials/index.md) to see applications
2. Follow [Core Workflows](../user-guide/core-workflows/index.md) for your use case
3. Dive into [API Reference](../api/index.md) for customization

### For Developers
1. Review [API Reference](../api/index.md) for interfaces
2. Check [Framework Integration](../user-guide/framework-integration/index.md) for your framework
3. See [Contributing](../contribute.md) to help improve PrimateFace

### For Students
1. Read [Concepts](../user-guide/concepts.md) for theory
2. Try [Tutorials](../tutorials/index.md) for hands-on learning
3. Use [Decision Tree](decision-tree.md) to find your workflow

## Hardware Requirements

### Minimum
- CPU: 4 cores
- RAM: 8GB
- GPU: Optional (CPU mode available)
- Storage: 10GB

### Recommended
- CPU: 8+ cores
- RAM: 16GB+
- GPU: NVIDIA with 6GB+ VRAM
- Storage: 50GB for models and data

## Getting Help

- 📖 Check [Troubleshooting](../troubleshooting.md) for common issues
- 💬 Ask questions in [GitHub Discussions](https://github.com/KordingLab/PrimateFace/discussions)
- 🐛 Report bugs via [GitHub Issues](https://github.com/KordingLab/PrimateFace/issues)
- 📧 Contact us at primateface@gmail.com

## Next Steps

Ready to dive in? Choose your path:

1. **[Quickstart Tutorial](./quickstart/)** - 5-minute hands-on introduction
2. **[Decision Tree](./decision-tree/)** - Find the right workflow for your needs
3. **[Tutorials](../tutorials/index.md)** - Learn through practical examples

Welcome to the PrimateFace community!