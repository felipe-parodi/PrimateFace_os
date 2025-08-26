# Core Workflows

Core workflows are the primary tools researchers use to accomplish end-to-end tasks in primate face analysis.

## Overview

These modules provide complete, integrated workflows for common research tasks:

- **Demos** - Complete inference pipeline for face detection and pose estimation
- **DINOv2** - Feature extraction, visualization, and intelligent subset selection
- **GUI** - Interactive annotation and pseudo-labeling workflow
- **Landmark Converter** - Training and applying format conversion models

## Workflow Integration

These modules are designed to work together:

1. **Data Collection** → Use GUI for annotation
2. **Feature Analysis** → Use DINOv2 for exploration
3. **Model Inference** → Use Demos for detection/pose
4. **Format Conversion** → Use Landmark Converter for compatibility

## Getting Started

Each module has its own quick start guide. Start with the module that matches your immediate need:

- **New to PrimateFace?** → Start with [Demos](demos.md)
- **Need to analyze image features?** → Use [DINOv2](dinov2.md)
- **Have unlabeled data?** → Try [GUI](gui.md)
- **Need format compatibility?** → Check [Landmark Converter](landmark-converter.md)

For detailed API documentation, see the [API Reference](../api/index.md).
