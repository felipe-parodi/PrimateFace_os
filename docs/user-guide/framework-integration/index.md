# Framework Integration

PrimateFace provides integration with popular computer vision and pose estimation frameworks.

## Overview

These guides help you use PrimateFace with external frameworks:

- **MMDetection/MMPose** - Primary framework for detection and pose estimation
- **DeepLabCut** - Alternative pose estimation with COCO training support
- **SLEAP** - Multi-animal tracking with COCO training support
- **YOLO** - Real-time detection and integration examples

## Integration Approach

PrimateFace doesn't replace these frameworks - it provides:

1. **Training Scripts** - Convert COCO data to framework-specific formats
2. **Model Compatibility** - Use framework models with PrimateFace pipelines
3. **Evaluation Tools** - Compare performance across frameworks
4. **Workflow Integration** - Seamless integration with existing workflows

## When to Use Each Framework

- **MMDetection/MMPose** - Production inference, best performance
- **DeepLabCut** - Markerless tracking, behavioral analysis
- **SLEAP** - Multi-animal scenarios, complex tracking
- **YOLO** - Real-time applications, edge deployment

## Getting Started

Each framework has specific setup requirements. See the individual guides for:

- Installation and setup
- Training from COCO data
- Integration with PrimateFace
- Performance optimization

For detailed API documentation, see the [API Reference](../api/index.md).
