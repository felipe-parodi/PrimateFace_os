# Changelog

All notable changes to PrimateFace will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-08-22

### Added
- **Core Framework**
  - Cross-species primate face detection using mmdetection
  - 68-point facial landmark estimation with mmpose
  - Support for multiple primate species (macaques, marmosets, lemurs, chimpanzees, etc.)
  - COCO format annotations for standardized data handling

- **Model Support**
  - Integration with MMDetection and MMPose frameworks
  - Ultralytics YOLO support for real-time inference
  - DeepLabCut and SLEAP integration for behavioral analysis
  - Pre-trained models optimized for primate faces

- **Tools & Features**
  - Interactive GUI for pseudo-labeling and annotation refinement
  - DINOv2 feature extraction for unsupervised analysis
  - Landmark converter for 68→49 point format conversion
  - Parallel GPU processing for large-scale video analysis
  - Temporal smoothing for stable video tracking

- **Documentation & Examples**
  - 6 interactive Jupyter notebook tutorials
  - Comprehensive API documentation
  - Installation guides for multiple platforms
  - Example scripts for common workflows

### Known Issues
- VLM genus classification requires >PyTorch 2.1.0 specifically
- High GPU memory usage for batch processing (recommend 8GB+ VRAM)

## Future Releases

### [0.2.0] - Planned Q4 2025
- Enhanced cross-species generalization
- Real-time video processing optimizations  
- Additional pre-trained models for rare species
- Docker container for easy deployment
- Extended dataset with additional annotations

### [0.3.0] - Planned Q1 2026
- Monocular 3D facial reconstruction from 2D landmarks
- Custom cross-species gaze tracking
- Automated pipelines for facial action unit recognition
- Integration with additional frameworks

## Contributing

See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for information on how to contribute to PrimateFace.

## Links

- [Documentation](https://primateface.studio)
- [Paper](https://www.biorxiv.org/content/10.1101/2025.08.12.669927v2)
- [Dataset](https://huggingface.co/datasets/fparodi/PrimateFace)
- [GitHub](https://github.com/PrimateFace/primateface_oss)