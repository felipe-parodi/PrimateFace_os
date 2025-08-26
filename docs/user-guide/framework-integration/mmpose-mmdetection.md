# MMPose & MMDetection Integration

Integration guide for using PrimateFace with MMPose and MMDetection frameworks.

## Overview

MMPose and MMDetection are the primary frameworks used by PrimateFace for production inference. This guide covers integration points and how to use PrimateFace models with these frameworks.

## Quick Start

```python
from demos.process import PrimateFaceProcessor

# Initialize with MMPose/MMDetection models
processor = PrimateFaceProcessor(
    det_config="demos/mmdet_config.py",
    det_checkpoint="demos/mmdet_checkpoint.pth",
    pose_config="demos/mmpose_config.py", 
    pose_checkpoint="demos/mmpose_checkpoint.pth"
)
```

## Integration Points

### Using PrimateFace Models in MMPose

1. **Model Loading**
   - Config files: `demos/mmdet_config.py`, `demos/mmpose_config.py`
   - Checkpoint files: Downloaded via `demos/download_models.py`

2. **Data Format**
   - Input: Standard image formats (JPEG, PNG)
   - Output: COCO-format JSON annotations

3. **Custom Training**
   - See MMPose documentation for training pipelines
   - Use PrimateFace COCO datasets directly

### Configuration

Key configuration parameters in `demos/mmdet_config.py`:
- Model architecture: Cascade R-CNN
- Backbone: ResNet-50
- Input size: 800x800

Key configuration parameters in `demos/mmpose_config.py`:
- Model architecture: HRNet-W32
- Keypoints: 68 facial landmarks
- Input size: 256x192

## Performance Optimization

- **Batch Processing**: Use `processor.process_batch()` for multiple images
- **GPU Acceleration**: Ensure CUDA is properly configured
- **Model Caching**: Models are cached after first load

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size
   - Use smaller input dimensions

2. **Model loading errors**
   - Ensure models are downloaded: `python demos/download_models.py`
   - Check config file paths

## See Also

- [MMPose Documentation](https://mmpose.readthedocs.io/)
- [MMDetection Documentation](https://mmdetection.readthedocs.io/)
- [Demos Core Workflow](../core-workflows/demos.md)
- [API Reference](../../api/demos.md)