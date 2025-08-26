# API Reference

Comprehensive API documentation organized by task.

## Core APIs

### [Detection API](detection.md)
Face detection interfaces and utilities.
- `PrimateFaceProcessor`: Main detection pipeline
- MMDetection integration
- YOLO detection
- Batch processing utilities

### [Pose Estimation API](pose.md)
Facial landmark detection and pose estimation.
- `PoseEstimator`: Landmark detection
- MMPose integration
- Keypoint operations
- Heatmap processing

### [Annotation API](annotation.md)
Interactive annotation and pseudo-labeling.
- `PseudoLabelGUI`: Main annotation interface
- SAM integration
- Data converters
- Quality control

## Feature APIs

### [DINOv2 API](dinov2.md)
Feature extraction and visualization.
- `DINOv2Extractor`: Extract visual features
- `DiverseImageSelector`: Smart subset selection
- UMAP/t-SNE visualization
- Attention maps

### [Converter API](converter.md)
Cross-dataset landmark conversion.
- `LandmarkConverter`: Format conversion
- Neural network models
- Training pipelines
- Format mappings

### [Evaluation API](evaluation.md)
Model comparison and metrics.
- Performance metrics (NME, PCK, mAP)
- Cross-framework comparison
- Statistical analysis
- Visualization tools

## Module APIs

### [Demos Module](demos.md)
Complete detection and pose pipeline.
- End-to-end processing
- Video support
- Model management

### [GUI Module](gui.md)
Interactive tools and utilities.
- Annotation interfaces
- Refinement tools
- Batch operations

## Quick Import Reference

```python
# Detection and pose
from demos.process import PrimateFaceProcessor, PoseEstimator
from demos.utils import crop_face, expand_bbox

# Feature extraction
from dinov2.core import DINOv2Extractor
from dinov2.selection import DiverseImageSelector
from dinov2.visualization import plot_umap

# Landmark conversion  
from landmark_converter.apply_model import LandmarkConverter
from landmark_converter.train import train_converter

# Annotation
from gui.pseudolabel_gui_fm import PseudoLabelGUI
from gui.refine_boxes import COCORefinementGUI
from gui.core.sam import SAMAnnotator

# Evaluation
from evals.compare_pose_models import compare_models
from evals.core.metrics import calculate_nme, calculate_pck
from evals.visualize_eval_results import plot_comparison
```

## API Patterns

### Common Parameters

Most APIs follow these conventions:
- `device`: "cuda:0" or "cpu"
- `batch_size`: Processing batch size
- `num_workers`: Parallel workers
- `verbose`: Print progress

### Error Handling

All APIs use consistent exception types:
```python
try:
    result = api_call()
except ModelNotFoundError:
    # Handle missing model
except InvalidInputError:
    # Handle bad input
except ProcessingError:
    # Handle processing failure
```

## See Also

- [User Guide](../user-guide/index.md) - Workflow tutorials
- [Getting Started](../getting-started/index.md) - Quick start
- [Troubleshooting](../troubleshooting.md) - Common issues