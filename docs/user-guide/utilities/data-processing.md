# Data Processing Utilities

Tools for format conversion and data manipulation.

## Overview

PrimateFace provides utilities for converting between annotation formats, processing COCO datasets, and preparing data for training.

## Quick Start

```python
from gui.converters.image import ImageDirToCOCO

# Convert image directory to COCO format
converter = ImageDirToCOCO(
    image_dir="path/to/images",
    output_json="annotations.json"
)
converter.convert()
```

## Format Converters

### Image Directory to COCO
```python
# For face detection annotations
from gui.imgdir2coco_facedet import convert_to_coco

convert_to_coco(
    image_dir="images/",
    output_json="face_det.json",
    default_bbox_size=100
)
```

### Video Directory to COCO
```python
# Extract frames and create annotations
from gui.viddir2coco_facedet import video_to_coco

video_to_coco(
    video_dir="videos/",
    output_json="video_annotations.json",
    sample_rate=5  # frames per second
)
```

### COCO Utilities

#### Merge Datasets
```python
from evals.core.utils import merge_coco

# Combine multiple COCO files
merge_coco(
    json_files=["dataset1.json", "dataset2.json"],
    output="merged.json"
)
```

#### Split Dataset
```python
from evals.core.utils import split_coco

# Create train/val/test splits
split_coco(
    input_json="full_dataset.json",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)
```

#### Filter by Category
```python
from evals.core.utils import filter_coco

# Keep only specific species
filter_coco(
    input_json="all_species.json",
    output_json="macaques_only.json",
    categories=["macaque"]
)
```

## Landmark Format Conversion

### 68-point to 48-point
```python
from landmark_converter.apply_model import apply_converter

# Convert human to primate landmarks
converter = apply_converter(
    model_path="converter_model.pth",
    input_format=68,
    output_format=48
)
results = converter.convert(annotations)
```

## Data Validation

### Check Annotation Integrity
```python
from evals.core.utils import validate_coco

# Validate COCO format
issues = validate_coco("annotations.json")
if issues:
    print(f"Found {len(issues)} issues")
```

### Visualize Annotations
```python
from demos.visualize_coco_annotations import visualize

# Preview annotations
visualize(
    coco_json="annotations.json",
    image_dir="images/",
    num_samples=10
)
```

## Temporal Smoothing

For video data:
```python
from demos.smooth_utils import smooth_trajectory

# Smooth keypoint trajectories
smoothed = smooth_trajectory(
    keypoints=keypoints_sequence,
    window_size=5,
    method="savgol"
)
```

## Genus Classification

Add species labels:
```python
from demos.classify_genus import classify_images

# Classify genus using SmolVLM
results = classify_images(
    image_dir="images/",
    model="SmolVLM",
    output_json="genus_labels.json"
)
```

## Batch Processing

### Parallel Processing
```python
from evals.core.utils import parallel_process

# Process multiple files in parallel
parallel_process(
    files=image_list,
    function=process_image,
    num_workers=8
)
```

## See Also

- [COCO Format Guide](../../data-models/coco-format.md)
- [API Reference](../../api/index.md)
- [GUI Converters](../core-workflows/gui.md)