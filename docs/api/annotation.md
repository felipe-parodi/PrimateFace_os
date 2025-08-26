# Annotation API

Interactive annotation and pseudo-labeling interfaces.

## GUI Components

### Main Annotation GUI

```python
from gui.pseudolabel_gui_fm import PseudoLabelGUI

gui = PseudoLabelGUI(
    img_dir="path/to/images",
    review_stage="det",  # or "pose"
    model_path="path/to/model.pth"
)
gui.run()
```

#### Configuration Options

```python
gui_config = {
    "window_size": (1200, 800),
    "point_size": 5,
    "line_width": 2,
    "colors": {
        "bbox": (0, 255, 0),
        "keypoints": (255, 0, 0),
        "skeleton": (0, 0, 255)
    },
    "shortcuts": {
        "next": "n",
        "previous": "p",
        "save": "s",
        "delete": "d"
    }
}
```

### Refinement GUI

```python
from gui.refine_boxes import COCORefinementGUI

refiner = COCORefinementGUI(
    image_dir="images/",
    coco_json="annotations.json",
    output_json="refined.json"
)
refiner.launch()
```

## Pseudo-Labeling

### Automatic Annotation

```python
from gui.pseudolabel import PseudoLabeler

labeler = PseudoLabeler(
    detector=detection_model,
    pose_estimator=pose_model,
    confidence_threshold=0.7
)

# Generate initial annotations
annotations = labeler.annotate_directory(
    image_dir="unlabeled_images/",
    save_low_confidence=True
)
```

### Human-in-the-Loop

```python
# TODO: Verify if HumanInTheLoop class exists in gui.core.models
# This may be conceptual - actual implementation is via pseudolabel_gui_fm.py

from gui.pseudolabel_gui_fm import PseudoLabelGUI

# The GUI provides human-in-the-loop functionality
gui = PseudoLabelGUI(
    img_dir="path/to/images",
    review_stage="det",  # or "pose"
    model_path="path/to/model.pth"
)
gui.run()
```

## SAM Integration

### Segment Anything Model

```python
from gui.core.sam import SAMAnnotator

sam = SAMAnnotator(
    model_type="vit_h",
    checkpoint="sam_vit_h.pth"
)

# Interactive segmentation
mask = sam.segment_with_click(
    image=image,
    click_point=(x, y),
    click_label=1  # 1 for foreground, 0 for background
)

# Box prompt
mask = sam.segment_with_box(
    image=image,
    box=[x1, y1, x2, y2]
)
```

## Data Converters

### Image Directory to COCO

```python
from gui.imgdir2coco_facedet import ImageDirConverter

converter = ImageDirConverter(
    image_dir="raw_images/",
    output_json="face_annotations.json",
    default_category="primate_face"
)

# Convert with auto-detection
coco_data = converter.convert(
    use_detector=True,
    detector_model=detection_model
)
```

### Video to COCO

```python
from gui.viddir2coco_facedet import VideoConverter

video_converter = VideoConverter(
    video_dir="videos/",
    output_json="video_annotations.json",
    sample_rate=5,  # Extract 5 fps
    detector=detection_model
)

# Process videos
annotations = video_converter.process_videos()
```

## Annotation Formats

### COCO Format Structure

```python
def create_coco_annotation(image_id, bbox, keypoints, category_id=1):
    """Create a COCO format annotation."""
    return {
        "id": generate_annotation_id(),
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,  # [x, y, width, height]
        "area": bbox[2] * bbox[3],
        "keypoints": keypoints,  # [x1,y1,v1, x2,y2,v2, ...]
        "num_keypoints": sum(1 for i in range(2, len(keypoints), 3) if keypoints[i] > 0),
        "iscrowd": 0
    }
```

### Validation

```python
from gui.utils import validate_annotations

# Check annotation consistency
errors = validate_annotations(
    coco_json="annotations.json",
    check_images=True,
    check_keypoints=True
)

if errors:
    for error in errors:
        print(f"Error: {error}")
```

## Keyboard Shortcuts

### Default Bindings

```python
KEYBOARD_SHORTCUTS = {
    "navigation": {
        "next_image": "Right",
        "prev_image": "Left",
        "first_image": "Home",
        "last_image": "End"
    },
    "annotation": {
        "add_point": "a",
        "delete_point": "d",
        "clear_all": "c",
        "toggle_bbox": "b"
    },
    "file": {
        "save": "Ctrl+s",
        "save_as": "Ctrl+Shift+s",
        "load": "Ctrl+o",
        "export": "Ctrl+e"
    },
    "view": {
        "zoom_in": "+",
        "zoom_out": "-",
        "fit_window": "f",
        "toggle_skeleton": "s"
    }
}
```

## Batch Operations

### Batch Annotation

```python
# TODO: Verify if BatchAnnotator exists in gui.batch
# Batch processing may be done via CLI tools instead:

# Use primateface_demo.py for batch processing:
import subprocess

subprocess.run([
    "python", "demos/primateface_demo.py", "process",
    "--input", "./images/",
    "--input-type", "images",
    "--det-config", "demos/mmdet_config.py",
    "--det-checkpoint", "demos/mmdet_checkpoint.pth",
    "--output-dir", "./annotations/",
    "--save-predictions"
])
```

### Progress Tracking

```python
from gui.utils import ProgressTracker

tracker = ProgressTracker(total_items=len(images))

for i, image in enumerate(images):
    # Process image
    result = process(image)
    
    # Update progress
    tracker.update(
        current=i+1,
        message=f"Processing {image}"
    )
    
    # Get statistics
    stats = tracker.get_stats()
    print(f"Speed: {stats['items_per_second']:.1f} img/s")
```

## Quality Control

### Annotation Quality Metrics

```python
from gui.quality import QualityChecker

checker = QualityChecker()

# Check annotation quality
quality_report = checker.check_annotations(
    annotations=coco_data,
    checks=[
        "bbox_size",
        "keypoint_visibility",
        "keypoint_consistency",
        "duplicate_detection"
    ]
)

print(f"Quality score: {quality_report['overall_score']:.2f}")
```

## Export Options

### Export Formats

```python
from gui.export import AnnotationExporter

exporter = AnnotationExporter(coco_data)

# Export to different formats
exporter.to_yolo("yolo_format/")
exporter.to_pascal_voc("voc_format/")
exporter.to_csv("annotations.csv")
exporter.to_json_lines("annotations.jsonl")
```

## See Also

- [GUI Module](./gui.md)
- [Detection API](./detection.md)
- [User Guide](../user-guide/core-workflows/gui.md)
- [Data Processing](../user-guide/utilities/data-processing.md)