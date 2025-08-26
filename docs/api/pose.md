# Pose Estimation API

Facial landmark detection and pose estimation interfaces.

## Core Classes

### Pose Estimator

Main pose estimation interface.

```python
from demos.process import PoseEstimator

estimator = PoseEstimator(
    config="configs/hrnet_w32.py",
    checkpoint="checkpoints/hrnet.pth",
    device="cuda:0"
)
```

#### Methods

##### estimate_keypoints()
```python
keypoints = estimator.estimate_keypoints(
    image=face_crop,
    return_heatmaps=False,
    return_confidence=True
)
```

##### batch_estimate()
```python
all_keypoints = estimator.batch_estimate(
    images=face_crops,
    batch_size=16
)
```

### MMPose Integration

```python
from mmpose.apis import init_pose_model, inference_top_down_pose_model

# Initialize model
pose_model = init_pose_model(
    config="configs/hrnet_w32_coco_256x192.py",
    checkpoint="checkpoints/hrnet_w32.pth"
)

# Run inference
results, _ = inference_top_down_pose_model(
    pose_model,
    image,
    person_results=bboxes,
    return_heatmap=True
)
```

## Keypoint Operations

### Keypoint Format

```python
# Standard keypoint format: [x, y, confidence]
keypoints = [
    [120.5, 85.2, 0.95],  # Point 0
    [125.1, 88.7, 0.92],  # Point 1
    # ... 68 points total
]

# Flatten format for COCO
flat_keypoints = []
for x, y, conf in keypoints:
    flat_keypoints.extend([x, y, 2 if conf > 0.5 else 0])
```

### Keypoint Utilities

```python
from demos.utils import (
    normalize_keypoints,
    denormalize_keypoints,
    calculate_nme,
    smooth_keypoints
)

# Normalize to [0, 1]
norm_kpts = normalize_keypoints(keypoints, image_size)

# Calculate error
nme = calculate_nme(predicted_kpts, ground_truth_kpts, face_size)

# Smooth trajectory
smoothed = smooth_keypoints(keypoint_sequence, window_size=5)
```

## Landmark Conversion

### Format Conversion

```python
from landmark_converter.apply_model import LandmarkConverter

converter = LandmarkConverter(
    model_path="converters/68to48.pth",
    input_format=68,
    output_format=48
)

# Convert landmarks
primate_landmarks = converter.convert(human_landmarks)
```

### Cross-Dataset Mapping

```python
from landmark_converter.utils import (
    map_dlib_to_coco,
    map_coco_to_openpose,
    map_mediapipe_to_coco
)

# Convert between formats
coco_landmarks = map_dlib_to_coco(dlib_landmarks)
openpose_landmarks = map_coco_to_openpose(coco_landmarks)
```

## Pose Configuration

### HRNet Configuration

```python
pose_config = dict(
    model=dict(
        type='TopDown',
        backbone=dict(
            type='HRNet',
            in_channels=3,
            extra=dict(
                stage2=dict(
                    num_modules=1,
                    num_branches=2,
                    num_channels=[48, 96]
                ),
                stage3=dict(
                    num_modules=4,
                    num_branches=3,
                    num_channels=[48, 96, 192]
                )
            )
        ),
        keypoint_head=dict(
            type='TopdownHeatmapSimpleHead',
            in_channels=48,
            out_channels=68,
            loss_keypoint=dict(type='JointsMSELoss')
        )
    )
)
```

## Heatmap Processing

### Heatmap Operations

```python
from mmpose.core import get_max_preds, transform_preds

# Extract keypoints from heatmaps
coords, confidence = get_max_preds(heatmaps)

# Transform to original image space
keypoints = transform_preds(
    coords,
    center=bbox_center,
    scale=bbox_scale,
    output_size=heatmap_size
)
```

### Visualization

```python
from demos.viz_utils import (
    draw_heatmaps,
    overlay_keypoints,
    create_skeleton
)

# Visualize heatmaps
heatmap_vis = draw_heatmaps(image, heatmaps, alpha=0.5)

# Draw keypoints
kpt_vis = overlay_keypoints(
    image,
    keypoints,
    connections=FACIAL_CONNECTIONS,
    point_size=3
)
```

## Multi-Person Pose

### Top-Down Approach

```python
# Detect faces first, then estimate pose
detections = detector.detect(image)
all_poses = []
for bbox in detections:
    pose = estimator.estimate(image, bbox)
    all_poses.append(pose)
```

### Bottom-Up Approach

```python
# Detect all keypoints, then group
from mmpose.apis import inference_bottom_up_pose_model

results = inference_bottom_up_pose_model(
    model,
    image,
    return_heatmap=True
)
```

## Performance Metrics

### Evaluation Metrics

```python
from evals.core.metrics import (
    calculate_pck,
    calculate_oks,
    calculate_auc
)

# Percentage of Correct Keypoints
pck = calculate_pck(
    predictions,
    ground_truth,
    threshold=0.2  # 20% of face size
)

# Object Keypoint Similarity
oks = calculate_oks(predictions, ground_truth)

# Area Under Curve
auc = calculate_auc(pck_curve)
```

## Error Handling

```python
from demos.exceptions import (
    NoFaceDetectedError,
    KeypointEstimationError,
    InvalidKeypointFormatError
)

try:
    keypoints = estimator.estimate(image)
except NoFaceDetectedError:
    print("No face found in image")
except KeypointEstimationError as e:
    print(f"Estimation failed: {e}")
```

## See Also

- [Detection API](./detection.md)
- [Landmark Converter](./converter.md)
- [Evaluation Metrics](./evaluation.md)
- [User Guide](../user-guide/core-workflows/demos.md)