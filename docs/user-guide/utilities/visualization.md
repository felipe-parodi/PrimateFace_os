# Visualization Utilities

Tools for plotting, attention maps, and visual analysis.

## Overview

PrimateFace provides visualization utilities for model interpretation, result analysis, and figure generation.

## Quick Start

```python
from dinov2.visualization import create_visualization

# Generate feature visualization
viz = create_visualization(
    images=["primate1.jpg", "primate2.jpg"],
    features=extracted_features,
    method="umap"
)
```

## Feature Visualization

### UMAP Plots
```python
from dinov2.visualization import plot_umap

# Visualize feature space
plot_umap(
    embeddings=dinov2_features,
    labels=species_labels,
    save_path="umap_plot.png",
    perplexity=30
)
```

### t-SNE Plots
```python
from dinov2.visualization import plot_tsne

# Alternative dimensionality reduction
plot_tsne(
    embeddings=features,
    labels=labels,
    n_components=2
)
```

## Attention Maps

### DINOv2 Attention
```python
from dinov2.visualization import visualize_attention

# Show what the model focuses on
attention_map = visualize_attention(
    model=dinov2_model,
    image="primate_face.jpg",
    layer=-1,  # Last layer
    head=0     # Attention head
)
```

### Pose Model Heatmaps
```python
from evals.core.visualization import plot_heatmaps

# Visualize keypoint confidence
plot_heatmaps(
    image="primate.jpg",
    heatmaps=model_output.heatmaps,
    keypoint_names=KEYPOINT_NAMES
)
```

## Result Visualization

### Detection Results
```python
from demos.viz_utils import draw_boxes

# Draw bounding boxes
result_img = draw_boxes(
    image=img,
    boxes=detections,
    scores=confidence_scores,
    threshold=0.5
)
```

### Pose Results
```python
from demos.viz_utils import draw_keypoints

# Draw facial landmarks
result_img = draw_keypoints(
    image=img,
    keypoints=landmarks,
    connections=FACIAL_CONNECTIONS,
    point_size=3
)
```

## Performance Plots

### Model Comparison
```python
from evals.visualize_eval_results import plot_comparison

# Compare multiple models
plot_comparison(
    results={
        "MMPose": mmpose_results,
        "DeepLabCut": dlc_results,
        "SLEAP": sleap_results
    },
    metric="nme",
    save_path="comparison.png"
)
```

### Learning Curves
```python
from evals.core.visualization import plot_learning_curve

# Training progress
plot_learning_curve(
    train_losses=train_loss_history,
    val_losses=val_loss_history,
    save_path="learning_curve.png"
)
```

## Distribution Plots

### Genus Distribution
```python
from evals.plot_genus_distribution import plot_distribution

# Dataset composition
plot_distribution(
    coco_json="annotations.json",
    save_path="genus_distribution.png"
)
```

### Error Distribution
```python
from evals.core.visualization import plot_error_distribution

# Analyze error patterns
plot_error_distribution(
    errors=nme_scores,
    bins=50,
    save_path="error_dist.png"
)
```

## Interactive Visualization

### Gradio Interface
```python
from docs.gradio.primateface_server import launch_demo

# Launch web interface
launch_demo(
    model_path="model.pth",
    share=True  # Get public URL
)
```

## Video Visualization

### Trajectory Overlay
```python
from demos.viz_utils import visualize_trajectory

# Show movement over time
visualize_trajectory(
    video_path="primate_video.mp4",
    keypoints_sequence=tracked_keypoints,
    output_path="trajectory_video.mp4"
)
```

## Export Options

### High-Quality Figures
```python
import matplotlib.pyplot as plt

# Publication-ready figures
plt.figure(dpi=300, figsize=(10, 6))
# ... create plot ...
plt.savefig("figure.pdf", bbox_inches='tight')
```

### Animation Export
```python
from matplotlib import animation

# Create animated visualization
anim = animation.FuncAnimation(fig, update_func, frames=100)
anim.save("animation.gif", writer='pillow', fps=30)
```

## See Also

- [DINOv2 Visualization](../core-workflows/dinov2.md)
- [Evaluation Metrics](./evaluation.md)
- [API Reference](../../api/index.md)