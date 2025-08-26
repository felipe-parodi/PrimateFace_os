# Evaluation Utilities

Tools for model comparison and performance metrics.

## Overview

PrimateFace provides comprehensive evaluation utilities for comparing models across frameworks and computing standard metrics.

## Quick Start

```python
from evals.compare_pose_models import compare_models

# Compare multiple models
results = compare_models(
    models=["mmpose", "deeplabcut", "sleap"],
    test_data="test_annotations.json"
)
```

## Available Metrics

### Detection Metrics
- **mAP**: Mean Average Precision at various IoU thresholds
- **Precision/Recall**: Detection accuracy metrics
- **F1 Score**: Harmonic mean of precision and recall

### Pose Estimation Metrics
- **NME**: Normalized Mean Error (percentage of face size)
- **PCK**: Percentage of Correct Keypoints
- **OKS**: Object Keypoint Similarity
- **AUC**: Area Under Curve for PCK

## Evaluation Scripts

### Compare Detection Models
```bash
python evals/compare_det_models.py \
  --models mmdet yolo \
  --test-json test.json \
  --output results.csv
```

### Compare Pose Models
```bash
python evals/compare_pose_models.py \
  --models mmpose dlc sleap \
  --test-json test.json \
  --metric nme
```

### Per-Genus Evaluation
```bash
python evals/eval_genera.py \
  --model-path model.pth \
  --test-json test.json \
  --by-genus
```

## Visualization

### Performance Plots
```python
from evals.visualize_eval_results import plot_metrics

# Generate comparison plots
plot_metrics(
    results_file="evaluation_results.csv",
    output_dir="plots/"
)
```

### Attention Maps
```python
from dinov2.visualization import plot_attention

# Visualize model attention
plot_attention(
    model=model,
    image="primate.jpg",
    save_path="attention.png"
)
```

## Cross-Dataset Evaluation

Test generalization across datasets:
```python
from evals.core.metrics import cross_dataset_eval

results = cross_dataset_eval(
    model=model,
    datasets=["primateface", "macaquepose", "chimpface"],
    metric="nme"
)
```

## Statistical Analysis

### Significance Testing
```python
from evals.core.metrics import compare_significance

# Test if model A is significantly better than B
p_value = compare_significance(
    model_a_results=results_a,
    model_b_results=results_b,
    test="wilcoxon"
)
```

## Export Results

Results can be exported in multiple formats:
```python
# CSV for analysis
results.to_csv("evaluation.csv")

# JSON for web visualization
results.to_json("evaluation.json")

# LaTeX for papers
results.to_latex("evaluation.tex")
```

## See Also

- [API Reference](../../api/evaluation.md)
- [Model Comparison Guide](../../guides/model-comparison.md)
- [Visualization Tools](./visualization.md)