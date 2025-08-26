# Evaluation API

The `evals` module provides comprehensive tools for evaluating pose estimation models across multiple frameworks, calculating standard metrics, and generating comparison reports.

## Quick Reference

```python
from evals import (
    NMECalculator,
    PCKCalculator,
    OKSCalculator,
    ModelManager,
    ModelConfig,
    CrossFrameworkEvaluator,
    EvalVisualizer
)
```

## Main Classes

### NMECalculator

Calculate Normalized Mean Error with various normalization strategies.

```python
calculator = NMECalculator(
    normalize_by='bbox',  # 'bbox', 'interocular', or 'head_size'
    use_visible_only=True
)

# Calculate NME
result = calculator.calculate(
    predictions,  # Shape: (N, K, 2) or (N, K, 3)
    ground_truth,  # Shape: (N, K, 2) or (N, K, 3)
    metadata={'bbox': [x, y, w, h]}
)

print(f"NME: {result['nme']:.3f}")
print(f"Per-keypoint NME: {result['per_keypoint_nme']}")
```

### PCKCalculator

Calculate Percentage of Correct Keypoints at multiple thresholds.

```python
calculator = PCKCalculator(
    thresholds=[0.05, 0.1, 0.2],  # Multiple thresholds
    normalize_by='bbox'
)

result = calculator.calculate(predictions, ground_truth, metadata)

# Access PCK at different thresholds
print(f"PCK@0.1: {result['pck'][0.1]:.2%}")
print(f"AUC: {result['auc']:.3f}")
```

### OKSCalculator

COCO-style Object Keypoint Similarity metric.

```python
calculator = OKSCalculator(
    sigmas=None,  # Use default or provide custom
    use_area=True
)

result = calculator.calculate(
    predictions,
    ground_truth,
    areas  # Object areas for normalization
)

print(f"Mean OKS: {result['mean_oks']:.3f}")
```

### ModelManager

Unified interface for loading models from different frameworks.

```python
manager = ModelManager()

# Load model from config
config = ModelConfig(
    name='pf_hrnet',
    framework='mmpose',
    config_path='configs/hrnet.py',
    checkpoint_path='models/hrnet.pth',
    device='cuda:0'
)

model = manager.load_model(config)

# Run inference
results = manager.run_inference(
    model,
    test_images,
    framework='mmpose'
)
```

### CrossFrameworkEvaluator

Compare models across different frameworks.

```python
evaluator = CrossFrameworkEvaluator(
    test_data='test_annotations.json',
    output_dir='evaluation_results/'
)

# Add models for comparison
evaluator.add_model('configs/mmpose_model.yaml')
evaluator.add_model('configs/dlc_model.yaml')
evaluator.add_model('configs/sleap_model.yaml')

# Evaluate all models
evaluator.evaluate_all(metrics=['nme', 'pck', 'oks'])

# Generate comparison report
df = evaluator.generate_comparison_report()
evaluator.plot_comparison(metric='nme')
```

## Common Usage Patterns

### Basic Model Evaluation

```python
from evals import NMECalculator, PCKCalculator

# Load predictions and ground truth
predictions = load_predictions('predictions.json')
ground_truth = load_annotations('annotations.json')

# Calculate multiple metrics
nme_calc = NMECalculator(normalize_by='bbox')
nme = nme_calc.calculate(predictions, ground_truth, metadata)

pck_calc = PCKCalculator(thresholds=[0.1, 0.2])
pck = pck_calc.calculate(predictions, ground_truth, metadata)

print(f"NME: {nme['nme']:.3f}")
print(f"PCK@0.2: {pck['pck'][0.2]:.2%}")
```

### Per-Genus Evaluation

```python
from evals import GenusEvaluator

evaluator = GenusEvaluator(
    predictions_path='predictions.json',
    annotations_path='annotations.json'
)

# Evaluate per genus
evaluator.evaluate_per_genus(metrics=['nme', 'pck'])

# Generate report
df = evaluator.generate_report()
print(df.to_string())

# Plot comparison
fig = evaluator.plot_genus_comparison(metric='nme')
```

### Visualization

```python
from evals import EvalVisualizer

viz = EvalVisualizer()

# Plot training curves
history = {
    'loss': [0.5, 0.4, 0.3],
    'val_loss': [0.6, 0.5, 0.4],
    'nme': [0.1, 0.08, 0.06]
}
viz.plot_training_curves(history, metrics=['loss', 'nme'])

# Visualize predictions
viz.plot_predictions(
    images=test_images,
    predictions=model_predictions,
    ground_truth=annotations,
    max_images=9
)

# Error distribution
viz.plot_error_distribution(
    errors,
    keypoint_names=KEYPOINT_NAMES
)
```

## CLI Scripts

### Detection Model Comparison

```bash
python compare_det_models.py \
    --coco test_annotations.json \
    --model-config cascade_rcnn.py \
    --model-checkpoint cascade_rcnn.pth \
    --output detection_eval.json
```

### Pose Model Comparison

```bash
python compare_pose_models.py \
    --coco test_annotations.json \
    --pose-config hrnet.py \
    --pose-checkpoint hrnet.pth \
    --output pose_eval.json
```

### Genus-Specific Evaluation

```bash
python eval_genera.py \
    --predictions predictions.json \
    --annotations annotations.json \
    --output genus_metrics.json
```

## Metrics Overview

| Metric | Description | Range | Lower is Better |
|--------|-------------|-------|-----------------|
| NME | Normalized Mean Error | 0-∞ | ✓ |
| PCK | Percentage Correct Keypoints | 0-1 | ✗ |
| OKS | Object Keypoint Similarity | 0-1 | ✗ |
| mAP | Mean Average Precision | 0-1 | ✗ |

## Framework Support

| Framework | Detection | Pose | Training | Evaluation |
|-----------|-----------|------|----------|------------|
| MMPose | ✓ | ✓ | ✓ | ✓ |
| DeepLabCut | ✗ | ✓ | ✓ | ✓ |
| SLEAP | ✗ | ✓ | ✓ | ✓ |
| YOLO | ✓ | ✗ | ✓ | ✓ |

## Configuration

### Model Configuration

```yaml
# model_config.yaml
name: pf_hrnet_68kpt
framework: mmpose
config_path: configs/hrnet_w32_primateface.py
checkpoint_path: checkpoints/pf_hrnet_best.pth
device: cuda:0
additional_params:
  bbox_thr: 0.3
  kpt_thr: 0.3
```

### Load and Use

```python
from evals import ModelConfig

# Load from file
config = ModelConfig.from_file('model_config.yaml')

# Or create programmatically
config = ModelConfig(
    name='my_model',
    framework='mmpose',
    config_path='config.py',
    checkpoint_path='model.pth'
)
```

## Advanced Features

### Temporal Consistency

```python
from evals import TemporalConsistencyEvaluator

evaluator = TemporalConsistencyEvaluator()
result = evaluator.evaluate_video(
    predictions,  # Shape: (T, K, 2)
    fps=30.0
)

print(f"Jitter: {result['mean_jitter']:.2f}")
print(f"Smoothness: {result['smooth_score']:.2f}")
print(f"Problematic frames: {result['problematic_frames']}")
```

### Multi-Scale Evaluation

```python
from evals import MultiScaleEvaluator

evaluator = MultiScaleEvaluator(scales=[0.5, 1.0, 1.5])
results = evaluator.evaluate(
    model,
    test_images,
    ground_truth
)

for scale, metrics in results.items():
    print(f"Scale {scale}: NME={metrics['nme']:.3f}")
```

## Detailed Documentation

For comprehensive technical documentation including:
- Full metric implementations
- Cross-framework comparison details
- Advanced evaluation pipelines
- Performance optimization
- Testing strategies

See: **[evals/eval_docs.md](../../evals/eval_docs.md)**

## Related Modules

- [Demos](demos.md) - Run inference with models
- [GUI](gui.md) - Visualize evaluation results
- [Converter](converter.md) - Convert between formats

## Best Practices

1. **Use configuration files** instead of hardcoded paths
2. **Store results** in standardized formats (COCO JSON)
3. **Batch processing** for efficient evaluation
4. **Multiple metrics** for comprehensive assessment
5. **Cross-validation** for robust results

## Troubleshooting

### Common Issues

1. **Framework conflicts**: Use separate environments
2. **Memory errors**: Reduce batch size
3. **Metric discrepancies**: Check normalization methods
4. **Path errors**: Use absolute paths in configs

### Getting Help

- Check [Technical Documentation](../../evals/eval_docs.md)
- See [Framework Training Guide](../guides/framework-training.md)
- Review examples in `test_evals.py`

## Next Steps

- Compare models using [CrossFrameworkEvaluator](#crossframeworkevaluator)
- Visualize results with [EvalVisualizer](#evalvisualizer)
- Train new models following [framework guides](../guides/framework-training.md)
- Analyze per-genus performance with [GenusEvaluator](#per-genus-evaluation)