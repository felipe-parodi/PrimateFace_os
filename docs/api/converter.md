# Landmark Converter API

The `landmark-converter` module provides machine learning models for converting between different facial landmark annotation schemes, enabling cross-dataset compatibility.

## Quick Reference

```python
from src.models import (
    SimplerLinearConverter,
    KeypointConverterMLP, 
    KeypointConverterMLPWithAttention,
    KeypointConverterGNN
)
from utils.data_utils import (
    normalize_keypoints_bbox,
    denormalize_keypoints_bbox,
    load_coco_keypoints
)
```

## Main Classes

### KeypointConverterMLP

General-purpose MLP for keypoint conversion.

```python
model = KeypointConverterMLP(
    num_source_kpts=68,
    num_target_kpts=49,
    hidden_dims=[256, 512],
    dropout=0.2
)

# Convert keypoints
source = torch.randn(10, 68, 2)  # [batch, source_kpts, 2]
converted = model(source.view(10, -1))  # [batch, target_kpts, 2]
```

**Architecture:**
- Two hidden layers with ReLU activation
- Dropout regularization
- ~85K parameters for 68→49 conversion

### KeypointConverterMLPWithAttention

Attention-based model for complex mappings.

```python
model = KeypointConverterMLPWithAttention(
    num_source_kpts=68,
    num_target_kpts=49,
    embed_dim=128,
    num_heads=4,
    hidden_dim=256
)

# Process with attention mechanism
converted = model(source_keypoints)
```

**Features:**
- Multi-head self-attention
- Positional embeddings
- Captures long-range dependencies
- ~51K parameters

### SimplerLinearConverter

Lightweight linear transformation.

```python
model = SimplerLinearConverter(
    num_source_kpts=68,
    num_target_kpts=49
)

# Fast inference
converted = model(source_keypoints)
```

**Use Cases:**
- Baseline model
- Real-time applications
- Linear relationships

## Training Pipeline

### Basic Training

```python
from src.training_pipeline import TrainingPipeline
from utils.data_utils import create_dataloaders

# Load data
train_loader, val_loader, test_loader = create_dataloaders(
    coco_json='annotations.json',
    batch_size=64,
    val_split=0.15
)

# Configure training
config = {
    'lr': 1e-3,
    'epochs': 500,
    'patience': 50,
    'device': 'cuda'
}

# Train model
pipeline = TrainingPipeline(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config
)

best_model = pipeline.train()
```

### Custom Training Loop

```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(epochs):
    for batch in train_loader:
        source, target = batch
        
        # Forward pass
        pred = model(source)
        loss = criterion(pred, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Data Utilities

### Loading COCO Annotations

```python
from utils.data_utils import load_coco_keypoints

# Load keypoints from COCO JSON
keypoints_dict = load_coco_keypoints(
    coco_json_path='annotations.json',
    source_field='keypoints',      # 68-point field
    target_field='keypoints_49'     # 49-point field
)
```

### Normalization

```python
from utils.data_utils import normalize_keypoints_bbox

# Normalize to bounding box
bbox = torch.tensor([[100, 100, 200, 200]])  # [x, y, w, h]
keypoints = torch.randn(1, 68, 2)

normalized = normalize_keypoints_bbox(keypoints, bbox)
# Now in range [0, 1] relative to bbox
```

### Data Splitting

```python
from utils.data_splitting_utils import DataSplitter

splitter = DataSplitter(strategy='pca_kmeans_stratified')

train_idx, val_idx, test_idx = splitter.split(
    keypoints_dict,
    val_ratio=0.15,
    test_ratio=0.15
)
```

**Strategies:**
- `random` - Random splitting
- `stratified` - Maintain distribution
- `pca_kmeans` - Cluster-based
- `pca_kmeans_stratified` - Best overall

## Common Usage Patterns

### Inference Pipeline

```python
# Load trained model
checkpoint = torch.load('best_model.pth')
model = KeypointConverterMLP(68, 49)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Process new data
with torch.no_grad():
    for image_data in test_data:
        keypoints = image_data['keypoints']
        bbox = image_data['bbox']
        
        # Normalize
        normalized = normalize_keypoints_bbox(keypoints, bbox)
        
        # Convert
        converted = model(normalized)
        
        # Denormalize
        result = denormalize_keypoints_bbox(converted, bbox)
```

### Batch Processing

```python
def batch_convert(model, coco_json, output_path):
    """Convert all annotations in COCO file."""
    
    data = load_coco_keypoints(coco_json)
    results = []
    
    model.eval()
    with torch.no_grad():
        for item in data:
            converted = model(item['keypoints'])
            results.append({
                'image_id': item['image_id'],
                'converted_keypoints': converted.cpu().numpy()
            })
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f)
```

### Model Ensemble

```python
# Load multiple models
models = [
    SimplerLinearConverter(68, 49),
    KeypointConverterMLP(68, 49),
    KeypointConverterMLPWithAttention(68, 49)
]

# Load weights
for i, model in enumerate(models):
    model.load_state_dict(torch.load(f'model_{i}.pth'))
    model.eval()

# Ensemble prediction
def ensemble_predict(models, keypoints):
    predictions = []
    for model in models:
        pred = model(keypoints)
        predictions.append(pred)
    
    # Average predictions
    return torch.stack(predictions).mean(dim=0)
```

## Evaluation Metrics

### Computing Metrics

```python
from utils.train_utils import compute_mpjpe, compute_pck

# Mean Per Joint Position Error
mpjpe = compute_mpjpe(predictions, targets)
print(f"MPJPE: {mpjpe:.2f} pixels")

# Percentage of Correct Keypoints
bbox_sizes = torch.tensor([[200, 200]])  # Width, height
pck_20 = compute_pck(predictions, targets, 0.2, bbox_sizes)
print(f"PCK@0.2: {pck_20:.2%}")
```

### Per-Keypoint Analysis

```python
def analyze_keypoint_errors(predictions, targets):
    """Analyze error per keypoint."""
    errors = torch.norm(predictions - targets, dim=-1)
    
    return {
        'mean_error': errors.mean(dim=0),
        'std_error': errors.std(dim=0),
        'max_error': errors.max(dim=0)[0],
        'worst_keypoint': errors.mean(dim=0).argmax()
    }
```

## Model Export

### ONNX Export

```python
import torch.onnx

dummy_input = torch.randn(1, 68*2)
torch.onnx.export(
    model,
    dummy_input,
    "converter.onnx",
    input_names=['keypoints'],
    output_names=['converted'],
    dynamic_axes={
        'keypoints': {0: 'batch'},
        'converted': {0: 'batch'}
    }
)
```

### TorchScript Export

```python
scripted = torch.jit.script(model)
scripted.save("converter.pt")

# Load and use
loaded = torch.jit.load("converter.pt")
output = loaded(input_keypoints)
```

## Configuration

### Training Configuration

```python
default_config = {
    'lr': 1e-3,
    'epochs': 500,
    'batch_size': 64,
    'patience': 50,
    'scheduler_patience': 20,
    'min_lr': 1e-6,
    'clip_grad': 1.0,
    'augment': True
}
```

### Model Selection Guide

| Use Case | Recommended Model | Config |
|----------|------------------|--------|
| Fast inference | SimplerLinearConverter | Default |
| General purpose | KeypointConverterMLP | 2 hidden layers |
| Complex mapping | MLPWithAttention | 4 attention heads |
| Structure preservation | KeypointConverterGNN | 5 neighbors |

## Detailed Documentation

For comprehensive technical documentation including:
- Full architecture details
- Custom conversion implementations
- Advanced training techniques
- Performance optimization
- Deployment strategies

See: **[landmark-converter/converter_docs.md](../../landmark-converter/converter_docs.md)**

## Related Modules

- [Demos](demos.md) - Use converted keypoints
- [Evaluation](evaluation.md) - Evaluate conversion quality
- [GUI](gui.md) - Visualize conversions

## Troubleshooting

### Common Issues

1. **Poor convergence**: Lower learning rate, try attention model
2. **Overfitting**: Add dropout, use data augmentation
3. **Memory issues**: Reduce batch size, use gradient accumulation
4. **Format errors**: Verify COCO JSON structure

### Getting Help

- Check [Technical Documentation](../../landmark-converter/converter_docs.md)
- See [Training Guide](../guides/landmark-training.md)
- Review examples in `test_landmark_converter.py`

## Next Steps

For practical workflows and step-by-step guides, see:
- [Landmark Training Guide](../user-guide/core-workflows/landmark-converter.md) - Complete training workflow
- [Pseudo-labeling Guide](../user-guide/core-workflows/pseudo-labeling.md) - Generate training data
- [MMPose Training](../user-guide/core-workflows/mmpose-training.md) - Use converted data
- [Evaluation Guide](../user-guide/core-workflows/evaluation-metrics.md) - Measure performance