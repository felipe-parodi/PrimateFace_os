# Landmark Converter - Technical Documentation

This document provides comprehensive technical documentation for the landmark converter module, focusing on internal architecture and implementation details.

For practical workflows and API reference, see:
- [Landmark Converter API](../docs/api/converter.md) - Complete API documentation
- [Training Guide](../docs/user-guide/core-workflows/landmark-converter.md) - Step-by-step workflows
- [Interactive Tutorials](../docs/tutorials/index.md) - Practical applications

## Internal Architecture

### Module Organization
```
landmark-converter/
├── train.py                # Training entrypoint
├── apply_model.py          # Inference entrypoint
├── src/
│   ├── models.py          # Model architectures
│   └── training_pipeline.py
├── utils/
│   ├── data_utils.py      # Data processing
│   └── metrics.py         # Evaluation metrics
└── constants.py           # Configuration
```

### Core Components

#### Model Registry
```python
MODEL_CONFIGS: Dict[str, Dict] = {
    'simple_linear': {
        'class_name': 'SimpleLinearConverter',
        'params': {}
    },
    'mlp': {
        'class_name': 'KeypointConverterMLP',
        'params': {
            'hidden_dim1': DEFAULT_HIDDEN_DIM1,
            'hidden_dim2': DEFAULT_HIDDEN_DIM2
        }
    },
    'mlp_attention': {
        'class_name': 'KeypointConverterMLPWithAttention',
        'params': {
            'embed_dim': DEFAULT_EMBED_DIM,
            'num_heads': DEFAULT_NUM_HEADS
        }
    }
}
```

#### Model Factory
```python
def create_model(model_name: str, **kwargs) -> nn.Module:
    """Create model instance from registry."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")
    
    config = MODEL_CONFIGS[model_name]
    cls = getattr(models, config['class_name'])
    params = {**config['params'], **kwargs}
    return cls(**params)
```

### Model Architectures

#### SimpleLinearConverter
- Direct linear transformation
- Single layer: `Linear(source_dim, target_dim)`
- No activation functions
- ~7K parameters for 68→48 conversion

#### KeypointConverterMLP
- Two-layer perceptron with dropout
- Architecture:
  ```
  Linear(source_dim, hidden_dim1)
  ReLU()
  Dropout(0.2)
  Linear(hidden_dim1, hidden_dim2)
  ReLU()
  Linear(hidden_dim2, target_dim)
  ```
- ~85K parameters for 68→48 conversion

#### KeypointConverterMLPWithAttention
- Self-attention enhanced MLP
- Architecture:
  ```
  Embedding(source_dim, embed_dim)
  MultiHeadAttention(embed_dim, num_heads)
  MLP(embed_dim, hidden_dim)
  Linear(hidden_dim, target_dim)
  ```
- ~51K parameters for 68→48 conversion

### Data Processing Pipeline

#### Keypoint Normalization
```python
def normalize_keypoints_bbox(
    keypoints: torch.Tensor,  # [N, K, 2]
    bbox: torch.Tensor,      # [N, 4]
) -> torch.Tensor:          # [N, K, 2]
    """Normalize keypoints to [0,1] range within bbox."""
    x1, y1, w, h = bbox.unbind(-1)
    normalized = (keypoints - torch.stack([x1,y1], -1).unsqueeze(1)) / \
                 torch.stack([w,h], -1).unsqueeze(1)
    return normalized
```

#### COCO Format Processing
```python
def process_coco_keypoints(
    keypoints: List[float],  # [x1,y1,v1,...,xN,yN,vN]
    num_keypoints: int
) -> torch.Tensor:         # [N, 2]
    """Extract (x,y) coordinates from COCO format."""
    points = torch.tensor(keypoints).reshape(-1, 3)
    return points[:num_keypoints, :2]
```

### Memory Management

#### Batch Processing
```python
def process_in_chunks(
    dataset: Dataset,
    model: nn.Module,
    chunk_size: int = 1000
) -> List[torch.Tensor]:
    """Process large datasets in memory-efficient chunks."""
    results = []
    for i in range(0, len(dataset), chunk_size):
        chunk = dataset[i:i+chunk_size]
        with torch.no_grad():
            output = model(chunk)
        results.append(output.cpu())
        torch.cuda.empty_cache()
    return torch.cat(results)
```

### Performance Optimization

#### Model Inference
- Use `torch.no_grad()` for inference
- Process in batches (32-64 samples)
- Keep models in eval mode
- Use half precision when possible

#### Data Loading
- Use `num_workers` for parallel loading
- Pin memory for GPU transfers
- Prefetch next batch during processing

### Error Handling

#### Input Validation
```python
def validate_keypoints(
    keypoints: torch.Tensor,
    expected_points: int
) -> None:
    """Validate keypoint tensor shape and values."""
    if keypoints.ndim != 3:
        raise ValueError(f"Expected 3D tensor [N,K,2], got shape {keypoints.shape}")
    if keypoints.shape[1] != expected_points:
        raise ValueError(f"Expected {expected_points} keypoints, got {keypoints.shape[1]}")
    if not torch.isfinite(keypoints).all():
        raise ValueError("Keypoints contain inf/nan values")
```

#### Model Validation
```python
def validate_model_output(
    output: torch.Tensor,
    batch_size: int,
    num_target_points: int
) -> None:
    """Validate model output shape and values."""
    expected_shape = (batch_size, num_target_points, 2)
    if output.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {output.shape}")
    if not torch.isfinite(output).all():
        raise ValueError("Model output contains inf/nan values")
```

### Testing Strategy

#### Unit Tests
- Model architecture tests
- Data processing tests
- Conversion accuracy tests
- Error handling tests

#### Integration Tests
- End-to-end conversion tests
- COCO format compatibility
- Memory management tests
- Performance benchmarks

### Deployment Considerations

#### Model Export
- ONNX format support
- TorchScript compilation
- CPU/GPU compatibility
- Batch size flexibility

#### Production Usage
- Error recovery strategies
- Logging and monitoring
- Performance profiling
- Resource management