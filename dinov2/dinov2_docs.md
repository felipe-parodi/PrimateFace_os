## Moved: DINOv2 Technical Documentation

The detailed technical documentation for this module now lives in the central docs site.

- Internals and deep dive: ../docs/api/dinov2.md#internals
- User guide: ../docs/user-guide/core-workflows/dinov2.md

This file remains as a pointer to avoid broken links.

## Module Architecture

```
dinov2/
├── dinov2_cli.py       # CLI entry point with subcommands
├── core.py             # Feature extraction engine
├── visualization.py    # UMAP and patch visualization
├── selection.py        # Active learning subset selection
├── constants.py        # Configuration constants
├── utils.py           # Helper utilities
└── test_dinov2.py    # Unit tests
```

## Core API Reference

### `core.py` - Feature Extraction Engine

#### Class: `DINOv2Extractor`

Main interface for extracting DINOv2 features from images.

```python
class DINOv2Extractor:
    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32,
        num_workers: int = 4
    )
```

**Parameters:**
- `model_name`: HuggingFace model identifier
  - `facebook/dinov2-small`: 21M params, 384-dim features
  - `facebook/dinov2-base`: 86M params, 768-dim features (recommended)
  - `facebook/dinov2-large`: 300M params, 1024-dim features
  - `facebook/dinov2-giant`: 1.1B params, 1536-dim features
- `device`: CUDA device or 'cpu'
- `batch_size`: Number of images per batch
- `num_workers`: Parallel data loading threads

**Methods:**

##### `extract_from_directory(directory: str, pattern: str = "*.jpg") -> Tuple[torch.Tensor, List[str]]`
Extract features from all images in a directory.

**Returns:**
```python
(
    embeddings,  # torch.Tensor of shape (N, feature_dim)
    image_ids    # List of image filenames
)
```

##### `extract_from_csv(csv_path: str, image_col: str = "image_path") -> Tuple[torch.Tensor, List[str]]`
Extract features from images listed in CSV file.

##### `extract_from_list(image_paths: List[str]) -> Tuple[torch.Tensor, List[str]]`
Extract features from a list of image paths.

##### `save_embeddings(embeddings: torch.Tensor, image_ids: List[str], output_path: str)`
Save embeddings with metadata to PyTorch file.

#### Class: `ImageDataset`

PyTorch dataset for loading images with various input formats.

```python
class ImageDataset(Dataset):
    def __init__(
        self,
        image_source: Union[str, List[str], pd.DataFrame],
        transform: Optional[Callable] = None
    )
```

### `visualization.py` - Visualization Tools

#### Class: `UMAPVisualizer`

UMAP dimensionality reduction and visualization.

```python
class UMAPVisualizer:
    def __init__(
        self,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        n_components: int = 2,
        metric: str = "cosine",
        random_state: int = 42
    )
```

**Methods:**

##### `fit_transform(embeddings: np.ndarray) -> np.ndarray`
Compute UMAP projection of embeddings.

##### `cluster(n_clusters: int = 100, method: str = "kmeans") -> np.ndarray`
Cluster embeddings for visualization.

##### `plot_static(output_path: str, dot_size: int = 50, dpi: int = 150, figsize: Tuple[int, int] = (12, 10))`
Generate static matplotlib visualization.

##### `plot_interactive(output_path: str, image_paths: Optional[List[str]] = None, show_images: bool = False)`
Generate interactive HTML visualization with plotly.

#### Class: `PatchVisualizer`

Visualize attention maps and patch features.

```python
class PatchVisualizer:
    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        device: str = "cuda"
    )
    
    def visualize_attention(
        self,
        image_path: str,
        output_path: str,
        layer: int = -1,
        head: Optional[int] = None
    )
```

**Features:**
- Extract attention weights from any layer
- Visualize individual attention heads
- Generate heatmap overlays
- Save patch-level features

### `selection.py` - Active Learning Selection

#### Class: `DiverseImageSelector`

Select diverse subsets using various strategies.

```python
class DiverseImageSelector:
    def __init__(
        self,
        strategy: str = "hybrid",
        random_state: int = 42
    )
```

**Strategies:**

##### Random Selection
```python
selector = DiverseImageSelector(strategy="random")
indices = selector.select(embeddings, n_samples=1000)
```

##### Cluster-Based Selection
```python
selector = DiverseImageSelector(strategy="cluster")
indices = selector.select(
    embeddings, 
    n_samples=1000,
    n_clusters=100
)
```
- Uses K-means clustering
- Proportional sampling from each cluster
- Ensures coverage of embedding space

##### Farthest Point Sampling (FPS)
```python
selector = DiverseImageSelector(strategy="fps")
indices = selector.select(embeddings, n_samples=1000)
```
- Iteratively selects most distant points
- Maximum diversity but computationally expensive
- O(n²) complexity

##### Hybrid Strategy (Recommended)
```python
selector = DiverseImageSelector(strategy="hybrid")
indices = selector.select(
    embeddings,
    n_samples=1000,
    n_clusters=100,
    fps_ratio=0.5
)
```
- Combines clustering with FPS
- Balances diversity and efficiency
- Applies FPS within each cluster

### `constants.py` - Configuration

Central configuration for DINOv2 module:

```python
# Model configurations
DINOV2_MODELS = {
    "small": {
        "name": "facebook/dinov2-small",
        "dim": 384,
        "patch_size": 14
    },
    "base": {
        "name": "facebook/dinov2-base", 
        "dim": 768,
        "patch_size": 14
    },
    "large": {
        "name": "facebook/dinov2-large",
        "dim": 1024,
        "patch_size": 14
    },
    "giant": {
        "name": "facebook/dinov2-giant",
        "dim": 1536,
        "patch_size": 14
    }
}

# Default parameters
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 4
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# UMAP parameters
UMAP_DEFAULTS = {
    "n_neighbors": 15,
    "min_dist": 0.1,
    "metric": "cosine"
}

# Image processing
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
IMAGE_SIZE = 224  # DINOv2 input size
```

## Advanced Usage Patterns

### Custom Feature Extraction Pipeline

```python
from dinov2.core import DINOv2Extractor
from dinov2.visualization import UMAPVisualizer
from dinov2.selection import DiverseImageSelector
import torch

# Initialize extractor with custom settings
extractor = DINOv2Extractor(
    model_name="facebook/dinov2-large",
    batch_size=64,
    num_workers=8
)

# Extract features with progress tracking
from tqdm import tqdm

class ProgressExtractor(DINOv2Extractor):
    def extract_batch(self, batch):
        with tqdm(total=len(batch), desc="Extracting") as pbar:
            features = super().extract_batch(batch)
            pbar.update(len(batch))
        return features

# Chain operations
embeddings, ids = extractor.extract_from_directory("images/")
visualizer = UMAPVisualizer()
projection = visualizer.fit_transform(embeddings)
selector = DiverseImageSelector(strategy="hybrid")
selected_indices = selector.select(embeddings, n_samples=1000)
```

### Multi-Scale Feature Extraction

```python
import torch
from dinov2.core import DINOv2Extractor

def extract_multiscale_features(image_path, scales=[0.5, 1.0, 1.5]):
    """Extract features at multiple scales and concatenate."""
    extractor = DINOv2Extractor()
    features = []
    
    for scale in scales:
        # Resize image
        resized = resize_image(image_path, scale)
        # Extract features
        feat = extractor.extract_single(resized)
        features.append(feat)
    
    # Concatenate multi-scale features
    return torch.cat(features, dim=-1)
```

### Incremental UMAP Updates

```python
from dinov2.visualization import UMAPVisualizer
import numpy as np

class IncrementalUMAP(UMAPVisualizer):
    def update(self, new_embeddings):
        """Update UMAP with new data points."""
        if hasattr(self, 'reducer'):
            # Transform new points using existing UMAP
            new_projection = self.reducer.transform(new_embeddings)
            # Update internal state
            self.projection = np.vstack([self.projection, new_projection])
            return new_projection
        else:
            return self.fit_transform(new_embeddings)
```

### Custom Selection Strategies

```python
from dinov2.selection import DiverseImageSelector
import numpy as np

class UncertaintySelector(DiverseImageSelector):
    """Select based on model uncertainty."""
    
    def select(self, embeddings, uncertainties, n_samples):
        # Combine diversity and uncertainty
        diversity_scores = self.compute_diversity(embeddings)
        combined_scores = diversity_scores * uncertainties
        
        # Select top scoring samples
        indices = np.argsort(combined_scores)[-n_samples:]
        return indices
```

## Performance Optimization

### GPU Memory Management

```python
# Process large datasets in chunks
def process_large_dataset(image_dir, chunk_size=10000):
    extractor = DINOv2Extractor(batch_size=32)
    all_embeddings = []
    
    for chunk in chunk_directory(image_dir, chunk_size):
        embeddings, ids = extractor.extract_from_list(chunk)
        all_embeddings.append(embeddings.cpu())
        
        # Clear GPU memory
        torch.cuda.empty_cache()
    
    return torch.cat(all_embeddings)
```

### Parallel Processing

```python
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def parallel_extract(image_paths, n_workers=4):
    """Extract features in parallel across multiple GPUs."""
    
    def extract_on_device(paths, device_id):
        device = f"cuda:{device_id}"
        extractor = DINOv2Extractor(device=device)
        return extractor.extract_from_list(paths)
    
    # Split paths across workers
    chunks = np.array_split(image_paths, n_workers)
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for i, chunk in enumerate(chunks):
            device_id = i % torch.cuda.device_count()
            future = executor.submit(extract_on_device, chunk, device_id)
            futures.append(future)
        
        results = [f.result() for f in futures]
    
    # Combine results
    embeddings = torch.cat([r[0] for r in results])
    ids = sum([r[1] for r in results], [])
    
    return embeddings, ids
```

### Caching Strategies

```python
import pickle
from pathlib import Path

class CachedExtractor(DINOv2Extractor):
    def __init__(self, cache_dir="cache/", **kwargs):
        super().__init__(**kwargs)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def extract_from_directory(self, directory):
        cache_file = self.cache_dir / f"{Path(directory).name}.pkl"
        
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # Extract and cache
        embeddings, ids = super().extract_from_directory(directory)
        with open(cache_file, 'wb') as f:
            pickle.dump((embeddings, ids), f)
        
        return embeddings, ids
```

## Input/Output Specifications

### Input Formats

#### Directory Structure
```
images/
├── class1/
│   ├── img1.jpg
│   └── img2.jpg
└── class2/
    ├── img3.jpg
    └── img4.jpg
```

#### CSV Format
```csv
image_path,label,metadata
/path/to/img1.jpg,lemur,additional_info
/path/to/img2.jpg,macaque,additional_info
```

#### Text File Format
```
/path/to/image1.jpg
/path/to/image2.jpg
/path/to/image3.jpg
```

### Output Format

#### Embeddings File Structure
```python
torch.save({
    'embeddings': tensor,  # Shape: (N, feature_dim)
    'image_ids': list,     # Length: N
    'model_name': str,     # e.g., "facebook/dinov2-base"
    'timestamp': str,      # ISO format
    'metadata': dict       # Additional information
}, 'embeddings.pt')
```

## Testing

### Unit Tests

```python
# test_extractor.py
import unittest
from dinov2.core import DINOv2Extractor

class TestExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = DINOv2Extractor(
            model_name="facebook/dinov2-small",
            device="cpu"
        )
    
    def test_single_image(self):
        embeddings, ids = self.extractor.extract_from_list(["test.jpg"])
        self.assertEqual(embeddings.shape[0], 1)
        self.assertEqual(embeddings.shape[1], 384)  # Small model dim
    
    def test_batch_processing(self):
        images = ["test1.jpg", "test2.jpg", "test3.jpg"]
        embeddings, ids = self.extractor.extract_from_list(images)
        self.assertEqual(len(ids), 3)
```

### Integration Tests

```python
def test_full_pipeline():
    """Test complete extraction -> visualization -> selection pipeline."""
    
    # Extract
    extractor = DINOv2Extractor()
    embeddings, ids = extractor.extract_from_directory("test_images/")
    
    # Visualize
    visualizer = UMAPVisualizer()
    projection = visualizer.fit_transform(embeddings)
    visualizer.plot_static("test_umap.png")
    
    # Select
    selector = DiverseImageSelector(strategy="hybrid")
    indices = selector.select(embeddings, n_samples=10)
    
    assert len(indices) == 10
    assert Path("test_umap.png").exists()
```

## Troubleshooting

### Common Issues and Solutions

#### Memory Issues
```python
# Problem: CUDA out of memory
# Solution 1: Reduce batch size
extractor = DINOv2Extractor(batch_size=8)

# Solution 2: Use smaller model
extractor = DINOv2Extractor(model_name="facebook/dinov2-small")

# Solution 3: Use CPU
extractor = DINOv2Extractor(device="cpu")
```

#### Slow Processing
```python
# Problem: Feature extraction is slow
# Solution 1: Increase batch size
extractor = DINOv2Extractor(batch_size=64)

# Solution 2: Use more workers
extractor = DINOv2Extractor(num_workers=8)

# Solution 3: Use GPU
extractor = DINOv2Extractor(device="cuda:0")
```

#### Import Errors
```python
# Problem: transformers not found
# Solution: Install DINOv2 dependencies
# pip install -e ".[dinov2]"

# Problem: UMAP not found
# Solution: Install visualization dependencies
# pip install umap-learn plotly
```

## API Compatibility

### Scikit-learn Compatible Interface

```python
from sklearn.base import BaseEstimator, TransformerMixin

class DINOv2Transformer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible DINOv2 transformer."""
    
    def __init__(self, model_name="facebook/dinov2-base"):
        self.model_name = model_name
        self.extractor = DINOv2Extractor(model_name=model_name)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # X is list of image paths
        embeddings, _ = self.extractor.extract_from_list(X)
        return embeddings.numpy()
```

## Contributing

When adding new functionality:

1. Follow existing patterns for consistency
2. Add comprehensive docstrings
3. Include unit tests in `test_dinov2.py`
4. Update this documentation
5. Ensure backward compatibility

## See Also

- [Main README](README.md) - Quick start guide
- [API Documentation](../docs/api/dinov2.md) - API overview
- [Paper](https://arxiv.org/abs/2304.07193) - DINOv2 paper
- [Original Repository](https://github.com/facebookresearch/dinov2) - Facebook Research
