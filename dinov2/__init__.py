"""DINOv2 feature extraction and analysis package.

This package provides tools for extracting and analyzing DINOv2 features
from images, including UMAP visualization, clustering, and active learning
subset selection.

Example:
    from dinov2.core import DINOv2Extractor
    from dinov2.visualization import UMAPVisualizer
    
    # Extract features
    extractor = DINOv2Extractor()
    embeddings, ids = extractor.extract_from_directory("./images/")
    
    # Visualize
    visualizer = UMAPVisualizer()
    visualizer.fit_transform(embeddings)
    visualizer.plot_static("umap.svg")
"""

from .core import (
    DINOv2Extractor,
    ImageDataset,
    load_embeddings,
    save_embeddings,
)
from .selection import DiverseImageSelector, save_selection
from .visualization import PatchVisualizer, UMAPVisualizer

__all__ = [
    # Core functionality
    'DINOv2Extractor',
    'ImageDataset',
    'load_embeddings',
    'save_embeddings',
    # Visualization
    'UMAPVisualizer',
    'PatchVisualizer',
    # Selection
    'DiverseImageSelector',
    'save_selection',
]

__version__ = '0.1.0'