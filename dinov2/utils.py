"""Utility functions for DINOv2 module.

This module contains shared utility functions used across different
DINOv2 components.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def check_cuda_availability() -> str:
    """Check CUDA availability and return appropriate device.
    
    Returns:
        Device string ('cuda' or 'cpu').
    """
    if torch.cuda.is_available():
        logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    else:
        logger.info("CUDA not available. Using CPU")
        return "cpu"


def validate_path(
    path: Union[str, Path],
    must_exist: bool = True,
    create_parent: bool = False
) -> Path:
    """Validate and convert path.
    
    Args:
        path: Input path string or Path object.
        must_exist: Whether the path must exist.
        create_parent: Whether to create parent directories.
        
    Returns:
        Validated Path object.
        
    Raises:
        FileNotFoundError: If path doesn't exist and must_exist=True.
    """
    path = Path(path)
    
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    
    if create_parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created parent directory: {path.parent}")
    
    return path


def batch_iterator(
    items: List,
    batch_size: int,
    desc: Optional[str] = None
) -> List:
    """Iterate over items in batches.
    
    Args:
        items: List of items to batch.
        batch_size: Size of each batch.
        desc: Description for progress bar.
        
    Yields:
        Batches of items.
    """
    from tqdm import tqdm
    
    n_items = len(items)
    n_batches = (n_items + batch_size - 1) // batch_size
    
    iterator = range(0, n_items, batch_size)
    if desc:
        iterator = tqdm(iterator, total=n_batches, desc=desc)
    
    for i in iterator:
        yield items[i:i + batch_size]


def normalize_embeddings(
    embeddings: Union[torch.Tensor, np.ndarray],
    method: str = "l2"
) -> Union[torch.Tensor, np.ndarray]:
    """Normalize embeddings.
    
    Args:
        embeddings: Input embeddings [N, D].
        method: Normalization method ('l2', 'standard', 'minmax').
        
    Returns:
        Normalized embeddings.
    """
    is_tensor = isinstance(embeddings, torch.Tensor)
    
    if is_tensor:
        device = embeddings.device
        embeddings = embeddings.cpu().numpy()
    
    if method == "l2":
        # L2 normalization
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized = embeddings / norms
    elif method == "standard":
        # Standardization (zero mean, unit variance)
        mean = embeddings.mean(axis=0)
        std = embeddings.std(axis=0)
        std[std == 0] = 1  # Avoid division by zero
        normalized = (embeddings - mean) / std
    elif method == "minmax":
        # Min-max normalization to [0, 1]
        min_vals = embeddings.min(axis=0)
        max_vals = embeddings.max(axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # Avoid division by zero
        normalized = (embeddings - min_vals) / range_vals
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    if is_tensor:
        normalized = torch.from_numpy(normalized).to(device)
    
    return normalized


def compute_pairwise_distances(
    embeddings1: Union[torch.Tensor, np.ndarray],
    embeddings2: Optional[Union[torch.Tensor, np.ndarray]] = None,
    metric: str = "euclidean"
) -> np.ndarray:
    """Compute pairwise distances between embeddings.
    
    Args:
        embeddings1: First set of embeddings [N, D].
        embeddings2: Second set of embeddings [M, D]. If None, compute within embeddings1.
        metric: Distance metric ('euclidean', 'cosine', 'manhattan').
        
    Returns:
        Distance matrix [N, M] or [N, N].
    """
    from sklearn.metrics import pairwise_distances
    
    if isinstance(embeddings1, torch.Tensor):
        embeddings1 = embeddings1.cpu().numpy()
    
    if embeddings2 is not None and isinstance(embeddings2, torch.Tensor):
        embeddings2 = embeddings2.cpu().numpy()
    
    return pairwise_distances(embeddings1, embeddings2, metric=metric)


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics.
    
    Returns:
        Dictionary with memory usage in GB.
    """
    import psutil
    
    memory_stats = {}
    
    # System memory
    vm = psutil.virtual_memory()
    memory_stats['system_total_gb'] = vm.total / (1024**3)
    memory_stats['system_used_gb'] = vm.used / (1024**3)
    memory_stats['system_available_gb'] = vm.available / (1024**3)
    memory_stats['system_percent'] = vm.percent
    
    # GPU memory if available
    if torch.cuda.is_available():
        memory_stats['gpu_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
        memory_stats['gpu_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
        
    return memory_stats