"""Image selection algorithms for active learning.

This module provides methods for selecting diverse subsets of images
based on their DINOv2 embeddings, useful for active learning scenarios.
"""

import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from .constants import (
    DEFAULT_FPS_BATCH_SIZE,
    DEFAULT_NUM_CLUSTERS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TARGET_SUBSET_SIZE,
)

logger = logging.getLogger(__name__)


class DiverseImageSelector:
    """Select diverse subsets of images for active learning.
    
    This class implements various strategies for selecting representative
    and diverse subsets from large image collections based on embeddings.
    
    Attributes:
        random_state: Random seed for reproducibility.
        strategy: Selection strategy to use.
    """
    
    def __init__(
        self,
        strategy: str = "hybrid",
        random_state: int = DEFAULT_RANDOM_STATE
    ) -> None:
        """Initialize the selector.
        
        Args:
            strategy: Selection strategy ('random', 'cluster', 'fps', 'hybrid').
            random_state: Random seed for reproducibility.
        """
        self.strategy = strategy
        self.random_state = random_state
        
        # Set random seeds
        random.seed(random_state)
        np.random.seed(random_state)
    
    def select(
        self,
        embeddings: Union[torch.Tensor, np.ndarray],
        n_samples: int = DEFAULT_TARGET_SUBSET_SIZE,
        n_clusters: Optional[int] = None,
        image_ids: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, Optional[List[str]]]:
        """Select diverse subset of samples.
        
        Args:
            embeddings: Feature embeddings [N, D].
            n_samples: Number of samples to select.
            n_clusters: Number of clusters for cluster-based strategies.
            image_ids: Optional image identifiers.
            
        Returns:
            Tuple of (selected_indices, selected_ids).
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        
        n_total = len(embeddings)
        
        if n_samples >= n_total:
            logger.warning(f"Requested {n_samples} samples but only {n_total} available")
            indices = np.arange(n_total)
            selected_ids = image_ids if image_ids else None
            return indices, selected_ids
        
        logger.info(f"Selecting {n_samples} from {n_total} using {self.strategy} strategy")
        
        # Apply selection strategy
        if self.strategy == "random":
            indices = self._random_selection(n_total, n_samples)
        elif self.strategy == "cluster":
            indices = self._cluster_selection(embeddings, n_samples, n_clusters)
        elif self.strategy == "fps":
            indices = self._fps_selection(embeddings, n_samples)
        elif self.strategy == "hybrid":
            indices = self._hybrid_selection(embeddings, n_samples, n_clusters)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Ensure indices are unique (they should be, but double-check)
        unique_indices = np.unique(indices)
        if len(unique_indices) != len(indices):
            logger.warning(f"Found {len(indices) - len(unique_indices)} duplicate indices, removing them")
            indices = unique_indices
        
        # Get corresponding image IDs
        selected_ids = None
        if image_ids:
            selected_ids = [image_ids[i] for i in indices]
            # Check if image_ids themselves have duplicates
            unique_selected = list(dict.fromkeys(selected_ids))  # Preserves order
            if len(unique_selected) != len(selected_ids):
                logger.warning(f"Image IDs contain duplicates! {len(selected_ids)} selected but only {len(unique_selected)} unique")
        
        logger.info(f"Selected {len(indices)} samples with {len(np.unique(indices))} unique indices")
        
        return indices, selected_ids
    
    def _random_selection(self, n_total: int, n_samples: int) -> np.ndarray:
        """Random selection of samples.
        
        Args:
            n_total: Total number of samples.
            n_samples: Number to select.
            
        Returns:
            Array of selected indices.
        """
        return np.random.choice(n_total, n_samples, replace=False)
    
    def _cluster_selection(
        self,
        embeddings: np.ndarray,
        n_samples: int,
        n_clusters: Optional[int] = None
    ) -> np.ndarray:
        """Cluster-based selection with proportional sampling.
        
        Args:
            embeddings: Feature embeddings.
            n_samples: Number to select.
            n_clusters: Number of clusters.
            
        Returns:
            Array of selected indices.
        """
        if n_clusters is None:
            n_clusters = max(1, min(DEFAULT_NUM_CLUSTERS, n_samples // 5, len(embeddings)))
        
        # Perform clustering
        logger.info(f"Clustering with K={n_clusters}")
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        labels = kmeans.fit_predict(embeddings)
        
        # Calculate samples per cluster (proportional to cluster size)
        selected_indices = []
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        for label, count in zip(unique_labels, counts):
            # Proportional allocation
            n_from_cluster = max(1, int(n_samples * count / len(embeddings)))
            cluster_indices = np.where(labels == label)[0]
            
            if len(cluster_indices) <= n_from_cluster:
                selected_indices.extend(cluster_indices)
            else:
                # Random sample from cluster
                selected = np.random.choice(
                    cluster_indices,
                    n_from_cluster,
                    replace=False
                )
                selected_indices.extend(selected)
        
        # Remove duplicates (shouldn't happen but be safe)
        selected_indices = np.unique(selected_indices)
        
        # Adjust to exact number if needed
        if len(selected_indices) > n_samples:
            selected_indices = np.random.choice(
                selected_indices,
                n_samples,
                replace=False
            )
        elif len(selected_indices) < n_samples:
            # Need more samples - get from unselected
            all_indices = np.arange(len(embeddings))
            unselected = np.setdiff1d(all_indices, selected_indices)
            if len(unselected) > 0:
                n_additional = min(n_samples - len(selected_indices), len(unselected))
                additional = np.random.choice(unselected, n_additional, replace=False)
                selected_indices = np.concatenate([selected_indices, additional])
        
        return selected_indices
    
    def _fps_selection(
        self,
        embeddings: np.ndarray,
        n_samples: int,
        batch_size: int = DEFAULT_FPS_BATCH_SIZE
    ) -> np.ndarray:
        """Farthest Point Sampling selection.
        
        Args:
            embeddings: Feature embeddings.
            n_samples: Number to select.
            batch_size: Batch size for distance computation.
            
        Returns:
            Array of selected indices.
        """
        n_total = len(embeddings)
        
        # Initialize with random point
        selected_indices = [np.random.randint(n_total)]
        selected_mask = np.zeros(n_total, dtype=bool)
        selected_mask[selected_indices[0]] = True
        
        # Initialize distances to first point
        distances = pairwise_distances(
            embeddings[selected_indices[0:1]],
            embeddings,
            metric='euclidean'
        ).squeeze()
        
        logger.info("Running Farthest Point Sampling...")
        
        for _ in tqdm(range(1, n_samples), desc="FPS Selection"):
            # Find farthest point
            distances_masked = distances.copy()
            distances_masked[selected_mask] = -np.inf
            farthest_idx = np.argmax(distances_masked)
            
            # Add to selected
            selected_indices.append(farthest_idx)
            selected_mask[farthest_idx] = True
            
            # Update distances (minimum distance to any selected point)
            new_distances = pairwise_distances(
                embeddings[farthest_idx:farthest_idx+1],
                embeddings,
                metric='euclidean'
            ).squeeze()
            distances = np.minimum(distances, new_distances)
        
        return np.array(selected_indices)
    
    def _hybrid_selection(
        self,
        embeddings: np.ndarray,
        n_samples: int,
        n_clusters: Optional[int] = None
    ) -> np.ndarray:
        """Hybrid selection combining clustering and FPS.
        
        This method first clusters the data, then applies FPS within
        each cluster for maximum diversity.
        
        Args:
            embeddings: Feature embeddings.
            n_samples: Number to select.
            n_clusters: Number of clusters.
            
        Returns:
            Array of selected indices.
        """
        if n_clusters is None:
            # Ensure at least 1 cluster, and don't exceed n_samples
            n_clusters = max(1, min(DEFAULT_NUM_CLUSTERS, n_samples // 10, n_samples))
        
        # For very small n_samples, just use FPS
        if n_clusters <= 1 or n_samples <= 3:
            logger.info(f"Small sample size, using FPS selection instead of hybrid")
            return self._fps_selection(embeddings, n_samples)
        
        # Perform clustering
        logger.info(f"Hybrid selection: clustering with K={n_clusters}")
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        labels = kmeans.fit_predict(embeddings)
        
        # Calculate samples per cluster
        selected_indices = []
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Base allocation (at least 1 per cluster)
        base_per_cluster = n_samples // n_clusters
        remaining = n_samples - (base_per_cluster * n_clusters)
        
        # Allocate remaining samples proportionally
        proportions = counts / counts.sum()
        extra_samples = (proportions * remaining).astype(int)
        
        for i, (label, count) in enumerate(zip(unique_labels, counts)):
            cluster_indices = np.where(labels == label)[0]
            n_from_cluster = base_per_cluster + extra_samples[i]
            n_from_cluster = min(n_from_cluster, len(cluster_indices))
            
            if n_from_cluster == 1:
                # Just pick center
                cluster_embeddings = embeddings[cluster_indices]
                center = cluster_embeddings.mean(axis=0)
                distances = np.linalg.norm(cluster_embeddings - center, axis=1)
                selected_idx = cluster_indices[np.argmin(distances)]
                selected_indices.append(selected_idx)
            elif n_from_cluster >= len(cluster_indices):
                # Take all
                selected_indices.extend(cluster_indices)
            else:
                # FPS within cluster
                cluster_embeddings = embeddings[cluster_indices]
                fps_indices = self._fps_selection(
                    cluster_embeddings,
                    n_from_cluster
                )
                selected_indices.extend(cluster_indices[fps_indices])
        
        # Remove duplicates and ensure exact number
        selected_indices = np.unique(selected_indices)
        
        # If we have fewer than needed due to duplicates, add more using FPS
        if len(selected_indices) < n_samples:
            # Get remaining samples using FPS from unselected points
            mask = np.ones(len(embeddings), dtype=bool)
            mask[selected_indices] = False
            remaining_indices = np.where(mask)[0]
            
            if len(remaining_indices) > 0:
                n_additional = min(n_samples - len(selected_indices), len(remaining_indices))
                additional_embeddings = embeddings[remaining_indices]
                additional_selected = self._fps_selection(additional_embeddings, n_additional)
                selected_indices = np.concatenate([
                    selected_indices,
                    remaining_indices[additional_selected]
                ])
        
        # Ensure we don't exceed requested number
        selected_indices = selected_indices[:n_samples]
        
        return selected_indices


def save_selection(
    indices: np.ndarray,
    output_path: Union[str, Path],
    image_ids: Optional[List[str]] = None
) -> None:
    """Save selected indices/IDs to file.
    
    Args:
        indices: Selected indices.
        output_path: Path to save the selection.
        image_ids: Optional image identifiers.
        
    Returns:
        - A .txt file with the selected image ids.
    """
    # Normalize Windows paths
    if isinstance(output_path, str):
        output_path = output_path.replace('\\', '/')
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        if image_ids:
            for idx in indices:
                if idx < len(image_ids):
                    f.write(f"{image_ids[idx]}\n")
        else:
            for idx in indices:
                f.write(f"{idx}\n")
    
    logger.info(f"Saved {len(indices)} selections to {output_path}")