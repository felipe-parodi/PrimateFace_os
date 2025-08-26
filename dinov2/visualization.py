"""Visualization utilities for DINOv2 features.

This module provides classes and functions for visualizing DINOv2 embeddings,
including UMAP projections, cluster visualizations, and attention maps.
"""

import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from matplotlib.patches import Rectangle
from PIL import Image

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from umap import UMAP

from .constants import (
    DEFAULT_COLORMAP,
    DEFAULT_DOT_SIZE,
    DEFAULT_FIGURE_DPI,
    DEFAULT_NUM_CLUSTERS,
    DEFAULT_PLOT_FORMAT,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SAMPLES_PER_CLUSTER,
    DEFAULT_THUMBNAIL_SIZE,
    DEFAULT_UMAP_METRIC,
    DEFAULT_UMAP_MIN_DIST,
    DEFAULT_UMAP_N_NEIGHBORS,
    NUM_ATTENTION_HEADS,
    PATCH_VIZ_DPI,
    PATCH_VIZ_FIGSIZE,
)

logger = logging.getLogger(__name__)


class UMAPVisualizer:
    """UMAP visualization for high-dimensional embeddings.
    
    This class provides methods for dimensionality reduction using UMAP
    and creating both static and interactive visualizations.
    
    Attributes:
        n_neighbors: UMAP parameter for local neighborhood size.
        min_dist: UMAP parameter for minimum distance between points.
        metric: Distance metric for UMAP.
        random_state: Random seed for reproducibility.
    """
    
    def __init__(
        self,
        n_neighbors: int = DEFAULT_UMAP_N_NEIGHBORS,
        min_dist: float = DEFAULT_UMAP_MIN_DIST,
        metric: str = DEFAULT_UMAP_METRIC,
        random_state: int = DEFAULT_RANDOM_STATE
    ) -> None:
        """Initialize the UMAP visualizer.
        
        Args:
            n_neighbors: Number of neighbors for UMAP.
            min_dist: Minimum distance between points in UMAP.
            metric: Distance metric to use.
            random_state: Random seed for reproducibility.
        """
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        self.reducer = None
        self.umap_embeddings = None
        self.cluster_labels = None
    
    def fit_transform(
        self,
        embeddings: Union[torch.Tensor, np.ndarray],
        n_components: int = 2
    ) -> np.ndarray:
        """Apply UMAP dimensionality reduction.
        
        Args:
            embeddings: High-dimensional embeddings.
            n_components: Number of UMAP components.
            
        Returns:
            Low-dimensional UMAP embeddings.
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        
        logger.info(f"Running UMAP on {len(embeddings)} samples...")
        
        self.reducer = UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            n_components=n_components,
            random_state=self.random_state
        )
        
        self.umap_embeddings = self.reducer.fit_transform(embeddings)
        logger.info(f"UMAP complete. Shape: {self.umap_embeddings.shape}")
        
        return self.umap_embeddings
    
    def cluster(
        self,
        embeddings: Optional[Union[torch.Tensor, np.ndarray]] = None,
        n_clusters: int = DEFAULT_NUM_CLUSTERS
    ) -> np.ndarray:
        """Perform K-means clustering on embeddings.
        
        Args:
            embeddings: Embeddings to cluster (uses UMAP embeddings if None).
            n_clusters: Number of clusters.
            
        Returns:
            Cluster labels for each sample.
        """
        if embeddings is None:
            if self.umap_embeddings is None:
                raise ValueError("Must run fit_transform first or provide embeddings")
            embeddings = self.umap_embeddings
        elif isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        
        logger.info(f"Running K-means with {n_clusters} clusters...")
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        
        self.cluster_labels = kmeans.fit_predict(embeddings)
        logger.info(f"Clustering complete. Found {len(np.unique(self.cluster_labels))} clusters")
        
        return self.cluster_labels
    
    def plot_static(
        self,
        output_path: Union[str, Path],
        labels: Optional[np.ndarray] = None,
        title: str = "UMAP Projection",
        figsize: Tuple[int, int] = (12, 10),
        dpi: int = DEFAULT_FIGURE_DPI,
        dot_size: float = DEFAULT_DOT_SIZE,
        colormap: str = DEFAULT_COLORMAP,
        show_axes: bool = True
    ) -> None:
        """Create static UMAP plot.
        
        Args:
            output_path: Path to save the plot.
            labels: Optional labels for coloring (uses clusters if None).
            title: Plot title.
            figsize: Figure size in inches.
            dpi: Dots per inch for output.
            dot_size: Size of scatter plot points.
            colormap: Matplotlib colormap name.
            show_axes: Whether to show axes labels.
        """
        if self.umap_embeddings is None:
            raise ValueError("Must run fit_transform first")
        
        if labels is None:
            labels = self.cluster_labels if self.cluster_labels is not None else np.zeros(len(self.umap_embeddings))
        
        # Normalize Windows paths
        if isinstance(output_path, str):
            output_path = output_path.replace('\\', '/')
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Create scatter plot
        scatter = ax.scatter(
            self.umap_embeddings[:, 0],
            self.umap_embeddings[:, 1],
            c=labels,
            cmap=colormap,
            s=dot_size,
            alpha=0.7,
            edgecolors='none'
        )
        
        # Debug: log the data range
        logger.debug(f"UMAP X range: {self.umap_embeddings[:, 0].min():.3f} to {self.umap_embeddings[:, 0].max():.3f}")
        logger.debug(f"UMAP Y range: {self.umap_embeddings[:, 1].min():.3f} to {self.umap_embeddings[:, 1].max():.3f}")
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Cluster')
        
        # Set title and labels
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        if show_axes:
            ax.set_xlabel('UMAP 1', fontsize=12)
            ax.set_ylabel('UMAP 2', fontsize=12)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Save figure
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        
        # For SVG format, use the SVG backend and don't pass dpi
        if output_path.suffix.lower() == '.svg':
            plt.savefig(output_path, format='svg', bbox_inches='tight')
        else:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved static plot to {output_path}")
    
    def plot_interactive(
        self,
        output_path: Union[str, Path],
        image_paths: Optional[List[str]] = None,
        labels: Optional[np.ndarray] = None,
        title: str = "Interactive UMAP Projection",
        thumbnail_size: Tuple[int, int] = DEFAULT_THUMBNAIL_SIZE
    ) -> None:
        """Create interactive UMAP plot with Plotly.
        
        Args:
            output_path: Path to save the HTML plot.
            image_paths: Optional paths to images for hover previews.
            labels: Optional labels for coloring.
            title: Plot title.
            thumbnail_size: Size for image thumbnails in hover.
        """
        if self.umap_embeddings is None:
            raise ValueError("Must run fit_transform first")
        
        # Normalize Windows paths
        if isinstance(output_path, str):
            output_path = output_path.replace('\\', '/')
        
        if labels is None:
            labels = self.cluster_labels if self.cluster_labels is not None else np.zeros(len(self.umap_embeddings))
        
        # Prepare hover data
        hover_text = []
        customdata = []
        
        for i in range(len(labels)):
            hover_info = f"Index: {i}<br>Cluster: {labels[i]}"
            img_data = ""
            
            # Add image path info and encode image if available
            if image_paths and i < len(image_paths):
                path = image_paths[i]
                hover_info += f"<br>Image: {Path(path).name if path else 'N/A'}"
                
                if path and Path(path).exists():
                    img_b64 = encode_image_base64(path, thumbnail_size)
                    if img_b64:
                        img_data = f"data:image/jpeg;base64,{img_b64}"
            
            hover_text.append(hover_info)
            customdata.append([img_data])
        
        # Create plotly figure
        fig = go.Figure(data=[
            go.Scatter(
                x=self.umap_embeddings[:, 0],
                y=self.umap_embeddings[:, 1],
                mode='markers',
                marker=dict(
                    size=5,
                    color=labels,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Cluster")
                ),
                text=hover_text,
                customdata=customdata,
                hovertemplate='%{text}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            hovermode='closest',
            width=1200,
            height=800
        )
        
        # Save HTML with custom JavaScript for image display
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate HTML with custom hover image display
        html_str = fig.to_html(include_plotlyjs='cdn')
        
        # Add custom CSS and JavaScript for image hover
        if image_paths:
            custom_html = """
            <style>
                .custom-tooltip {
                    position: absolute;
                    background: white;
                    border: 2px solid #333;
                    border-radius: 5px;
                    padding: 10px;
                    display: none;
                    z-index: 1000;
                    pointer-events: none;
                }
                .custom-tooltip img {
                    max-width: 400px;
                    max-height: 400px;
                    display: block;
                    margin-top: 5px;
                }
            </style>
            <div class="custom-tooltip" id="image-tooltip"></div>
            <script>
                document.addEventListener('DOMContentLoaded', function() {
                    var tooltip = document.getElementById('image-tooltip');
                    var plot = document.getElementsByClassName('plotly-graph-div')[0];
                    
                    plot.on('plotly_hover', function(data) {
                        var point = data.points[0];
                        if (point.customdata && point.customdata[0]) {
                            var imgSrc = point.customdata[0];
                            if (imgSrc) {
                                tooltip.innerHTML = '<img src="' + imgSrc + '">';
                                tooltip.style.display = 'block';
                                tooltip.style.left = (data.event.pageX + 10) + 'px';
                                tooltip.style.top = (data.event.pageY + 10) + 'px';
                            }
                        }
                    });
                    
                    plot.on('plotly_unhover', function() {
                        tooltip.style.display = 'none';
                    });
                });
            </script>
            """
            
            # Insert custom HTML before closing body tag
            html_str = html_str.replace('</body>', custom_html + '</body>')
        
        # Write the modified HTML
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_str)
        
        logger.info(f"Saved interactive plot to {output_path}")


class PatchVisualizer:
    """Visualizer for DINOv2 patch features and attention maps.
    
    This class provides methods for visualizing the internal representations
    of DINOv2, including patch features and multi-head attention patterns.
    """
    
    def __init__(self, model_name: str = "facebook/dinov2-base") -> None:
        """Initialize the patch visualizer.
        
        Args:
            model_name: Name of the DINOv2 model for configuration.
        """
        self.model_name = model_name
        self.patch_size = 14  # DINOv2 uses 14x14 patches
    
    def visualize_patches(
        self,
        image: Union[Image.Image, np.ndarray],
        patch_features: torch.Tensor,
        output_path: Union[str, Path],
        attention_maps: Optional[torch.Tensor] = None
    ) -> None:
        """Create comprehensive patch visualization.
        
        Args:
            image: Original image.
            patch_features: Patch token features [N, D].
            output_path: Path to save visualization.
            attention_maps: Optional attention maps [L, H, N, N].
        """
        # Normalize Windows paths
        if isinstance(output_path, str):
            output_path = output_path.replace('\\', '/')
        # Convert image to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Setup figure - simpler layout
        if attention_maps is not None:
            # 2 rows: first row for original and PCA, second row for selected attention maps
            fig, axes = plt.subplots(2, 6, figsize=(18, 6), dpi=PATCH_VIZ_DPI)
        else:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=PATCH_VIZ_DPI)
            axes = axes.reshape(1, -1)
        
        # Plot original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # PCA visualization of patches
        pca_image = self._create_pca_image(patch_features, image.shape[:2])
        axes[0, 1].imshow(pca_image)
        axes[0, 1].set_title("PCA of Patches")
        axes[0, 1].axis('off')
        
        # Hide unused axes in first row if attention maps present
        if attention_maps is not None:
            for col in range(2, 6):
                axes[0, col].axis('off')
            
            # Plot attention maps in second row
            self._plot_attention_maps(axes, attention_maps, image)
        
        # Save figure
        plt.tight_layout()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=PATCH_VIZ_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved patch visualization to {output_path}")
    
    def _create_pca_image(
        self,
        patch_features: torch.Tensor,
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Create PCA visualization of patch features.
        
        Args:
            patch_features: Patch features [N, D].
            image_shape: Original image shape (H, W).
            
        Returns:
            RGB image showing PCA components.
        """
        if isinstance(patch_features, torch.Tensor):
            patch_features = patch_features.cpu().numpy()
        
        # Handle batch dimension if present
        if patch_features.ndim == 3:
            # Shape is [B, N, D], take first item in batch
            patch_features = patch_features[0]
        
        # Apply PCA to get 3 components
        pca = PCA(n_components=3)
        pca_features = pca.fit_transform(patch_features)
        
        # Reshape to grid
        h, w = image_shape
        h_patches = h // self.patch_size
        w_patches = w // self.patch_size
        
        # Ensure we have the right number of patches
        expected_patches = h_patches * w_patches
        if len(pca_features) != expected_patches:
            # Resize if mismatch - DINOv2 may use different patch grid
            # Try to find the closest square grid
            n_patches = len(pca_features)
            grid_size = int(np.sqrt(n_patches))
            if grid_size * grid_size != n_patches:
                # Not a perfect square, use rectangular grid
                h_patches = grid_size
                w_patches = n_patches // h_patches
                if h_patches * w_patches != n_patches:
                    # Still doesn't match, just use square approximation
                    h_patches = grid_size
                    w_patches = grid_size
                    pca_features = pca_features[:h_patches * w_patches]
            else:
                h_patches = grid_size
                w_patches = grid_size
        
        pca_image = pca_features.reshape(h_patches, w_patches, 3)
        
        # Normalize to [0, 1]
        pca_image = (pca_image - pca_image.min()) / (pca_image.max() - pca_image.min())
        
        # Resize to original image size
        pca_image = Image.fromarray((pca_image * 255).astype(np.uint8))
        pca_image = pca_image.resize((w, h), Image.BILINEAR)
        
        return np.array(pca_image)
    
    def _plot_attention_maps(
        self,
        axes: np.ndarray,
        attention_maps: torch.Tensor,
        image: np.ndarray
    ) -> None:
        """Plot attention maps for different layers.
        
        Args:
            axes: Matplotlib axes array.
            attention_maps: Attention maps [L, H, N, N].
            image: Original image for reference (not used for overlay).
        """
        if isinstance(attention_maps, torch.Tensor):
            attention_maps = attention_maps.cpu().numpy()
        
        # Select 6 layers evenly spaced through the network
        n_total_layers = len(attention_maps)
        if n_total_layers <= 6:
            selected_layers = list(range(n_total_layers))
        else:
            # Select layers 1, 3, 5, 7, 9, 11 for 12-layer model
            step = n_total_layers // 6
            selected_layers = [i * step for i in range(6)]
            if selected_layers[-1] >= n_total_layers:
                selected_layers[-1] = n_total_layers - 1
        
        for idx, layer_idx in enumerate(selected_layers):
            row = 1  # Always use second row
            col = idx
            
            if row < axes.shape[0] and col < axes.shape[1]:
                # Get attention for this layer [B, H, N, N]
                layer_attn = attention_maps[layer_idx]
                
                # Take first batch item and average across heads
                if layer_attn.ndim == 4:  # [B, H, N, N]
                    attn = layer_attn[0].mean(axis=0)  # [N, N]
                elif layer_attn.ndim == 3:  # [H, N, N]
                    attn = layer_attn.mean(axis=0)  # [N, N]
                else:
                    attn = layer_attn  # Already [N, N]
                
                # Get CLS token attention (first row)
                cls_attn = attn[0, 1:]  # Skip CLS to CLS
                
                # Reshape to grid
                grid_size = int(np.sqrt(len(cls_attn)))
                cls_attn = cls_attn.reshape(grid_size, grid_size)
                
                # Just show the attention map without overlay
                im = axes[row, col].imshow(cls_attn, cmap='jet', interpolation='nearest')
                axes[row, col].set_title(f"Layer {layer_idx + 1}")
                axes[row, col].axis('off')


def encode_image_base64(
    image_path: Union[str, Path],
    size: Tuple[int, int] = DEFAULT_THUMBNAIL_SIZE
) -> Optional[str]:
    """Encode image to base64 for embedding in HTML.
    
    Args:
        image_path: Path to the image file.
        size: Target size for thumbnail.
        
    Returns:
        Base64 encoded string or None if error.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img.thumbnail(size)
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    except Exception as e:
        logger.warning(f"Failed to encode image {image_path}: {e}")
        return None