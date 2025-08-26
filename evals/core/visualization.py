"""Visualization utilities for evaluation results.

This module provides unified visualization functions for training curves,
predictions, and evaluation metrics across all frameworks.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import cv2


class EvalVisualizer:
    """Unified visualization for evaluation results."""
    
    def __init__(
        self,
        figure_size: Tuple[int, int] = (12, 8),
        dpi: int = 100,
        style: str = 'seaborn-v0_8'
    ):
        """Initialize visualizer.
        
        Args:
            figure_size: Default figure size in inches
            dpi: Dots per inch for saved figures
            style: Matplotlib style
        """
        self.figure_size = figure_size
        self.dpi = dpi
        if style and style in plt.style.available:
            plt.style.use(style)
    
    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        output_path: Optional[Union[str, Path]] = None,
        metrics: Optional[List[str]] = None,
        title: str = "Training Progress"
    ) -> plt.Figure:
        """Plot training and validation curves.
        
        Args:
            history: Dictionary with metric histories
            output_path: Path to save the figure
            metrics: Specific metrics to plot (None for all)
            title: Plot title
            
        Returns:
            Matplotlib figure object
        """
        if metrics is None:
            metrics = list(history.keys())
        
        # Determine subplot layout
        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(
            n_rows, n_cols, 
            figsize=(self.figure_size[0], self.figure_size[1] * n_rows / 2),
            squeeze=False
        )
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            if metric not in history:
                continue
                
            ax = axes[idx]
            values = history[metric]
            epochs = range(1, len(values) + 1)
            
            # Plot metric
            ax.plot(epochs, values, 'b-', label=metric, linewidth=2)
            
            # Check for validation metric
            val_metric = f'val_{metric}'
            if val_metric in history:
                val_values = history[val_metric]
                ax.plot(epochs, val_values, 'r--', label=val_metric, linewidth=2)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Mark best epoch if loss metric
            if 'loss' in metric.lower():
                best_epoch = np.argmin(values) + 1
                best_value = min(values)
                ax.axvline(x=best_epoch, color='g', linestyle=':', alpha=0.5)
                ax.annotate(
                    f'Best: {best_value:.4f}',
                    xy=(best_epoch, best_value),
                    xytext=(best_epoch + 1, best_value),
                    fontsize=8
                )
        
        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def plot_predictions(
        self,
        images: Union[np.ndarray, List[np.ndarray]],
        predictions: Union[np.ndarray, List[np.ndarray]],
        ground_truth: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        output_path: Optional[Union[str, Path]] = None,
        max_images: int = 6,
        keypoint_names: Optional[List[str]] = None
    ) -> plt.Figure:
        """Visualize pose predictions on images.
        
        Args:
            images: Input images [N, H, W, C] or list of images
            predictions: Predicted keypoints [N, K, 2/3] or list
            ground_truth: Ground truth keypoints (optional)
            output_path: Path to save the figure
            max_images: Maximum number of images to display
            keypoint_names: Names of keypoints for labeling
            
        Returns:
            Matplotlib figure object
        """
        # Convert to lists for uniform handling
        if isinstance(images, np.ndarray):
            images = [images[i] for i in range(min(len(images), max_images))]
        if isinstance(predictions, np.ndarray):
            predictions = [predictions[i] for i in range(min(len(predictions), max_images))]
        if ground_truth is not None and isinstance(ground_truth, np.ndarray):
            ground_truth = [ground_truth[i] for i in range(min(len(ground_truth), max_images))]
        
        n_images = min(len(images), max_images)
        n_cols = min(3, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(self.figure_size[0], self.figure_size[1] * n_rows / 2),
            squeeze=False
        )
        axes = axes.flatten()
        
        for idx in range(n_images):
            ax = axes[idx]
            img = images[idx]
            pred = predictions[idx]
            
            # Display image
            if img.dtype == np.float32 or img.dtype == np.float64:
                img = (img * 255).astype(np.uint8)
            ax.imshow(img)
            
            # Plot predictions
            self._plot_keypoints(
                ax, pred, color='red', marker='o', label='Prediction'
            )
            
            # Plot ground truth if available
            if ground_truth is not None:
                gt = ground_truth[idx]
                self._plot_keypoints(
                    ax, gt, color='green', marker='x', label='Ground Truth'
                )
            
            ax.set_title(f'Image {idx + 1}')
            ax.axis('off')
            if idx == 0:
                ax.legend(loc='upper right', fontsize=8)
        
        # Hide unused subplots
        for idx in range(n_images, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Pose Predictions', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def plot_confusion_matrix(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        class_names: Optional[List[str]] = None,
        output_path: Optional[Union[str, Path]] = None,
        normalize: bool = True
    ) -> plt.Figure:
        """Plot confusion matrix for classification results.
        
        Args:
            predictions: Predicted class indices
            labels: True class indices
            class_names: Names of classes
            output_path: Path to save the figure
            normalize: Whether to normalize the matrix
            
        Returns:
            Matplotlib figure object
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(labels, predictions)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
        fig, ax = plt.subplots(figsize=self.figure_size)
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        if class_names is not None:
            ax.set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   xticklabels=class_names,
                   yticklabels=class_names)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")
        
        # Add text annotations
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def plot_metric_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        output_path: Optional[Union[str, Path]] = None,
        metrics: Optional[List[str]] = None
    ) -> plt.Figure:
        """Plot comparison of metrics across models.
        
        Args:
            results: Nested dict {model_name: {metric_name: value}}
            output_path: Path to save the figure
            metrics: Specific metrics to plot (None for all)
            
        Returns:
            Matplotlib figure object
        """
        model_names = list(results.keys())
        
        if metrics is None:
            # Get all unique metrics
            metrics = set()
            for model_results in results.values():
                metrics.update(model_results.keys())
            metrics = sorted(list(metrics))
        
        n_metrics = len(metrics)
        n_models = len(model_names)
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        x = np.arange(n_metrics)
        width = 0.8 / n_models
        
        colors = plt.cm.Set3(np.linspace(0, 1, n_models))
        
        for i, model in enumerate(model_names):
            values = [results[model].get(m, 0) for m in metrics]
            offset = (i - n_models / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=model, color=colors[i])
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{val:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Value')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def _plot_keypoints(
        self,
        ax: plt.Axes,
        keypoints: np.ndarray,
        color: str = 'red',
        marker: str = 'o',
        label: Optional[str] = None,
        connections: Optional[List[Tuple[int, int]]] = None
    ) -> None:
        """Plot keypoints on an axis.
        
        Args:
            ax: Matplotlib axis
            keypoints: Keypoints array [K, 2/3]
            color: Color for keypoints
            marker: Marker style
            label: Label for legend
            connections: List of (start_idx, end_idx) for skeleton
        """
        # Handle visibility
        if keypoints.shape[-1] >= 3:
            visible = keypoints[:, 2] > 0
            kpts = keypoints[visible, :2]
        else:
            kpts = keypoints[:, :2]
            visible = np.ones(len(keypoints), dtype=bool)
        
        if len(kpts) > 0:
            ax.scatter(
                kpts[:, 0], kpts[:, 1],
                c=color, marker=marker, s=30,
                label=label, zorder=5
            )
        
        # Draw skeleton connections if provided
        if connections is not None:
            for start_idx, end_idx in connections:
                if visible[start_idx] and visible[end_idx]:
                    start = keypoints[start_idx, :2]
                    end = keypoints[end_idx, :2]
                    ax.plot(
                        [start[0], end[0]], [start[1], end[1]],
                        color=color, linewidth=1, alpha=0.7
                    )


def plot_training_curves(
    history: Dict[str, List[float]],
    output_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> plt.Figure:
    """Convenience function to plot training curves.
    
    Args:
        history: Dictionary with metric histories
        output_path: Path to save the figure
        **kwargs: Additional arguments for EvalVisualizer
        
    Returns:
        Matplotlib figure object
    """
    viz = EvalVisualizer()
    return viz.plot_training_curves(history, output_path, **kwargs)


def plot_predictions(
    images: Union[np.ndarray, List[np.ndarray]],
    predictions: Union[np.ndarray, List[np.ndarray]],
    ground_truth: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    output_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> plt.Figure:
    """Convenience function to plot predictions.
    
    Args:
        images: Input images
        predictions: Predicted keypoints
        ground_truth: Ground truth keypoints (optional)
        output_path: Path to save the figure
        **kwargs: Additional arguments for EvalVisualizer
        
    Returns:
        Matplotlib figure object
    """
    viz = EvalVisualizer()
    return viz.plot_predictions(
        images, predictions, ground_truth, output_path, **kwargs
    )