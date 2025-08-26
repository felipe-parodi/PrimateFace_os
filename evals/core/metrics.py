"""Metrics calculation for pose estimation evaluation.

This module provides unified metrics calculation for evaluating pose estimation
models across different frameworks (MMPose, DeepLabCut, SLEAP).
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from abc import ABC, abstractmethod


class BaseMetric(ABC):
    """Abstract base class for evaluation metrics."""
    
    @abstractmethod
    def calculate(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        **kwargs
    ) -> Union[float, Dict[str, float]]:
        """Calculate the metric.
        
        Args:
            predictions: Predicted keypoints or bboxes
            ground_truth: Ground truth annotations
            **kwargs: Additional metric-specific parameters
            
        Returns:
            Metric value(s)
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the metric."""
        pass


class NMECalculator(BaseMetric):
    """Normalized Mean Error calculator for keypoint evaluation."""
    
    def __init__(self, normalize_by: str = 'bbox'):
        """Initialize NME calculator.
        
        Args:
            normalize_by: Normalization method ('bbox', 'torso', 'head', 'interocular')
        """
        self.normalize_by = normalize_by
        
    def calculate(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        **kwargs
    ) -> float:
        """Calculate Normalized Mean Error.
        
        Args:
            predictions: Predicted keypoints [N, K, 2 or 3]
            ground_truth: Ground truth keypoints [N, K, 2 or 3]
            bboxes: Bounding boxes for normalization [N, 4]
            
        Returns:
            NME value
        """
        if predictions.shape != ground_truth.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} "
                f"vs ground_truth {ground_truth.shape}"
            )
            
        # Extract coordinates and visibility
        pred_coords = predictions[..., :2]
        gt_coords = ground_truth[..., :2]
        
        if predictions.shape[-1] > 2:
            visibility = ground_truth[..., 2] > 0
        else:
            visibility = np.ones(ground_truth.shape[:-1], dtype=bool)
            
        # Calculate distances
        distances = np.linalg.norm(pred_coords - gt_coords, axis=-1)
        
        # Get normalization factor
        norm_factor = self._get_normalization_factor(
            gt_coords, bboxes, self.normalize_by
        )
        
        # Normalize and average
        normalized_distances = distances / norm_factor[:, None]
        valid_distances = normalized_distances[visibility]
        
        if len(valid_distances) == 0:
            return float('inf')
            
        return float(np.mean(valid_distances))
    
    def _get_normalization_factor(
        self,
        keypoints: np.ndarray,
        bboxes: Optional[np.ndarray],
        method: str
    ) -> np.ndarray:
        """Get normalization factor based on method.
        
        Args:
            keypoints: Ground truth keypoints [N, K, 2]
            bboxes: Bounding boxes [N, 4]
            method: Normalization method
            
        Returns:
            Normalization factors [N]
        """
        N = keypoints.shape[0]
        
        if method == 'bbox':
            if bboxes is None:
                # Calculate bbox from keypoints
                valid_mask = ~np.isnan(keypoints).any(axis=-1)
                norm_factors = []
                for i in range(N):
                    valid_kpts = keypoints[i][valid_mask[i]]
                    if len(valid_kpts) > 0:
                        bbox_w = valid_kpts[:, 0].max() - valid_kpts[:, 0].min()
                        bbox_h = valid_kpts[:, 1].max() - valid_kpts[:, 1].min()
                        norm_factors.append(np.sqrt(bbox_w * bbox_h))
                    else:
                        norm_factors.append(1.0)
                return np.array(norm_factors)
            else:
                # Use provided bboxes
                bbox_w = bboxes[:, 2] - bboxes[:, 0]
                bbox_h = bboxes[:, 3] - bboxes[:, 1]
                return np.sqrt(bbox_w * bbox_h)
                
        elif method == 'interocular':
            # Assume eye keypoints are at indices 1 and 2
            left_eye = keypoints[:, 1]
            right_eye = keypoints[:, 2]
            return np.linalg.norm(left_eye - right_eye, axis=-1)
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def get_name(self) -> str:
        """Get metric name."""
        return f"NME_{self.normalize_by}"


class PCKCalculator(BaseMetric):
    """Percentage of Correct Keypoints calculator."""
    
    def __init__(self, threshold: float = 0.2, reference: str = 'bbox_diagonal'):
        """Initialize PCK calculator.
        
        Args:
            threshold: Threshold as fraction of reference distance
            reference: Reference distance ('bbox_diagonal', 'head_size', 'torso_size')
        """
        self.threshold = threshold
        self.reference = reference
        
    def calculate(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, float]:
        """Calculate PCK metrics.
        
        Args:
            predictions: Predicted keypoints [N, K, 2 or 3]
            ground_truth: Ground truth keypoints [N, K, 2 or 3]
            bboxes: Bounding boxes for reference [N, 4]
            
        Returns:
            Dictionary with PCK values
        """
        # Extract coordinates and visibility
        pred_coords = predictions[..., :2]
        gt_coords = ground_truth[..., :2]
        
        if predictions.shape[-1] > 2:
            visibility = ground_truth[..., 2] > 0
        else:
            visibility = np.ones(ground_truth.shape[:-1], dtype=bool)
            
        # Calculate distances
        distances = np.linalg.norm(pred_coords - gt_coords, axis=-1)
        
        # Get reference distance
        ref_dist = self._get_reference_distance(gt_coords, bboxes, self.reference)
        
        # Calculate PCK
        threshold_dist = ref_dist[:, None] * self.threshold
        correct = (distances <= threshold_dist) & visibility
        
        # Calculate per-keypoint and overall PCK
        pck_per_kpt = []
        for k in range(correct.shape[1]):
            valid = visibility[:, k]
            if valid.sum() > 0:
                pck_per_kpt.append(correct[valid, k].mean())
                
        overall_pck = correct[visibility].mean() if visibility.sum() > 0 else 0.0
        
        return {
            'pck': float(overall_pck),
            'pck_per_keypoint': pck_per_kpt,
            'threshold': self.threshold,
        }
    
    def _get_reference_distance(
        self,
        keypoints: np.ndarray,
        bboxes: Optional[np.ndarray],
        method: str
    ) -> np.ndarray:
        """Get reference distance for PCK calculation.
        
        Args:
            keypoints: Ground truth keypoints [N, K, 2]
            bboxes: Bounding boxes [N, 4]
            method: Reference method
            
        Returns:
            Reference distances [N]
        """
        N = keypoints.shape[0]
        
        if method == 'bbox_diagonal':
            if bboxes is None:
                # Calculate from keypoints
                ref_dists = []
                for i in range(N):
                    valid_mask = ~np.isnan(keypoints[i]).any(axis=-1)
                    valid_kpts = keypoints[i][valid_mask]
                    if len(valid_kpts) > 0:
                        bbox_w = valid_kpts[:, 0].max() - valid_kpts[:, 0].min()
                        bbox_h = valid_kpts[:, 1].max() - valid_kpts[:, 1].min()
                        ref_dists.append(np.sqrt(bbox_w**2 + bbox_h**2))
                    else:
                        ref_dists.append(1.0)
                return np.array(ref_dists)
            else:
                bbox_w = bboxes[:, 2] - bboxes[:, 0]
                bbox_h = bboxes[:, 3] - bboxes[:, 1]
                return np.sqrt(bbox_w**2 + bbox_h**2)
                
        elif method == 'head_size':
            # Assume specific keypoint indices for head
            # This is a simplified version - adjust indices as needed
            nose = keypoints[:, 0]
            chin = keypoints[:, -1]
            return np.linalg.norm(nose - chin, axis=-1)
            
        else:
            raise ValueError(f"Unknown reference method: {method}")
    
    def get_name(self) -> str:
        """Get metric name."""
        return f"PCK@{self.threshold}"


class OKSCalculator(BaseMetric):
    """Object Keypoint Similarity calculator (COCO-style)."""
    
    def __init__(self, sigmas: Optional[np.ndarray] = None):
        """Initialize OKS calculator.
        
        Args:
            sigmas: Per-keypoint standard deviations for OKS
        """
        self.sigmas = sigmas
        
    def calculate(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        areas: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, float]:
        """Calculate OKS metrics.
        
        Args:
            predictions: Predicted keypoints [N, K, 3]
            ground_truth: Ground truth keypoints [N, K, 3]
            areas: Object areas for normalization [N]
            
        Returns:
            Dictionary with OKS values
        """
        N, K = predictions.shape[:2]
        
        # Use default sigmas if not provided
        if self.sigmas is None:
            self.sigmas = np.ones(K) * 0.05
            
        # Calculate areas if not provided
        if areas is None:
            areas = self._calculate_areas_from_keypoints(ground_truth)
            
        # Calculate OKS for each instance
        oks_values = []
        for i in range(N):
            pred = predictions[i]
            gt = ground_truth[i]
            area = areas[i]
            
            # Visibility mask
            vis = gt[:, 2] > 0
            
            if vis.sum() == 0 or area <= 0:
                continue
                
            # Calculate distances
            dx = pred[vis, 0] - gt[vis, 0]
            dy = pred[vis, 1] - gt[vis, 1]
            
            # OKS calculation
            vars = (self.sigmas[vis] * 2) ** 2
            e = (dx**2 + dy**2) / vars / (area + 1e-8) / 2
            oks = np.exp(-e).mean()
            oks_values.append(oks)
            
        return {
            'oks': float(np.mean(oks_values)) if oks_values else 0.0,
            'oks_per_instance': oks_values,
        }
    
    def _calculate_areas_from_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """Calculate areas from keypoint bounding boxes.
        
        Args:
            keypoints: Keypoints [N, K, 3]
            
        Returns:
            Areas [N]
        """
        areas = []
        for kpts in keypoints:
            vis = kpts[:, 2] > 0
            if vis.sum() > 0:
                visible_kpts = kpts[vis, :2]
                w = visible_kpts[:, 0].max() - visible_kpts[:, 0].min()
                h = visible_kpts[:, 1].max() - visible_kpts[:, 1].min()
                areas.append(w * h)
            else:
                areas.append(0.0)
        return np.array(areas)
    
    def get_name(self) -> str:
        """Get metric name."""
        return "OKS"


class MetricsCalculator:
    """Unified metrics calculator for all evaluation needs."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.nme_calculator = NMECalculator()
        self.pck_calculator = PCKCalculator()
        self.oks_calculator = OKSCalculator()
        
    def calculate_all(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        metrics: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Calculate all requested metrics.
        
        Args:
            predictions: Predicted keypoints
            ground_truth: Ground truth keypoints
            metrics: List of metrics to calculate (None for all)
            **kwargs: Additional parameters for specific metrics
            
        Returns:
            Dictionary with all calculated metrics
        """
        if metrics is None:
            metrics = ['nme', 'pck', 'oks']
            
        results = {}
        
        if 'nme' in metrics:
            results['nme'] = self.nme_calculator.calculate(
                predictions, ground_truth, **kwargs
            )
            
        if 'pck' in metrics:
            results.update(self.pck_calculator.calculate(
                predictions, ground_truth, **kwargs
            ))
            
        if 'oks' in metrics:
            results.update(self.oks_calculator.calculate(
                predictions, ground_truth, **kwargs
            ))
            
        return results
    
    def calculate_nme(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        normalize_by: str = 'bbox',
        **kwargs
    ) -> float:
        """Calculate Normalized Mean Error.
        
        Args:
            predictions: Predicted keypoints
            ground_truth: Ground truth keypoints
            normalize_by: Normalization method
            **kwargs: Additional parameters
            
        Returns:
            NME value
        """
        calculator = NMECalculator(normalize_by=normalize_by)
        return calculator.calculate(predictions, ground_truth, **kwargs)
    
    def calculate_pck(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        threshold: float = 0.2,
        **kwargs
    ) -> Dict[str, float]:
        """Calculate Percentage of Correct Keypoints.
        
        Args:
            predictions: Predicted keypoints
            ground_truth: Ground truth keypoints
            threshold: PCK threshold
            **kwargs: Additional parameters
            
        Returns:
            PCK results dictionary
        """
        calculator = PCKCalculator(threshold=threshold)
        return calculator.calculate(predictions, ground_truth, **kwargs)
    
    def calculate_oks(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        sigmas: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, float]:
        """Calculate Object Keypoint Similarity.
        
        Args:
            predictions: Predicted keypoints
            ground_truth: Ground truth keypoints
            sigmas: Per-keypoint standard deviations
            **kwargs: Additional parameters
            
        Returns:
            OKS results dictionary
        """
        calculator = OKSCalculator(sigmas=sigmas)
        return calculator.calculate(predictions, ground_truth, **kwargs)