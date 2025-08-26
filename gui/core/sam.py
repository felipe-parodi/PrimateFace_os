"""SAM (Segment Anything Model) masking utilities.

This module provides integration with SAM for improved keypoint projection
and instance segmentation.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from .models import FrameworkType, ModelManager


class SAMMasker:
    """SAM mask generator for improved pose estimation.
    
    This class provides SAM-based mask generation to improve keypoint
    localization by projecting keypoints onto instance masks.
    
    Attributes:
        model: SAM model instance.
        predictor: SAM predictor for original implementation.
        use_ultralytics: Whether using Ultralytics SAM.
        model_type: Type of SAM model.
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_type: str = "vit_h",
        device: str = "cuda:0",
        model_manager: Optional[ModelManager] = None
    ) -> None:
        """Initialize SAMMasker.
        
        Args:
            checkpoint_path: Path to SAM checkpoint.
            model_type: SAM model type ('vit_b', 'vit_l', 'vit_h').
            device: Device to run model on.
            model_manager: Optional ModelManager for caching.
            
        Raises:
            ImportError: If SAM is not installed.
        """
        self.model_type = model_type
        self.device = device
        
        if model_manager is None:
            model_manager = ModelManager()
        
        try:
            from ultralytics import SAM
            if checkpoint_path:
                self.model = SAM(checkpoint_path)
            else:
                self.model = SAM(f'sam_{model_type}.pt')
            self.use_ultralytics = True
            self.predictor = None
        except ImportError:
            try:
                if not checkpoint_path:
                    raise ValueError(
                        "checkpoint_path required for original SAM implementation"
                    )
                self.predictor, _ = model_manager.load_model(
                    FrameworkType.SAM,
                    checkpoint_path=checkpoint_path,
                    device=device,
                    model_type=model_type
                )
                self.use_ultralytics = False
                self.model = None
            except ImportError as e:
                raise ImportError(
                    "SAM is required for mask generation. Install with: "
                    "pip install segment-anything ultralytics"
                ) from e
    
    def generate_masks(
        self,
        image: np.ndarray,
        bboxes: np.ndarray
    ) -> np.ndarray:
        """Generate masks for multiple bounding boxes.
        
        Args:
            image: Input image (H, W, C).
            bboxes: Array of bounding boxes (N, 4) in xyxy format.
            
        Returns:
            Array of binary masks (N, H, W).
        """
        if bboxes.size == 0:
            return np.zeros((0, image.shape[0], image.shape[1]), dtype=np.uint8)
        
        masks = []
        
        if not self.use_ultralytics and self.predictor:
            self.predictor.set_image(image)
        
        for bbox in bboxes:
            mask = self.generate_mask(image, bbox)
            masks.append(mask)
        
        return np.array(masks, dtype=np.uint8)
    
    def generate_mask(
        self,
        image: np.ndarray,
        bbox: Union[List[float], np.ndarray],
        point_prompts: Optional[List[Tuple[int, int]]] = None
    ) -> np.ndarray:
        """Generate mask for a single bounding box.
        
        Args:
            image: Input image (H, W, C).
            bbox: Bounding box [x1, y1, x2, y2].
            point_prompts: Optional list of (x, y) points.
            
        Returns:
            Binary mask array (H, W).
        """
        if self.use_ultralytics:
            return self._generate_mask_ultralytics(image, bbox, point_prompts)
        else:
            return self._generate_mask_original(image, bbox, point_prompts)
    
    def _generate_mask_ultralytics(
        self,
        image: np.ndarray,
        bbox: Union[List[float], np.ndarray],
        point_prompts: Optional[List[Tuple[int, int]]] = None
    ) -> np.ndarray:
        """Generate mask using Ultralytics SAM."""
        x1, y1, x2, y2 = bbox
        
        results = self.model(image, bboxes=[[x1, y1, x2, y2]])
        
        if results and len(results[0].masks) > 0:
            mask = results[0].masks.data[0].cpu().numpy()
            
            if mask.shape != image.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.float32),
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )
            
            return (mask > 0.5).astype(np.uint8)
        else:
            return np.zeros(image.shape[:2], dtype=np.uint8)
    
    def _generate_mask_original(
        self,
        image: np.ndarray,
        bbox: Union[List[float], np.ndarray],
        point_prompts: Optional[List[Tuple[int, int]]] = None
    ) -> np.ndarray:
        """Generate mask using original SAM implementation."""
        sam_bbox = np.array(bbox)
        
        if point_prompts:
            point_coords = np.array(point_prompts)
            point_labels = np.ones(len(point_prompts))
        else:
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            point_coords = np.array([[cx, cy]])
            point_labels = np.array([1])
        
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=sam_bbox,
            multimask_output=False
        )
        
        return masks[0].astype(np.uint8)
    
    @staticmethod
    def composite_mask(
        masks: np.ndarray,
        img_hw: Tuple[int, int]
    ) -> np.ndarray:
        """Create composite mask from multiple masks.
        
        Args:
            masks: Array of masks (N, H, W) or empty.
            img_hw: Image height and width.
            
        Returns:
            Composite mask (H, W) with values 0-255.
        """
        if masks is None or masks.size == 0:
            return np.zeros(img_hw, dtype=np.uint8)
        
        composite = (np.any(masks > 0, axis=0)).astype(np.uint8) * 255
        return composite
    
    @staticmethod
    def project_keypoints_to_mask(
        keypoints: np.ndarray,
        scores: np.ndarray,
        mask: np.ndarray,
        threshold: float = 0.3
    ) -> np.ndarray:
        """Project keypoints to nearest mask point if outside.
        
        Args:
            keypoints: Keypoint coordinates (N, 2 or 3).
            scores: Keypoint scores (N,).
            mask: Binary mask (H, W).
            threshold: Score threshold for projection.
            
        Returns:
            Projected keypoint coordinates.
        """
        if mask.ndim != 2 or mask.size == 0 or np.count_nonzero(mask) == 0:
            return keypoints
        
        H, W = mask.shape
        ys_mask, xs_mask = np.where(mask > 0)
        
        if ys_mask.size == 0:
            return keypoints
        
        projected = keypoints.copy()
        
        for i in range(projected.shape[0]):
            if scores[i] < threshold:
                continue
            
            x, y = float(projected[i, 0]), float(projected[i, 1])
            xi, yi = int(round(x)), int(round(y))
            
            if 0 <= yi < H and 0 <= xi < W and mask[yi, xi] > 0:
                continue
            
            dy = ys_mask.astype(np.int32) - yi
            dx = xs_mask.astype(np.int32) - xi
            distances = dx * dx + dy * dy
            idx = int(np.argmin(distances))
            
            projected[i, 0] = float(xs_mask[idx])
            projected[i, 1] = float(ys_mask[idx])
        
        return projected
    
    def refine_bbox_with_mask(
        self,
        bbox: List[float],
        mask: np.ndarray,
        padding: int = 5
    ) -> List[float]:
        """Refine bounding box based on mask.
        
        Args:
            bbox: Original bounding box [x1, y1, x2, y2].
            mask: Binary mask.
            padding: Padding to add around mask bbox.
            
        Returns:
            Refined bounding box.
        """
        if mask.sum() == 0:
            return bbox
        
        ys, xs = np.where(mask > 0)
        
        x1 = max(0, xs.min() - padding)
        y1 = max(0, ys.min() - padding)
        x2 = min(mask.shape[1] - 1, xs.max() + padding)
        y2 = min(mask.shape[0] - 1, ys.max() + padding)
        
        return [float(x1), float(y1), float(x2), float(y2)]
    
    def process_detections(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        refine_bboxes: bool = True,
        generate_composite: bool = True
    ) -> Tuple[List[List[float]], Optional[np.ndarray], List[np.ndarray]]:
        """Process detections with SAM masks.
        
        Args:
            image: Input image.
            bboxes: Detection bounding boxes.
            refine_bboxes: Whether to refine bboxes with masks.
            generate_composite: Whether to generate composite mask.
            
        Returns:
            Tuple of (refined_bboxes, composite_mask, individual_masks).
        """
        if not bboxes:
            return bboxes, None, []
        
        bboxes_np = np.array(bboxes, dtype=np.float32)
        masks = self.generate_masks(image, bboxes_np)
        
        refined_bboxes = bboxes
        if refine_bboxes and masks.size > 0:
            refined_bboxes = []
            for i, bbox in enumerate(bboxes):
                if i < len(masks):
                    refined_bbox = self.refine_bbox_with_mask(bbox, masks[i])
                    refined_bboxes.append(refined_bbox)
                else:
                    refined_bboxes.append(bbox)
        
        composite = None
        if generate_composite and masks.size > 0:
            composite = self.composite_mask(masks, image.shape[:2])
        
        return refined_bboxes, composite, list(masks)
    
    @staticmethod
    def is_available() -> bool:
        """Check if SAM is available.
        
        Returns:
            True if SAM is installed and available.
        """
        try:
            from ultralytics import SAM
            return True
        except ImportError:
            try:
                from segment_anything import sam_model_registry
                return True
            except ImportError:
                return False