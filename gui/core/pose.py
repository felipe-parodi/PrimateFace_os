"""Unified pose estimation module supporting multiple frameworks.

This module provides consolidated pose estimation with support for
MMPose, Ultralytics, and future frameworks. Includes SAM masking
and flexible keypoint/skeleton handling.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm

from ..constants import (
    DEFAULT_DEVICE,
    DEFAULT_KPT_THR,
    DEFAULT_MASK_KPT_THR,
    DEFAULT_MIN_KEYPOINTS,
)
from .models import FrameworkType, ModelManager


class PoseEstimator:
    """Unified pose estimation supporting multiple frameworks.
    
    This class provides pose estimation capabilities with support for
    SAM mask projection and flexible keypoint/skeleton formats based
    on the model or COCO JSON configuration.
    
    Attributes:
        framework: Pose estimation framework type.
        model: Loaded pose model.
        metadata: Model metadata including keypoints and skeleton.
        kpt_thr: Keypoint confidence threshold.
        min_keypoints: Minimum number of valid keypoints required.
        device: Device for inference.
        keypoint_names: Names of keypoints from model or JSON.
        skeleton_links: Skeleton connectivity from model or JSON.
    """
    
    def __init__(
        self,
        framework: Union[str, FrameworkType] = "mmpose",
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: str = DEFAULT_DEVICE,
        kpt_thr: float = DEFAULT_KPT_THR,
        min_keypoints: int = DEFAULT_MIN_KEYPOINTS,
        model_manager: Optional[ModelManager] = None,
        coco_metadata: Optional[Dict] = None
    ) -> None:
        """Initialize PoseEstimator.
        
        Args:
            framework: Framework type for pose estimation.
            config_path: Path to model configuration.
            checkpoint_path: Path to model checkpoint.
            device: Device for inference.
            kpt_thr: Keypoint confidence threshold.
            min_keypoints: Minimum valid keypoints required.
            model_manager: Optional ModelManager for caching.
            coco_metadata: Optional COCO metadata for keypoints/skeleton.
        """
        self.framework = FrameworkType(framework.lower()) if isinstance(framework, str) else framework
        self.kpt_thr = kpt_thr
        self.min_keypoints = min_keypoints
        self.device = device
        
        if model_manager is None:
            model_manager = ModelManager()
        
        self.model, self.metadata = model_manager.load_model(
            self.framework,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=device,
            task="pose" if self.framework == FrameworkType.ULTRALYTICS else None
        )
        
        self._setup_keypoint_info(coco_metadata)
    
    def _load_keypoint_definitions(self) -> Dict:
        """Loads predefined keypoint sets from a YAML file."""
        definitions_path = Path(__file__).parent.parent / "keypoint_definitions.yml"
        if not definitions_path.exists():
            return {}
        with open(definitions_path, 'r') as f:
            return yaml.safe_load(f)

    def _setup_keypoint_info(self, coco_metadata: Optional[Dict]) -> None:
        """Setup keypoint names and skeleton from model or COCO metadata."""
        if coco_metadata and "keypoints" in coco_metadata:
            self.keypoint_names = coco_metadata["keypoints"]
            self.skeleton_links = coco_metadata.get("skeleton", [])
            return

        if self.framework == FrameworkType.MMPOSE:
            try:
                # Get the number of keypoints from the model's config
                num_kpts = self.model.cfg.model.head.out_channels
                
                # Load predefined keypoint sets
                definitions = self._load_keypoint_definitions()
                
                if num_kpts in definitions:
                    config = definitions[num_kpts]
                    self.keypoint_names = config.get('keypoint_names', [])
                    self.skeleton_links = config.get('skeleton_links', [])
                    return
            except AttributeError:
                pass  # Fallback to other methods if config structure is unexpected

        if self.metadata:
            if self.framework == FrameworkType.MMPOSE:
                self.keypoint_names = self.metadata.get("keypoint_names", [])
                self.skeleton_links = self.metadata.get("skeleton_links", [])
            elif self.framework == FrameworkType.ULTRALYTICS:
                num_kpts = self.metadata.get("num_keypoints", 17)
                self.keypoint_names = [f"kpt_{i}" for i in range(num_kpts)]
                self.skeleton_links = []
        else:
            self.keypoint_names = []
            self.skeleton_links = []
    
    def estimate_pose(
        self,
        frame: np.ndarray,
        bboxes: List[List[float]],
        return_named: bool = False,
        composite_mask: Optional[np.ndarray] = None,
        mask_kpt_thr: float = DEFAULT_MASK_KPT_THR
    ) -> List[Dict[str, Any]]:
        """Estimate poses for given bounding boxes.
        
        Args:
            frame: Input image as numpy array (BGR).
            bboxes: List of bounding boxes [x1, y1, x2, y2].
            return_named: Whether to return named keypoint format.
            composite_mask: Optional SAM mask for keypoint projection.
            mask_kpt_thr: Threshold for mask-based keypoint filtering.
            
        Returns:
            List of pose annotations with keypoints.
        """
        if not bboxes:
            return []
        
        if self.framework == FrameworkType.MMPOSE:
            return self._estimate_mmpose(
                frame, bboxes, return_named, composite_mask, mask_kpt_thr
            )
        elif self.framework == FrameworkType.ULTRALYTICS:
            return self._estimate_ultralytics(
                frame, bboxes, return_named, composite_mask, mask_kpt_thr
            )
        else:
            raise ValueError(f"Pose estimation not supported for {self.framework}")
    
    def _estimate_mmpose(
        self,
        frame: np.ndarray,
        bboxes: List[List[float]],
        return_named: bool,
        composite_mask: Optional[np.ndarray],
        mask_kpt_thr: float
    ) -> List[Dict[str, Any]]:
        """Estimate poses using MMPose."""
        from mmpose.apis import inference_topdown
        from mmpose.structures import merge_data_samples
        
        bboxes_xyxy = np.array(bboxes, dtype=np.float32)
        
        batch_results = inference_topdown(self.model, frame, bboxes_xyxy)
        results = merge_data_samples(batch_results)
        
        if not hasattr(results, 'pred_instances'):
            return []
        
        keypoints = results.pred_instances.keypoints
        keypoint_scores = results.pred_instances.keypoint_scores
        
        annotations = []
        for i, (kpts, scores) in enumerate(zip(keypoints, keypoint_scores)):
            kpts_with_scores = self._process_keypoints(
                kpts, scores, composite_mask, mask_kpt_thr
            )
            
            valid_kpts = (kpts_with_scores[:, 2] > 0).sum()
            if valid_kpts < self.min_keypoints:
                continue
            
            ann = {
                "bbox": bboxes[i],
                "keypoints": kpts_with_scores.flatten().tolist(),
                "num_keypoints": int(valid_kpts)
            }
            
            if return_named and self.keypoint_names:
                ann["keypoints_named"] = self._create_named_keypoints(kpts_with_scores)
            
            annotations.append(ann)
        
        return annotations
    
    def _estimate_ultralytics(
        self,
        frame: np.ndarray,
        bboxes: List[List[float]],
        return_named: bool,
        composite_mask: Optional[np.ndarray],
        mask_kpt_thr: float
    ) -> List[Dict[str, Any]]:
        """Estimate poses using Ultralytics YOLO."""
        results = self.model(frame)
        
        if not results or not hasattr(results[0], 'keypoints'):
            return []
        
        keypoints = results[0].keypoints
        if keypoints is None:
            return []
        
        kpts_xy = keypoints.xy.cpu().numpy()
        kpts_conf = keypoints.conf.cpu().numpy()
        
        annotations = []
        for i, bbox in enumerate(bboxes):
            if i >= len(kpts_xy):
                break
            
            kpts = kpts_xy[i]
            scores = kpts_conf[i]
            
            kpts_with_scores = np.concatenate([
                kpts,
                scores.reshape(-1, 1)
            ], axis=1)
            
            kpts_with_scores = self._process_keypoints(
                kpts[:, :2], scores, composite_mask, mask_kpt_thr
            )
            
            valid_kpts = (kpts_with_scores[:, 2] > 0).sum()
            if valid_kpts < self.min_keypoints:
                continue
            
            ann = {
                "bbox": bbox,
                "keypoints": kpts_with_scores.flatten().tolist(),
                "num_keypoints": int(valid_kpts)
            }
            
            if return_named and self.keypoint_names:
                ann["keypoints_named"] = self._create_named_keypoints(kpts_with_scores)
            
            annotations.append(ann)
        
        return annotations
    
    def _process_keypoints(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        composite_mask: Optional[np.ndarray],
        mask_kpt_thr: float
    ) -> np.ndarray:
        """Process keypoints with confidence filtering and mask projection.
        
        Args:
            keypoints: Keypoint coordinates (N, 2).
            scores: Keypoint confidence scores (N,).
            composite_mask: Optional SAM mask.
            mask_kpt_thr: Threshold for mask filtering.
            
        Returns:
            Processed keypoints with visibility (N, 3).
        """
        kpts_with_scores = np.zeros((len(keypoints), 3))
        kpts_with_scores[:, :2] = keypoints
        
        for j, score in enumerate(scores):
            if score > self.kpt_thr:
                visibility = 2
                
                if composite_mask is not None:
                    x, y = int(keypoints[j, 0]), int(keypoints[j, 1])
                    if 0 <= y < composite_mask.shape[0] and 0 <= x < composite_mask.shape[1]:
                        if composite_mask[y, x] < mask_kpt_thr * 255:
                            visibility = 1
                
                kpts_with_scores[j, 2] = visibility
        
        return kpts_with_scores
    
    def _create_named_keypoints(self, kpts_with_scores: np.ndarray) -> Dict[str, Dict]:
        """Create named keypoint dictionary.
        
        Args:
            kpts_with_scores: Keypoints with visibility (N, 3).
            
        Returns:
            Dictionary mapping keypoint names to coordinates.
        """
        named_kpts = {}
        for j, name in enumerate(self.keypoint_names):
            if j < len(kpts_with_scores):
                named_kpts[name] = {
                    "x": float(kpts_with_scores[j, 0]),
                    "y": float(kpts_with_scores[j, 1]),
                    "visibility": int(kpts_with_scores[j, 2])
                }
        return named_kpts
    
    def process_frame(
        self,
        frame: np.ndarray,
        bboxes: List[List[float]],
        img_id: int,
        ann_id_start: int,
        composite_mask: Optional[np.ndarray] = None,
        mask_kpt_thr: float = DEFAULT_MASK_KPT_THR,
        return_named: bool = True
    ) -> Tuple[List[Dict], int]:
        """Process single frame for COCO format annotations.
        
        Args:
            frame: Input image.
            bboxes: Detection bounding boxes.
            img_id: COCO image ID.
            ann_id_start: Starting annotation ID.
            composite_mask: Optional SAM mask.
            mask_kpt_thr: Mask threshold.
            return_named: Whether to include named keypoints.
            
        Returns:
            Tuple of (annotations, next_ann_id).
        """
        pose_results = self.estimate_pose(
            frame, bboxes, return_named, composite_mask, mask_kpt_thr
        )
        
        annotations = []
        ann_id = ann_id_start
        
        for pose in pose_results:
            bbox = pose["bbox"]
            area = bbox[2] * bbox[3] if len(bbox) == 4 else (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            ann = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": bbox,
                "area": float(area),
                "keypoints": pose["keypoints"],
                "num_keypoints": pose["num_keypoints"],
                "iscrowd": 0
            }
            
            if "keypoints_named" in pose:
                ann["keypoints_named"] = pose["keypoints_named"]
            
            annotations.append(ann)
            ann_id += 1
        
        return annotations, ann_id
    
    def process_directory(
        self,
        img_dir: Union[str, Path],
        detections: Dict[str, List[List[float]]],
        return_named: bool = True
    ) -> Dict[str, List[Dict]]:
        """Process directory of images with detections.
        
        Args:
            img_dir: Path to image directory.
            detections: Dictionary of image paths to bboxes.
            return_named: Whether to include named keypoints.
            
        Returns:
            Dictionary mapping image paths to pose annotations.
        """
        img_dir = Path(img_dir)
        results = {}
        
        for img_path, bboxes in tqdm(detections.items(), desc="Estimating poses"):
            if not bboxes:
                results[img_path] = []
                continue
            
            full_path = img_dir / Path(img_path).name if not Path(img_path).is_absolute() else Path(img_path)
            frame = cv2.imread(str(full_path))
            
            if frame is None:
                results[img_path] = []
                continue
            
            poses = self.estimate_pose(frame, bboxes, return_named)
            results[img_path] = poses
        
        return results
    
    def update_thresholds(
        self,
        kpt_thr: Optional[float] = None,
        min_keypoints: Optional[int] = None
    ) -> None:
        """Update pose estimation thresholds.
        
        Args:
            kpt_thr: New keypoint confidence threshold.
            min_keypoints: New minimum keypoints requirement.
        """
        if kpt_thr is not None:
            self.kpt_thr = kpt_thr
        if min_keypoints is not None:
            self.min_keypoints = min_keypoints
    
    def get_model_info(self) -> Dict:
        """Get information about loaded model.
        
        Returns:
            Dictionary with model information.
        """
        return {
            "framework": self.framework.value,
            "device": self.device,
            "kpt_thr": self.kpt_thr,
            "min_keypoints": self.min_keypoints,
            "num_keypoints": len(self.keypoint_names),
            "keypoint_names": self.keypoint_names,
            "skeleton_links": self.skeleton_links,
            "metadata": self.metadata
        }