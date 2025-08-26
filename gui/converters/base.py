"""Base COCO converter with flexible skeleton handling.

This module provides the base class for converting various input formats
to COCO annotation format with dynamic keypoint/skeleton configuration.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..core import Detector, PoseEstimator, SAMMasker
from ..utils import load_coco_categories


class COCOConverter:
    """Base converter for creating COCO-format annotations.
    
    This class provides shared functionality for converting detection
    and pose estimation results to COCO format with flexible skeleton
    handling from model configuration or existing COCO JSON.
    
    Attributes:
        detector: Detector instance for object detection.
        pose_estimator: Optional PoseEstimator for pose estimation.
        sam_masker: Optional SAMMasker for mask generation.
        categories: COCO category definitions with keypoints/skeleton.
        output_dir: Directory for saving outputs.
    """
    
    def __init__(
        self,
        detector: Detector,
        pose_estimator: Optional[PoseEstimator] = None,
        sam_masker: Optional[SAMMasker] = None,
        output_dir: Optional[Union[str, Path]] = None,
        coco_template: Optional[Union[str, Path]] = None
    ) -> None:
        """Initialize COCOConverter.
        
        Args:
            detector: Detector instance.
            pose_estimator: Optional PoseEstimator instance.
            sam_masker: Optional SAMMasker instance.
            output_dir: Directory for outputs.
            coco_template: Optional COCO JSON to copy skeleton from.
        """
        self.detector = detector
        self.pose_estimator = pose_estimator
        self.sam_masker = sam_masker
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        
        self.categories = self._setup_categories(coco_template)
        self._image_id_counter = 0
        self._annotation_id_counter = 0
    
    def _setup_categories(
        self,
        coco_template: Optional[Union[str, Path]]
    ) -> List[Dict[str, Any]]:
        """Setup COCO categories with flexible skeleton.
        
        Args:
            coco_template: Optional COCO JSON template.
            
        Returns:
            List of category definitions.
        """
        if coco_template and Path(coco_template).exists():
            keypoint_names, skeleton_links = load_coco_categories(coco_template)
        elif self.pose_estimator:
            keypoint_names = self.pose_estimator.keypoint_names
            skeleton_links = self.pose_estimator.skeleton_links
        else:
            keypoint_names = []
            skeleton_links = []
        
        category = {
            "id": 1,
            "name": "primate",
            "supercategory": "animal"
        }
        
        if keypoint_names:
            category["keypoints"] = keypoint_names
            category["skeleton"] = skeleton_links
        
        return [category]
    
    def create_image_info(
        self,
        file_name: str,
        width: int,
        height: int,
        image_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create COCO image information.
        
        Args:
            file_name: Image file name.
            width: Image width.
            height: Image height.
            image_id: Optional specific image ID.
            
        Returns:
            COCO image info dictionary.
        """
        if image_id is None:
            image_id = self._image_id_counter
            self._image_id_counter += 1
        
        return {
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height,
            "date_captured": datetime.now().isoformat(),
            "license": 1,
            "coco_url": "",
            "flickr_url": ""
        }
    
    def create_annotation(
        self,
        image_id: int,
        bbox: List[float],
        keypoints: Optional[List[float]] = None,
        segmentation: Optional[Any] = None,
        annotation_id: Optional[int] = None,
        category_id: int = 1,
        score: Optional[float] = None
    ) -> Dict[str, Any]:
        """Create COCO annotation.
        
        Args:
            image_id: Associated image ID.
            bbox: Bounding box [x, y, w, h] or [x1, y1, x2, y2].
            keypoints: Optional keypoint list.
            segmentation: Optional segmentation data.
            annotation_id: Optional specific annotation ID.
            category_id: Category ID (default 1).
            score: Optional confidence score.
            
        Returns:
            COCO annotation dictionary.
        """
        if annotation_id is None:
            annotation_id = self._annotation_id_counter
            self._annotation_id_counter += 1
        
        if len(bbox) == 4 and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            x, y, x2, y2 = bbox
            w, h = x2 - x, y2 - y
            bbox = [x, y, w, h]
        
        area = float(bbox[2] * bbox[3]) if len(bbox) == 4 else 0.0
        
        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox,
            "area": area,
            "iscrowd": 0
        }
        
        if keypoints is not None:
            annotation["keypoints"] = keypoints
            num_keypoints = sum(1 for i in range(2, len(keypoints), 3) if keypoints[i] > 0)
            annotation["num_keypoints"] = num_keypoints
        
        if segmentation is not None:
            annotation["segmentation"] = segmentation
        
        if score is not None:
            annotation["score"] = score
        
        return annotation
    
    def process_image(
        self,
        image: np.ndarray,
        image_name: str,
        return_visualizations: bool = False
    ) -> Tuple[Dict, List[Dict], Optional[np.ndarray]]:
        """Process single image through detection and pose pipeline.
        
        Args:
            image: Input image array.
            image_name: Image file name.
            return_visualizations: Whether to return visualization.
            
        Returns:
            Tuple of (image_info, annotations, visualization).
        """
        h, w = image.shape[:2]
        image_info = self.create_image_info(image_name, w, h)
        
        bboxes, scores = self.detector.detect_frame(image, return_scores=True)
        
        annotations = []
        composite_mask = None
        
        if self.sam_masker and bboxes:
            bboxes, composite_mask, _ = self.sam_masker.process_detections(
                image, bboxes, refine_bboxes=True, generate_composite=True
            )
        
        if self.pose_estimator and bboxes:
            pose_results = self.pose_estimator.estimate_pose(
                image, bboxes, return_named=False,
                composite_mask=composite_mask
            )
            
            for i, (bbox, score) in enumerate(zip(bboxes, scores)):
                if i < len(pose_results):
                    annotation = self.create_annotation(
                        image_info["id"],
                        bbox,
                        keypoints=pose_results[i]["keypoints"],
                        score=score
                    )
                else:
                    annotation = self.create_annotation(
                        image_info["id"],
                        bbox,
                        score=score
                    )
                annotations.append(annotation)
        else:
            for bbox, score in zip(bboxes, scores):
                annotation = self.create_annotation(
                    image_info["id"],
                    bbox,
                    score=score
                )
                annotations.append(annotation)
        
        visualization = None
        if return_visualizations:
            from ..utils import visualize_skeleton
            visualization = self._create_visualization(
                image, annotations
            )
        
        return image_info, annotations, visualization
    
    def _create_visualization(
        self,
        image: np.ndarray,
        annotations: List[Dict]
    ) -> np.ndarray:
        """Create visualization with bboxes and skeletons.
        
        Args:
            image: Input image.
            annotations: List of annotations.
            
        Returns:
            Visualized image.
        """
        import cv2
        vis_image = image.copy()
        
        for ann in annotations:
            bbox = ann["bbox"]
            if len(bbox) == 4:
                x, y, w, h = bbox
                x2, y2 = x + w, y + h
            else:
                x, y, x2, y2 = bbox
            
            cv2.rectangle(
                vis_image,
                (int(x), int(y)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )
            
            if "keypoints" in ann and self.categories[0].get("skeleton"):
                kpts_flat = ann["keypoints"]
                kpts = np.array(kpts_flat).reshape(-1, 3)
                
                from ..utils import visualize_skeleton
                vis_image = visualize_skeleton(
                    vis_image,
                    kpts,
                    self.categories[0]["skeleton"],
                    self.categories[0].get("keypoints")
                )
        
        return vis_image
    
    def create_coco_dataset(
        self,
        images: List[Dict],
        annotations: List[Dict],
        info: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Create complete COCO dataset dictionary.
        
        Args:
            images: List of image info dictionaries.
            annotations: List of annotation dictionaries.
            info: Optional dataset info.
            
        Returns:
            Complete COCO dataset dictionary.
        """
        if info is None:
            info = {
                "description": "PrimateFace Dataset",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "PrimateFace",
                "date_created": datetime.now().isoformat()
            }
        
        return {
            "info": info,
            "licenses": [
                {
                    "id": 1,
                    "name": "MIT License",
                    "url": "https://opensource.org/licenses/MIT"
                }
            ],
            "categories": self.categories,
            "images": images,
            "annotations": annotations
        }
    
    def save_coco_json(
        self,
        coco_data: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None
    ) -> str:
        """Save COCO data to JSON file.
        
        Args:
            coco_data: COCO dataset dictionary.
            output_path: Output file path.
            
        Returns:
            Path to saved file.
        """
        if output_path is None:
            output_path = self.output_dir / "annotations.json"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"Saved COCO annotations to {output_path}")
        return str(output_path)
    
    def reset_counters(self) -> None:
        """Reset image and annotation ID counters."""
        self._image_id_counter = 0
        self._annotation_id_counter = 0