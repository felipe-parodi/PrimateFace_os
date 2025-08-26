"""Unified detection module supporting multiple frameworks.

This module provides a consolidated Detector class that handles
detection across images and videos with batch processing support.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm

from ..constants import (
    DEFAULT_BBOX_THR,
    DEFAULT_DEVICE,
    DEFAULT_MAX_MONKEYS,
    DEFAULT_NMS_THR,
)
from .models import FrameworkType, ModelManager


class Detector:
    """Unified detector supporting multiple frameworks.
    
    This class provides detection capabilities for both single frames
    and batch processing, with support for NMS and confidence thresholding.
    Works with MMDetection, Ultralytics, and future frameworks.
    
    Attributes:
        framework: Detection framework type.
        model: Loaded detection model.
        metadata: Model metadata including classes.
        bbox_thr: Bounding box confidence threshold.
        nms_thr: Non-maximum suppression threshold.
        max_instances: Maximum number of instances per image.
        device: Device for inference.
    """
    
    def __init__(
        self,
        framework: Union[str, FrameworkType] = "mmdet",
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: str = DEFAULT_DEVICE,
        bbox_thr: float = DEFAULT_BBOX_THR,
        nms_thr: float = DEFAULT_NMS_THR,
        max_instances: int = DEFAULT_MAX_MONKEYS,
        model_manager: Optional[ModelManager] = None
    ) -> None:
        """Initialize Detector.
        
        Args:
            framework: Framework type for detection.
            config_path: Path to model configuration.
            checkpoint_path: Path to model checkpoint.
            device: Device for inference.
            bbox_thr: Confidence threshold for bounding boxes.
            nms_thr: IoU threshold for NMS.
            max_instances: Maximum instances to detect per image.
            model_manager: Optional ModelManager for caching.
        """
        self.framework = FrameworkType(framework.lower()) if isinstance(framework, str) else framework
        self.bbox_thr = bbox_thr
        self.nms_thr = nms_thr
        self.max_instances = max_instances
        self.device = device
        
        if model_manager is None:
            model_manager = ModelManager()
        
        self.model, self.metadata = model_manager.load_model(
            self.framework,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=device
        )
    
    def detect_frame(
        self, 
        frame: np.ndarray,
        return_scores: bool = False
    ) -> Union[List[List[float]], Tuple[List[List[float]], List[float]]]:
        """Detect objects in a single frame.
        
        Args:
            frame: Input image as numpy array (BGR).
            return_scores: Whether to return confidence scores.
            
        Returns:
            List of bounding boxes [x1, y1, x2, y2] or tuple with scores.
        """
        if self.framework == FrameworkType.MMDET:
            return self._detect_mmdet(frame, return_scores)
        elif self.framework == FrameworkType.ULTRALYTICS:
            return self._detect_ultralytics(frame, return_scores)
        else:
            raise ValueError(f"Detection not supported for {self.framework}")
    
    def _detect_mmdet(
        self,
        frame: np.ndarray,
        return_scores: bool
    ) -> Union[List[List[float]], Tuple[List[List[float]], List[float]]]:
        """Detect using MMDetection."""
        from mmdet.apis import inference_detector
        from mmpose.evaluation.functional import nms
        
        results = inference_detector(self.model, frame)
        
        if len(results.pred_instances) == 0:
            return ([], []) if return_scores else []
        
        if len(results.pred_instances) > self.max_instances:
            results.pred_instances = results.pred_instances[:self.max_instances]
        
        pred_instance = results.pred_instances.cpu().numpy()
        bboxes = pred_instance.bboxes
        scores = pred_instance.scores
        labels = pred_instance.labels
        
        bboxes_with_scores = np.concatenate((bboxes, scores[:, None]), axis=1)
        valid_indices = np.where((labels == 0) & (scores > self.bbox_thr))[0]
        bboxes_with_scores = bboxes_with_scores[valid_indices]
        
        if len(bboxes_with_scores) == 0:
            return ([], []) if return_scores else []
        
        keep_indices = nms(bboxes_with_scores, self.nms_thr)
        final_bboxes = bboxes_with_scores[keep_indices, :4].tolist()
        final_scores = bboxes_with_scores[keep_indices, 4].tolist()
        
        if len(final_bboxes) > self.max_instances:
            final_bboxes = final_bboxes[:self.max_instances]
            final_scores = final_scores[:self.max_instances]
        
        return (final_bboxes, final_scores) if return_scores else final_bboxes
    
    def _detect_ultralytics(
        self,
        frame: np.ndarray,
        return_scores: bool
    ) -> Union[List[List[float]], Tuple[List[List[float]], List[float]]]:
        """Detect using Ultralytics YOLO."""
        results = self.model(frame, conf=self.bbox_thr, iou=self.nms_thr)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return ([], []) if return_scores else []
        
        boxes = results[0].boxes
        bboxes = boxes.xyxy.cpu().numpy().tolist()
        scores = boxes.conf.cpu().numpy().tolist()
        
        if len(bboxes) > self.max_instances:
            bboxes = bboxes[:self.max_instances]
            scores = scores[:self.max_instances]
        
        return (bboxes, scores) if return_scores else bboxes
    
    def detect_batch(
        self,
        frames: List[np.ndarray],
        return_scores: bool = False
    ) -> List[Union[List[List[float]], Tuple[List[List[float]], List[float]]]]:
        """Detect objects in batch of frames.
        
        Args:
            frames: List of input images.
            return_scores: Whether to return confidence scores.
            
        Returns:
            List of detection results per frame.
        """
        results = []
        for frame in frames:
            results.append(self.detect_frame(frame, return_scores))
        return results
    
    def detect_directory(
        self,
        img_dir: Union[str, Path],
        extensions: Optional[List[str]] = None,
        return_paths: bool = True
    ) -> Dict[str, List[List[float]]]:
        """Run detection on all images in directory.
        
        Args:
            img_dir: Path to image directory.
            extensions: Image file extensions to process.
            return_paths: Whether to use full paths as keys.
            
        Returns:
            Dictionary mapping image paths/names to detections.
        """
        img_dir = Path(img_dir)
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        image_files = []
        for ext in extensions:
            image_files.extend(img_dir.glob(f"*{ext}"))
            image_files.extend(img_dir.glob(f"*{ext.upper()}"))
        
        if not image_files:
            raise ValueError(f"No valid images found in: {img_dir}")
        
        results = {}
        for img_path in tqdm(image_files, desc="Running detection"):
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            
            detections = self.detect_frame(frame)
            key = str(img_path) if return_paths else img_path.name
            results[key] = detections
        
        return results
    
    def process_video(
        self,
        video_path: Union[str, Path],
        frame_interval: int = 1,
        max_frames: Optional[int] = None,
        callback: Optional[callable] = None
    ) -> List[Tuple[int, List[List[float]]]]:
        """Process video file for detections.
        
        Args:
            video_path: Path to video file.
            frame_interval: Process every Nth frame.
            max_frames: Maximum frames to process.
            callback: Optional callback for each frame result.
            
        Returns:
            List of (frame_index, detections) tuples.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        results = []
        frame_idx = 0
        
        pbar = tqdm(total=total_frames, desc="Processing video")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                detections = self.detect_frame(frame)
                results.append((frame_idx, detections))
                
                if callback:
                    callback(frame_idx, frame, detections)
            
            frame_idx += 1
            pbar.update(1)
            
            if max_frames and frame_idx >= max_frames:
                break
        
        cap.release()
        pbar.close()
        
        return results
    
    def update_thresholds(
        self,
        bbox_thr: Optional[float] = None,
        nms_thr: Optional[float] = None,
        max_instances: Optional[int] = None
    ) -> None:
        """Update detection thresholds.
        
        Args:
            bbox_thr: New confidence threshold.
            nms_thr: New NMS IoU threshold.
            max_instances: New maximum instances.
        """
        if bbox_thr is not None:
            self.bbox_thr = bbox_thr
        if nms_thr is not None:
            self.nms_thr = nms_thr
        if max_instances is not None:
            self.max_instances = max_instances
    
    def warmup(self, size: Tuple[int, int] = (640, 480)) -> None:
        """Warmup model with dummy inference.
        
        Args:
            size: Size of dummy image (width, height).
        """
        dummy_frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        _ = self.detect_frame(dummy_frame)
    
    def get_model_info(self) -> Dict:
        """Get information about loaded model.
        
        Returns:
            Dictionary with model information.
        """
        return {
            "framework": self.framework.value,
            "device": self.device,
            "bbox_thr": self.bbox_thr,
            "nms_thr": self.nms_thr,
            "max_instances": self.max_instances,
            "metadata": self.metadata
        }