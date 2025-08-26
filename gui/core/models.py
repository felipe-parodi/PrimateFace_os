"""Model management for multiple detection and pose frameworks.

This module provides a unified interface for loading and caching models
from different frameworks including MMDetection, MMPose, Ultralytics,
DeepLabCut, and SLEAP.
"""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch


class FrameworkType(Enum):
    """Supported model frameworks."""
    MMDET = "mmdet"
    MMPOSE = "mmpose"
    ULTRALYTICS = "ultralytics"
    SAM = "sam"
    # Planned for future releases:
    # DEEPLABCUT = "deeplabcut"
    # SLEAP = "sleap"


class ModelManager:
    """Manages model loading and caching across frameworks.
    
    This class provides a unified interface for loading models from
    different frameworks and caches them to avoid redundant loading.
    
    Attributes:
        _model_cache: Dictionary storing loaded models.
        _config_cache: Dictionary storing model configurations.
    """
    
    def __init__(self) -> None:
        """Initialize ModelManager with empty caches."""
        self._model_cache: Dict[str, Any] = {}
        self._config_cache: Dict[str, Any] = {}
        self._available_gpus = self._detect_gpus()
    
    def _detect_gpus(self) -> List[int]:
        """Detect available CUDA GPUs.
        
        Returns:
            List of GPU indices.
        """
        if not torch.cuda.is_available():
            return []
        return list(range(torch.cuda.device_count()))
    
    def load_model(
        self,
        framework: Union[str, FrameworkType],
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda:0",
        **kwargs
    ) -> Tuple[Any, Optional[Dict]]:
        """Load a model from specified framework.
        
        Args:
            framework: Framework type or string identifier.
            config_path: Path to model configuration file.
            checkpoint_path: Path to model checkpoint/weights.
            device: Device to load model on.
            **kwargs: Additional framework-specific arguments.
            
        Returns:
            Tuple of (model, metadata) where metadata contains
            framework-specific information like keypoints, skeleton, etc.
            
        Raises:
            ImportError: If required framework is not installed.
            ValueError: If framework is not supported.
        """
        if isinstance(framework, str):
            framework = FrameworkType(framework.lower())
        
        cache_key = f"{framework.value}_{checkpoint_path}_{device}"
        
        if cache_key in self._model_cache:
            return self._model_cache[cache_key], self._config_cache.get(cache_key)
        
        model, metadata = self._load_framework_model(
            framework, config_path, checkpoint_path, device, **kwargs
        )
        
        self._model_cache[cache_key] = model
        if metadata:
            self._config_cache[cache_key] = metadata
        
        return model, metadata
    
    def _load_framework_model(
        self,
        framework: FrameworkType,
        config_path: Optional[str],
        checkpoint_path: Optional[str],
        device: str,
        **kwargs
    ) -> Tuple[Any, Optional[Dict]]:
        """Load model based on framework type.
        
        Args:
            framework: Framework type enum.
            config_path: Path to configuration.
            checkpoint_path: Path to checkpoint.
            device: Device to load on.
            **kwargs: Framework-specific arguments.
            
        Returns:
            Tuple of (model, metadata).
            
        Raises:
            ImportError: If framework not installed.
            ValueError: If framework not supported.
        """
        if framework == FrameworkType.MMDET:
            return self._load_mmdet(config_path, checkpoint_path, device, **kwargs)
        elif framework == FrameworkType.MMPOSE:
            return self._load_mmpose(config_path, checkpoint_path, device, **kwargs)
        elif framework == FrameworkType.ULTRALYTICS:
            return self._load_ultralytics(checkpoint_path, device, **kwargs)
        elif framework == FrameworkType.DEEPLABCUT:
            return self._load_deeplabcut(config_path, checkpoint_path, device, **kwargs)
        elif framework == FrameworkType.SLEAP:
            return self._load_sleap(checkpoint_path, device, **kwargs)
        elif framework == FrameworkType.SAM:
            return self._load_sam(checkpoint_path, device, **kwargs)
        else:
            raise ValueError(f"Unsupported framework: {framework}")
    
    def _load_mmdet(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str,
        **kwargs
    ) -> Tuple[Any, Dict]:
        """Load MMDetection model."""
        try:
            from mmdet.apis import init_detector
            from mmpose.utils import adapt_mmdet_pipeline
        except ImportError as e:
            raise ImportError(
                "MMDetection not installed. Install with: "
                "mim install mmdet"
            ) from e
        
        model = init_detector(config_path, checkpoint_path, device=device)
        model.cfg = adapt_mmdet_pipeline(model.cfg)
        
        metadata = {
            "framework": "mmdet",
            "config": model.cfg,
            "classes": model.CLASSES if hasattr(model, "CLASSES") else None
        }
        
        return model, metadata
    
    def _load_mmpose(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str,
        **kwargs
    ) -> Tuple[Any, Dict]:
        """Load MMPose model."""
        try:
            from mmpose.apis import init_model
        except ImportError as e:
            raise ImportError(
                "MMPose not installed. Install with: "
                "mim install mmpose"
            ) from e
        
        model = init_model(config_path, checkpoint_path, device=device)
        
        metadata = {
            "framework": "mmpose",
            "config": model.cfg,
            "dataset_meta": model.dataset_meta if hasattr(model, "dataset_meta") else {},
        }
        
        if hasattr(model, "dataset_meta"):
            metadata["keypoint_names"] = model.dataset_meta.get("keypoint_names", [])
            metadata["skeleton_links"] = model.dataset_meta.get("skeleton_links", [])
            
        return model, metadata
    
    def _load_ultralytics(
        self,
        checkpoint_path: str,
        device: str,
        **kwargs
    ) -> Tuple[Any, Dict]:
        """Load Ultralytics YOLO model."""
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "Ultralytics not installed. Install with: "
                "pip install ultralytics"
            ) from e
        
        model = YOLO(checkpoint_path)
        model.to(device)
        
        task = kwargs.get("task", "detect")
        metadata = {
            "framework": "ultralytics",
            "task": task,
            "names": model.names if hasattr(model, "names") else None
        }
        
        if task == "pose" and hasattr(model, "model") and hasattr(model.model, "kpt_shape"):
            n_kpts, _ = model.model.kpt_shape
            metadata["num_keypoints"] = n_kpts
        
        return model, metadata
    
    def _load_deeplabcut(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str,
        **kwargs
    ) -> Tuple[Any, Dict]:
        """Load DeepLabCut model."""
        try:
            import deeplabcut
        except ImportError as e:
            raise ImportError(
                "DeepLabCut not installed. See: "
                "https://github.com/DeepLabCut/DeepLabCut"
            ) from e
        
        raise NotImplementedError(
            "DeepLabCut integration coming soon. "
            "For now, export to COCO format and use MMPose."
        )
    
    def _load_sleap(
        self,
        checkpoint_path: str,
        device: str,
        **kwargs
    ) -> Tuple[Any, Dict]:
        """Load SLEAP model."""
        try:
            import sleap
        except ImportError as e:
            raise ImportError(
                "SLEAP not installed. See: "
                "https://github.com/talmolab/sleap"
            ) from e
        
        raise NotImplementedError(
            "SLEAP integration coming soon. "
            "For now, export to COCO format and use MMPose."
        )
    
    def _load_sam(
        self,
        checkpoint_path: str,
        device: str,
        **kwargs
    ) -> Tuple[Any, Dict]:
        """Load Segment Anything Model."""
        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError as e:
            raise ImportError(
                "Segment Anything not installed. Install with: "
                "pip install segment-anything"
            ) from e
        
        model_type = kwargs.get("model_type", "vit_h")
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        
        metadata = {
            "framework": "sam",
            "model_type": model_type,
            "device": device
        }
        
        return predictor, metadata
    
    def distribute_gpus(self, num_workers: int) -> List[str]:
        """Distribute GPUs among workers for parallel processing.
        
        Args:
            num_workers: Number of worker processes.
            
        Returns:
            List of device strings for each worker.
        """
        if not self._available_gpus:
            return ["cpu"] * num_workers
        
        devices = []
        for i in range(num_workers):
            gpu_idx = self._available_gpus[i % len(self._available_gpus)]
            devices.append(f"cuda:{gpu_idx}")
        
        return devices
    
    def clear_cache(self) -> None:
        """Clear all cached models and configurations."""
        self._model_cache.clear()
        self._config_cache.clear()
        
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_cached_models(self) -> List[str]:
        """Get list of cached model keys.
        
        Returns:
            List of cache keys for loaded models.
        """
        return list(self._model_cache.keys())