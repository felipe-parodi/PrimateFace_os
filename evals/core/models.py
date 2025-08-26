"""Model management utilities for different frameworks.

This module provides a unified interface for loading and managing models
from different pose estimation frameworks.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
import json
import yaml


@dataclass
class ModelConfig:
    """Configuration for a pose estimation model."""
    
    name: str
    framework: str  # 'mmpose', 'dlc', 'sleap'
    config_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    model_type: Optional[str] = None  # For DLC: 'hrnet_w32', etc.
    profile: Optional[str] = None  # For SLEAP: profile name
    device: str = 'cuda:0'
    additional_params: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'framework': self.framework,
            'config_path': self.config_path,
            'checkpoint_path': self.checkpoint_path,
            'model_type': self.model_type,
            'profile': self.profile,
            'device': self.device,
            'additional_params': self.additional_params or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """Create from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'ModelConfig':
        """Load from JSON or YAML file."""
        path = Path(path)
        with open(path, 'r') as f:
            if path.suffix == '.json':
                data = json.load(f)
            elif path.suffix in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        return cls.from_dict(data)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save to JSON or YAML file."""
        path = Path(path)
        data = self.to_dict()
        with open(path, 'w') as f:
            if path.suffix == '.json':
                json.dump(data, f, indent=2)
            elif path.suffix in ['.yaml', '.yml']:
                yaml.dump(data, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")


class ModelManager:
    """Manages model loading and inference for different frameworks."""
    
    def __init__(self):
        """Initialize model manager."""
        self._models_cache = {}
        self._check_available_frameworks()
    
    def _check_available_frameworks(self) -> None:
        """Check which frameworks are available."""
        self.available_frameworks = []
        
        try:
            import mmpose
            import mmdet
            self.available_frameworks.append('mmpose')
        except ImportError:
            pass
        
        try:
            import deeplabcut
            self.available_frameworks.append('dlc')
        except ImportError:
            pass
        
        try:
            import sleap
            self.available_frameworks.append('sleap')
        except ImportError:
            pass
    
    def load_model(
        self,
        config: Union[ModelConfig, Dict[str, Any], str, Path]
    ) -> Any:
        """Load a model based on configuration.
        
        Args:
            config: Model configuration (ModelConfig, dict, or path to config file)
            
        Returns:
            Loaded model object
        """
        if isinstance(config, (str, Path)):
            config = ModelConfig.from_file(config)
        elif isinstance(config, dict):
            config = ModelConfig.from_dict(config)
        
        if config.framework not in self.available_frameworks:
            raise ImportError(
                f"Framework '{config.framework}' is not available. "
                f"Available frameworks: {self.available_frameworks}"
            )
        
        # Check cache
        cache_key = f"{config.framework}_{config.name}"
        if cache_key in self._models_cache:
            return self._models_cache[cache_key]
        
        # Load model based on framework
        if config.framework == 'mmpose':
            model = self.load_mmpose_model(
                config.config_path,
                config.checkpoint_path,
                config.device
            )
        elif config.framework == 'dlc':
            model = self.load_dlc_model(
                config.checkpoint_path or config.model_type,
                config.additional_params
            )
        elif config.framework == 'sleap':
            model = self.load_sleap_model(
                config.checkpoint_path or config.profile,
                config.additional_params
            )
        else:
            raise ValueError(f"Unknown framework: {config.framework}")
        
        # Cache the model
        self._models_cache[cache_key] = model
        return model
    
    def load_mmpose_model(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = 'cuda:0'
    ) -> Tuple[Any, Any]:
        """Load MMPose model with its detector.
        
        Args:
            config_path: Path to model config
            checkpoint_path: Path to checkpoint
            device: Device to load model on
            
        Returns:
            Tuple of (pose_model, detector_model)
        """
        from mmpose.apis import init_model as init_pose_model
        from mmdet.apis import init_detector
        
        # Load pose model
        pose_model = init_pose_model(config_path, checkpoint_path, device=device)
        
        # Try to load detector if config specifies one
        detector = None
        if hasattr(pose_model.cfg, 'detector'):
            det_config = pose_model.cfg.detector.get('config')
            det_checkpoint = pose_model.cfg.detector.get('checkpoint')
            if det_config and det_checkpoint:
                detector = init_detector(det_config, det_checkpoint, device=device)
        
        return pose_model, detector
    
    def load_dlc_model(
        self,
        model_path_or_type: str,
        additional_params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Load DeepLabCut model.
        
        Args:
            model_path_or_type: Path to model or model type name
            additional_params: Additional parameters for model loading
            
        Returns:
            Loaded DLC model
        """
        try:
            from deeplabcut.pose_estimation_pytorch.models import PoseModel
            from deeplabcut.pose_estimation_pytorch.config import make_pytorch_pose_config
        except ImportError:
            raise ImportError("DeepLabCut is not installed")
        
        params = additional_params or {}
        
        # Check if it's a path or a model type
        path = Path(model_path_or_type)
        if path.exists() and path.suffix == '.pt':
            # Load from checkpoint
            import torch
            checkpoint = torch.load(path, map_location='cpu')
            
            # Extract config from checkpoint if available
            if 'config' in checkpoint:
                config = checkpoint['config']
            else:
                # Create default config
                config = make_pytorch_pose_config()
            
            # Create and load model
            model = PoseModel.from_config(config)
            model.load_state_dict(checkpoint['model_state_dict'])
            
        else:
            # Create new model from type
            config = make_pytorch_pose_config(
                model_name=model_path_or_type,
                **params
            )
            model = PoseModel.from_config(config)
        
        return model
    
    def load_sleap_model(
        self,
        model_path_or_profile: str,
        additional_params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Load SLEAP model.
        
        Args:
            model_path_or_profile: Path to model or profile name
            additional_params: Additional parameters for model loading
            
        Returns:
            Loaded SLEAP model
        """
        try:
            import sleap
            from sleap.nn.config import TrainingJobConfig
        except ImportError:
            raise ImportError("SLEAP is not installed")
        
        params = additional_params or {}
        
        # Check if it's a model path or profile
        path = Path(model_path_or_profile)
        if path.exists():
            if path.is_dir():
                # Load from model directory
                model = sleap.load_model(path, **params)
            elif path.suffix == '.json':
                # Load from profile
                config = TrainingJobConfig.load_json(path)
                model = config.model
            else:
                model = sleap.load_model(path, **params)
        else:
            # Assume it's a profile name
            config = TrainingJobConfig.from_preset(model_path_or_profile)
            model = config.model
        
        return model
    
    def run_inference(
        self,
        model: Any,
        image: Any,
        framework: str,
        **kwargs
    ) -> Any:
        """Run inference with a loaded model.
        
        Args:
            model: Loaded model
            image: Input image
            framework: Framework name
            **kwargs: Framework-specific inference parameters
            
        Returns:
            Inference results
        """
        if framework == 'mmpose':
            return self._run_mmpose_inference(model, image, **kwargs)
        elif framework == 'dlc':
            return self._run_dlc_inference(model, image, **kwargs)
        elif framework == 'sleap':
            return self._run_sleap_inference(model, image, **kwargs)
        else:
            raise ValueError(f"Unknown framework: {framework}")
    
    def _run_mmpose_inference(
        self,
        model: Tuple[Any, Any],
        image: Any,
        **kwargs
    ) -> Any:
        """Run MMPose inference."""
        from mmpose.apis import inference_topdown
        
        pose_model, detector = model
        
        # Run detection first if detector is available
        if detector is not None:
            from mmdet.apis import inference_detector
            det_results = inference_detector(detector, image)
            # Extract bboxes from detection results
            # TODO: Implement proper parsing of detection results
            bboxes = det_results
        else:
            # Use provided bboxes or full image
            bboxes = kwargs.get('bboxes', [[0, 0, image.shape[1], image.shape[0]]])
        
        # Run pose estimation
        results = inference_topdown(pose_model, image, bboxes, **kwargs)
        return results
    
    def _run_dlc_inference(
        self,
        model: Any,
        image: Any,
        **kwargs
    ) -> Any:
        """Run DeepLabCut inference."""
        import torch
        
        # Prepare image
        if not isinstance(image, torch.Tensor):
            import torchvision.transforms as transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            image = transform(image).unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            outputs = model(image)
        
        return outputs
    
    def _run_sleap_inference(
        self,
        model: Any,
        image: Any,
        **kwargs
    ) -> Any:
        """Run SLEAP inference."""
        # SLEAP inference is more complex and depends on the model type
        # TODO: Implement proper parsing of detection results
        predictions = model.predict(image, **kwargs)
        return predictions
    
    def clear_cache(self) -> None:
        """Clear the model cache."""
        self._models_cache.clear()
    
    def list_cached_models(self) -> List[str]:
        """List currently cached models."""
        return list(self._models_cache.keys())