# Evaluation Module - Technical Documentation

This document provides comprehensive technical documentation for the PrimateFace evaluation module, including detailed metric implementations, cross-framework comparison strategies, and advanced evaluation pipelines.

## Module Architecture

```
evals/
├── core/                    # Core evaluation components
│   ├── metrics.py          # Metric implementations
│   ├── models.py           # Model management
│   └── visualization.py    # Plotting utilities
├── dlc/                    # DeepLabCut-specific
│   ├── train_with_coco.py
│   └── evaluate_nme.py
├── sleap/                  # SLEAP-specific
│   ├── train_sleap_with_coco.py
│   └── evaluate_sleap_nme.py
├── compare_det_models.py   # Detection comparison
├── compare_pose_models.py  # Pose comparison
├── eval_genera.py         # Per-genus evaluation
├── constants.py           # Configuration
└── test_evals.py         # Unit tests
```

## Core Components

### Metrics Implementation

#### Class: `NMECalculator`

Normalized Mean Error calculator with multiple normalization strategies.

```python
class NMECalculator:
    def __init__(
        self,
        normalize_by: str = 'bbox',  # 'bbox', 'interocular', 'head_size'
        use_visible_only: bool = True
    ):
        self.normalize_by = normalize_by
        self.use_visible_only = use_visible_only
    
    def calculate(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        metadata: dict
    ) -> dict:
        """
        Calculate NME with various normalizations.
        
        Args:
            predictions: Shape (N, K, 2) or (N, K, 3) with visibility
            ground_truth: Shape (N, K, 2) or (N, K, 3) with visibility
            metadata: Contains bbox, interocular distance, etc.
        
        Returns:
            dict: {
                'nme': float,
                'per_keypoint_nme': np.ndarray,
                'per_sample_nme': np.ndarray,
                'normalization_factor': float
            }
        """
        # Extract normalization factor
        norm_factor = self._get_normalization_factor(metadata)
        
        # Calculate distances
        distances = np.linalg.norm(predictions - ground_truth, axis=-1)
        
        # Apply visibility mask if needed
        if self.use_visible_only:
            visibility = ground_truth[..., 2] if ground_truth.shape[-1] == 3 else np.ones_like(distances)
            distances = distances * visibility
            valid_count = visibility.sum()
        else:
            valid_count = distances.size
        
        # Normalize
        normalized_distances = distances / norm_factor
        
        return {
            'nme': normalized_distances.sum() / valid_count,
            'per_keypoint_nme': normalized_distances.mean(axis=0),
            'per_sample_nme': normalized_distances.mean(axis=1),
            'normalization_factor': norm_factor
        }
    
    def _get_normalization_factor(self, metadata: dict) -> float:
        """Get normalization factor based on strategy."""
        if self.normalize_by == 'bbox':
            bbox = metadata['bbox']  # [x, y, w, h]
            return np.sqrt(bbox[2] * bbox[3])  # Geometric mean
        elif self.normalize_by == 'interocular':
            return metadata['interocular_distance']
        elif self.normalize_by == 'head_size':
            # Define head size as distance from top to bottom of face
            return metadata['head_size']
        else:
            raise ValueError(f"Unknown normalization: {self.normalize_by}")
```

#### Class: `PCKCalculator`

Percentage of Correct Keypoints with configurable thresholds.

```python
class PCKCalculator:
    def __init__(
        self,
        thresholds: Union[float, List[float]] = [0.05, 0.1, 0.2],
        normalize_by: str = 'bbox',
        use_visible_only: bool = True
    ):
        self.thresholds = [thresholds] if isinstance(thresholds, float) else thresholds
        self.normalize_by = normalize_by
        self.use_visible_only = use_visible_only
    
    def calculate(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        metadata: dict
    ) -> dict:
        """
        Calculate PCK at multiple thresholds.
        
        Returns:
            dict: {
                'pck': dict,  # {threshold: accuracy}
                'per_keypoint_pck': dict,  # {threshold: per-keypoint accuracy}
                'auc': float  # Area under PCK curve
            }
        """
        distances = np.linalg.norm(predictions - ground_truth, axis=-1)
        norm_factor = self._get_normalization_factor(metadata)
        normalized_distances = distances / norm_factor
        
        results = {'pck': {}, 'per_keypoint_pck': {}}
        
        for threshold in self.thresholds:
            correct = normalized_distances <= threshold
            
            if self.use_visible_only and ground_truth.shape[-1] == 3:
                visibility = ground_truth[..., 2] > 0
                correct = correct * visibility
                accuracy = correct.sum() / visibility.sum()
                per_kpt_acc = correct.sum(axis=0) / visibility.sum(axis=0)
            else:
                accuracy = correct.mean()
                per_kpt_acc = correct.mean(axis=0)
            
            results['pck'][threshold] = accuracy
            results['per_keypoint_pck'][threshold] = per_kpt_acc
        
        # Calculate AUC
        results['auc'] = self._calculate_auc(results['pck'])
        
        return results
    
    def _calculate_auc(self, pck_dict: dict) -> float:
        """Calculate area under PCK curve."""
        thresholds = sorted(pck_dict.keys())
        accuracies = [pck_dict[t] for t in thresholds]
        
        # Trapezoidal integration
        auc = 0
        for i in range(len(thresholds) - 1):
            dt = thresholds[i+1] - thresholds[i]
            avg_acc = (accuracies[i] + accuracies[i+1]) / 2
            auc += dt * avg_acc
        
        return auc
```

#### Class: `OKSCalculator`

Object Keypoint Similarity for COCO-style evaluation.

```python
class OKSCalculator:
    def __init__(
        self,
        sigmas: Optional[np.ndarray] = None,
        use_area: bool = True
    ):
        self.sigmas = sigmas or self._get_default_sigmas()
        self.use_area = use_area
    
    def calculate(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        areas: np.ndarray
    ) -> dict:
        """
        Calculate OKS following COCO evaluation.
        
        Args:
            predictions: Shape (N, K, 3) with [x, y, visibility]
            ground_truth: Shape (N, K, 3) with [x, y, visibility]
            areas: Shape (N,) with object areas
        
        Returns:
            dict: {
                'oks': np.ndarray,  # Per-sample OKS scores
                'mean_oks': float,
                'oks_per_keypoint': np.ndarray
            }
        """
        N, K = predictions.shape[:2]
        oks_scores = np.zeros(N)
        
        for i in range(N):
            pred = predictions[i]
            gt = ground_truth[i]
            area = areas[i] if self.use_area else 1.0
            
            # Visibility from ground truth
            visibility = gt[:, 2]
            
            # Calculate distances
            dx = pred[:, 0] - gt[:, 0]
            dy = pred[:, 1] - gt[:, 1]
            distances = np.sqrt(dx**2 + dy**2)
            
            # OKS calculation
            k = 2 * self.sigmas
            oks_per_kpt = np.exp(-(distances**2) / (2 * area * k**2))
            oks_per_kpt = oks_per_kpt * visibility
            
            # Average over visible keypoints
            if visibility.sum() > 0:
                oks_scores[i] = oks_per_kpt.sum() / visibility.sum()
            else:
                oks_scores[i] = 0
        
        return {
            'oks': oks_scores,
            'mean_oks': oks_scores.mean(),
            'oks_per_keypoint': self._calculate_per_keypoint_oks(predictions, ground_truth, areas)
        }
    
    def _get_default_sigmas(self) -> np.ndarray:
        """Get default sigma values for primate faces."""
        # These are tuned for 68-point primate faces
        return np.array([
            0.025, 0.025, 0.025, 0.025, 0.025,  # Jaw line
            0.02, 0.02, 0.02, 0.02,              # Eyebrows
            0.015, 0.015, 0.015, 0.015,          # Eyes
            0.02, 0.02, 0.02,                    # Nose
            0.025, 0.025, 0.025, 0.025,          # Mouth
            # ... continue for all keypoints
        ])
```

### Model Management

#### Class: `ModelManager`

Unified interface for loading and running models across frameworks.

```python
class ModelManager:
    """Manage models from different frameworks."""
    
    def __init__(self):
        self.supported_frameworks = ['mmpose', 'dlc', 'sleap', 'yolo']
        self.loaded_models = {}
    
    def load_model(self, config: Union[str, ModelConfig]) -> Any:
        """
        Load model from configuration.
        
        Args:
            config: Path to config file or ModelConfig object
        
        Returns:
            Loaded model object
        """
        if isinstance(config, str):
            config = ModelConfig.from_file(config)
        
        if config.framework == 'mmpose':
            return self._load_mmpose_model(config)
        elif config.framework == 'dlc':
            return self._load_dlc_model(config)
        elif config.framework == 'sleap':
            return self._load_sleap_model(config)
        elif config.framework == 'yolo':
            return self._load_yolo_model(config)
        else:
            raise ValueError(f"Unsupported framework: {config.framework}")
    
    def _load_mmpose_model(self, config: ModelConfig):
        """Load MMPose model."""
        from mmpose.apis import init_model
        
        model = init_model(
            config.config_path,
            config.checkpoint_path,
            device=config.device
        )
        
        self.loaded_models[config.name] = model
        return model
    
    def _load_dlc_model(self, config: ModelConfig):
        """Load DeepLabCut model."""
        import deeplabcut as dlc
        
        # DLC uses project config
        cfg = dlc.auxiliaryfunctions.read_config(config.config_path)
        
        # Load specific snapshot
        model = dlc.load_model_from_snapshot(
            config.checkpoint_path,
            cfg
        )
        
        self.loaded_models[config.name] = model
        return model
    
    def run_inference(
        self,
        model: Any,
        images: Union[np.ndarray, List[str]],
        framework: str
    ) -> dict:
        """
        Run inference with loaded model.
        
        Returns:
            dict: {
                'predictions': np.ndarray,  # Keypoint predictions
                'scores': np.ndarray,       # Confidence scores
                'bboxes': np.ndarray        # Detection boxes (if applicable)
            }
        """
        if framework == 'mmpose':
            return self._run_mmpose_inference(model, images)
        elif framework == 'dlc':
            return self._run_dlc_inference(model, images)
        # ... other frameworks
```

#### Class: `ModelConfig`

Configuration management for models.

```python
class ModelConfig:
    """Model configuration container."""
    
    def __init__(
        self,
        name: str,
        framework: str,
        config_path: str,
        checkpoint_path: str,
        device: str = 'cuda:0',
        **kwargs
    ):
        self.name = name
        self.framework = framework
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.additional_params = kwargs
    
    @classmethod
    def from_file(cls, path: str) -> 'ModelConfig':
        """Load configuration from YAML/JSON file."""
        if path.endswith('.yaml') or path.endswith('.yml'):
            import yaml
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
        elif path.endswith('.json'):
            import json
            with open(path, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path}")
        
        return cls(**data)
    
    def save(self, path: str):
        """Save configuration to file."""
        data = {
            'name': self.name,
            'framework': self.framework,
            'config_path': self.config_path,
            'checkpoint_path': self.checkpoint_path,
            'device': self.device,
            **self.additional_params
        }
        
        if path.endswith('.yaml') or path.endswith('.yml'):
            import yaml
            with open(path, 'w') as f:
                yaml.dump(data, f)
        elif path.endswith('.json'):
            import json
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
```

### Cross-Framework Comparison

#### Class: `CrossFrameworkEvaluator`

Compare models across different frameworks on same data.

```python
class CrossFrameworkEvaluator:
    """Evaluate and compare models across frameworks."""
    
    def __init__(
        self,
        test_data: str,  # Path to COCO annotations
        output_dir: str
    ):
        self.test_data = test_data
        self.output_dir = output_dir
        self.manager = ModelManager()
        self.metrics_calc = MetricsCalculator()
        self.results = {}
    
    def add_model(self, config: Union[str, ModelConfig], name: Optional[str] = None):
        """Add model for comparison."""
        if isinstance(config, str):
            config = ModelConfig.from_file(config)
        
        name = name or config.name
        model = self.manager.load_model(config)
        
        self.results[name] = {
            'config': config,
            'model': model,
            'metrics': {}
        }
    
    def evaluate_all(
        self,
        metrics: List[str] = ['nme', 'pck', 'oks'],
        batch_size: int = 32
    ):
        """Evaluate all models on test data."""
        # Load test data
        from pycocotools.coco import COCO
        coco = COCO(self.test_data)
        
        for name, model_info in self.results.items():
            print(f"Evaluating {name}...")
            
            # Run inference
            predictions = self._run_inference_on_coco(
                model_info['model'],
                model_info['config'].framework,
                coco,
                batch_size
            )
            
            # Calculate metrics
            for metric in metrics:
                if metric == 'nme':
                    calc = NMECalculator(normalize_by='bbox')
                    result = calc.calculate(
                        predictions['keypoints'],
                        self._get_ground_truth(coco),
                        self._get_metadata(coco)
                    )
                    model_info['metrics']['nme'] = result['nme']
                    model_info['metrics']['nme_per_kpt'] = result['per_keypoint_nme']
                
                elif metric == 'pck':
                    calc = PCKCalculator(thresholds=[0.05, 0.1, 0.2])
                    result = calc.calculate(
                        predictions['keypoints'],
                        self._get_ground_truth(coco),
                        self._get_metadata(coco)
                    )
                    model_info['metrics']['pck'] = result['pck']
                    model_info['metrics']['pck_auc'] = result['auc']
                
                elif metric == 'oks':
                    calc = OKSCalculator()
                    result = calc.calculate(
                        predictions['keypoints'],
                        self._get_ground_truth(coco),
                        self._get_areas(coco)
                    )
                    model_info['metrics']['oks'] = result['mean_oks']
    
    def generate_comparison_report(self) -> pd.DataFrame:
        """Generate comparison report as DataFrame."""
        import pandas as pd
        
        rows = []
        for name, info in self.results.items():
            row = {
                'Model': name,
                'Framework': info['config'].framework,
                **info['metrics']
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Save to CSV
        df.to_csv(f"{self.output_dir}/comparison_results.csv", index=False)
        
        return df
    
    def plot_comparison(self, metric: str = 'nme'):
        """Generate comparison plots."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar plot of overall metric
        models = list(self.results.keys())
        values = [self.results[m]['metrics'].get(metric, 0) for m in models]
        
        axes[0].bar(models, values)
        axes[0].set_ylabel(metric.upper())
        axes[0].set_title(f'Overall {metric.upper()} Comparison')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Per-keypoint comparison (if available)
        if f'{metric}_per_kpt' in self.results[models[0]]['metrics']:
            for model in models:
                per_kpt = self.results[model]['metrics'][f'{metric}_per_kpt']
                axes[1].plot(per_kpt, label=model)
            
            axes[1].set_xlabel('Keypoint Index')
            axes[1].set_ylabel(metric.upper())
            axes[1].set_title(f'Per-Keypoint {metric.upper()}')
            axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{metric}_comparison.png")
        plt.show()
```

### Visualization Components

#### Class: `EvalVisualizer`

Comprehensive visualization for evaluation results.

```python
class EvalVisualizer:
    """Visualization utilities for evaluation."""
    
    def __init__(self, style: str = 'seaborn'):
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.style.use(style)
        sns.set_palette("husl")
        
        self.plt = plt
        self.sns = sns
    
    def plot_training_curves(
        self,
        history: dict,
        metrics: List[str] = ['loss', 'nme'],
        output_path: Optional[str] = None
    ):
        """Plot training and validation curves."""
        n_metrics = len(metrics)
        fig, axes = self.plt.subplots(1, n_metrics, figsize=(6*n_metrics, 4))
        
        if n_metrics == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics):
            # Training curve
            if metric in history:
                ax.plot(history[metric], label=f'Train {metric}')
            
            # Validation curve
            val_metric = f'val_{metric}'
            if val_metric in history:
                ax.plot(history[val_metric], label=f'Val {metric}')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        self.plt.tight_layout()
        
        if output_path:
            self.plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_predictions(
        self,
        images: np.ndarray,
        predictions: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
        keypoint_names: Optional[List[str]] = None,
        max_images: int = 9,
        output_path: Optional[str] = None
    ):
        """Plot prediction grid with optional ground truth."""
        n_images = min(len(images), max_images)
        grid_size = int(np.ceil(np.sqrt(n_images)))
        
        fig, axes = self.plt.subplots(grid_size, grid_size, figsize=(15, 15))
        axes = axes.flatten() if n_images > 1 else [axes]
        
        for idx in range(n_images):
            ax = axes[idx]
            
            # Display image
            ax.imshow(images[idx])
            
            # Plot predictions
            pred_kpts = predictions[idx]
            ax.scatter(pred_kpts[:, 0], pred_kpts[:, 1], 
                      c='red', s=30, marker='o', label='Prediction')
            
            # Plot ground truth if available
            if ground_truth is not None:
                gt_kpts = ground_truth[idx]
                ax.scatter(gt_kpts[:, 0], gt_kpts[:, 1],
                          c='green', s=30, marker='x', label='Ground Truth')
                
                # Draw lines between predictions and ground truth
                for p, g in zip(pred_kpts, gt_kpts):
                    ax.plot([p[0], g[0]], [p[1], g[1]], 
                           'b-', alpha=0.3, linewidth=0.5)
            
            # Add keypoint labels if provided
            if keypoint_names:
                for kpt_idx, name in enumerate(keypoint_names[:5]):  # Show first 5
                    ax.text(pred_kpts[kpt_idx, 0], pred_kpts[kpt_idx, 1],
                           name, fontsize=8, color='white',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor='black', alpha=0.5))
            
            ax.axis('off')
            if idx == 0:
                ax.legend(loc='upper right')
        
        # Hide unused subplots
        for idx in range(n_images, len(axes)):
            axes[idx].axis('off')
        
        self.plt.suptitle('Prediction Visualization', fontsize=16)
        self.plt.tight_layout()
        
        if output_path:
            self.plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_error_distribution(
        self,
        errors: np.ndarray,
        keypoint_names: Optional[List[str]] = None,
        output_path: Optional[str] = None
    ):
        """Plot error distribution per keypoint."""
        fig, axes = self.plt.subplots(2, 1, figsize=(12, 8))
        
        # Box plot of errors per keypoint
        axes[0].boxplot(errors.T, labels=keypoint_names if keypoint_names else None)
        axes[0].set_ylabel('Error (pixels)')
        axes[0].set_title('Error Distribution per Keypoint')
        axes[0].grid(True, alpha=0.3)
        
        if keypoint_names:
            axes[0].tick_params(axis='x', rotation=90)
        
        # Histogram of overall errors
        axes[1].hist(errors.flatten(), bins=50, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Error (pixels)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Overall Error Distribution')
        axes[1].grid(True, alpha=0.3)
        
        # Add statistics
        mean_error = errors.mean()
        median_error = np.median(errors)
        axes[1].axvline(mean_error, color='red', linestyle='--', 
                       label=f'Mean: {mean_error:.2f}')
        axes[1].axvline(median_error, color='green', linestyle='--',
                       label=f'Median: {median_error:.2f}')
        axes[1].legend()
        
        self.plt.tight_layout()
        
        if output_path:
            self.plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
        return fig
```

### Per-Genus Evaluation

#### Class: `GenusEvaluator`

Evaluate performance across different primate genera.

```python
class GenusEvaluator:
    """Evaluate model performance per genus."""
    
    def __init__(self, predictions_path: str, annotations_path: str):
        self.predictions = self._load_json(predictions_path)
        self.annotations = self._load_json(annotations_path)
        self.genus_mapping = self._create_genus_mapping()
        self.results = {}
    
    def _create_genus_mapping(self) -> dict:
        """Map image IDs to genus labels."""
        mapping = {}
        
        # Assuming genus info is in image metadata
        for img in self.annotations['images']:
            if 'genus' in img:
                mapping[img['id']] = img['genus']
            else:
                # Extract from filename pattern
                filename = img['file_name']
                genus = self._extract_genus_from_filename(filename)
                mapping[img['id']] = genus
        
        return mapping
    
    def evaluate_per_genus(self, metrics: List[str] = ['nme', 'pck']):
        """Calculate metrics for each genus."""
        # Group predictions by genus
        genus_groups = defaultdict(list)
        
        for pred in self.predictions:
            img_id = pred['image_id']
            genus = self.genus_mapping.get(img_id, 'unknown')
            genus_groups[genus].append(pred)
        
        # Evaluate each genus
        for genus, preds in genus_groups.items():
            self.results[genus] = {}
            
            # Get corresponding ground truth
            gt = self._get_ground_truth_for_genus(genus)
            
            # Calculate metrics
            if 'nme' in metrics:
                nme_calc = NMECalculator()
                nme_result = nme_calc.calculate(
                    self._format_predictions(preds),
                    gt,
                    self._get_metadata_for_genus(genus)
                )
                self.results[genus]['nme'] = nme_result['nme']
                self.results[genus]['sample_count'] = len(preds)
            
            if 'pck' in metrics:
                pck_calc = PCKCalculator()
                pck_result = pck_calc.calculate(
                    self._format_predictions(preds),
                    gt,
                    self._get_metadata_for_genus(genus)
                )
                self.results[genus]['pck'] = pck_result['pck']
    
    def plot_genus_comparison(self, metric: str = 'nme'):
        """Plot comparison across genera."""
        import matplotlib.pyplot as plt
        
        genera = list(self.results.keys())
        values = [self.results[g].get(metric, 0) for g in genera]
        sample_counts = [self.results[g].get('sample_count', 0) for g in genera]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create bar plot
        bars = ax.bar(genera, values)
        
        # Color bars by sample count
        norm = plt.Normalize(min(sample_counts), max(sample_counts))
        colors = plt.cm.viridis(norm(sample_counts))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add sample count labels
        for i, (bar, count) in enumerate(zip(bars, sample_counts)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'n={count}', ha='center', va='bottom', fontsize=8)
        
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} by Genus')
        ax.tick_params(axis='x', rotation=45)
        
        # Add colorbar for sample counts
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Sample Count')
        
        plt.tight_layout()
        return fig
    
    def generate_report(self) -> pd.DataFrame:
        """Generate detailed report per genus."""
        import pandas as pd
        
        rows = []
        for genus, metrics in self.results.items():
            row = {'Genus': genus, **metrics}
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df = df.sort_values('nme', ascending=True)  # Sort by best performance
        
        return df
```

### Advanced Evaluation Pipelines

#### Temporal Consistency Evaluation

```python
class TemporalConsistencyEvaluator:
    """Evaluate temporal consistency in video sequences."""
    
    def __init__(self):
        self.jitter_threshold = 5.0  # pixels
    
    def evaluate_video(
        self,
        predictions: np.ndarray,  # Shape: (T, K, 2)
        fps: float = 30.0
    ) -> dict:
        """
        Evaluate temporal consistency of predictions.
        
        Returns:
            dict: {
                'mean_jitter': float,
                'per_keypoint_jitter': np.ndarray,
                'smooth_score': float,
                'problematic_frames': List[int]
            }
        """
        T, K, _ = predictions.shape
        
        # Calculate frame-to-frame differences
        velocities = np.diff(predictions, axis=0)
        accelerations = np.diff(velocities, axis=0)
        
        # Jitter metric (high acceleration)
        jitter = np.linalg.norm(accelerations, axis=-1)
        mean_jitter = jitter.mean()
        per_kpt_jitter = jitter.mean(axis=0)
        
        # Identify problematic frames
        frame_jitter = jitter.mean(axis=1)
        problematic = np.where(frame_jitter > self.jitter_threshold)[0]
        
        # Smoothness score (inverse of jitter)
        smooth_score = 1.0 / (1.0 + mean_jitter)
        
        return {
            'mean_jitter': mean_jitter,
            'per_keypoint_jitter': per_kpt_jitter,
            'smooth_score': smooth_score,
            'problematic_frames': problematic_frames.tolist(),
            'velocities': velocities,
            'accelerations': accelerations
        }
```

#### Multi-Scale Evaluation

```python
class MultiScaleEvaluator:
    """Evaluate model at different image scales."""
    
    def __init__(self, scales: List[float] = [0.5, 0.75, 1.0, 1.25, 1.5]):
        self.scales = scales
    
    def evaluate(
        self,
        model,
        test_images: List[np.ndarray],
        ground_truth: np.ndarray
    ) -> dict:
        """
        Evaluate model at multiple scales.
        
        Returns:
            dict: {scale: metrics} for each scale
        """
        results = {}
        
        for scale in self.scales:
            # Resize images
            scaled_images = [
                cv2.resize(img, None, fx=scale, fy=scale)
                for img in test_images
            ]
            
            # Run inference
            predictions = model.predict(scaled_images)
            
            # Scale predictions back
            predictions = predictions / scale
            
            # Calculate metrics
            nme_calc = NMECalculator()
            nme = nme_calc.calculate(predictions, ground_truth, {})
            
            results[scale] = {
                'nme': nme['nme'],
                'inference_time': self._measure_inference_time(model, scaled_images)
            }
        
        return results
```

## Configuration

### Constants and Defaults

```python
# constants.py

# Keypoint definitions
PRIMATE_KEYPOINTS_68 = [
    'jaw_1', 'jaw_2', 'jaw_3', 'jaw_4', 'jaw_5',  # ... continue
]

PRIMATE_KEYPOINTS_49 = [
    'jaw_left', 'jaw_center', 'jaw_right',  # ... continue
]

# Skeleton definitions
PRIMATE_SKELETON_68 = [
    (0, 1), (1, 2), (2, 3),  # Jaw connections
    # ... continue
]

# Evaluation thresholds
DEFAULT_PCK_THRESHOLDS = [0.05, 0.1, 0.15, 0.2]
DEFAULT_OKS_SIGMAS = np.array([0.025] * 68)  # Uniform for simplicity

# Normalization factors
BBOX_NORMALIZATION = 'geometric_mean'  # or 'diagonal', 'area'
INTEROCULAR_INDICES = (36, 45)  # Left and right eye corners

# Visualization settings
EVAL_COLORS = {
    'prediction': (255, 0, 0),    # Red
    'ground_truth': (0, 255, 0),  # Green
    'error': (0, 0, 255)          # Blue
}

# Framework-specific settings
FRAMEWORK_CONFIGS = {
    'mmpose': {
        'default_device': 'cuda:0',
        'default_batch_size': 32
    },
    'dlc': {
        'default_pcutoff': 0.6,
        'default_batch_size': 16
    },
    'sleap': {
        'default_peak_threshold': 0.2,
        'default_batch_size': 32
    }
}
```

## Testing

### Unit Tests

```python
import unittest
from evals.core.metrics import NMECalculator, PCKCalculator

class TestMetrics(unittest.TestCase):
    def setUp(self):
        # Create test data
        self.predictions = np.random.randn(10, 68, 2)
        self.ground_truth = self.predictions + np.random.randn(10, 68, 2) * 0.1
        self.metadata = {
            'bbox': [100, 100, 200, 200],
            'interocular_distance': 50
        }
    
    def test_nme_calculation(self):
        calc = NMECalculator(normalize_by='bbox')
        result = calc.calculate(
            self.predictions,
            self.ground_truth,
            self.metadata
        )
        
        self.assertIn('nme', result)
        self.assertIsInstance(result['nme'], float)
        self.assertGreaterEqual(result['nme'], 0)
    
    def test_pck_calculation(self):
        calc = PCKCalculator(thresholds=[0.1, 0.2])
        result = calc.calculate(
            self.predictions,
            self.ground_truth,
            self.metadata
        )
        
        self.assertIn('pck', result)
        self.assertEqual(len(result['pck']), 2)
        
        # PCK values should be between 0 and 1
        for threshold, accuracy in result['pck'].items():
            self.assertGreaterEqual(accuracy, 0)
            self.assertLessEqual(accuracy, 1)
    
    def test_cross_framework_comparison(self):
        from evals.core.models import CrossFrameworkEvaluator
        
        evaluator = CrossFrameworkEvaluator(
            test_data='test_annotations.json',
            output_dir='test_output'
        )
        
        # Add mock models
        evaluator.add_model('mock_config.yaml')
        
        # Test evaluation
        evaluator.evaluate_all(metrics=['nme'])
        
        # Check results structure
        self.assertIsInstance(evaluator.results, dict)
```

## Performance Optimization

### Batch Processing

```python
def batch_evaluate(
    model,
    test_loader: DataLoader,
    metrics: List[str],
    device: str = 'cuda'
) -> dict:
    """Efficiently evaluate model on large dataset."""
    
    all_predictions = []
    all_ground_truth = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch['images'].to(device)
            gt = batch['keypoints']
            
            # Run inference
            pred = model(images)
            
            all_predictions.append(pred.cpu().numpy())
            all_ground_truth.append(gt.numpy())
    
    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions)
    all_ground_truth = np.concatenate(all_ground_truth)
    
    # Calculate metrics
    results = {}
    for metric in metrics:
        if metric == 'nme':
            calc = NMECalculator()
            results['nme'] = calc.calculate(
                all_predictions, 
                all_ground_truth,
                {}
            )['nme']
    
    return results
```

### Caching Results

```python
class CachedEvaluator:
    """Cache evaluation results to avoid recomputation."""
    
    def __init__(self, cache_dir: str = '.eval_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_key(self, model_name: str, dataset: str, metric: str) -> str:
        """Generate cache key."""
        return f"{model_name}_{dataset}_{metric}"
    
    def evaluate(self, model, dataset, metric):
        """Evaluate with caching."""
        cache_key = self.get_cache_key(model.name, dataset.name, metric)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # Check cache
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # Compute if not cached
        result = self._compute_metric(model, dataset, metric)
        
        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        
        return result
```

## See Also

- [Main README](README.md) - Module overview
- [API Documentation](../docs/api/evaluation.md) - API reference
- [Framework Training Guide](../docs/guides/framework-training.md) - Training models
- [DLC Documentation](dlc/README.md) - DeepLabCut specifics
- [SLEAP Documentation](sleap/README.md) - SLEAP specifics