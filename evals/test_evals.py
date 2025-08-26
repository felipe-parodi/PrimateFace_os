"""Unit tests for evaluation module.

Tests core functionality without requiring actual models or large datasets.
"""

import sys
import os
import unittest
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path so we can import evals module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from evals.constants import (
    get_default_config,
    get_model_config,
    validate_image_extensions,
    get_color_rgb
)
from evals.core.metrics import (
    NMECalculator,
    PCKCalculator,
    OKSCalculator,
    MetricsCalculator
)
from evals.core.models import ModelConfig, ModelManager
from evals.core.visualization import EvalVisualizer


class TestConstants(unittest.TestCase):
    """Test configuration and constants."""
    
    def test_get_default_config(self):
        """Test getting default configs for frameworks."""
        # Valid framework
        config = get_default_config('mmpose')
        self.assertIsInstance(config, dict)
        self.assertIn('device', config)
        self.assertIn('batch_size', config)
        
        # Invalid framework
        with self.assertRaises(ValueError):
            get_default_config('invalid_framework')
    
    def test_validate_image_extensions(self):
        """Test image extension validation."""
        extensions = ['jpg', '.png', 'JPEG']
        normalized = validate_image_extensions(extensions)
        self.assertEqual(normalized, ['.jpg', '.png', '.jpeg'])
    
    def test_get_color_rgb(self):
        """Test color conversion."""
        # Hex color
        color = get_color_rgb('gt_bbox')
        self.assertIsInstance(color, tuple)
        self.assertEqual(len(color), 3)
        
        # RGB tuple
        color = get_color_rgb('keypoint_left')
        self.assertEqual(color, (0, 255, 0))


class TestMetrics(unittest.TestCase):
    """Test metric calculators."""
    
    def setUp(self):
        """Set up test data."""
        # Create synthetic keypoint data
        np.random.seed(42)
        self.n_samples = 10
        self.n_keypoints = 17
        
        # Ground truth keypoints [N, K, 3] (x, y, visibility)
        self.gt_keypoints = np.random.rand(
            self.n_samples, self.n_keypoints, 3
        ) * 100
        self.gt_keypoints[..., 2] = (self.gt_keypoints[..., 2] > 0.3).astype(float)
        
        # Predictions with some noise
        noise = np.random.randn(self.n_samples, self.n_keypoints, 2) * 5
        self.pred_keypoints = self.gt_keypoints.copy()
        self.pred_keypoints[..., :2] += noise
        
        # Bounding boxes [N, 4] (x1, y1, x2, y2)
        self.bboxes = np.array([
            [10, 10, 90, 90] for _ in range(self.n_samples)
        ])
    
    def test_nme_calculator(self):
        """Test NME calculation."""
        calculator = NMECalculator(normalize_by='bbox')
        
        nme = calculator.calculate(
            self.pred_keypoints,
            self.gt_keypoints,
            self.bboxes
        )
        
        self.assertIsInstance(nme, float)
        self.assertGreater(nme, 0)
        self.assertLess(nme, 1.0)  # Should be normalized
    
    def test_pck_calculator(self):
        """Test PCK calculation."""
        calculator = PCKCalculator(threshold=0.2)
        
        results = calculator.calculate(
            self.pred_keypoints,
            self.gt_keypoints,
            self.bboxes
        )
        
        self.assertIn('pck', results)
        self.assertIn('pck_per_keypoint', results)
        self.assertIn('threshold', results)
        
        pck = results['pck']
        self.assertGreaterEqual(pck, 0)
        self.assertLessEqual(pck, 1)
    
    def test_oks_calculator(self):
        """Test OKS calculation."""
        calculator = OKSCalculator()
        
        # Calculate areas from bboxes
        areas = (self.bboxes[:, 2] - self.bboxes[:, 0]) * \
                (self.bboxes[:, 3] - self.bboxes[:, 1])
        
        results = calculator.calculate(
            self.pred_keypoints,
            self.gt_keypoints,
            areas
        )
        
        self.assertIn('oks', results)
        self.assertIn('oks_per_instance', results)
        
        oks = results['oks']
        self.assertGreaterEqual(oks, 0)
        self.assertLessEqual(oks, 1)
    
    def test_metrics_calculator(self):
        """Test unified metrics calculator."""
        calculator = MetricsCalculator()
        
        results = calculator.calculate_all(
            self.pred_keypoints,
            self.gt_keypoints,
            metrics=['nme', 'pck'],
            bboxes=self.bboxes
        )
        
        self.assertIn('nme', results)
        self.assertIn('pck', results)
        
        # Test individual methods
        nme = calculator.calculate_nme(
            self.pred_keypoints,
            self.gt_keypoints,
            bboxes=self.bboxes
        )
        self.assertIsInstance(nme, float)


class TestModelConfig(unittest.TestCase):
    """Test model configuration."""
    
    def test_model_config_creation(self):
        """Test creating model config."""
        config = ModelConfig(
            name='test_model',
            framework='mmpose',
            config_path='path/to/config.py',
            checkpoint_path='path/to/model.pth'
        )
        
        self.assertEqual(config.name, 'test_model')
        self.assertEqual(config.framework, 'mmpose')
        self.assertEqual(config.device, 'cuda:0')
    
    def test_model_config_serialization(self):
        """Test saving and loading config."""
        config = ModelConfig(
            name='test_model',
            framework='dlc',
            model_type='hrnet_w32'
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test JSON
            json_path = Path(tmpdir) / 'config.json'
            config.save(json_path)
            loaded = ModelConfig.from_file(json_path)
            self.assertEqual(loaded.name, config.name)
            self.assertEqual(loaded.framework, config.framework)
            
            # Test YAML
            yaml_path = Path(tmpdir) / 'config.yaml'
            config.save(yaml_path)
            loaded = ModelConfig.from_file(yaml_path)
            self.assertEqual(loaded.name, config.name)
    
    def test_model_config_dict_conversion(self):
        """Test dictionary conversion."""
        data = {
            'name': 'test',
            'framework': 'sleap',
            'profile': 'baseline.json'
        }
        
        config = ModelConfig.from_dict(data)
        self.assertEqual(config.name, 'test')
        self.assertEqual(config.framework, 'sleap')
        
        dict_data = config.to_dict()
        self.assertEqual(dict_data['name'], 'test')
        self.assertEqual(dict_data['framework'], 'sleap')


class TestModelManager(unittest.TestCase):
    """Test model management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = ModelManager()
    
    def test_available_frameworks(self):
        """Test framework availability check."""
        # Should be a list (may be empty if no frameworks installed)
        self.assertIsInstance(self.manager.available_frameworks, list)
    
    @patch('evals.core.models.ModelManager.load_mmpose_model')
    def test_load_model_from_config(self, mock_load):
        """Test loading model from config."""
        mock_load.return_value = (Mock(), Mock())
        
        # Assume mmpose is available for this test
        self.manager.available_frameworks = ['mmpose']
        
        config = ModelConfig(
            name='test',
            framework='mmpose',
            config_path='config.py',
            checkpoint_path='model.pth'
        )
        
        model = self.manager.load_model(config)
        self.assertIsNotNone(model)
        mock_load.assert_called_once()
    
    def test_model_caching(self):
        """Test model caching."""
        self.manager.available_frameworks = ['mmpose']
        
        with patch.object(self.manager, 'load_mmpose_model') as mock_load:
            mock_load.return_value = (Mock(), Mock())
            
            config = ModelConfig(
                name='cached_model',
                framework='mmpose',
                config_path='config.py',
                checkpoint_path='model.pth'
            )
            
            # First load
            model1 = self.manager.load_model(config)
            self.assertEqual(mock_load.call_count, 1)
            
            # Second load should use cache
            model2 = self.manager.load_model(config)
            self.assertEqual(mock_load.call_count, 1)
            self.assertIs(model1, model2)
    
    def test_clear_cache(self):
        """Test clearing model cache."""
        self.manager._models_cache = {'test': Mock()}
        self.assertEqual(len(self.manager._models_cache), 1)
        
        self.manager.clear_cache()
        self.assertEqual(len(self.manager._models_cache), 0)


class TestVisualization(unittest.TestCase):
    """Test visualization utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.viz = EvalVisualizer()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_plot_training_curves(self, mock_show, mock_save):
        """Test plotting training curves."""
        history = {
            'loss': [0.5, 0.4, 0.3, 0.2],
            'val_loss': [0.6, 0.5, 0.4, 0.3],
            'accuracy': [0.7, 0.8, 0.85, 0.9]
        }
        
        fig = self.viz.plot_training_curves(history)
        self.assertIsNotNone(fig)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_plot_predictions(self, mock_show, mock_save):
        """Test plotting predictions."""
        # Create dummy data
        images = np.random.rand(3, 100, 100, 3)
        predictions = np.random.rand(3, 17, 3) * 100
        ground_truth = np.random.rand(3, 17, 3) * 100
        
        fig = self.viz.plot_predictions(
            images, predictions, ground_truth
        )
        self.assertIsNotNone(fig)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_plot_metric_comparison(self, mock_show, mock_save):
        """Test plotting metric comparison."""
        results = {
            'model1': {'nme': 0.05, 'pck': 0.85, 'oks': 0.75},
            'model2': {'nme': 0.06, 'pck': 0.82, 'oks': 0.73},
            'model3': {'nme': 0.04, 'pck': 0.88, 'oks': 0.78}
        }
        
        fig = self.viz.plot_metric_comparison(results)
        self.assertIsNotNone(fig)


class TestIntegration(unittest.TestCase):
    """Integration tests for the full pipeline."""
    
    def test_full_evaluation_pipeline(self):
        """Test complete evaluation workflow."""
        # Create synthetic data
        np.random.seed(42)
        n_samples = 5
        n_keypoints = 17
        
        gt = np.random.rand(n_samples, n_keypoints, 3) * 100
        pred = gt + np.random.randn(n_samples, n_keypoints, 3) * 2
        
        # Calculate all metrics
        calculator = MetricsCalculator()
        results = calculator.calculate_all(pred, gt)
        
        # Check all metrics present
        self.assertIn('nme', results)
        self.assertIn('pck', results)
        self.assertIn('oks', results)
        
        # All should be valid numbers
        self.assertIsInstance(results['nme'], float)
        self.assertTrue(0 <= results['pck'] <= 1)
        self.assertTrue(0 <= results['oks'] <= 1)


if __name__ == '__main__':
    unittest.main()