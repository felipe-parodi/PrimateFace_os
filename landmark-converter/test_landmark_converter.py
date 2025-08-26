"""Unit tests for the landmark converter module.

This module contains tests for model initialization, data loading,
normalization/denormalization, and basic training functionality.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

# Import modules to test
from constants import MODEL_CONFIGS, CONVERSION_MODES, FIXED_MAPPING_68_TO_49
from src.models import (
    SimpleLinearConverter,
    KeypointConverterMLP,
    MinimalMLPConverter,
    KeypointConverterMLPWithAttention
)
from utils.data_utils import normalize_keypoints_bbox, denormalize_keypoints_bbox


class TestModels(unittest.TestCase):
    """Test model initialization and forward pass."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_source_kpts = 68
        self.num_target_kpts = 49
        self.batch_size = 4
        self.device = torch.device('cpu')
    
    def test_simple_linear_converter(self):
        """Test SimpleLinearConverter initialization and forward pass."""
        model = SimpleLinearConverter(self.num_source_kpts, self.num_target_kpts)
        model.to(self.device)
        
        # Create random input
        input_tensor = torch.randn(self.batch_size, self.num_source_kpts * 2)
        
        # Forward pass
        output = model(input_tensor)
        
        # Check output shape
        expected_shape = (self.batch_size, self.num_target_kpts, 2)
        self.assertEqual(output.shape, expected_shape)
    
    def test_keypoint_converter_mlp(self):
        """Test KeypointConverterMLP initialization and forward pass."""
        model = KeypointConverterMLP(
            self.num_source_kpts,
            self.num_target_kpts,
            hidden_dim1=256,
            hidden_dim2=256
        )
        model.to(self.device)
        
        # Create random input
        input_tensor = torch.randn(self.batch_size, self.num_source_kpts * 2)
        
        # Forward pass
        output = model(input_tensor)
        
        # Check output shape
        expected_shape = (self.batch_size, self.num_target_kpts, 2)
        self.assertEqual(output.shape, expected_shape)
    
    def test_minimal_mlp_converter(self):
        """Test MinimalMLPConverter initialization and forward pass."""
        model = MinimalMLPConverter(
            self.num_source_kpts,
            self.num_target_kpts,
            hidden_dim=128
        )
        model.to(self.device)
        
        # Create random input
        input_tensor = torch.randn(self.batch_size, self.num_source_kpts * 2)
        
        # Forward pass
        output = model(input_tensor)
        
        # Check output shape
        expected_shape = (self.batch_size, self.num_target_kpts, 2)
        self.assertEqual(output.shape, expected_shape)
    
    def test_attention_mlp_converter(self):
        """Test KeypointConverterMLPWithAttention initialization and forward pass."""
        model = KeypointConverterMLPWithAttention(
            self.num_source_kpts,
            self.num_target_kpts,
            embed_dim=128,
            num_heads=4,
            mlp_hidden_dim=256
        )
        model.to(self.device)
        
        # Create random input
        input_tensor = torch.randn(self.batch_size, self.num_source_kpts * 2)
        
        # Forward pass
        output = model(input_tensor)
        
        # Check output shape
        expected_shape = (self.batch_size, self.num_target_kpts, 2)
        self.assertEqual(output.shape, expected_shape)


class TestDataUtils(unittest.TestCase):
    """Test data utility functions."""
    
    def test_normalize_denormalize_keypoints(self):
        """Test keypoint normalization and denormalization."""
        batch_size = 2
        num_keypoints = 10
        
        # Create random keypoints and bounding boxes
        bboxes = torch.tensor([
            [100, 100, 200, 200],  # [x, y, width, height]
            [150, 150, 100, 100]
        ], dtype=torch.float32)
        
        # Create keypoints that are mostly within the bounding boxes
        # For bbox1 [100,100,200,200]: valid range is x:[100,300], y:[100,300]
        # For bbox2 [150,150,100,100]: valid range is x:[150,250], y:[150,250]
        keypoints = torch.zeros(batch_size, num_keypoints, 2)
        # Add some randomness but keep points reasonably close to bbox
        keypoints[0] = torch.rand(num_keypoints, 2) * 300 + 50  # Points in [50, 350]
        keypoints[1] = torch.rand(num_keypoints, 2) * 150 + 125  # Points in [125, 275]
        
        # Normalize
        normalized = normalize_keypoints_bbox(keypoints, bboxes)
        
        # Check normalized values are in reasonable range
        # Since points can be outside bbox, allow for values outside [0,1]
        self.assertTrue(torch.all(normalized >= -1))
        self.assertTrue(torch.all(normalized <= 3))
        
        # Denormalize
        denormalized = denormalize_keypoints_bbox(normalized, bboxes)
        
        # Check that denormalization is inverse of normalization
        torch.testing.assert_close(denormalized, keypoints, rtol=1e-5, atol=1e-5)
    
    def test_normalization_with_zero_bbox(self):
        """Test normalization handles zero-sized bounding boxes gracefully."""
        keypoints = torch.randn(1, 5, 2)
        bbox_zero_width = torch.tensor([[100, 100, 0, 200]], dtype=torch.float32)
        
        # Should handle zero width/height without error
        normalized = normalize_keypoints_bbox(keypoints, bbox_zero_width)
        self.assertEqual(normalized.shape, keypoints.shape)


class TestConstants(unittest.TestCase):
    """Test constants and configurations."""
    
    def test_model_configs(self):
        """Test that all model configurations are valid."""
        required_keys = {'description', 'class_name', 'params'}
        
        for model_name, config in MODEL_CONFIGS.items():
            with self.subTest(model=model_name):
                # Check required keys exist
                self.assertTrue(required_keys.issubset(config.keys()))
                
                # Check description is non-empty
                self.assertTrue(len(config['description']) > 0)
                
                # Check class_name is non-empty
                self.assertTrue(len(config['class_name']) > 0)
                
                # Check params is a dict
                self.assertIsInstance(config['params'], dict)
    
    def test_conversion_modes(self):
        """Test conversion mode configurations."""
        for mode_name, config in CONVERSION_MODES.items():
            with self.subTest(mode=mode_name):
                # Check target_kpt_slice_idx exists
                self.assertIn('target_kpt_slice_idx', config)
                
                # For non-custom modes, check keypoint numbers
                if mode_name != 'custom':
                    self.assertIn('num_source_kpt', config)
                    self.assertIn('num_target_kpt', config)
                    self.assertIsInstance(config['num_source_kpt'], int)
                    self.assertIsInstance(config['num_target_kpt'], int)
                    self.assertGreater(config['num_source_kpt'], 0)
                    self.assertGreater(config['num_target_kpt'], 0)
    
    def test_fixed_mapping(self):
        """Test the fixed 68 to 49 keypoint mapping."""
        # Check all source indices are valid for 68 keypoints
        for source_idx in FIXED_MAPPING_68_TO_49.keys():
            self.assertGreaterEqual(source_idx, 0)
            self.assertLess(source_idx, 68)
        
        # Check all target indices are valid for 49 keypoints
        for target_idx in FIXED_MAPPING_68_TO_49.values():
            self.assertGreaterEqual(target_idx, 0)
            self.assertLess(target_idx, 49)


class TestCOCODataCreation(unittest.TestCase):
    """Test synthetic COCO data creation for testing."""
    
    def create_synthetic_coco_json(
        self,
        num_images: int = 5,
        num_source_kpts: int = 68,
        num_target_kpts: int = 49
    ) -> Dict:
        """Create a synthetic COCO JSON for testing.
        
        Args:
            num_images: Number of images to create.
            num_source_kpts: Number of source keypoints.
            num_target_kpts: Number of target keypoints.
            
        Returns:
            Dictionary in COCO format.
        """
        coco_data = {
            'images': [],
            'annotations': [],
            'categories': [{'id': 1, 'name': 'primate'}]
        }
        
        for i in range(num_images):
            # Add image
            coco_data['images'].append({
                'id': i,
                'file_name': f'image_{i:04d}.jpg',
                'width': 640,
                'height': 480
            })
            
            # Add annotation with both keypoint sets
            source_kpts = []
            for _ in range(num_source_kpts):
                x = np.random.randint(100, 540)
                y = np.random.randint(100, 380)
                v = 2  # visible
                source_kpts.extend([x, y, v])
            
            target_kpts = []
            for _ in range(num_target_kpts):
                x = np.random.randint(100, 540)
                y = np.random.randint(100, 380)
                v = 2  # visible
                target_kpts.extend([x, y, v])
            
            coco_data['annotations'].append({
                'id': i,
                'image_id': i,
                'category_id': 1,
                'bbox': [100, 100, 200, 200],
                'keypoints': source_kpts,
                'num_keypoints': num_source_kpts,
                'keypoints_49': target_kpts,
                'num_keypoints_49': num_target_kpts
            })
        
        return coco_data
    
    def test_create_and_save_synthetic_data(self):
        """Test creating and saving synthetic COCO data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create synthetic data
            coco_data = self.create_synthetic_coco_json(num_images=3)
            
            # Save to file
            json_path = Path(tmpdir) / 'test_annotations.json'
            with open(json_path, 'w') as f:
                json.dump(coco_data, f)
            
            # Verify file exists and can be loaded
            self.assertTrue(json_path.exists())
            
            with open(json_path, 'r') as f:
                loaded_data = json.load(f)
            
            self.assertEqual(len(loaded_data['images']), 3)
            self.assertEqual(len(loaded_data['annotations']), 3)


class TestImports(unittest.TestCase):
    """Test that all modules can be imported."""
    
    def test_import_constants(self):
        """Test importing constants module."""
        import constants
        self.assertTrue(hasattr(constants, 'MODEL_CONFIGS'))
        self.assertTrue(hasattr(constants, 'DEFAULT_EPOCHS'))
    
    def test_import_models(self):
        """Test importing model classes."""
        from src import models
        self.assertTrue(hasattr(models, 'SimpleLinearConverter'))
        self.assertTrue(hasattr(models, 'KeypointConverterMLP'))
        self.assertTrue(hasattr(models, 'MinimalMLPConverter'))
        self.assertTrue(hasattr(models, 'KeypointConverterMLPWithAttention'))
    
    def test_import_training_pipeline(self):
        """Test importing training pipeline."""
        from src import training_pipeline
        self.assertTrue(hasattr(training_pipeline, 'ModelTrainer'))
    
    def test_import_data_utils(self):
        """Test importing data utilities."""
        from utils import data_utils
        self.assertTrue(hasattr(data_utils, 'CocoPairedKeypointDataset'))
        self.assertTrue(hasattr(data_utils, 'normalize_keypoints_bbox'))
        self.assertTrue(hasattr(data_utils, 'denormalize_keypoints_bbox'))


if __name__ == '__main__':
    unittest.main()