"""Unit tests for GUI module.

These tests verify basic functionality without requiring actual models
or external dependencies.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import numpy as np

# Mock imports for modules that require external dependencies
sys_modules_backup = {}

def mock_imports():
    """Mock external dependencies for testing."""
    import sys
    
    # Mock mmdet
    sys.modules['mmdet'] = MagicMock()
    sys.modules['mmdet.apis'] = MagicMock()
    
    # Mock mmpose
    sys.modules['mmpose'] = MagicMock()
    sys.modules['mmpose.apis'] = MagicMock()
    sys.modules['mmpose.utils'] = MagicMock()
    sys.modules['mmpose.evaluation.functional'] = MagicMock()
    sys.modules['mmpose.structures'] = MagicMock()
    sys.modules['mmpose.registry'] = MagicMock()
    
    # Mock ultralytics
    sys.modules['ultralytics'] = MagicMock()
    sys.modules['ultralytics.models.sam'] = MagicMock()
    sys.modules['segment_anything'] = MagicMock()

mock_imports()

# Now import our modules
from constants import *
from core.models import ModelConfig, ModelManager
from converters.base import COCOConverter


class TestConstants(unittest.TestCase):
    """Test constants module."""
    
    def test_color_constants(self):
        """Test color constants are valid BGR tuples."""
        self.assertEqual(len(KEYPOINT_COLOR_LEFT), 3)
        self.assertEqual(len(KEYPOINT_COLOR_RIGHT), 3)
        self.assertEqual(len(KEYPOINT_COLOR_CENTER), 3)
        
        # Check values are in valid range
        for color in [KEYPOINT_COLOR_LEFT, KEYPOINT_COLOR_RIGHT, KEYPOINT_COLOR_CENTER]:
            for val in color:
                self.assertGreaterEqual(val, 0)
                self.assertLessEqual(val, 255)
                
    def test_default_parameters(self):
        """Test default parameter values."""
        self.assertIsInstance(DEFAULT_BBOX_THR, float)
        self.assertGreater(DEFAULT_BBOX_THR, 0)
        self.assertLess(DEFAULT_BBOX_THR, 1)
        
        self.assertIsInstance(DEFAULT_MAX_MONKEYS, int)
        self.assertGreater(DEFAULT_MAX_MONKEYS, 0)
        
    def test_keypoint_names(self):
        """Test keypoint name lists."""
        self.assertEqual(len(COCO17_KEYPOINTS), 17)
        self.assertIn("nose", COCO17_KEYPOINTS)
        self.assertIn("left_shoulder", COCO17_KEYPOINTS)
        
    def test_file_extensions(self):
        """Test file extension lists."""
        self.assertIn('.jpg', IMAGE_EXTENSIONS)
        self.assertIn('.mp4', VIDEO_EXTENSIONS)
        
        # Check all extensions start with dot
        for ext in IMAGE_EXTENSIONS + VIDEO_EXTENSIONS:
            self.assertTrue(ext.startswith('.'))


class TestModelConfig(unittest.TestCase):
    """Test ModelConfig class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.py")
        self.checkpoint_path = os.path.join(self.temp_dir, "model.pth")
        
        # Create dummy files
        Path(self.config_path).touch()
        Path(self.checkpoint_path).touch()
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_model_config_creation(self):
        """Test creating ModelConfig instance."""
        config = ModelConfig(
            name="test_model",
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            device="cuda:0",
            model_type="detection"
        )
        
        self.assertEqual(config.name, "test_model")
        self.assertEqual(config.config_path, self.config_path)
        self.assertEqual(config.device, "cuda:0")
        self.assertEqual(config.model_type, "detection")
        
    def test_model_config_validation(self):
        """Test ModelConfig validation."""
        config = ModelConfig(
            name="test",
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path
        )
        self.assertTrue(config.validate())
        
        # Test with non-existent files
        config_invalid = ModelConfig(
            name="test",
            config_path="/nonexistent/path.py",
            checkpoint_path="/nonexistent/model.pth"
        )
        self.assertFalse(config_invalid.validate())
        
    def test_model_config_serialization(self):
        """Test ModelConfig to/from dict conversion."""
        config = ModelConfig(
            name="test_model",
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            description="Test model"
        )
        
        # Convert to dict
        config_dict = config.to_dict()
        self.assertEqual(config_dict['name'], "test_model")
        self.assertEqual(config_dict['description'], "Test model")
        
        # Convert from dict
        config_restored = ModelConfig.from_dict(config_dict)
        self.assertEqual(config_restored.name, config.name)
        self.assertEqual(config_restored.description, config.description)


class TestModelManager(unittest.TestCase):
    """Test ModelManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ModelManager()
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_model_registration(self):
        """Test registering models."""
        # Create dummy files
        config_path = os.path.join(self.temp_dir, "config.py")
        checkpoint_path = os.path.join(self.temp_dir, "model.pth")
        Path(config_path).touch()
        Path(checkpoint_path).touch()
        
        # Register model
        self.manager.register_model(
            name="test_detector",
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            model_type="detection"
        )
        
        self.assertIn("test_detector", self.manager.configs)
        
        # Test duplicate registration
        with self.assertRaises(ValueError):
            self.manager.register_model(
                name="test_detector",
                config_path=config_path,
                checkpoint_path=checkpoint_path
            )
            
    def test_model_listing(self):
        """Test listing registered models."""
        # Create dummy files
        config_path = os.path.join(self.temp_dir, "config.py")
        checkpoint_path = os.path.join(self.temp_dir, "model.pth")
        Path(config_path).touch()
        Path(checkpoint_path).touch()
        
        # Register models
        self.manager.register_model(
            "detector1", config_path, checkpoint_path, "detection"
        )
        self.manager.register_model(
            "pose1", config_path, checkpoint_path, "pose"
        )
        
        # List all models
        all_models = self.manager.list_models()
        self.assertEqual(len(all_models), 2)
        self.assertIn("detector1", all_models)
        self.assertIn("pose1", all_models)
        
        # List by type
        detectors = self.manager.list_models("detection")
        self.assertEqual(len(detectors), 1)
        self.assertIn("detector1", detectors)
        
    def test_config_persistence(self):
        """Test saving and loading configurations."""
        config_file = os.path.join(self.temp_dir, "models.json")
        
        # Create dummy model files
        model_config = os.path.join(self.temp_dir, "config.py")
        model_checkpoint = os.path.join(self.temp_dir, "model.pth")
        Path(model_config).touch()
        Path(model_checkpoint).touch()
        
        # Register and save
        self.manager.register_model(
            "test_model", model_config, model_checkpoint
        )
        self.manager.save_configs(config_file)
        
        # Load in new manager
        new_manager = ModelManager(config_file)
        self.assertIn("test_model", new_manager.configs)
        self.assertEqual(
            new_manager.configs["test_model"].config_path,
            model_config
        )


class TestCOCOConverter(unittest.TestCase):
    """Test COCO converter base class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock detector and pose estimator
        self.mock_detector = MagicMock()
        self.mock_pose = MagicMock()
        
        # Create converter with mocks
        self.converter = COCOConverter(
            detector=self.mock_detector,
            pose_estimator=self.mock_pose
        )
        
    def test_coco_structure_initialization(self):
        """Test COCO data structure initialization."""
        coco_data = self.converter.coco_data
        
        # Check required fields
        self.assertIn('info', coco_data)
        self.assertIn('images', coco_data)
        self.assertIn('annotations', coco_data)
        self.assertIn('categories', coco_data)
        
        # Check categories
        self.assertEqual(len(coco_data['categories']), 1)
        self.assertEqual(coco_data['categories'][0]['name'], 'primate')
        
    def test_add_image(self):
        """Test adding image to COCO data."""
        img_id = self.converter.add_image(
            file_name="test.jpg",
            width=640,
            height=480
        )
        
        self.assertEqual(img_id, 1)
        self.assertEqual(len(self.converter.coco_data['images']), 1)
        
        image = self.converter.coco_data['images'][0]
        self.assertEqual(image['file_name'], "test.jpg")
        self.assertEqual(image['width'], 640)
        self.assertEqual(image['height'], 480)
        
    def test_add_annotation(self):
        """Test adding annotation to COCO data."""
        # Add image first
        img_id = self.converter.add_image("test.jpg", 640, 480)
        
        # Add annotation with bbox
        ann_id = self.converter.add_annotation(
            image_id=img_id,
            bbox=[100, 100, 200, 200],  # x1, y1, x2, y2
            score=0.95
        )
        
        self.assertEqual(ann_id, 1)
        self.assertEqual(len(self.converter.coco_data['annotations']), 1)
        
        annotation = self.converter.coco_data['annotations'][0]
        self.assertEqual(annotation['image_id'], img_id)
        self.assertEqual(annotation['bbox'], [100, 100, 100, 100])  # Converted to x,y,w,h
        self.assertEqual(annotation['score'], 0.95)
        
    def test_add_annotation_with_keypoints(self):
        """Test adding annotation with keypoints."""
        img_id = self.converter.add_image("test.jpg", 640, 480)
        
        # Test with flat keypoints
        flat_kpts = [100, 100, 2, 110, 110, 2, 120, 120, 1]
        ann_id = self.converter.add_annotation(
            image_id=img_id,
            bbox=[100, 100, 50, 50],
            keypoints=flat_kpts
        )
        
        annotation = self.converter.coco_data['annotations'][0]
        self.assertEqual(annotation['keypoints'], flat_kpts)
        self.assertEqual(annotation['num_keypoints'], 2)  # Two visible
        
    def test_statistics(self):
        """Test getting statistics."""
        # Add some data
        img_id1 = self.converter.add_image("test1.jpg", 640, 480)
        img_id2 = self.converter.add_image("test2.jpg", 640, 480)
        
        self.converter.add_annotation(img_id1, [10, 10, 50, 50])
        self.converter.add_annotation(img_id1, [60, 60, 50, 50])
        self.converter.add_annotation(img_id2, [10, 10, 50, 50])
        
        stats = self.converter.get_statistics()
        
        self.assertEqual(stats['num_images'], 2)
        self.assertEqual(stats['num_annotations'], 3)
        self.assertAlmostEqual(stats['avg_annotations_per_image'], 1.5)
        
    def test_filter_by_confidence(self):
        """Test filtering annotations by confidence."""
        img_id = self.converter.add_image("test.jpg", 640, 480)
        
        # Add annotations with different scores
        self.converter.add_annotation(img_id, [10, 10, 50, 50], score=0.9)
        self.converter.add_annotation(img_id, [60, 60, 50, 50], score=0.5)
        self.converter.add_annotation(img_id, [110, 110, 50, 50], score=0.3)
        
        # Filter by confidence
        self.converter.filter_by_confidence(min_score=0.6)
        
        self.assertEqual(len(self.converter.coco_data['annotations']), 1)
        self.assertEqual(self.converter.coco_data['annotations'][0]['score'], 0.9)
        
    def test_save_json(self):
        """Test saving COCO data to JSON."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test.json")
            
            # Add some data
            img_id = self.converter.add_image("test.jpg", 640, 480)
            self.converter.add_annotation(img_id, [10, 10, 50, 50])
            
            # Save
            self.converter.save(output_path)
            
            # Verify file exists and is valid JSON
            self.assertTrue(os.path.exists(output_path))
            
            with open(output_path, 'r') as f:
                loaded_data = json.load(f)
                
            self.assertEqual(len(loaded_data['images']), 1)
            self.assertEqual(len(loaded_data['annotations']), 1)


class TestPseudolabelScript(unittest.TestCase):
    """Test main pseudolabel script functions."""
    
    @patch('pseudolabel.Path')
    def test_detect_input_type(self, mock_path_class):
        """Test input type detection."""
        from pseudolabel import detect_input_type
        
        # Test JSON file
        mock_path = MagicMock()
        mock_path.suffix.lower.return_value = '.json'
        mock_path_class.return_value = mock_path
        
        result = detect_input_type("/path/to/annotations.json")
        self.assertEqual(result, 'coco')
        
        # Test video file
        mock_path.suffix.lower.return_value = '.mp4'
        mock_path.is_dir.return_value = False
        
        result = detect_input_type("/path/to/video.mp4")
        self.assertEqual(result, 'videos')
        
    def test_argparser_creation(self):
        """Test argument parser setup."""
        from pseudolabel import setup_argparser
        
        parser = setup_argparser()
        
        # Test required arguments
        args = parser.parse_args([
            '--input', './images',
            '--det-config', 'config.py',
            '--det-checkpoint', 'model.pth'
        ])
        
        self.assertEqual(args.input, './images')
        self.assertEqual(args.det_config, 'config.py')
        self.assertEqual(args.det_checkpoint, 'model.pth')
        self.assertEqual(args.type, 'auto')  # Default
        
        # Test optional arguments
        args = parser.parse_args([
            '--input', './videos',
            '--det-config', 'config.py',
            '--det-checkpoint', 'model.pth',
            '--type', 'videos',
            '--frame-interval', '2.0',
            '--max-frames', '100'
        ])
        
        self.assertEqual(args.type, 'videos')
        self.assertEqual(args.frame_interval, 2.0)
        self.assertEqual(args.max_frames, 100)


if __name__ == '__main__':
    unittest.main()