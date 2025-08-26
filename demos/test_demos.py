"""Simple unit tests for PrimateFace demo modules.

These tests check basic functionality without requiring actual models or data.
Run with: python -m pytest demos/test_demos.py
"""

import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from demos import constants
from demos.smooth_utils import MedianSavgolSmoother
from demos.viz_utils import FastPoseVisualizer


class TestConstants(unittest.TestCase):
    """Test constants module."""
    
    def test_color_constants(self):
        """Test that color constants are properly defined."""
        self.assertIsInstance(constants.ID_COLORS_BGR, dict)
        self.assertIsInstance(constants.DEFAULT_COLOR_BGR, tuple)
        self.assertEqual(len(constants.DEFAULT_COLOR_BGR), 3)
        
        # Check BGR values are in valid range
        for color_id, color in constants.ID_COLORS_BGR.items():
            self.assertEqual(len(color), 3)
            for channel in color:
                self.assertGreaterEqual(channel, 0)
                self.assertLessEqual(channel, 255)
    
    def test_threshold_constants(self):
        """Test threshold constants are valid."""
        self.assertIsInstance(constants.DEFAULT_BBOX_THR, float)
        self.assertIsInstance(constants.DEFAULT_KPT_THR, float)
        self.assertIsInstance(constants.DEFAULT_NMS_THR, float)
        
        # Check thresholds are in valid range [0, 1]
        self.assertGreaterEqual(constants.DEFAULT_BBOX_THR, 0)
        self.assertLessEqual(constants.DEFAULT_BBOX_THR, 1)
        self.assertGreaterEqual(constants.DEFAULT_KPT_THR, 0)
        self.assertLessEqual(constants.DEFAULT_KPT_THR, 1)
        self.assertGreaterEqual(constants.DEFAULT_NMS_THR, 0)
        self.assertLessEqual(constants.DEFAULT_NMS_THR, 1)
    
    def test_primate_genera(self):
        """Test primate genera list."""
        self.assertIsInstance(constants.PRIMATE_GENERA, list)
        self.assertGreater(len(constants.PRIMATE_GENERA), 0)
        # Don't check exact count as the genera list can change
        self.assertGreater(len(constants.PRIMATE_GENERA), 50) 
        
        # Check all are strings
        for genus in constants.PRIMATE_GENERA:
            self.assertIsInstance(genus, str)
            self.assertTrue(genus)  # Not empty
    
    def test_file_extensions(self):
        """Test file extension lists."""
        self.assertIsInstance(constants.IMAGE_EXTENSIONS, list)
        self.assertIsInstance(constants.VIDEO_EXTENSIONS, list)
        
        # Check they contain expected formats
        self.assertIn("*.jpg", constants.IMAGE_EXTENSIONS)
        self.assertIn("*.png", constants.IMAGE_EXTENSIONS)
        self.assertIn("*.mp4", constants.VIDEO_EXTENSIONS)


class TestMedianSavgolSmoother(unittest.TestCase):
    """Test MedianSavgolSmoother class."""
    
    def test_initialization(self):
        """Test smoother initialization."""
        smoother = MedianSavgolSmoother(
            median_window=5,
            savgol_window=7,
            savgol_order=3
        )
        self.assertEqual(smoother.median_window, 5)
        self.assertEqual(smoother.savgol_window, 7)
        self.assertEqual(smoother.savgol_order, 3)
    
    def test_invalid_initialization(self):
        """Test smoother with invalid parameters."""
        # Even window size should raise error
        with self.assertRaises(ValueError):
            MedianSavgolSmoother(median_window=4)
        
        # Negative window size should raise error
        with self.assertRaises(ValueError):
            MedianSavgolSmoother(median_window=-1)
        
        # Order >= window should raise error
        with self.assertRaises(ValueError):
            MedianSavgolSmoother(savgol_window=5, savgol_order=5)
    
    def test_update_with_insufficient_history(self):
        """Test that update returns original keypoints when history is insufficient."""
        smoother = MedianSavgolSmoother()
        
        # Create dummy keypoints
        keypoints = np.array([[100, 200], [150, 250]])
        scores = np.array([0.9, 0.8])
        
        # First update should return original
        result = smoother.update(
            instance_id=1,
            keypoints=keypoints,
            keypoint_scores=scores
        )
        np.testing.assert_array_equal(result, keypoints)
    
    def test_reset_history(self):
        """Test history reset functionality."""
        smoother = MedianSavgolSmoother()
        
        # Add some history
        keypoints = np.array([[100, 200], [150, 250]])
        scores = np.array([0.9, 0.8])
        smoother.update(1, keypoints, scores)
        
        # Reset specific ID
        smoother.reset_history(1)
        self.assertNotIn(1, smoother.history)
        
        # Add history for multiple IDs
        smoother.update(1, keypoints, scores)
        smoother.update(2, keypoints, scores)
        
        # Reset all
        smoother.reset_history()
        self.assertEqual(len(smoother.history), 0)


class TestFastPoseVisualizer(unittest.TestCase):
    """Test FastPoseVisualizer class."""
    
    def test_initialization(self):
        """Test visualizer initialization."""
        viz = FastPoseVisualizer(
            draw_keypoints=True,
            draw_skeleton=False,
            draw_bbox=True,
            keypoint_radius=5,
            line_thickness=2
        )
        
        self.assertTrue(viz.draw_keypoints)
        self.assertFalse(viz.draw_skeleton)
        self.assertTrue(viz.draw_bbox)
        self.assertEqual(viz.keypoint_radius, 5)
        self.assertEqual(viz.line_thickness, 2)
    
    def test_draw_poses_empty(self):
        """Test drawing with no poses."""
        viz = FastPoseVisualizer()
        
        # Create dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw with None instances
        result = viz.draw_poses(frame, None)
        np.testing.assert_array_equal(result, frame)
    
    def test_color_selection(self):
        """Test that colors are properly selected from constants."""
        viz = FastPoseVisualizer()
        
        # Check that ID colors match constants
        self.assertEqual(viz.ID_COLORS, constants.ID_COLORS_BGR)
        self.assertEqual(viz.DEFAULT_COLOR, constants.DEFAULT_COLOR_BGR)
        self.assertEqual(viz.TEXT_COLOR, constants.TEXT_COLOR_BGR)


class TestImports(unittest.TestCase):
    """Test that all modules can be imported."""
    
    def test_import_constants(self):
        """Test importing constants module."""
        from demos import constants
        self.assertIsNotNone(constants)
    
    def test_import_smooth_utils(self):
        """Test importing smooth_utils module."""
        from demos.smooth_utils import MedianSavgolSmoother
        self.assertIsNotNone(MedianSavgolSmoother)
    
    def test_import_viz_utils(self):
        """Test importing viz_utils module."""
        from demos.viz_utils import FastPoseVisualizer
        self.assertIsNotNone(FastPoseVisualizer)
    
    def test_import_process(self):
        """Test importing process module (without models)."""
        try:
            from demos.process import PrimateFaceProcessor
            # Will fail at runtime without models, but import should work
        except ImportError as e:
            if "mmdet" in str(e) or "mmpose" in str(e):
                self.skipTest("MMDetection/MMPose not installed")
            else:
                raise


def create_dummy_instances():
    """Create dummy instances for testing visualization."""
    class DummyInstances:
        def __init__(self):
            self.keypoints = np.array([
                [[100, 100], [150, 120], [200, 140]],
                [[300, 300], [350, 320], [400, 340]]
            ])
            self.keypoint_scores = np.array([
                [0.9, 0.8, 0.7],
                [0.95, 0.85, 0.75]
            ])
    
    return DummyInstances()


if __name__ == '__main__':
    unittest.main()