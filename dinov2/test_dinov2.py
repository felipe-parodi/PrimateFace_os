"""Unit tests for DINOv2 modules.

These tests check basic functionality without requiring actual models or data.
Run with: python -m pytest dinov2/test_dinov2.py
"""

import sys
import tempfile
import unittest
from pathlib import Path
from typing import List

import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dinov2 import constants
from dinov2.selection import DiverseImageSelector
from dinov2.visualization import UMAPVisualizer


class TestConstants(unittest.TestCase):
    """Test constants module."""
    
    def test_model_configs(self):
        """Test that model configurations are properly defined."""
        self.assertIsInstance(constants.DINOV2_MODELS, dict)
        self.assertIn("base", constants.DINOV2_MODELS)
        self.assertIn("facebook/dinov2", constants.DEFAULT_MODEL)
    
    def test_default_parameters(self):
        """Test default parameter values."""
        self.assertGreater(constants.DEFAULT_BATCH_SIZE, 0)
        self.assertGreater(constants.DEFAULT_NUM_CLUSTERS, 0)
        self.assertGreater(constants.DEFAULT_UMAP_N_NEIGHBORS, 0)
        self.assertGreater(constants.DEFAULT_UMAP_MIN_DIST, 0)
        self.assertLessEqual(constants.DEFAULT_UMAP_MIN_DIST, 1)
    
    def test_file_extensions(self):
        """Test supported file extensions."""
        self.assertIsInstance(constants.SUPPORTED_IMAGE_EXTENSIONS, list)
        self.assertIn(".jpg", constants.SUPPORTED_IMAGE_EXTENSIONS)
        self.assertIn(".png", constants.SUPPORTED_IMAGE_EXTENSIONS)


class TestDiverseImageSelector(unittest.TestCase):
    """Test DiverseImageSelector class."""
    
    def setUp(self):
        """Create test embeddings."""
        np.random.seed(42)
        self.embeddings = np.random.randn(100, 768)  # 100 samples, 768 dims
        self.image_ids = [f"image_{i}.jpg" for i in range(100)]
    
    def test_initialization(self):
        """Test selector initialization."""
        selector = DiverseImageSelector(strategy="random", random_state=42)
        self.assertEqual(selector.strategy, "random")
        self.assertEqual(selector.random_state, 42)
    
    def test_random_selection(self):
        """Test random selection strategy."""
        selector = DiverseImageSelector(strategy="random", random_state=42)
        indices, ids = selector.select(
            self.embeddings,
            n_samples=10,
            image_ids=self.image_ids
        )
        
        self.assertEqual(len(indices), 10)
        self.assertEqual(len(ids), 10)
        self.assertTrue(all(0 <= i < 100 for i in indices))
    
    def test_cluster_selection(self):
        """Test cluster-based selection."""
        selector = DiverseImageSelector(strategy="cluster", random_state=42)
        indices, ids = selector.select(
            self.embeddings,
            n_samples=20,
            n_clusters=5,
            image_ids=self.image_ids
        )
        
        self.assertEqual(len(indices), 20)
        self.assertEqual(len(set(indices)), 20)  # All unique
    
    def test_fps_selection(self):
        """Test farthest point sampling."""
        selector = DiverseImageSelector(strategy="fps", random_state=42)
        indices, _ = selector.select(
            self.embeddings[:20],  # Use smaller subset for speed
            n_samples=5
        )
        
        self.assertEqual(len(indices), 5)
        self.assertTrue(all(0 <= i < 20 for i in indices))
    
    def test_hybrid_selection(self):
        """Test hybrid selection strategy."""
        selector = DiverseImageSelector(strategy="hybrid", random_state=42)
        indices, ids = selector.select(
            self.embeddings,
            n_samples=15,
            n_clusters=3,
            image_ids=self.image_ids
        )
        
        self.assertEqual(len(indices), 15)
    
    def test_select_more_than_available(self):
        """Test selection when requesting more samples than available."""
        selector = DiverseImageSelector(strategy="random")
        indices, _ = selector.select(
            self.embeddings[:10],
            n_samples=20  # More than available
        )
        
        self.assertEqual(len(indices), 10)  # Should return all available


class TestUMAPVisualizer(unittest.TestCase):
    """Test UMAPVisualizer class."""
    
    def setUp(self):
        """Create test embeddings."""
        np.random.seed(42)
        self.embeddings = np.random.randn(50, 768)
    
    def test_initialization(self):
        """Test visualizer initialization."""
        viz = UMAPVisualizer(
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=42
        )
        
        self.assertEqual(viz.n_neighbors, 15)
        self.assertEqual(viz.min_dist, 0.1)
        self.assertEqual(viz.metric, "cosine")
    
    def test_fit_transform(self):
        """Test UMAP dimensionality reduction."""
        viz = UMAPVisualizer(random_state=42)
        umap_embeddings = viz.fit_transform(self.embeddings)
        
        self.assertEqual(umap_embeddings.shape, (50, 2))
        self.assertIsNotNone(viz.umap_embeddings)
        self.assertIsNotNone(viz.reducer)
    
    def test_cluster(self):
        """Test clustering functionality."""
        viz = UMAPVisualizer(random_state=42)
        viz.fit_transform(self.embeddings)
        labels = viz.cluster(n_clusters=5)
        
        self.assertEqual(len(labels), 50)
        self.assertEqual(len(np.unique(labels)), 5)
        self.assertIsNotNone(viz.cluster_labels)
    
    def test_cluster_without_umap(self):
        """Test clustering on original embeddings."""
        viz = UMAPVisualizer(random_state=42)
        labels = viz.cluster(self.embeddings, n_clusters=3)
        
        self.assertEqual(len(labels), 50)
        self.assertEqual(len(np.unique(labels)), 3)


class TestImports(unittest.TestCase):
    """Test that all modules can be imported."""
    
    def test_import_core(self):
        """Test importing core module."""
        try:
            from dinov2.core import DINOv2Extractor, ImageDataset
            self.assertIsNotNone(DINOv2Extractor)
            self.assertIsNotNone(ImageDataset)
        except ImportError as e:
            if "transformers" in str(e):
                self.skipTest("Transformers not installed")
            else:
                raise
    
    def test_import_visualization(self):
        """Test importing visualization module."""
        from dinov2.visualization import PatchVisualizer, UMAPVisualizer
        self.assertIsNotNone(UMAPVisualizer)
        self.assertIsNotNone(PatchVisualizer)
    
    def test_import_selection(self):
        """Test importing selection module."""
        from dinov2.selection import DiverseImageSelector, save_selection
        self.assertIsNotNone(DiverseImageSelector)
        self.assertIsNotNone(save_selection)
    
    def test_import_cli(self):
        """Test importing CLI module."""
        from dinov2.dinov2_cli import create_parser, main
        self.assertIsNotNone(create_parser)
        self.assertIsNotNone(main)


class TestHelperFunctions(unittest.TestCase):
    """Test helper functions."""
    
    def test_save_and_load_embeddings(self):
        """Test saving and loading embeddings."""
        from dinov2.core import load_embeddings, save_embeddings
        
        # Create test data
        embeddings = torch.randn(10, 768)
        image_ids = [f"img_{i}.jpg" for i in range(10)]
        metadata = {"test": "value"}
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            # Save
            save_embeddings(embeddings, image_ids, tmp_path, metadata)
            self.assertTrue(tmp_path.exists())
            
            # Load
            loaded_emb, loaded_ids, loaded_meta = load_embeddings(tmp_path)
            
            # Check
            torch.testing.assert_close(embeddings, loaded_emb)
            self.assertEqual(image_ids, loaded_ids)
            self.assertEqual(metadata, loaded_meta)
        finally:
            # Cleanup
            if tmp_path.exists():
                tmp_path.unlink()
    
    def test_save_selection(self):
        """Test saving selection to file."""
        from dinov2.selection import save_selection
        
        indices = np.array([0, 5, 10, 15])
        image_ids = [f"img_{i}.jpg" for i in range(20)]
        
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            save_selection(indices, tmp_path, image_ids)
            self.assertTrue(tmp_path.exists())
            
            # Read and verify
            with open(tmp_path, 'r') as f:
                lines = f.readlines()
            
            self.assertEqual(len(lines), 4)
            self.assertEqual(lines[0].strip(), "img_0.jpg")
            self.assertEqual(lines[1].strip(), "img_5.jpg")
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


if __name__ == '__main__':
    unittest.main()