"""Tests for the PrimateFace orchestrator class."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from primateface.core import PrimateFace


class TestLoadImage:
    """Test PrimateFace._load_image static method."""

    def test_load_from_file_path(self):
        """Load a real image file."""
        img_path = Path(__file__).resolve().parent.parent / "demos" / "ateles_000003.jpeg"
        if not img_path.exists():
            pytest.skip("Test image not found")
        bgr = PrimateFace._load_image(str(img_path))
        assert isinstance(bgr, np.ndarray)
        assert bgr.ndim == 3
        assert bgr.shape[2] == 3

    def test_load_from_path_object(self):
        img_path = Path(__file__).resolve().parent.parent / "demos" / "ateles_000003.jpeg"
        if not img_path.exists():
            pytest.skip("Test image not found")
        bgr = PrimateFace._load_image(img_path)
        assert bgr.ndim == 3

    def test_load_from_numpy_bgr(self, dummy_image):
        bgr = PrimateFace._load_image(dummy_image)
        assert bgr is dummy_image  # should return same reference

    def test_load_from_numpy_grayscale(self):
        gray = np.zeros((100, 100), dtype=np.uint8)
        bgr = PrimateFace._load_image(gray)
        assert bgr.ndim == 3
        assert bgr.shape == (100, 100, 3)

    def test_load_from_numpy_rgba(self):
        rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        bgr = PrimateFace._load_image(rgba)
        assert bgr.ndim == 3
        assert bgr.shape == (100, 100, 3)

    def test_load_from_pil_image(self):
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not installed")
        pil_img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        bgr = PrimateFace._load_image(pil_img)
        assert bgr.shape == (100, 100, 3)

    def test_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            PrimateFace._load_image("/nonexistent/image.jpg")

    def test_raises_type_error(self):
        with pytest.raises(TypeError):
            PrimateFace._load_image(12345)

    def test_raises_value_error_bad_shape(self):
        bad = np.zeros((100, 100, 5), dtype=np.uint8)
        with pytest.raises(ValueError):
            PrimateFace._load_image(bad)


class TestAnalyze:
    """Test PrimateFace.analyze with mocked processor."""

    @pytest.fixture
    def pf_mocked(self, mock_processor):
        """PrimateFace instance with mocked internals."""
        with patch.object(PrimateFace, "__init__", lambda self, **kw: None):
            pf = PrimateFace.__new__(PrimateFace)
            pf._processor = mock_processor
            pf.det_threshold = 0.5
            pf.nms_threshold = 0.3
            pf.device = "cpu"
        return pf

    def test_analyze_returns_faces(self, pf_mocked, dummy_image):
        faces = pf_mocked.analyze(dummy_image)
        assert isinstance(faces, list)
        assert len(faces) == 2
        from primateface.face import Face
        assert all(isinstance(f, Face) for f in faces)

    def test_analyze_sorted_by_confidence(self, pf_mocked, dummy_image):
        faces = pf_mocked.analyze(dummy_image)
        scores = [f.score for f in faces]
        assert scores == sorted(scores, reverse=True)

    def test_analyze_no_detections(self, pf_mocked, dummy_image):
        pf_mocked._processor.detect_primates.return_value = (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
        faces = pf_mocked.analyze(dummy_image)
        assert faces == []

    def test_analyze_keypoints_shape(self, pf_mocked, dummy_image):
        faces = pf_mocked.analyze(dummy_image)
        assert faces[0].keypoints.shape == (68, 3)

    def test_analyze_batch(self, pf_mocked, dummy_image):
        results = pf_mocked.analyze_batch([dummy_image, dummy_image])
        assert len(results) == 2
        assert all(isinstance(r, list) for r in results)
