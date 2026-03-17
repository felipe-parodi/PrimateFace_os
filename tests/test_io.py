"""Tests for primateface.io export utilities."""

import json

import pandas as pd
import pytest

from primateface.face import Face
from primateface.io import (
    from_coco_json,
    to_coco_json,
    to_csv,
    to_dataframe,
    to_dlc_csv,
)


@pytest.fixture
def sample_faces(synthetic_keypoints_68x3, dummy_image, sample_bbox):
    """Two Face objects for testing exports."""
    faces = []
    for score in [0.95, 0.72]:
        faces.append(Face(
            bbox=sample_bbox.copy(),
            score=score,
            keypoints=synthetic_keypoints_68x3.copy(),
            _image=dummy_image,
            _image_size=(200, 200),
        ))
    return faces


class TestToDataFrame:
    """Test to_dataframe conversion."""

    def test_returns_dataframe(self, sample_faces):
        df = to_dataframe(sample_faces)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_has_detection_columns(self, sample_faces):
        df = to_dataframe(sample_faces)
        for col in ("face_idx", "score", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"):
            assert col in df.columns

    def test_has_keypoint_columns(self, sample_faces):
        df = to_dataframe(sample_faces)
        assert "kpt_0_x" in df.columns
        assert "kpt_67_score" in df.columns

    def test_has_kinematic_columns(self, sample_faces):
        df = to_dataframe(sample_faces)
        assert "mouth_aperture" in df.columns
        assert "interocular_distance" in df.columns

    def test_has_head_pose_columns(self, sample_faces):
        df = to_dataframe(sample_faces)
        for col in ("yaw", "pitch", "roll"):
            assert col in df.columns

    def test_has_quality_columns(self, sample_faces):
        df = to_dataframe(sample_faces)
        assert "quality_blur" in df.columns
        assert "quality_score" in df.columns

    def test_has_symmetry_columns(self, sample_faces):
        df = to_dataframe(sample_faces)
        assert "symmetry_fa" in df.columns
        assert "symmetry_jaw" in df.columns

    def test_image_path_column(self, sample_faces):
        df = to_dataframe(sample_faces, image_path="monkey.jpg")
        assert "image" in df.columns
        assert df["image"].iloc[0] == "monkey.jpg"

    def test_exclude_features(self, sample_faces):
        df = to_dataframe(
            sample_faces,
            include_kinematics=False,
            include_head_pose=False,
            include_quality=False,
            include_symmetry=False,
        )
        assert "mouth_aperture" not in df.columns
        assert "yaw" not in df.columns
        assert "quality_blur" not in df.columns
        assert "symmetry_fa" not in df.columns


class TestToCsv:
    """Test CSV export."""

    def test_creates_file(self, sample_faces, tmp_path):
        out = to_csv(sample_faces, tmp_path / "test.csv")
        assert out.exists()

    def test_roundtrip(self, sample_faces, tmp_path):
        out = to_csv(sample_faces, tmp_path / "test.csv")
        df = pd.read_csv(out)
        assert len(df) == 2
        assert "score" in df.columns


class TestToCocoJson:
    """Test COCO JSON export."""

    def test_creates_file(self, sample_faces, tmp_path):
        out = to_coco_json(sample_faces, tmp_path / "test.json", image_path="img.jpg")
        assert out.exists()

    def test_valid_coco_structure(self, sample_faces, tmp_path):
        out = to_coco_json(sample_faces, tmp_path / "test.json", image_path="img.jpg")
        with open(out) as f:
            coco = json.load(f)
        assert "images" in coco
        assert "annotations" in coco
        assert "categories" in coco
        assert len(coco["annotations"]) == 2

    def test_coco_bbox_is_xywh(self, sample_faces, tmp_path):
        out = to_coco_json(sample_faces, tmp_path / "test.json")
        with open(out) as f:
            coco = json.load(f)
        bbox = coco["annotations"][0]["bbox"]
        assert len(bbox) == 4
        # bbox should be [x, y, w, h] where w, h > 0
        assert bbox[2] > 0
        assert bbox[3] > 0

    def test_keypoints_length(self, sample_faces, tmp_path):
        out = to_coco_json(sample_faces, tmp_path / "test.json")
        with open(out) as f:
            coco = json.load(f)
        kpts = coco["annotations"][0]["keypoints"]
        assert len(kpts) == 68 * 3


class TestFromCocoJson:
    """Test COCO JSON import."""

    def test_roundtrip(self, sample_faces, tmp_path):
        out = to_coco_json(sample_faces, tmp_path / "test.json", image_path="img.jpg")
        df = from_coco_json(out)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "kpt_0_x" in df.columns


class TestToDlcCsv:
    """Test DeepLabCut CSV export."""

    def test_creates_file(self, sample_faces, tmp_path):
        out = to_dlc_csv(sample_faces, tmp_path / "dlc.csv")
        assert out.exists()

    def test_multiindex_header(self, sample_faces, tmp_path):
        out = to_dlc_csv(sample_faces, tmp_path / "dlc.csv")
        df = pd.read_csv(out, header=[0, 1, 2], index_col=0)
        assert df.columns.nlevels == 3
        assert len(df) == 2
