"""Face dataclass with lazy analysis properties.

Each ``Face`` is produced by :meth:`PrimateFace.analyze` and carries
detection outputs (bbox, score, keypoints) plus derived features that
are computed on first access and cached thereafter.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np


@dataclass
class Face:
    """A single detected primate face with landmarks and derived features.

    Core detection fields are set at construction time. Analysis features
    (head_pose, symmetry, quality, kinematics) are computed lazily on first
    access via ``cached_property``.

    Attributes:
        bbox: Bounding box as ``[x1, y1, x2, y2]`` in pixel coordinates.
        score: Detection confidence in ``[0, 1]``.
        keypoints: Facial landmarks, shape ``(68, 3)`` with
            ``[x, y, confidence]``.
    """

    bbox: np.ndarray
    score: float
    keypoints: np.ndarray
    _image: np.ndarray = field(repr=False, compare=False)
    _image_size: Tuple[int, int] = field(repr=False, compare=False)

    # -- Eager convenience properties --

    @functools.cached_property
    def crop(self) -> np.ndarray:
        """Face crop extracted from the source image using the bbox."""
        x1, y1, x2, y2 = self.bbox.astype(int)
        h, w = self._image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return self._image[y1:y2, x1:x2].copy()

    # -- Lazy analysis properties --

    @functools.cached_property
    def head_pose(self) -> Tuple[float, float, float]:
        """Head pose as ``(yaw, pitch, roll)`` in degrees."""
        from analysis.head_pose import estimate_head_pose

        return estimate_head_pose(self.keypoints, self._image_size)

    @functools.cached_property
    def quality(self) -> Dict[str, float]:
        """Face quality metrics: blur, size, visibility, brightness, score."""
        from analysis.quality import face_quality

        return face_quality(self._image, self.bbox, self.keypoints)

    @functools.cached_property
    def symmetry(self) -> float:
        """Fluctuating asymmetry score (0 = perfect symmetry)."""
        from analysis.symmetry import facial_symmetry

        return facial_symmetry(self.keypoints)

    @functools.cached_property
    def region_symmetry(self) -> Dict[str, float]:
        """Per-region symmetry: jaw, eyebrows, eyes, nose, mouth."""
        from analysis.symmetry import per_region_symmetry

        return per_region_symmetry(self.keypoints)

    @functools.cached_property
    def kinematics(self) -> Dict[str, float]:
        """All scalar kinematic/geometric features (~14 keys)."""
        from analysis.kinematics import extract_kinematics

        return extract_kinematics(self.keypoints)

    # -- Shortcut properties that delegate to kinematics --

    @functools.cached_property
    def mouth_aperture(self) -> float:
        """Mouth opening (normalized by interocular distance)."""
        return self.kinematics["mouth_aperture"]

    @functools.cached_property
    def eye_aperture(self) -> Tuple[float, float]:
        """Eye opening as ``(right, left)``, normalized."""
        return (
            self.kinematics["right_eye_aperture"],
            self.kinematics["left_eye_aperture"],
        )

    @functools.cached_property
    def brow_position(self) -> Tuple[float, float]:
        """Brow height as ``(right, left)``, normalized."""
        return (
            self.kinematics["right_brow_height"],
            self.kinematics["left_brow_height"],
        )

    @functools.cached_property
    def interocular_distance(self) -> float:
        """Distance between eye centers in pixels."""
        return self.kinematics["interocular_distance"]

    # -- Serialization --

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (excludes image reference)."""
        return {
            "bbox": self.bbox.tolist(),
            "score": self.score,
            "keypoints": self.keypoints.tolist(),
        }

    def __repr__(self) -> str:
        x1, y1, x2, y2 = self.bbox.astype(int)
        return (
            f"Face(score={self.score:.2f}, "
            f"bbox=[{x1}, {y1}, {x2}, {y2}], "
            f"landmarks=68)"
        )
