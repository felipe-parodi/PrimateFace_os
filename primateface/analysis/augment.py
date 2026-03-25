"""Skeleton / landmark augmentation for temporal sequences.

Vectorized augmentations for (T, K, 2) landmark arrays:
jitter, rotation, flip, temporal speed, temporal crop.
"""

from typing import List, Optional, Tuple

import numpy as np

from .constants import SYMMETRIC_PAIRS


def jitter_landmarks(
    keypoints: np.ndarray,
    sigma: float = 0.02,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Add Gaussian noise to landmark coordinates.

    Noise is scaled relative to interocular distance for each frame.

    Args:
        keypoints: (T, K, 2) or (K, 2) landmark array.
        sigma: Noise std as fraction of IOD.
        rng: Random state for reproducibility.

    Returns:
        Augmented keypoints (same shape).
    """
    if rng is None:
        rng = np.random.RandomState()
    kpts = keypoints.copy()
    single = kpts.ndim == 2
    if single:
        kpts = kpts[np.newaxis]

    for t in range(kpts.shape[0]):
        left_eye = kpts[t, 42:48, :2].mean(axis=0)
        right_eye = kpts[t, 36:42, :2].mean(axis=0)
        iod = max(np.linalg.norm(left_eye - right_eye), 1e-6)
        noise = rng.randn(kpts.shape[1], 2).astype(np.float32) * sigma * iod
        kpts[t, :, :2] += noise

    return kpts[0] if single else kpts


def random_rotation(
    keypoints: np.ndarray,
    max_deg: float = 15.0,
    center_idx: int = 30,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Apply random in-plane rotation around a center landmark.

    Args:
        keypoints: (T, K, 2) or (K, 2) landmark array.
        max_deg: Maximum rotation angle in degrees.
        center_idx: Landmark index to rotate around (default: nose tip).
        rng: Random state.

    Returns:
        Rotated keypoints (same shape).
    """
    if rng is None:
        rng = np.random.RandomState()
    kpts = keypoints.copy()
    single = kpts.ndim == 2
    if single:
        kpts = kpts[np.newaxis]

    angle_rad = rng.uniform(-max_deg, max_deg) * np.pi / 180.0
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    for t in range(kpts.shape[0]):
        center = kpts[t, center_idx, :2].copy()
        coords = kpts[t, :, :2] - center
        kpts[t, :, :2] = (coords @ rot.T) + center

    return kpts[0] if single else kpts


def horizontal_flip(
    keypoints: np.ndarray,
    image_width: Optional[float] = None,
    symmetric_pairs: Optional[List[Tuple[int, int]]] = None,
) -> np.ndarray:
    """Mirror landmarks left-right and swap symmetric pairs.

    Args:
        keypoints: (T, K, 2) or (K, 2) landmark array.
        image_width: If provided, flip x = width - x. If None, flip
            around the midpoint of all x coordinates.
        symmetric_pairs: List of (left_idx, right_idx) pairs to swap.
            Defaults to dlib-68 SYMMETRIC_PAIRS.

    Returns:
        Flipped keypoints (same shape).
    """
    if symmetric_pairs is None:
        symmetric_pairs = SYMMETRIC_PAIRS

    kpts = keypoints.copy()
    single = kpts.ndim == 2
    if single:
        kpts = kpts[np.newaxis]

    for t in range(kpts.shape[0]):
        # Flip x coordinates
        if image_width is not None:
            kpts[t, :, 0] = image_width - kpts[t, :, 0]
        else:
            cx = kpts[t, :, 0].mean()
            kpts[t, :, 0] = 2 * cx - kpts[t, :, 0]

        # Swap symmetric landmark pairs
        for li, ri in symmetric_pairs:
            kpts[t, li, :2], kpts[t, ri, :2] = (
                kpts[t, ri, :2].copy(),
                kpts[t, li, :2].copy(),
            )

    return kpts[0] if single else kpts


def temporal_speed(
    keypoints: np.ndarray,
    factor_range: Tuple[float, float] = (0.8, 1.2),
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Resample temporal axis to simulate speed changes.

    Args:
        keypoints: (T, K, 2) landmark sequence.
        factor_range: (min_speed, max_speed) multiplier.
        rng: Random state.

    Returns:
        Resampled sequence (T, K, 2) — same length, different sampling.
    """
    if rng is None:
        rng = np.random.RandomState()
    t_orig = keypoints.shape[0]
    if t_orig < 3:
        return keypoints.copy()

    factor = rng.uniform(*factor_range)
    t_new = max(3, int(t_orig * factor))
    indices = np.linspace(0, t_orig - 1, t_new).astype(np.float32)

    # Linear interpolation
    result = np.zeros((t_new, *keypoints.shape[1:]), dtype=np.float32)
    for i, idx in enumerate(indices):
        lo = int(np.floor(idx))
        hi = min(lo + 1, t_orig - 1)
        alpha = idx - lo
        result[i] = (1 - alpha) * keypoints[lo] + alpha * keypoints[hi]

    # Resample back to original length
    if t_new != t_orig:
        final_indices = np.linspace(0, t_new - 1, t_orig, dtype=int)
        result = result[final_indices]

    return result


def temporal_crop_pad(
    keypoints: np.ndarray,
    crop_frac: float = 0.8,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Random temporal crop and pad back to original length.

    Args:
        keypoints: (T, K, 2) landmark sequence.
        crop_frac: Fraction of sequence to keep.
        rng: Random state.

    Returns:
        Cropped+padded sequence (T, K, 2).
    """
    if rng is None:
        rng = np.random.RandomState()
    t_orig = keypoints.shape[0]
    t_crop = max(3, int(t_orig * crop_frac))

    start = rng.randint(0, max(1, t_orig - t_crop + 1))
    cropped = keypoints[start : start + t_crop]

    # Pad back to original length by repeating last frame
    if cropped.shape[0] < t_orig:
        pad = np.repeat(cropped[-1:], t_orig - cropped.shape[0], axis=0)
        return np.concatenate([cropped, pad], axis=0)
    return cropped[:t_orig]
