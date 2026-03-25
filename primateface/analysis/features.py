"""Landmark-derived geometric features for downstream modeling.

Provides pairwise landmark distances, temporal aggregation, and
face graph adjacency for skeleton-based models. All operations
are vectorized with NumPy for performance.
"""

from typing import Dict, List, Tuple

import numpy as np

from .utils import interocular_distance

# ============================================================================
# Face graph edges (68-point dlib skeleton connectivity)
# ============================================================================

SKELETON_EDGES: List[Tuple[int, int]] = []
# Jaw chain: 0-1-2-...-16
SKELETON_EDGES.extend([(i, i + 1) for i in range(16)])
# Right eyebrow: 17-21
SKELETON_EDGES.extend([(i, i + 1) for i in range(17, 21)])
# Left eyebrow: 22-26
SKELETON_EDGES.extend([(i, i + 1) for i in range(22, 26)])
# Nose bridge: 27-30
SKELETON_EDGES.extend([(i, i + 1) for i in range(27, 30)])
# Nose tip: 31-35
SKELETON_EDGES.extend([(i, i + 1) for i in range(31, 35)])
# Nose bridge → tip center
SKELETON_EDGES.append((30, 33))
# Right eye loop: 36-41-36
SKELETON_EDGES.extend([(i, i + 1) for i in range(36, 41)])
SKELETON_EDGES.append((41, 36))
# Left eye loop: 42-47-42
SKELETON_EDGES.extend([(i, i + 1) for i in range(42, 47)])
SKELETON_EDGES.append((47, 42))
# Outer mouth loop: 48-59-48
SKELETON_EDGES.extend([(i, i + 1) for i in range(48, 59)])
SKELETON_EDGES.append((59, 48))
# Inner mouth loop: 60-67-60
SKELETON_EDGES.extend([(i, i + 1) for i in range(60, 67)])
SKELETON_EDGES.append((67, 60))
# Cross-connections
SKELETON_EDGES.extend([(17, 36), (21, 39), (22, 42), (26, 45)])
SKELETON_EDGES.extend([(27, 39), (27, 42)])
SKELETON_EDGES.extend([(33, 51), (33, 62)])
SKELETON_EDGES.extend([(5, 48), (11, 54), (8, 57)])

# Default anatomical distance pairs (landmark index pairs)
DEFAULT_DISTANCE_PAIRS: Dict[str, List[Tuple[int, int]]] = {
    "brow_eye": [
        (17, 36), (18, 37), (19, 38), (21, 39),
        (22, 42), (23, 43), (24, 44), (26, 45),
    ],
    "eye_nose": [(39, 30), (42, 30), (36, 31), (45, 35)],
    "nose_mouth": [(30, 51), (30, 57), (33, 62), (33, 66)],
    "mouth_jaw": [(48, 5), (54, 11), (57, 8)],
    "eye_mouth": [(39, 48), (42, 54)],
    "brow_brow": [(21, 22)],
}


def build_face_adjacency(num_nodes: int = 68) -> np.ndarray:
    """Build the 68-point face graph adjacency matrix.

    Symmetric normalized: D^{-1/2} A D^{-1/2} with self-loops.

    Args:
        num_nodes: Number of landmark nodes.

    Returns:
        (N, N) float32 normalized adjacency matrix.
    """
    adj = np.eye(num_nodes, dtype=np.float32)
    for i, j in SKELETON_EDGES:
        if i < num_nodes and j < num_nodes:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    degree = adj.sum(axis=1)
    d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    d_mat = np.diag(d_inv_sqrt)
    return (d_mat @ adj @ d_mat).astype(np.float32)


def pairwise_distances(
    keypoints: np.ndarray,
    pairs: List[Tuple[int, int]],
    normalize_by_iod: bool = True,
) -> np.ndarray:
    """Compute pairwise Euclidean distances between landmark pairs.

    Vectorized: handles single frame (68, 2+) or batch (T, 68, 2+).

    Args:
        keypoints: (68, 2+) or (T, 68, 2+) landmark array.
        pairs: List of (i, j) index pairs.
        normalize_by_iod: Normalize each distance by interocular distance.

    Returns:
        (D,) or (T, D) array of distances where D = len(pairs).
    """
    single = keypoints.ndim == 2
    if single:
        keypoints = keypoints[np.newaxis]  # (1, 68, 2+)

    coords = keypoints[:, :, :2]  # (T, 68, 2)
    t = coords.shape[0]
    idx_i = np.array([p[0] for p in pairs])
    idx_j = np.array([p[1] for p in pairs])

    # Vectorized distance: (T, D)
    diffs = coords[:, idx_i] - coords[:, idx_j]  # (T, D, 2)
    dists = np.linalg.norm(diffs, axis=2)  # (T, D)

    if normalize_by_iod:
        for t_idx in range(t):
            iod = interocular_distance(keypoints[t_idx])
            if iod > 1e-6:
                dists[t_idx] /= iod

    return dists[0] if single else dists


def aggregate_timeseries(
    per_frame: np.ndarray,
    stats: List[str] = ("mean", "std", "min", "max", "range"),
) -> np.ndarray:
    """Aggregate per-frame features into a fixed-length vector.

    Args:
        per_frame: (T, D) array of per-frame feature vectors.
        stats: Which summary statistics to compute.

    Returns:
        (D * len(stats),) aggregated feature vector.
    """
    if per_frame.ndim == 1:
        per_frame = per_frame[np.newaxis]
    if per_frame.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)

    parts = []
    for stat in stats:
        if stat == "mean":
            parts.append(per_frame.mean(axis=0))
        elif stat == "std":
            parts.append(per_frame.std(axis=0))
        elif stat == "min":
            parts.append(per_frame.min(axis=0))
        elif stat == "max":
            parts.append(per_frame.max(axis=0))
        elif stat == "range":
            parts.append(per_frame.max(axis=0) - per_frame.min(axis=0))

    return np.concatenate(parts).astype(np.float32)
