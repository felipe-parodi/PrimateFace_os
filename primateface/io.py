"""Data export utilities for PrimateFace results.

Converts :class:`Face` objects to common interchange formats:
CSV, COCO JSON, DeepLabCut-compatible CSV/H5, and pandas DataFrames.

Example:
    >>> from primateface.io import to_csv, to_dataframe
    >>> faces = pf.analyze("monkey.jpg")
    >>> to_csv(faces, "results.csv", image_path="monkey.jpg")
    >>> df = to_dataframe(faces, image_path="monkey.jpg")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .face import Face


def to_dataframe(
    faces: List[Face],
    image_path: Optional[str] = None,
    include_kinematics: bool = True,
    include_head_pose: bool = True,
    include_quality: bool = True,
    include_symmetry: bool = True,
) -> pd.DataFrame:
    """Convert Face objects to a pandas DataFrame.

    One row per detected face, with columns for detection outputs and
    optionally all analysis features.

    Args:
        faces: List of Face objects from ``PrimateFace.analyze()``.
        image_path: Source image path (added as a column if provided).
        include_kinematics: Include kinematic feature columns.
        include_head_pose: Include yaw/pitch/roll columns.
        include_quality: Include quality metric columns.
        include_symmetry: Include symmetry columns.

    Returns:
        DataFrame with one row per face.
    """
    rows: List[Dict[str, Any]] = []
    for i, face in enumerate(faces):
        row: Dict[str, Any] = {
            "face_idx": i,
            "score": face.score,
            "bbox_x1": face.bbox[0],
            "bbox_y1": face.bbox[1],
            "bbox_x2": face.bbox[2],
            "bbox_y2": face.bbox[3],
        }

        if image_path:
            row["image"] = image_path

        # Keypoint coordinates as flat columns
        for j in range(68):
            row[f"kpt_{j}_x"] = float(face.keypoints[j, 0])
            row[f"kpt_{j}_y"] = float(face.keypoints[j, 1])
            row[f"kpt_{j}_score"] = float(face.keypoints[j, 2])

        if include_kinematics:
            for key, val in face.kinematics.items():
                row[key] = val

        if include_head_pose:
            yaw, pitch, roll = face.head_pose
            row["yaw"] = yaw
            row["pitch"] = pitch
            row["roll"] = roll

        if include_quality:
            for key, val in face.quality.items():
                row[f"quality_{key}"] = val

        if include_symmetry:
            row["symmetry_fa"] = face.symmetry
            for key, val in face.region_symmetry.items():
                row[f"symmetry_{key}"] = val

        rows.append(row)

    return pd.DataFrame(rows)


def to_csv(
    faces: List[Face],
    output_path: Union[str, Path],
    image_path: Optional[str] = None,
    **kwargs: Any,
) -> Path:
    """Export Face results to CSV.

    Args:
        faces: List of Face objects.
        output_path: Output CSV file path.
        image_path: Source image path (added as a column).
        **kwargs: Passed to :func:`to_dataframe`.

    Returns:
        Path to the written CSV file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = to_dataframe(faces, image_path=image_path, **kwargs)
    df.to_csv(output_path, index=False)
    return output_path


def to_coco_json(
    faces: List[Face],
    output_path: Union[str, Path],
    image_path: Optional[str] = None,
    image_id: int = 1,
) -> Path:
    """Export Face results to COCO keypoints JSON format.

    Args:
        faces: List of Face objects.
        output_path: Output JSON file path.
        image_path: Source image filename (stored in the ``images`` list).
        image_id: Image ID for the COCO annotation.

    Returns:
        Path to the written JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    images = []
    if image_path:
        h, w = faces[0]._image.shape[:2] if faces else (0, 0)
        images.append({
            "id": image_id,
            "file_name": str(Path(image_path).name),
            "width": w,
            "height": h,
        })

    annotations = []
    for i, face in enumerate(faces):
        # COCO keypoints: [x1, y1, v1, x2, y2, v2, ...]
        kpts_flat = []
        num_visible = 0
        for j in range(68):
            x = float(face.keypoints[j, 0])
            y = float(face.keypoints[j, 1])
            s = float(face.keypoints[j, 2])
            # Map score to COCO visibility: 0=not labeled, 2=visible
            v = 2 if s > 0.3 else 0
            if v > 0:
                num_visible += 1
            kpts_flat.extend([x, y, v])

        x1, y1, x2, y2 = face.bbox.tolist()
        annotations.append({
            "id": i + 1,
            "image_id": image_id,
            "category_id": 1,
            "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO uses [x, y, w, h]
            "keypoints": kpts_flat,
            "num_keypoints": num_visible,
            "score": face.score,
            "area": (x2 - x1) * (y2 - y1),
            "iscrowd": 0,
        })

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{
            "id": 1,
            "name": "primate_face",
            "keypoints": [f"point_{i}" for i in range(68)],
            "skeleton": [],
        }],
    }

    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

    return output_path


def to_dlc_csv(
    faces: List[Face],
    output_path: Union[str, Path],
    scorer: str = "primateface",
) -> Path:
    """Export to DeepLabCut-compatible CSV format.

    DLC CSV has a MultiIndex header: scorer / bodypart / (x, y, likelihood).
    This exports single-frame data for each detected face.

    Args:
        faces: List of Face objects.
        output_path: Output CSV file path.
        scorer: Name of the scorer/network (DLC convention).

    Returns:
        Path to the written CSV file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bodyparts = [f"point_{i}" for i in range(68)]

    # Build MultiIndex columns: (scorer, bodypart, coord)
    columns = pd.MultiIndex.from_tuples(
        [(scorer, bp, coord) for bp in bodyparts for coord in ("x", "y", "likelihood")],
        names=["scorer", "bodyparts", "coords"],
    )

    rows = []
    for face in faces:
        row = []
        for j in range(68):
            row.extend([
                float(face.keypoints[j, 0]),
                float(face.keypoints[j, 1]),
                float(face.keypoints[j, 2]),
            ])
        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_path)
    return output_path


def to_dlc_h5(
    faces: List[Face],
    output_path: Union[str, Path],
    scorer: str = "primateface",
) -> Path:
    """Export to DeepLabCut-compatible HDF5 format.

    Args:
        faces: List of Face objects.
        output_path: Output H5 file path.
        scorer: Name of the scorer/network.

    Returns:
        Path to the written H5 file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bodyparts = [f"point_{i}" for i in range(68)]

    columns = pd.MultiIndex.from_tuples(
        [(scorer, bp, coord) for bp in bodyparts for coord in ("x", "y", "likelihood")],
        names=["scorer", "bodyparts", "coords"],
    )

    rows = []
    for face in faces:
        row = []
        for j in range(68):
            row.extend([
                float(face.keypoints[j, 0]),
                float(face.keypoints[j, 1]),
                float(face.keypoints[j, 2]),
            ])
        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    df.to_hdf(output_path, key="df_with_missing", mode="w")
    return output_path


def from_coco_json(
    json_path: Union[str, Path],
) -> pd.DataFrame:
    """Load keypoints from a COCO JSON file into a DataFrame.

    Args:
        json_path: Path to COCO keypoints JSON file.

    Returns:
        DataFrame with one row per annotation, columns for bbox,
        score, and per-keypoint x/y/visibility.
    """
    with open(json_path) as f:
        coco = json.load(f)

    rows = []
    for ann in coco.get("annotations", []):
        row: Dict[str, Any] = {
            "image_id": ann.get("image_id"),
            "score": ann.get("score", 1.0),
        }
        bbox = ann.get("bbox", [0, 0, 0, 0])
        row["bbox_x1"] = bbox[0]
        row["bbox_y1"] = bbox[1]
        row["bbox_x2"] = bbox[0] + bbox[2]
        row["bbox_y2"] = bbox[1] + bbox[3]

        kpts = ann.get("keypoints", [])
        n_kpts = len(kpts) // 3
        for j in range(n_kpts):
            row[f"kpt_{j}_x"] = kpts[j * 3]
            row[f"kpt_{j}_y"] = kpts[j * 3 + 1]
            row[f"kpt_{j}_v"] = kpts[j * 3 + 2]

        rows.append(row)

    return pd.DataFrame(rows)
