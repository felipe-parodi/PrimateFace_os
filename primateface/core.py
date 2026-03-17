"""High-level PrimateFace interface.

Provides the ``PrimateFace`` class — the "3 lines of code" entry point
for primate face detection, landmark estimation, and analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import cv2
import numpy as np

from .face import Face
from ._model_manager import ModelManager, POSE_MODEL_VARIANTS

InputType = Union[str, Path, np.ndarray]


class PrimateFace:
    """High-level interface for primate face analysis.

    Handles model download, initialization, and provides a simple
    :meth:`analyze` method that returns rich :class:`Face` objects.

    Example:
        >>> import primateface
        >>> pf = primateface.PrimateFace()
        >>> faces = pf.analyze("monkey.jpg")
        >>> faces[0].head_pose
        (5.2, -3.1, 1.0)

    Args:
        device: PyTorch device string. Defaults to ``"cuda:0"`` if
            available, else ``"cpu"``.
        pose_model: Pose estimation backend. ``"hrnet"`` (default,
            38 MB) or ``"vitpose"`` (1.2 GB, more accurate).
        det_threshold: Detection confidence threshold.
        nms_threshold: NMS threshold for overlapping detections.
        model_dir: Directory to cache model files. If *None*, uses
            HuggingFace Hub default cache (``~/.cache/huggingface/``).
    """

    def __init__(
        self,
        device: Optional[str] = None,
        pose_model: str = "hrnet",
        det_threshold: float = 0.5,
        nms_threshold: float = 0.3,
        model_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        import torch

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.pose_model = pose_model
        self.det_threshold = det_threshold
        self.nms_threshold = nms_threshold

        # Validate pose_model
        if pose_model not in POSE_MODEL_VARIANTS:
            raise ValueError(
                f"Unknown pose_model={pose_model!r}. "
                f"Choose from: {list(POSE_MODEL_VARIANTS.keys())}"
            )

        # Resolve model paths (downloads from HF if needed)
        pose_variant = POSE_MODEL_VARIANTS[pose_model]
        mgr = ModelManager(model_dir=model_dir, pose_variant=pose_variant)
        det_cfg, det_ckpt, pose_cfg, pose_ckpt = mgr.ensure_models()

        # Initialize the underlying processor
        from demos.process import PrimateFaceProcessor

        self._processor = PrimateFaceProcessor(
            det_config=str(det_cfg),
            det_checkpoint=str(det_ckpt),
            pose_config=str(pose_cfg),
            pose_checkpoint=str(pose_ckpt),
            device=self.device,
        )

    def analyze(
        self,
        image: InputType,
        det_threshold: Optional[float] = None,
        nms_threshold: Optional[float] = None,
    ) -> List[Face]:
        """Detect and analyze all primate faces in an image.

        Args:
            image: Input image as a file path (``str``/``Path``),
                numpy array (BGR), or PIL Image.
            det_threshold: Override default detection threshold.
            nms_threshold: Override default NMS threshold.

        Returns:
            List of :class:`Face` objects, sorted by detection
            confidence (highest first). Empty list if no faces detected.

        Raises:
            FileNotFoundError: If image path does not exist.
            ValueError: If image cannot be loaded or decoded.
        """
        bgr = self._load_image(image)
        h, w = bgr.shape[:2]
        image_size = (w, h)

        det_thr = det_threshold if det_threshold is not None else self.det_threshold
        nms_thr = nms_threshold if nms_threshold is not None else self.nms_threshold

        # Run detection
        bboxes, scores = self._processor.detect_primates(
            bgr, bbox_thr=det_thr, nms_thr=nms_thr
        )

        if len(bboxes) == 0:
            return []

        # Run pose estimation
        pose_results = self._processor.estimate_poses(bgr, bboxes)
        keypoints_xy = pose_results.keypoints  # (N, 68, 2)
        keypoint_scores = pose_results.keypoint_scores  # (N, 68)

        # Build Face objects
        faces: List[Face] = []
        for i in range(len(bboxes)):
            kpts_with_vis = np.column_stack([
                keypoints_xy[i],
                keypoint_scores[i][:, np.newaxis],
            ]).astype(np.float32)

            face = Face(
                bbox=bboxes[i].astype(np.float32),
                score=float(scores[i]),
                keypoints=kpts_with_vis,
                _image=bgr,
                _image_size=image_size,
            )
            faces.append(face)

        faces.sort(key=lambda f: f.score, reverse=True)
        return faces

    def analyze_batch(
        self,
        images: Sequence[InputType],
    ) -> List[List[Face]]:
        """Analyze multiple images.

        Convenience wrapper that calls :meth:`analyze` for each image.

        Args:
            images: Sequence of image inputs.

        Returns:
            List of Face lists, one per input image.
        """
        return [self.analyze(img) for img in images]

    @staticmethod
    def draw(
        faces: List[Face],
        image: InputType,
        output: Optional[Union[str, Path]] = None,
        draw_keypoints: bool = True,
        draw_skeleton: bool = True,
        draw_bbox: bool = True,
    ) -> np.ndarray:
        """Draw detected faces on an image.

        Args:
            faces: List of Face objects from :meth:`analyze`.
            image: Source image (path, numpy BGR, or PIL Image).
            output: If provided, save the visualization to this path.
            draw_keypoints: Draw landmark points.
            draw_skeleton: Draw skeleton connections.
            draw_bbox: Draw bounding boxes.

        Returns:
            BGR numpy array with faces drawn.
        """
        bgr = PrimateFace._load_image(image)
        canvas = bgr.copy()

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            kpts = face.keypoints[:, :2]  # (68, 2)
            scores = face.keypoints[:, 2]  # (68,)

            if draw_bbox:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{face.score:.2f}"
                cv2.putText(
                    canvas, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                )

            if draw_keypoints:
                for j in range(68):
                    if scores[j] > 0.3:
                        px, py = int(kpts[j, 0]), int(kpts[j, 1])
                        cv2.circle(canvas, (px, py), 2, (0, 0, 255), -1)

            if draw_skeleton:
                # Jaw contour
                for j in range(16):
                    if scores[j] > 0.3 and scores[j + 1] > 0.3:
                        p1 = (int(kpts[j, 0]), int(kpts[j, 1]))
                        p2 = (int(kpts[j + 1, 0]), int(kpts[j + 1, 1]))
                        cv2.line(canvas, p1, p2, (255, 200, 0), 1)
                # Eye, brow, nose, mouth contours
                for start, end in [
                    (17, 22), (22, 27),  # eyebrows
                    (27, 31),            # nose bridge
                    (31, 36),            # nose base
                    (36, 42), (42, 48),  # eyes
                    (48, 60), (60, 68),  # mouth
                ]:
                    for j in range(start, end - 1):
                        if scores[j] > 0.3 and scores[j + 1] > 0.3:
                            p1 = (int(kpts[j, 0]), int(kpts[j, 1]))
                            p2 = (int(kpts[j + 1, 0]), int(kpts[j + 1, 1]))
                            cv2.line(canvas, p1, p2, (255, 200, 0), 1)

        if output is not None:
            cv2.imwrite(str(output), canvas)

        return canvas

    @staticmethod
    def _load_image(image: InputType) -> np.ndarray:
        """Convert various input types to BGR numpy array.

        Args:
            image: File path, numpy array, or PIL Image.

        Returns:
            BGR numpy array.

        Raises:
            FileNotFoundError: If path does not exist.
            ValueError: If image cannot be decoded.
            TypeError: If input type is not supported.
        """
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise ValueError(f"Could not decode image: {path}")
            return bgr

        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if image.ndim == 3 and image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            if image.ndim == 3 and image.shape[2] == 3:
                return image
            raise ValueError(
                f"Unexpected array shape: {image.shape}. "
                "Expected (H, W), (H, W, 3), or (H, W, 4)."
            )

        # PIL Image
        try:
            import PIL.Image

            if isinstance(image, PIL.Image.Image):
                rgb = np.array(image.convert("RGB"))
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except ImportError:
            pass

        raise TypeError(
            f"Unsupported image type: {type(image)}. "
            "Expected str, Path, numpy array, or PIL Image."
        )
