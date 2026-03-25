"""Face embedding extraction via pre-trained ViT (FMAE-IAT).

Extracts fixed-size embeddings from face crops using a Vision Transformer
pre-trained on 9M human faces with MAE + identity adversarial training.
Used as a frozen feature extractor for downstream tasks (AU prediction,
expression recognition, cross-species transfer).

Reference:
    Li et al., "FMAE-IAT: Face MAE with Identity Adversarial Training",
    arXiv:2407.11243.
"""

import logging
import pathlib
import platform
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("primateface.analysis.face_encoder")

# ImageNet normalization (standard for ViT)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _build_vit_small() -> nn.Module:
    """Build a ViT-Small encoder (384-dim, 12 blocks, 6 heads).

    Uses timm if available, otherwise a minimal implementation.

    Returns:
        ViT-Small module (encoder only, no decoder/head).
    """
    try:
        import timm

        model = timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,
            num_classes=0,  # no classification head → returns features
        )
        return model
    except ImportError:
        raise ImportError(
            "timm is required for FMAE-IAT ViT. Install with: pip install timm"
        )


class FaceEncoder:
    """Frozen ViT face encoder for embedding extraction.

    Loads a pre-trained FMAE-IAT ViT-Small and extracts CLS token
    embeddings from aligned face crops.

    Args:
        model_name: Model variant ("vit_small", "vit_base", "vit_large").
        device: Torch device string.
        checkpoint_path: Local path to .pth file. If None, downloads
            from HuggingFace.
    """

    def __init__(
        self,
        model_name: str = "vit_small",
        device: str = "cuda:0",
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.device = device
        self.embed_dim = {"vit_small": 384, "vit_base": 768, "vit_large": 1024}[
            model_name
        ]

        # Build model
        self.model = _build_vit_small()  # extend for base/large later

        # Load checkpoint
        if checkpoint_path is None:
            checkpoint_path = self._download_checkpoint(model_name)
        self._load_checkpoint(checkpoint_path)

        self.model = self.model.to(device)
        self.model.eval()
        logger.info(
            "FaceEncoder loaded: %s (%d-dim) on %s",
            model_name,
            self.embed_dim,
            device,
        )

    @staticmethod
    def _download_checkpoint(model_name: str) -> Path:
        """Download checkpoint from HuggingFace."""
        from huggingface_hub import hf_hub_download

        filename_map = {
            "vit_small": "FMAE_ViT_small.pth",
            "vit_base": "FMAE_ViT_base.pth",
            "vit_large": "FMAE_ViT_large.pth",
        }
        path = hf_hub_download(
            "forever208/FMAE-IAT",
            filename_map[model_name],
        )
        return Path(path)

    def _load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load pre-trained weights (encoder only, skip decoder)."""
        # Fix PosixPath issue on Windows
        if platform.system() == "Windows":
            pathlib.PosixPath = pathlib.WindowsPath

        ckpt = torch.load(str(path), map_location="cpu")
        state = ckpt.get("model", ckpt)

        # Filter to encoder-only keys (skip decoder, mask_token)
        encoder_state = {}
        for k, v in state.items():
            if k.startswith("decoder_") or k == "mask_token":
                continue
            encoder_state[k] = v

        # Load with strict=False (some keys may not match timm model)
        missing, unexpected = self.model.load_state_dict(
            encoder_state, strict=False
        )
        if missing:
            logger.debug("Missing keys: %s", missing[:5])
        if unexpected:
            logger.debug("Unexpected keys: %s", unexpected[:5])

    def preprocess(
        self,
        image_bgr: np.ndarray,
        bbox: Optional[np.ndarray] = None,
        output_size: int = 224,
    ) -> np.ndarray:
        """Preprocess a face crop for ViT input.

        Args:
            image_bgr: BGR image.
            bbox: [x1, y1, x2, y2] face bounding box. If None, uses
                full image.
            output_size: Target size (224 for ViT).

        Returns:
            (3, H, W) float32 array, normalized.
        """
        if bbox is not None:
            x1, y1, x2, y2 = bbox.astype(int)
            # Add 10% padding
            w, h = x2 - x1, y2 - y1
            pad = int(max(w, h) * 0.1)
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(image_bgr.shape[1], x2 + pad)
            y2 = min(image_bgr.shape[0], y2 + pad)
            crop = image_bgr[y1:y2, x1:x2]
        else:
            crop = image_bgr

        if crop.size == 0:
            return np.zeros((3, output_size, output_size), dtype=np.float32)

        # Resize and normalize
        crop = cv2.resize(crop, (output_size, output_size))
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - _MEAN) / _STD
        return rgb.transpose(2, 0, 1)  # (3, H, W)

    @torch.no_grad()
    def encode(
        self,
        images: Union[np.ndarray, List[np.ndarray]],
    ) -> np.ndarray:
        """Extract embeddings from preprocessed images.

        Args:
            images: (N, 3, 224, 224) array or list of (3, 224, 224) arrays.

        Returns:
            (N, embed_dim) embedding array.
        """
        if isinstance(images, list):
            images = np.stack(images)
        if images.ndim == 3:
            images = images[np.newaxis]

        tensor = torch.from_numpy(images).to(self.device)
        # Process in batches to avoid OOM
        batch_size = 32
        all_embs = []
        for i in range(0, tensor.shape[0], batch_size):
            batch = tensor[i : i + batch_size]
            emb = self.model(batch)  # (B, embed_dim) from timm with num_classes=0
            all_embs.append(emb.cpu().numpy())

        return np.concatenate(all_embs, axis=0)

    def encode_video_result(
        self,
        frames: List[np.ndarray],
        bboxes: np.ndarray,
    ) -> np.ndarray:
        """Extract per-frame embeddings from video frames + bboxes.

        Args:
            frames: List of BGR frame arrays.
            bboxes: (T, 4) bounding boxes.

        Returns:
            (T, embed_dim) per-frame embeddings.
        """
        preprocessed = []
        for i, frame in enumerate(frames):
            bbox = bboxes[i] if i < len(bboxes) else None
            preprocessed.append(self.preprocess(frame, bbox))
        return self.encode(preprocessed)
