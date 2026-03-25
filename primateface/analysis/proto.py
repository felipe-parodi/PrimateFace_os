"""Prototypical classifier for few-shot classification from embeddings.

Computes per-class prototypes (mean embeddings of support examples)
and classifies by cosine or Euclidean distance to prototypes.
Designed for multi-label: each class has independent positive/negative
prototypes.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("primateface.analysis.proto")


class PrototypicalClassifier:
    """Few-shot prototypical classifier for multi-label prediction.

    Computes per-class prototypes from support examples, then
    classifies query examples by distance to prototypes.

    Works with any encoder that maps inputs to fixed-size embeddings.

    Args:
        encoder: PyTorch module mapping input → (batch, embed_dim).
            If None, assumes inputs are already embeddings.
        distance: "cosine" or "euclidean".
        device: Torch device string.
    """

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        distance: str = "cosine",
        device: str = "cpu",
    ) -> None:
        self.encoder = encoder
        self.distance = distance
        self.device = device
        self.prototypes_pos: Optional[np.ndarray] = None  # (C, D)
        self.prototypes_neg: Optional[np.ndarray] = None  # (C, D)
        self.class_labels: List[int] = []

    def _encode(self, x: np.ndarray) -> np.ndarray:
        """Encode inputs to embeddings.

        Args:
            x: Input array (shape depends on encoder).

        Returns:
            (N, D) embedding array.
        """
        if self.encoder is None:
            return x

        self.encoder.eval()
        with torch.no_grad():
            t = torch.from_numpy(x.astype(np.float32)).to(self.device)
            emb = self.encoder(t)
            if emb.ndim > 2:
                emb = emb.mean(dim=tuple(range(1, emb.ndim - 1)))
            return emb.cpu().numpy()

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        class_labels: Optional[List[int]] = None,
    ) -> "PrototypicalClassifier":
        """Compute prototypes from support examples.

        For multi-label: computes separate positive and negative
        prototypes per class.

        Args:
            x: (N, ...) input features or embeddings.
            y: (N, C) binary label matrix.
            class_labels: Optional names for the C classes.

        Returns:
            self.
        """
        embeddings = self._encode(x)  # (N, D)
        n_classes = y.shape[1]
        d = embeddings.shape[1]

        self.prototypes_pos = np.zeros((n_classes, d), dtype=np.float32)
        self.prototypes_neg = np.zeros((n_classes, d), dtype=np.float32)
        self.class_labels = list(class_labels) if class_labels else list(range(n_classes))

        for c in range(n_classes):
            pos_mask = y[:, c] == 1
            neg_mask = y[:, c] == 0

            if pos_mask.sum() > 0:
                self.prototypes_pos[c] = embeddings[pos_mask].mean(axis=0)
            if neg_mask.sum() > 0:
                self.prototypes_neg[c] = embeddings[neg_mask].mean(axis=0)

        logger.info(
            "Fit prototypes: %d classes, %d-dim embeddings, %d support samples",
            n_classes, d, len(embeddings),
        )
        return self

    def predict_scores(self, x: np.ndarray) -> np.ndarray:
        """Predict per-class scores (higher = more likely positive).

        Score is the difference in distance: d(neg_proto) - d(pos_proto).
        Positive score means closer to positive prototype.

        Args:
            x: (N, ...) input features or embeddings.

        Returns:
            (N, C) score matrix.
        """
        if self.prototypes_pos is None:
            raise RuntimeError("Call fit() first")

        embeddings = self._encode(x)  # (N, D)
        n = embeddings.shape[0]
        n_classes = self.prototypes_pos.shape[0]

        scores = np.zeros((n, n_classes), dtype=np.float32)

        for c in range(n_classes):
            if self.distance == "cosine":
                # Cosine similarity
                emb_norm = embeddings / (
                    np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
                )
                pos_norm = self.prototypes_pos[c] / (
                    np.linalg.norm(self.prototypes_pos[c]) + 1e-8
                )
                neg_norm = self.prototypes_neg[c] / (
                    np.linalg.norm(self.prototypes_neg[c]) + 1e-8
                )
                sim_pos = emb_norm @ pos_norm
                sim_neg = emb_norm @ neg_norm
                scores[:, c] = sim_pos - sim_neg
            else:
                # Euclidean distance (negative = closer)
                d_pos = np.linalg.norm(
                    embeddings - self.prototypes_pos[c], axis=1
                )
                d_neg = np.linalg.norm(
                    embeddings - self.prototypes_neg[c], axis=1
                )
                scores[:, c] = d_neg - d_pos  # higher = closer to positive

        return scores

    def predict(
        self, x: np.ndarray, threshold: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict binary labels and scores.

        Args:
            x: (N, ...) input features.
            threshold: Score threshold for positive prediction.

        Returns:
            Tuple of (binary predictions, raw scores).
        """
        scores = self.predict_scores(x)
        preds = (scores > threshold).astype(int)
        return preds, scores
