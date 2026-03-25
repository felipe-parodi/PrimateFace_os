"""Loss functions for multi-label imbalanced classification.

Provides Asymmetric Loss (ASL) from Ridnik et al. (ICCV 2021) as a
drop-in replacement for BCEWithLogitsLoss.
"""

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification.

    Applies different focusing parameters for positive and negative
    samples. Hard-thresholds easy negatives for cleaner gradients.

    Reference:
        Ridnik et al., "Asymmetric Loss For Multi-Label Classification",
        ICCV 2021. https://github.com/Alibaba-MIIL/ASL

    Args:
        gamma_neg: Focusing parameter for negative samples (higher = more
            suppression of easy negatives). Default 4.
        gamma_pos: Focusing parameter for positive samples. Default 1.
        clip: Hard-threshold for negative probabilities. Probabilities
            below this value are clipped to zero. Default 0.05.
        reduction: Loss reduction mode ("mean" or "sum").
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute asymmetric loss.

        Args:
            logits: (N, C) raw model output (before sigmoid).
            targets: (N, C) binary labels (0 or 1).

        Returns:
            Scalar loss.
        """
        probs = torch.sigmoid(logits)
        pos_part = targets
        neg_part = 1 - targets

        # Asymmetric clipping for negatives
        if self.clip > 0:
            neg_probs = (probs + self.clip).clamp(max=1.0)
        else:
            neg_probs = probs

        # Compute cross-entropy per element
        loss_pos = pos_part * torch.log(probs.clamp(min=1e-8))
        loss_neg = neg_part * torch.log((1 - neg_probs).clamp(min=1e-8))

        loss = loss_pos + loss_neg

        # Asymmetric focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt = probs * pos_part + (1 - probs) * neg_part
            gamma = self.gamma_pos * pos_part + self.gamma_neg * neg_part
            modulator = (1 - pt).pow(gamma)
            loss = loss * modulator

        loss = -loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
