"""G2: Temporal Convolutional Network for AU prediction.

Operates on (T, 68*2) landmark sequences — skeleton/pose-based, not pixels.
Dilated causal convolutions capture temporal dynamics of facial movements.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger("animalfacs.models.temporal_cnn")


class _TemporalBlock(nn.Module):
    """Single dilated causal convolution block with residual connection."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size,
            padding=padding, dilation=dilation,
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        # Residual projection if channel mismatch
        self.residual = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, C_in, T) tensor.

        Returns:
            (B, C_out, T) tensor.
        """
        out = self.conv(x)
        # Trim to causal (remove future padding)
        out = out[:, :, :x.size(2)]
        out = self.bn(out)
        out = self.relu(out)
        out = self.drop(out)
        return out + self.residual(x)


class LandmarkTCN(nn.Module):
    """Temporal Convolutional Network for landmark-based AU prediction.

    Input: (B, T, 68*2) flattened landmark sequences.
    Output: (B, num_aus) AU logits.

    Args:
        num_aus: Number of AU outputs.
        channels: List of channel sizes per block.
        kernel_size: Convolution kernel size.
        dilations: List of dilation rates per block.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        num_aus: int,
        channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        dilations: Optional[List[int]] = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if channels is None:
            channels = [64, 128, 128]
        if dilations is None:
            dilations = [1, 2, 4]

        in_dim = 68 * 2  # flattened landmark coordinates

        blocks = []
        prev_ch = in_dim
        for ch, dil in zip(channels, dilations):
            blocks.append(_TemporalBlock(prev_ch, ch, kernel_size, dil, dropout))
            prev_ch = ch

        self.encoder = nn.Sequential(*blocks)
        self.head = nn.Linear(prev_ch, num_aus)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, 136) landmark sequence tensor.

        Returns:
            (B, num_aus) logit tensor.
        """
        # (B, T, 136) → (B, 136, T) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.encoder(x)  # (B, C, T)
        x = x.mean(dim=2)    # Global average pool → (B, C)
        return self.head(x)   # (B, num_aus)


class LandmarkSequenceDataset(Dataset):
    """PyTorch dataset for landmark sequences with optional augmentation.

    Args:
        sequences: (N, T, 68, 2) array.
        labels: (N, C) binary label matrix.
        augment: Apply skeleton augmentations during training.
    """

    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
    ) -> None:
        self.raw_sequences = sequences.astype(np.float32)  # keep (N, T, 68, 2)
        self.y = torch.from_numpy(labels.astype(np.float32))
        self.augment = augment
        self._rng = np.random.RandomState(42)

    def __len__(self) -> int:
        return self.raw_sequences.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.raw_sequences[idx].copy()  # (T, 68, 2)

        if self.augment:
            from primateface.analysis.augment import (
                jitter_landmarks,
                random_rotation,
                temporal_speed,
            )

            seq = jitter_landmarks(seq, sigma=0.02, rng=self._rng)
            seq = random_rotation(seq, max_deg=10, rng=self._rng)
            if self._rng.random() > 0.5:
                seq = temporal_speed(seq, rng=self._rng)

        # Flatten (T, 68, 2) → (T, 136)
        t_len = seq.shape[0]
        flat = seq.reshape(t_len, -1)
        return torch.from_numpy(flat), self.y[idx]


def train_tcn(
    train_seq: np.ndarray,
    train_labels: np.ndarray,
    val_seq: np.ndarray,
    val_labels: np.ndarray,
    cfg: Dict[str, Any],
    device: str = "cuda:0",
) -> Tuple[LandmarkTCN, Dict[str, List[float]]]:
    """Train the TCN model.

    Args:
        train_seq: (N_train, T, 68, 2) training sequences.
        train_labels: (N_train, C) training labels.
        val_seq: (N_val, T, 68, 2) validation sequences.
        val_labels: (N_val, C) validation labels.
        cfg: Model config from config.yaml["models"]["g2_tcn"].
        device: Torch device string.

    Returns:
        Tuple of (trained model, training history dict).
    """
    num_aus = train_labels.shape[1]

    from primateface.analysis.losses import AsymmetricLoss

    train_ds = LandmarkSequenceDataset(train_seq, train_labels, augment=True)
    val_ds = LandmarkSequenceDataset(val_seq, val_labels, augment=False)
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.get("batch_size", 16),
        shuffle=True,
        drop_last=False,
    )
    val_dl = DataLoader(val_ds, batch_size=cfg.get("batch_size", 16))

    model = LandmarkTCN(
        num_aus=num_aus,
        channels=cfg.get("channels", [64, 128, 128]),
        kernel_size=cfg.get("kernel_size", 3),
        dilations=cfg.get("dilations", [1, 2, 4]),
        dropout=cfg.get("dropout", 0.3),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("lr", 1e-3),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )
    epochs = cfg.get("epochs", 100)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
    patience = cfg.get("patience", 15)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_ds)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= max(len(val_ds), 1)

        scheduler.step()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if (epoch + 1) % 10 == 0:
            logger.info(
                "  TCN epoch %d/%d: train=%.4f val=%.4f",
                epoch + 1, epochs, train_loss, val_loss,
            )

        if wait >= patience:
            logger.info("  Early stopping at epoch %d", epoch + 1)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


def predict_tcn(
    model: LandmarkTCN,
    sequences: np.ndarray,
    device: str = "cuda:0",
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run TCN inference.

    Args:
        model: Trained LandmarkTCN.
        sequences: (N, T, 68, 2) arrays.
        device: Torch device.
        threshold: Sigmoid threshold for binary prediction.

    Returns:
        Tuple of (binary predictions, probability scores).
    """
    model.eval()
    ds = LandmarkSequenceDataset(sequences, np.zeros((sequences.shape[0], 1)))
    dl = DataLoader(ds, batch_size=32)

    all_probs = []
    with torch.no_grad():
        for xb, _ in dl:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    probs = np.concatenate(all_probs, axis=0)
    preds = (probs >= threshold).astype(int)
    return preds, probs
