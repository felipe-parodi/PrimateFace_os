"""G3: Spatial-Temporal Graph Convolutional Network for AU prediction.

Operates on (T, 68, 2) landmark sequences with face graph adjacency.
Manual implementation — no torch_geometric dependency required.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from primateface.analysis.features import build_face_adjacency as build_adjacency_matrix

logger = logging.getLogger("animalfacs.models.stgcn")


class SpatialGraphConv(nn.Module):
    """Spatial graph convolution: X' = A @ X @ W.

    Args:
        in_features: Input feature dimension per node.
        out_features: Output feature dimension per node.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(in_features, out_features)
        )
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, N, C_in) node features.
            adj: (N, N) normalized adjacency matrix.

        Returns:
            (B, T, N, C_out) transformed features.
        """
        # x @ W: (B, T, N, C_in) @ (C_in, C_out) → (B, T, N, C_out)
        xw = torch.matmul(x, self.weight)
        # adj @ xw: broadcast (N, N) @ (B, T, N, C_out) → (B, T, N, C_out)
        out = torch.matmul(adj, xw)
        return out + self.bias


class STGCNBlock(nn.Module):
    """Spatial-Temporal GCN block: spatial graph conv + temporal conv.

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
        temporal_kernel: Temporal convolution kernel size.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        temporal_kernel: int = 9,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.sgc = SpatialGraphConv(in_ch, out_ch)
        self.bn_s = nn.BatchNorm1d(out_ch)
        # Temporal conv: operates along time axis per node
        pad = (temporal_kernel - 1) // 2
        self.tcn = nn.Conv1d(out_ch, out_ch, temporal_kernel, padding=pad)
        self.bn_t = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.residual = (
            nn.Linear(in_ch, out_ch) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, N, C_in) input.
            adj: (N, N) adjacency.

        Returns:
            (B, T, N, C_out) output.
        """
        b, t, n, c = x.shape
        res = self.residual(x)  # (B, T, N, C_out)

        # Spatial graph conv
        out = self.sgc(x, adj)  # (B, T, N, C_out)
        # BN over channel dim: reshape to (B*T*N, C)
        out = self.bn_s(out.reshape(-1, out.size(-1))).reshape(b, t, n, -1)
        out = self.relu(out)

        # Temporal conv: reshape to (B*N, C, T)
        c_out = out.size(-1)
        out = out.permute(0, 2, 3, 1).reshape(b * n, c_out, t)
        out = self.tcn(out)        # (B*N, C, T)
        out = self.bn_t(out.reshape(-1, c_out)).reshape(b * n, c_out, t)
        out = out.reshape(b, n, c_out, t).permute(0, 3, 1, 2)  # (B, T, N, C)
        out = self.relu(out)
        out = self.drop(out)

        return out + res


class FaceSTGCN(nn.Module):
    """ST-GCN for facial AU prediction from 68-point landmarks.

    Args:
        num_aus: Number of AU outputs.
        channels: Channel sizes per ST-GCN block.
        temporal_kernel: Temporal conv kernel size.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        num_aus: int,
        channels: Optional[List[int]] = None,
        temporal_kernel: int = 9,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if channels is None:
            channels = [32, 64]

        # Build adjacency
        adj_np = build_adjacency_matrix(68)
        self.register_buffer("adj", torch.from_numpy(adj_np))

        in_ch = 2  # (x, y) per landmark
        blocks = []
        prev = in_ch
        for ch in channels:
            blocks.append(STGCNBlock(prev, ch, temporal_kernel, dropout))
            prev = ch
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(prev, num_aus)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, 68, 2) landmark sequence.

        Returns:
            (B, num_aus) logit tensor.
        """
        for block in self.blocks:
            x = block(x, self.adj)  # (B, T, 68, C)

        # Global pool: mean over time and nodes
        x = x.mean(dim=(1, 2))  # (B, C)
        return self.head(x)     # (B, num_aus)


class GraphSequenceDataset(Dataset):
    """Dataset for ST-GCN with optional augmentation.

    Args:
        sequences: (N, T, 68, 2) array.
        labels: (N, C) binary label matrix.
        augment: Apply skeleton augmentations.
    """

    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
    ) -> None:
        self.raw_sequences = sequences.astype(np.float32)
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

        return torch.from_numpy(seq), self.y[idx]


def train_stgcn(
    train_seq: np.ndarray,
    train_labels: np.ndarray,
    val_seq: np.ndarray,
    val_labels: np.ndarray,
    cfg: Dict[str, Any],
    device: str = "cuda:0",
) -> Tuple[FaceSTGCN, Dict[str, List[float]]]:
    """Train the ST-GCN model.

    Args:
        train_seq: (N_train, T, 68, 2) training sequences.
        train_labels: (N_train, C) training labels.
        val_seq: (N_val, T, 68, 2) validation sequences.
        val_labels: (N_val, C) validation labels.
        cfg: Model config from config.yaml["models"]["g3_stgcn"].
        device: Torch device string.

    Returns:
        Tuple of (trained model, training history dict).
    """
    num_aus = train_labels.shape[1]

    from primateface.analysis.losses import AsymmetricLoss

    train_ds = GraphSequenceDataset(train_seq, train_labels, augment=True)
    val_ds = GraphSequenceDataset(val_seq, val_labels, augment=False)
    batch_size = cfg.get("batch_size", 16)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    model = FaceSTGCN(
        num_aus=num_aus,
        channels=cfg.get("channels", [32, 64]),
        temporal_kernel=cfg.get("temporal_kernel", 9),
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
                "  ST-GCN epoch %d/%d: train=%.4f val=%.4f",
                epoch + 1, epochs, train_loss, val_loss,
            )

        if wait >= patience:
            logger.info("  Early stopping at epoch %d", epoch + 1)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


def predict_stgcn(
    model: FaceSTGCN,
    sequences: np.ndarray,
    device: str = "cuda:0",
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run ST-GCN inference.

    Args:
        model: Trained FaceSTGCN.
        sequences: (N, T, 68, 2) arrays.
        device: Torch device.
        threshold: Sigmoid threshold.

    Returns:
        Tuple of (binary predictions, probability scores).
    """
    model.eval()
    ds = GraphSequenceDataset(sequences, np.zeros((sequences.shape[0], 1)))
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
