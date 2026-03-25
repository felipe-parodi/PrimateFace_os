"""Pre-trained STGCN with weight transfer from body skeleton to face landmarks.

Loads MMAction2 STGCN weights pre-trained on NTU RGB+D 60 (17 body joints),
replaces the adjacency matrix with a 68-point face graph, and fine-tunes
for facial AU/expression prediction.

The key insight: spatial graph conv weights are (C_out*K, C_in, 1, 1) where
K=3 (adjacency partitions). These are independent of node count V. Only the
adjacency buffers (3, V, V) change from 17→68.
"""

import logging
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .features import SKELETON_EDGES

logger = logging.getLogger("primateface.analysis.pretrained_stgcn")


def _build_face_adjacency_partitioned(num_nodes: int = 68) -> np.ndarray:
    """Build 3-partition adjacency for STGCN (self, inward, outward).

    Partitions: [0] = identity, [1] = inward (toward center), [2] = outward.
    Center node = 30 (nose tip).

    Args:
        num_nodes: Number of landmark nodes.

    Returns:
        (3, V, V) float32 adjacency array.
    """
    center = 30  # nose tip

    adj = np.zeros((3, num_nodes, num_nodes), dtype=np.float32)

    # Partition 0: self-loops
    for i in range(num_nodes):
        adj[0, i, i] = 1.0

    # Build distance-from-center for partitioning
    # Simple BFS-like: edges closer to center = inward, farther = outward
    edge_set = set()
    for i, j in SKELETON_EDGES:
        if i < num_nodes and j < num_nodes:
            edge_set.add((i, j))
            edge_set.add((j, i))

    # Compute hop distance from center via BFS
    dist = {center: 0}
    queue = [center]
    visited = {center}
    neighbors = {i: [] for i in range(num_nodes)}
    for i, j in edge_set:
        neighbors[i].append(j)

    while queue:
        node = queue.pop(0)
        for nb in neighbors[node]:
            if nb not in visited:
                visited.add(nb)
                dist[nb] = dist[node] + 1
                queue.append(nb)

    # Assign unvisited nodes a large distance
    for i in range(num_nodes):
        if i not in dist:
            dist[i] = num_nodes

    # Partition edges into inward (toward center) and outward
    for i, j in edge_set:
        if dist[j] < dist[i]:
            adj[1, j, i] = 1.0  # inward: j is closer to center
        else:
            adj[2, j, i] = 1.0  # outward: j is farther

    # Normalize each partition
    for k in range(3):
        degree = adj[k].sum(axis=1, keepdims=True)
        degree = np.where(degree > 0, degree, 1.0)
        adj[k] /= degree

    return adj


class STGCNBlock(nn.Module):
    """Single STGCN block matching MMAction2 architecture."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_nodes: int,
        temporal_kernel: int = 9,
        stride: int = 1,
    ) -> None:
        super().__init__()
        # Spatial graph conv: (C_out * K, C_in, 1, 1)
        self.gcn = nn.ModuleDict({
            "conv": nn.Conv2d(in_ch, out_ch * 3, 1),
            "bn": nn.BatchNorm2d(out_ch),
        })
        # Adjacency buffers (replaced with face graph)
        adj = _build_face_adjacency_partitioned(num_nodes)
        self.gcn_A = nn.Parameter(torch.from_numpy(adj), requires_grad=False)
        self.gcn_PA = nn.Parameter(torch.zeros_like(self.gcn_A))  # learnable

        # Temporal conv
        pad = (temporal_kernel - 1) // 2
        self.tcn = nn.ModuleDict({
            "conv": nn.Conv2d(out_ch, out_ch, (temporal_kernel, 1),
                              padding=(pad, 0), stride=(stride, 1)),
            "bn": nn.BatchNorm2d(out_ch),
        })

        self.relu = nn.ReLU(inplace=True)

        # Residual
        if in_ch != out_ch or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: x shape (B, C, T, V)."""
        res = self.residual(x)

        # Spatial graph conv
        b, c, t, v = x.shape
        adj = self.gcn_A + self.gcn_PA  # (3, V, V)
        h = self.gcn["conv"](x)  # (B, C_out*3, T, V)
        c_out = h.size(1) // 3
        h = h.view(b, 3, c_out, t, v)

        # A @ h for each partition, sum
        out = torch.zeros(b, c_out, t, v, device=x.device)
        for k in range(3):
            # (V, V) @ (B, C, T, V) → need (B, C*T, V) @ (V, V)^T
            hk = h[:, k]  # (B, C_out, T, V)
            hk_flat = hk.reshape(b, c_out * t, v)  # (B, C*T, V)
            ak = adj[k]  # (V, V)
            out_flat = torch.matmul(hk_flat, ak.T)  # (B, C*T, V)
            out += out_flat.reshape(b, c_out, t, v)

        out = self.gcn["bn"](out)
        out = self.relu(out)

        # Temporal conv
        out = self.tcn["conv"](out)
        out = self.tcn["bn"](out)
        out = self.relu(out + res)

        return out


class PretrainedFaceSTGCN(nn.Module):
    """STGCN with pre-trained body skeleton weights for face landmarks.

    Architecture matches MMAction2 STGCN (10 blocks, channels 64→128→256).
    Adjacency is replaced with 68-point face graph. Pre-trained conv/bn
    weights are loaded; adjacency and classification head are re-initialized.

    Args:
        num_classes: Number of output classes.
        num_nodes: Number of landmark nodes (68).
        freeze_blocks: Number of early blocks to freeze (0 = train all).
        pretrained_path: Path to MMAction2 checkpoint. If None, downloads.
    """

    def __init__(
        self,
        num_classes: int,
        num_nodes: int = 68,
        freeze_blocks: int = 7,
        pretrained_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes

        # Data batch normalization (input: 3 channels for NTU, 2 for face)
        self.data_bn = nn.BatchNorm1d(num_nodes * 2)

        # 10 GCN blocks matching MMAction2 architecture
        channels = [
            (2, 64), (64, 64), (64, 64), (64, 64),
            (64, 128), (128, 128), (128, 128),
            (128, 256), (256, 256), (256, 256),
        ]
        strides = [1, 1, 1, 1, 2, 1, 1, 2, 1, 1]
        self.blocks = nn.ModuleList()
        for (ic, oc), s in zip(channels, strides):
            self.blocks.append(STGCNBlock(ic, oc, num_nodes, stride=s))

        # Classification head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

        # Load pre-trained weights
        if pretrained_path is None:
            pretrained_path = self._download_pretrained()
        self._load_pretrained(pretrained_path)

        # Freeze early blocks
        if freeze_blocks > 0:
            for i, block in enumerate(self.blocks[:freeze_blocks]):
                for param in block.parameters():
                    param.requires_grad = False
            logger.info("Froze %d/%d STGCN blocks", freeze_blocks, len(self.blocks))

    @staticmethod
    def _download_pretrained() -> str:
        """Download NTU60 pre-trained STGCN checkpoint."""
        url = (
            "https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/"
            "stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/"
            "stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221129-484a394a.pth"
        )
        cache_dir = os.path.expanduser("~/.cache/mmaction2")
        os.makedirs(cache_dir, exist_ok=True)
        out_path = os.path.join(cache_dir, "stgcn_ntu60_2d.pth")
        if not os.path.exists(out_path):
            logger.info("Downloading pre-trained STGCN from OpenMMLab...")
            torch.hub.download_url_to_file(url, out_path)
        return out_path

    def _load_pretrained(self, path: str) -> None:
        """Load pre-trained weights, skipping adjacency and mismatched layers."""
        ckpt = torch.load(path, map_location="cpu")
        src_state = ckpt.get("state_dict", ckpt)

        loaded = 0
        skipped = 0
        for src_key, src_val in src_state.items():
            # Map MMAction2 key format to our format
            dst_key = src_key
            dst_key = dst_key.replace("backbone.gcn.", "blocks.")
            dst_key = dst_key.replace("backbone.data_bn.", "data_bn.")
            dst_key = dst_key.replace("cls_head.fc.", "fc.")
            # Handle gcn.A → gcn_A, gcn.PA → gcn_PA
            dst_key = dst_key.replace(".gcn.A", ".gcn_A")
            dst_key = dst_key.replace(".gcn.PA", ".gcn_PA")

            if dst_key not in self.state_dict():
                skipped += 1
                continue

            dst_shape = self.state_dict()[dst_key].shape
            src_shape = src_val.shape

            # Skip mismatched shapes (adjacency, data_bn, fc head)
            if src_shape != dst_shape:
                skipped += 1
                continue

            self.state_dict()[dst_key].copy_(src_val)
            loaded += 1

        logger.info(
            "Loaded %d/%d pre-trained weights (skipped %d mismatched)",
            loaded, loaded + skipped, skipped,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, V, 2) landmark sequence.

        Returns:
            (B, num_classes) logits.
        """
        b, t, v, c = x.shape

        # Data BN: reshape to (B, V*C, T)
        x_bn = x.permute(0, 2, 3, 1).reshape(b, v * c, t)
        x_bn = self.data_bn(x_bn)
        x_bn = x_bn.reshape(b, v, c, t).permute(0, 2, 3, 1)  # (B, C, T, V)

        # GCN blocks
        out = x_bn
        for block in self.blocks:
            out = block(out)

        # Pool + classify
        out = self.pool(out).squeeze(-1).squeeze(-1)  # (B, 256)
        return self.fc(out)
