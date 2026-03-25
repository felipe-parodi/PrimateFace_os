"""Two-stream Adaptive Graph Convolutional Network for face landmarks.

Processes (T, 68, 2) landmark sequences through joint and bone streams
with learnable adjacency for facial AU/expression prediction.

Inspired by 2s-AGCN (Shi et al. 2019) and FER_PSTBLN_MCD (Hdr et al.).
Adapted for 68-point dlib face landmarks.
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .features import SKELETON_EDGES, build_face_adjacency

logger = logging.getLogger("primateface.analysis.face_gcn")

# Bone connections: for each edge (i,j), the bone vector is kpts[j] - kpts[i]
BONE_PAIRS: List[Tuple[int, int]] = SKELETON_EDGES.copy()


def _compute_bones(joints: torch.Tensor) -> torch.Tensor:
    """Compute bone vectors from joint positions.

    For each edge (i, j), bone = joints[j] - joints[i].

    Args:
        joints: (B, T, V, C) joint positions.

    Returns:
        (B, T, V, C) bone vectors (zero for non-connected nodes).
    """
    bones = torch.zeros_like(joints)
    for i, j in BONE_PAIRS:
        bones[:, :, j, :] = joints[:, :, j, :] - joints[:, :, i, :]
    return bones


class AdaptiveGraphConv(nn.Module):
    """Adaptive spatial graph convolution with learnable adjacency.

    A = A_fixed + A_learnable + A_data
    where A_data is computed from input features (attention-based).

    Args:
        in_ch: Input feature channels.
        out_ch: Output feature channels.
        num_nodes: Number of graph nodes.
        adaptive: Learn adjacency adjustments.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_nodes: int = 68,
        adaptive: bool = True,
    ) -> None:
        super().__init__()
        adj_np = build_face_adjacency(num_nodes)
        self.register_buffer("A_fixed", torch.from_numpy(adj_np))

        self.weight = nn.Linear(in_ch, out_ch, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)

        self.adaptive = adaptive
        if adaptive:
            # Learnable residual adjacency
            self.A_learnable = nn.Parameter(
                torch.zeros(num_nodes, num_nodes)
            )
            # Data-dependent attention adjacency
            self.attn_a = nn.Linear(in_ch, num_nodes // 4)
            self.attn_b = nn.Linear(in_ch, num_nodes // 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, V, C_in) node features.

        Returns:
            (B, T, V, C_out).
        """
        b, t, v, c = x.shape

        # Build adaptive adjacency
        adj = self.A_fixed
        if self.adaptive:
            adj = adj + torch.tanh(self.A_learnable)
            # Data-dependent attention
            x_mean = x.mean(dim=1)  # (B, V, C)
            a = self.attn_a(x_mean)  # (B, V, D)
            b_feat = self.attn_b(x_mean)  # (B, V, D)
            a_data = torch.softmax(
                torch.bmm(a, b_feat.transpose(1, 2)), dim=-1
            )  # (B, V, V)
            # Expand adj for batch
            adj = adj.unsqueeze(0) + a_data  # (B, V, V)

        # Graph conv: A @ X @ W
        xw = self.weight(x)  # (B, T, V, C_out)
        if adj.dim() == 2:
            out = torch.matmul(adj, xw)
        else:
            # (B, V, V) @ (B, T, V, C_out) -> need to handle T dimension
            out = torch.zeros_like(xw)
            for ti in range(t):
                out[:, ti] = torch.bmm(adj, xw[:, ti])

        c_out = out.size(-1)
        out = self.bn(out.reshape(-1, c_out)).reshape(b, t, v, c_out)
        return out


class TwoStreamFaceGCN(nn.Module):
    """Two-stream adaptive GCN for face landmark sequences.

    Joint stream: operates on landmark positions (T, 68, 2).
    Bone stream: operates on bone vectors (landmark differences).
    Streams are fused before classification.

    Args:
        num_classes: Number of output classes (AUs).
        channels: Channel sizes per GCN block.
        temporal_kernel: Temporal conv kernel size.
        dropout: Dropout rate.
        adaptive: Use adaptive (learnable) adjacency.
    """

    def __init__(
        self,
        num_classes: int,
        channels: Optional[List[int]] = None,
        temporal_kernel: int = 9,
        dropout: float = 0.3,
        adaptive: bool = True,
    ) -> None:
        super().__init__()
        if channels is None:
            channels = [32, 64]

        # Joint stream
        self.joint_blocks = nn.ModuleList()
        prev = 2
        for ch in channels:
            block = nn.ModuleDict({
                "gcn": AdaptiveGraphConv(prev, ch, 68, adaptive),
                "tcn": nn.Conv1d(ch, ch, temporal_kernel,
                                 padding=temporal_kernel // 2),
                "bn": nn.BatchNorm1d(ch),
                "relu": nn.ReLU(inplace=True),
                "drop": nn.Dropout(dropout),
                "residual": nn.Linear(prev, ch) if prev != ch else nn.Identity(),
            })
            self.joint_blocks.append(block)
            prev = ch

        # Bone stream (same architecture)
        self.bone_blocks = nn.ModuleList()
        prev = 2
        for ch in channels:
            block = nn.ModuleDict({
                "gcn": AdaptiveGraphConv(prev, ch, 68, adaptive),
                "tcn": nn.Conv1d(ch, ch, temporal_kernel,
                                 padding=temporal_kernel // 2),
                "bn": nn.BatchNorm1d(ch),
                "relu": nn.ReLU(inplace=True),
                "drop": nn.Dropout(dropout),
                "residual": nn.Linear(prev, ch) if prev != ch else nn.Identity(),
            })
            self.bone_blocks.append(block)
            prev = ch

        # Fusion + classification
        self.head = nn.Linear(channels[-1] * 2, num_classes)

    def _run_stream(
        self, x: torch.Tensor, blocks: nn.ModuleList
    ) -> torch.Tensor:
        """Run one stream through GCN blocks.

        Args:
            x: (B, T, V, C) input.
            blocks: List of GCN block dicts.

        Returns:
            (B, C_out) pooled features.
        """
        for block in blocks:
            res = block["residual"](x)  # (B, T, V, C_out)
            out = block["gcn"](x)       # (B, T, V, C_out)
            out = block["relu"](out)

            # Temporal conv: reshape to (B*V, C, T)
            b, t, v, c = out.shape
            out_t = out.permute(0, 2, 3, 1).reshape(b * v, c, t)
            out_t = block["tcn"](out_t)
            out_t = block["bn"](out_t.reshape(-1, c)).reshape(b * v, c, t)
            out = out_t.reshape(b, v, c, t).permute(0, 3, 1, 2)  # (B,T,V,C)

            out = block["relu"](out)
            out = block["drop"](out)
            x = out + res

        # Global pool over time and nodes
        return x.mean(dim=(1, 2))  # (B, C_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, 68, 2) landmark sequences.

        Returns:
            (B, num_classes) logits.
        """
        # Compute bone vectors
        bones = _compute_bones(x)

        # Run both streams
        joint_feat = self._run_stream(x, self.joint_blocks)    # (B, C)
        bone_feat = self._run_stream(bones, self.bone_blocks)  # (B, C)

        # Late fusion
        fused = torch.cat([joint_feat, bone_feat], dim=-1)  # (B, 2*C)
        return self.head(fused)
