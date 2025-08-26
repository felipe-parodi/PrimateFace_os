"""Model architectures for landmark conversion.

This module contains all neural network architectures for converting
between different keypoint annotation systems, including Linear, MLP,
Attention-based, Autoencoder, and GNN models.
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd

# Optional: Import PyG layers for GNN models
try:
    from torch_geometric.nn import GCNConv, global_mean_pool
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    GCNConv = None
    global_mean_pool = None

# --- Model Definition ---
class SimpleLinearConverter(nn.Module):
    def __init__(self, num_source_kpts, num_target_kpts):
        super().__init__()
        self.linear = nn.Linear(num_source_kpts * 2, num_target_kpts * 2)
        self.num_target_kpts = num_target_kpts

    def forward(self, source_kpts_flat):
        x = self.linear(source_kpts_flat)
        return x.view(-1, self.num_target_kpts, 2)
    
# --- Model Definition ---
class KeypointConverterMLP(nn.Module):
    def __init__(self, num_source_kpts, num_target_kpts, 
                 hidden_dim1=256, hidden_dim2=256):
        super().__init__()
        # Input: num_source_kpts * 2 (x, y)
        # Output: num_target_kpts * 2 (x, y)
        # Visibility flags are ignored for this MLP.
        self.fc1 = nn.Linear(num_source_kpts * 2, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3) # Added dropout
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3) # Added dropout
        self.fc3 = nn.Linear(hidden_dim2, num_target_kpts * 2)
        
        self.num_source_kpts = num_source_kpts
        self.num_target_kpts = num_target_kpts

    def forward(self, source_kpts_flat):
        # source_kpts_flat is expected to be [batch_size, num_source_kpts * 2]
        x = self.relu1(self.fc1(source_kpts_flat))
        x = self.dropout1(x)
        x = self.relu2(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        # Output will be [batch_size, num_target_kpts * 2]
        # Reshape to [batch_size, num_target_kpts, 2] for easier use with loss/evaluation if needed outside
        return x.view(-1, self.num_target_kpts, 2)


# --- Minimal MLP Converter --- 
class MinimalMLPConverter(nn.Module):
    def __init__(self, num_source_kpts, num_target_kpts, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(num_source_kpts * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_target_kpts * 2)
        self.num_target_kpts = num_target_kpts

    def forward(self, source_kpts_flat):
        # source_kpts_flat is [batch_size, num_source_kpts * 2]
        x = self.relu(self.fc1(source_kpts_flat))
        x = self.fc2(x)
        # Output is [batch_size, num_target_kpts * 2]
        return x.view(-1, self.num_target_kpts, 2)


# --- Attention-Enhanced MLP Converter --- 
class KeypointConverterMLPWithAttention(nn.Module):
    def __init__(self, num_source_kpts, num_target_kpts, 
                 embed_dim=128, num_heads=4, mlp_hidden_dim=256):
        super().__init__()
        self.num_source_kpts = num_source_kpts
        self.num_target_kpts = num_target_kpts
        input_dim = num_source_kpts * 2

        self.input_projection = nn.Linear(input_dim, embed_dim)
        # Using MultiheadAttention. Note: embed_dim must be divisible by num_heads.
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, num_target_kpts * 2)
        )

    def forward(self, source_kpts_flat):
        # source_kpts_flat: [batch_size, num_source_kpts * 2]
        # Project to embedding dimension
        projected_input = self.input_projection(source_kpts_flat) # [batch_size, embed_dim]
        
        # Reshape for attention: MultiheadAttention expects (N, L, E) for batch_first=True
        # Here, we treat the entire set of keypoints as a single "token" in a sequence of length 1.
        # Or, we could try to treat each keypoint as a token if we reshape differently and adjust embed_dim.
        # For simplicity, treating as one sequence item:
        attn_input = projected_input.unsqueeze(1) # [batch_size, 1, embed_dim]
        
        # Self-attention
        # Query, Key, Value are the same for self-attention
        attn_output, _ = self.attention(attn_input, attn_input, attn_input)
        
        # Add & Norm (skip connection)
        x = self.norm1(attn_input + attn_output) # [batch_size, 1, embed_dim]
        x = x.squeeze(1) # [batch_size, embed_dim]
        
        # MLP head
        output = self.mlp(x)
        return output.view(-1, self.num_target_kpts, 2)


# --- Autoencoder Converter --- 
class KeypointConverterAE(nn.Module):
    def __init__(self, num_source_kpts, num_target_kpts, 
                 latent_dim=64, hidden_dim_enc=128, hidden_dim_dec=128):
        super().__init__()
        self.num_target_kpts = num_target_kpts

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(num_source_kpts * 2, hidden_dim_enc),
            nn.ReLU(),
            nn.Linear(hidden_dim_enc, latent_dim),
            nn.ReLU() # Or Tanh, or nothing, depending on desired latent space properties
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim_dec),
            nn.ReLU(),
            nn.Linear(hidden_dim_dec, num_target_kpts * 2)
        )

    def forward(self, source_kpts_flat):
        # source_kpts_flat: [batch_size, num_source_kpts * 2]
        latent_representation = self.encoder(source_kpts_flat)
        output = self.decoder(latent_representation)
        return output.view(-1, self.num_target_kpts, 2)


# --- Graph Neural Network (GNN) Converter --- 
class KeypointConverterGNN(nn.Module):
    def __init__(self, num_target_kpts, node_input_features=2, gcn_hidden_channels=128, num_gcn_layers=2):
        super().__init__()
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError(
                "torch_geometric is required for GNN models. "
                "Install it with: pip install torch-geometric"
            )
        self.num_target_kpts = num_target_kpts
        self.node_input_features = node_input_features # Should be 2 for (x,y)
        self.gcn_hidden_channels = gcn_hidden_channels

        self.gcn_layers = nn.ModuleList()
        # Initial GCN layer: maps input node features (2 for x,y) to hidden_channels
        self.gcn_layers.append(GCNConv(node_input_features, gcn_hidden_channels))
        
        # Additional GCN layers
        for _ in range(max(0, num_gcn_layers - 1)):
            self.gcn_layers.append(GCNConv(gcn_hidden_channels, gcn_hidden_channels))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3) # Optional dropout after GCN layers

        # Output MLP: maps pooled graph features to target keypoints
        # The input dimension to this MLP depends on the global pooling strategy.
        # If global_mean_pool, it's gcn_hidden_channels.
        self.output_mlp = nn.Sequential(
            nn.Linear(gcn_hidden_channels, gcn_hidden_channels * 2),
            nn.ReLU(),
            nn.Linear(gcn_hidden_channels * 2, num_target_kpts * 2)
        )

    def forward(self, data):
        # data is a PyG Batch object, containing:
        # data.x: Node features [num_nodes_in_batch, node_feature_dim]
        # data.edge_index: Edge connectivity [2, num_edges_in_batch]
        # data.batch: Assignment vector [num_nodes_in_batch], maps each node to its graph in the batch
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply GCN layers
        for gcn_layer in self.gcn_layers:
            x = gcn_layer(x, edge_index)
            x = self.relu(x)
            # x = self.dropout(x) # Optional dropout between GCN layers
        
        # Global pooling: aggregates node features into a single graph-level representation
        # global_mean_pool requires the batch vector to know which nodes belong to which graph
        x_pooled = global_mean_pool(x, batch) # [batch_size, gcn_hidden_channels]
        x_pooled = self.dropout(x_pooled) # Apply dropout after pooling

        # Output MLP
        output_flat = self.output_mlp(x_pooled) # [batch_size, num_target_kpts * 2]
        
        # Reshape to [batch_size, num_target_kpts, 2]
        return output_flat.view(-1, self.num_target_kpts, 2)
