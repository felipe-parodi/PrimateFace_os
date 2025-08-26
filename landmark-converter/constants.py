"""Constants and default configurations for landmark converter.

This module contains shared constants, default values, and configurations
used across the landmark converter package.
"""

from typing import Dict

# Default directories
DEFAULT_OUTPUT_DIR = './outputs/landmark_converter'
DEFAULT_MODEL_DIR = './models'

# Model architecture defaults
DEFAULT_EMBED_DIM = 128
DEFAULT_NUM_HEADS = 4
DEFAULT_MLP_HIDDEN_DIM = 256
DEFAULT_HIDDEN_DIM1 = 256
DEFAULT_HIDDEN_DIM2 = 256
DEFAULT_GNN_HIDDEN_DIM = 128

# Training defaults
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_VAL_SPLIT = 0.15
DEFAULT_TEST_SPLIT = 0.15
DEFAULT_RANDOM_SEED = 42

# Data processing defaults
DEFAULT_IMAGE_SIZE = (256, 256)
DEFAULT_SOURCE_KPT_FIELD = 'keypoints'
DEFAULT_TARGET_KPT_FIELD = 'keypoints_49'
DEFAULT_SOURCE_NUM_KPT_FIELD = 'num_keypoints'
DEFAULT_TARGET_NUM_KPT_FIELD = 'num_keypoints_49'

# Visualization defaults
DEFAULT_VIS_EVERY_N_EPOCHS = 10
DEFAULT_NUM_VIS_SAMPLES = 3

# Model registry mapping model names to their configurations
MODEL_CONFIGS: Dict[str, Dict] = {
    'simple_linear': {
        'description': 'Simple linear transformation model',
        'class_name': 'SimpleLinearConverter',
        'params': {}
    },
    'mlp': {
        'description': 'Multi-layer perceptron with dropout',
        'class_name': 'KeypointConverterMLP',
        'params': {
            'hidden_dim1': DEFAULT_HIDDEN_DIM1,
            'hidden_dim2': DEFAULT_HIDDEN_DIM2
        }
    },
    'minimal_mlp': {
        'description': 'Minimal MLP with single hidden layer',
        'class_name': 'MinimalMLPConverter',
        'params': {
            'hidden_dim': 128
        }
    },
    'mlp_attention': {
        'description': 'MLP with multi-head attention',
        'class_name': 'KeypointConverterMLPWithAttention',
        'params': {
            'embed_dim': DEFAULT_EMBED_DIM,
            'num_heads': DEFAULT_NUM_HEADS,
            'mlp_hidden_dim': DEFAULT_MLP_HIDDEN_DIM
        }
    },
    'gnn': {
        'description': 'Graph Neural Network converter',
        'class_name': 'GNNConverter',
        'params': {
            'hidden_dim': DEFAULT_GNN_HIDDEN_DIM,
            'num_layers': 3
        }
    },
    'autoencoder': {
        'description': 'Autoencoder-based converter',
        'class_name': 'AutoencoderConverter',
        'params': {
            'latent_dim': 64
        }
    }
}

# Fixed mapping for 68 to 49 keypoint conversion
FIXED_MAPPING_68_TO_49: Dict[int, int] = {
    48: 6, 8: 7, 54: 8, 27: 13, 39: 14, 36: 16, 42: 19, 45: 21,
    32: 32, 34: 33, 30: 35, 51: 36, 57: 37, 28: 39, 29: 40,
    49: 41, 52: 43, 53: 44, 55: 45, 56: 46, 58: 47, 59: 48
}

# Conversion mode configurations
CONVERSION_MODES = {
    '68_to_49': {
        'num_source_kpt': 68,
        'num_target_kpt': 49,
        'target_kpt_slice_idx': 1  # Skip first keypoint (usually nose tip)
    },
    '49_to_68': {
        'num_source_kpt': 49,
        'num_target_kpt': 68,
        'target_kpt_slice_idx': 0
    },
    'custom': {
        # User must specify num_source_kpt and num_target_kpt
        'target_kpt_slice_idx': 0
    }
}