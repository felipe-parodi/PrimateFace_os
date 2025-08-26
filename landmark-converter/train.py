"""Unified training script for landmark converter models.

This script provides a single entry point for training all landmark converter
model architectures. Use --model to select the architecture.

Example:
    Train a simple linear model:
        $ python train.py --model simple_linear --coco_json data.json --epochs 100
    
    Train an attention-based MLP:
        $ python train.py --model mlp_attention --coco_json data.json --output_dir results/
    
    Train with custom conversion mode:
        $ python train.py --model mlp --conversion_mode 68_to_49 --coco_json data.json
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from constants import (
    MODEL_CONFIGS,
    CONVERSION_MODES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_VAL_SPLIT,
    DEFAULT_TEST_SPLIT,
    DEFAULT_RANDOM_SEED,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_VIS_EVERY_N_EPOCHS
)
from src.models import (
    SimpleLinearConverter,
    KeypointConverterMLP,
    MinimalMLPConverter,
    KeypointConverterMLPWithAttention
)
from src.training_pipeline import ModelTrainer
from utils.data_utils import CocoPairedKeypointDataset, prepare_data
from utils.parser_utils import add_common_training_args, validate_common_args


def get_model_class(model_name: str):
    """Get model class from registry.
    
    Args:
        model_name: Name of the model architecture.
        
    Returns:
        Model class corresponding to the name.
        
    Raises:
        ValueError: If model name is not recognized.
    """
    model_map = {
        'simple_linear': SimpleLinearConverter,
        'mlp': KeypointConverterMLP,
        'minimal_mlp': MinimalMLPConverter,
        'mlp_attention': KeypointConverterMLPWithAttention
    }
    
    if model_name not in model_map:
        available = ', '.join(model_map.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    
    return model_map[model_name]


def setup_device(gpu_id: Optional[int] = None) -> torch.device:
    """Setup and return the appropriate device for training.
    
    Args:
        gpu_id: Specific GPU ID to use. None for automatic selection.
        
    Returns:
        torch.device object configured for training.
    """
    if not torch.cuda.is_available():
        print("CUDA not available. Using CPU.")
        return torch.device("cpu")
    
    if gpu_id is not None:
        if gpu_id < torch.cuda.device_count():
            device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(gpu_id)
            print(f"Using specified GPU: cuda:{gpu_id}")
        else:
            print(f"Warning: GPU ID {gpu_id} is invalid. Using default CUDA device.")
            device = torch.device("cuda")
    else:
        device = torch.device("cuda")
        print("Using default CUDA device.")
    
    return device


def create_model(
    model_name: str,
    num_source_kpts: int,
    num_target_kpts: int,
    args: argparse.Namespace
) -> nn.Module:
    """Create and return a model instance.
    
    Args:
        model_name: Name of the model architecture.
        num_source_kpts: Number of source keypoints.
        num_target_kpts: Number of target keypoints to predict.
        args: Command-line arguments with model-specific parameters.
        
    Returns:
        Initialized model instance.
    """
    model_class = get_model_class(model_name)
    model_config = MODEL_CONFIGS.get(model_name, {})
    
    # Get model-specific parameters from args or use defaults
    if model_name == 'simple_linear':
        model = model_class(num_source_kpts, num_target_kpts)
    elif model_name == 'mlp':
        model = model_class(
            num_source_kpts, num_target_kpts,
            hidden_dim1=getattr(args, 'hidden_dim1', model_config['params']['hidden_dim1']),
            hidden_dim2=getattr(args, 'hidden_dim2', model_config['params']['hidden_dim2'])
        )
    elif model_name == 'minimal_mlp':
        model = model_class(
            num_source_kpts, num_target_kpts,
            hidden_dim=getattr(args, 'hidden_dim', model_config['params']['hidden_dim'])
        )
    elif model_name == 'mlp_attention':
        model = model_class(
            num_source_kpts, num_target_kpts,
            embed_dim=getattr(args, 'embed_dim', model_config['params']['embed_dim']),
            num_heads=getattr(args, 'num_heads', model_config['params']['num_heads']),
            mlp_hidden_dim=getattr(args, 'mlp_hidden_dim', model_config['params']['mlp_hidden_dim'])
        )
    else:
        raise ValueError(f"Model creation not implemented for: {model_name}")
    
    return model


def main(args: argparse.Namespace) -> None:
    """Main training function.
    
    Args:
        args: Parsed command-line arguments.
    """
    print(f"Starting {args.model} training...")
    print(f"Configuration: {args}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = setup_device(args.gpu_id)
    
    # Setup data transforms
    img_transforms = transforms.Compose([
        transforms.Resize(DEFAULT_IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    
    # Prepare data loaders
    print("Preparing data loaders...")
    train_loader, val_loader, test_loader, _ = prepare_data(
        args,
        img_transforms,
        CocoPairedKeypointDataset
    )
    
    # Determine number of keypoints to predict
    num_keypoints_to_predict = args.num_target_kpt - args.target_kpt_slice_idx
    
    # Create model
    print(f"Creating {args.model} model...")
    model = create_model(
        args.model,
        args.num_source_kpt,
        num_keypoints_to_predict,
        args
    )
    model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Setup optimizer and loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    criterion = nn.MSELoss()
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        output_dir=str(output_dir),
        args=args,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_source_kpt=args.num_source_kpt,
        target_kpt_slice_idx=args.target_kpt_slice_idx
    )
    
    # Train model
    print(f"Training for {args.epochs} epochs...")
    trainer.train(epochs=args.epochs)
    
    # Evaluate on test set if available
    if test_loader:
        print("Evaluating on test set...")
        trainer.evaluate_test()
    
    print(f"Training complete! Results saved to: {output_dir}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description='Unified training script for landmark converter models',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Model selection
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help='Model architecture to train'
    )
    
    # Add common training arguments
    add_common_training_args(parser)
    
    # Model-specific arguments
    parser.add_argument('--embed_dim', type=int, help='Embedding dimension (for attention model)')
    parser.add_argument('--num_heads', type=int, help='Number of attention heads')
    parser.add_argument('--mlp_hidden_dim', type=int, help='MLP hidden dimension')
    parser.add_argument('--hidden_dim', type=int, help='Hidden dimension (for minimal MLP)')
    parser.add_argument('--hidden_dim1', type=int, help='First hidden dimension (for MLP)')
    parser.add_argument('--hidden_dim2', type=int, help='Second hidden dimension (for MLP)')
    
    # Override defaults if not specified
    parser.set_defaults(
        output_dir=DEFAULT_OUTPUT_DIR,
        epochs=DEFAULT_EPOCHS,
        batch_size=DEFAULT_BATCH_SIZE,
        lr=DEFAULT_LEARNING_RATE,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        val_split=DEFAULT_VAL_SPLIT,
        test_split=DEFAULT_TEST_SPLIT,
        split_random_seed=DEFAULT_RANDOM_SEED,
        vis_every_n_epochs=DEFAULT_VIS_EVERY_N_EPOCHS
    )
    
    args = parser.parse_args()
    
    # Validate common arguments
    validate_common_args(args)
    
    # Apply conversion mode if specified
    if hasattr(args, 'conversion_mode') and args.conversion_mode in CONVERSION_MODES:
        mode_config = CONVERSION_MODES[args.conversion_mode]
        if not hasattr(args, 'num_source_kpt') or args.num_source_kpt is None:
            args.num_source_kpt = mode_config.get('num_source_kpt')
        if not hasattr(args, 'num_target_kpt') or args.num_target_kpt is None:
            args.num_target_kpt = mode_config.get('num_target_kpt')
        if not hasattr(args, 'target_kpt_slice_idx') or args.target_kpt_slice_idx is None:
            args.target_kpt_slice_idx = mode_config.get('target_kpt_slice_idx', 0)
    
    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(args)