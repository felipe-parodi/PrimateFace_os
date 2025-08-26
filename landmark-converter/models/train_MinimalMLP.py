"""Training script for Minimal MLP landmark converter.

This script trains a simple single-hidden-layer MLP model for fast
and lightweight keypoint conversion.

Example:
    $ python train_MinimalMLP.py --coco_json data.json --hidden_dim 128 --epochs 100
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

# --- Utility Imports ---
from utils.data_utils import CocoPairedKeypointDataset, prepare_data
from utils.parser_utils import add_common_training_args, validate_common_args
from src.models import MinimalMLPConverter
from src.training_pipeline import ModelTrainer

# --- Configuration & Hyperparameters (Defaults for this specific script) ---
DEFAULT_OUTPUT_DIR_MINIMAL_MLP = './outputs/minimal_mlp_training'
DEFAULT_HIDDEN_DIM = 128 # Default for MinimalMLPConverter

def main(args):
    print("Starting Minimal MLP Converter Training (using ModelTrainer)...")
    print(f"Args: {args}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = None
    if torch.cuda.is_available():
        if args.gpu_id is not None:
            if args.gpu_id < torch.cuda.device_count():
                device = torch.device(f"cuda:{args.gpu_id}")
                torch.cuda.set_device(args.gpu_id)
                print(f"Using specified GPU: cuda:{args.gpu_id}")
            else:
                print(f"Warning: GPU ID {args.gpu_id} is invalid. Using default CUDA device (cuda:0 if available). Recommended to check available GPUs.")
                device = torch.device("cuda")
        else:
            device = torch.device("cuda")
            print(f"Using default CUDA device (cuda:0 if available).")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    img_transforms = transforms.Compose([
        transforms.Resize((256, 256)), # Consider making this configurable via args
        transforms.ToTensor(),
    ])

    print("Preparing data loaders...")
    train_loader, val_loader, test_loader, _ = prepare_data(
        args,
        img_transforms,
        CocoPairedKeypointDataset
    )

    if not train_loader or not val_loader:
        print("Error: Training or Validation DataLoader is None. Cannot proceed.")
        return

    print("Initializing MinimalMLPConverter model...")
    num_keypoints_to_predict = args.num_target_kpt - args.target_kpt_slice_idx
    model = MinimalMLPConverter(
        num_source_kpts=args.num_source_kpt,
        num_target_kpts=num_keypoints_to_predict,
        hidden_dim=args.hidden_dim # Use the new argument
    ).to(device)
    print(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    print("Initializing ModelTrainer...")
    trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        output_dir=args.output_dir,
        args=args, 
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        # num_source_kpt and target_kpt_slice_idx are in args
    )

    print("Starting training via ModelTrainer...")
    trainer.train()

    if test_loader:
        print("Starting test set evaluation via ModelTrainer...")
        trainer.evaluate_on_test_set()
    else:
        print("Skipping test set evaluation as no test data is available or test split was zero.")

    print("Minimal MLP Converter Training Process (using ModelTrainer) completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Minimal MLP Converter for keypoint conversion using ModelTrainer.")
    
    # Add common arguments
    parser = add_common_training_args(parser)

    # Set script-specific defaults for common arguments
    parser.set_defaults(
        output_dir=DEFAULT_OUTPUT_DIR_MINIMAL_MLP,
        # epochs=100, # Uses default from parser_utils unless overridden here
        # learning_rate=1e-4, # Uses default from parser_utils
        # batch_size=32, # Uses default from parser_utils
        # vis_every_n_epochs = 20, # Uses default from parser_utils
        # vis_num_samples = 5, # Uses default from parser_utils
        # vis_num_samples_test = 5 # Uses default from parser_utils
    )

    # Add model-specific arguments for MinimalMLPConverter
    parser.add_argument("--hidden_dim", type=int, default=DEFAULT_HIDDEN_DIM, help="Size of the hidden layer for MinimalMLPConverter.")
    # Note: num_source_kpt and num_target_kpt (+slice_idx) are common args now

    args = parser.parse_args()
    
    # Validate common arguments
    validate_common_args(args, parser) # parser instance for error reporting
    # Add any script-specific validation if needed
    
    main(args) 