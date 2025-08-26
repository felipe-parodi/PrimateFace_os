"""Training script for MLP-based landmark converter.

This script trains a multi-layer perceptron model with dropout for
robust keypoint conversion between different landmark systems.

Example:
    $ python train_ConverterMLP.py --coco_json data.json --hidden_dim1 256 --hidden_dim2 128
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms

# --- Utility Imports ---
from utils.data_utils import CocoPairedKeypointDataset, visualize_dataset_samples, prepare_data

from utils.parser_utils import add_common_training_args, validate_common_args
from src.models import KeypointConverterMLP
from src.training_pipeline import ModelTrainer

DEFAULT_OUTPUT_DIR_MLP = './outputs/converter_mlp_training'
DEFAULT_HIDDEN_DIM1 = 256
DEFAULT_HIDDEN_DIM2 = 256


# --- Main Function ---
def main(args):
    print("Starting Keypoint Conversion MLP Training (using ModelTrainer)...")
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
                device = torch.device("cuda") # Default to cuda:0 or first available
        else:
            device = torch.device("cuda")
            print(f"Using default CUDA device (cuda:0 if available).")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")

    img_transforms = transforms.Compose([
        transforms.Resize((256, 256)), # Consider making this configurable via args
        transforms.ToTensor(),
    ])

    print("Preparing data loaders...")
    # The prepare_data function will internally handle CocoPairedKeypointDataset instantiation, 
    # raw data visualization, splitting, and DataLoader creation.
    train_loader, val_loader, test_loader, _ = prepare_data(
        args,
        img_transforms,
        CocoPairedKeypointDataset
    )

    if not train_loader or not val_loader:
        print("Error: Training or Validation DataLoader is None. Cannot proceed.")
        return

    print("Initializing KeypointConverterMLP model...")
    # num_target_kpt from args is total in data. Model predicts N - slice_idx.
    num_keypoints_to_predict = args.num_target_kpt - args.target_kpt_slice_idx
    model = KeypointConverterMLP(
        num_source_kpts=args.num_source_kpt,
        num_target_kpts=num_keypoints_to_predict,
        hidden_dim1=args.hidden_dim1,
        hidden_dim2=args.hidden_dim2
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
        args=args, # Pass all args; ModelTrainer will use what it needs
        train_loader=train_loader, 
        val_loader=val_loader, 
        test_loader=test_loader, # Can be None if no test set
        # num_source_kpt and target_kpt_slice_idx are in args, ModelTrainer can access them
    )
    
    print("Starting training via ModelTrainer...")
    trainer.train()

    if test_loader: # test_loader can be None if test split is 0
        print("Starting test set evaluation via ModelTrainer...")
        trainer.evaluate_on_test_set()
    else:
        print("Skipping test set evaluation as no test data is available or test split was zero.")

    print("Keypoint Conversion MLP Process (using ModelTrainer) completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an MLP to convert facial keypoint configurations using ModelTrainer.")
    
    # Add common arguments
    parser = add_common_training_args(parser)

    # Set script-specific defaults for common arguments
    parser.set_defaults(
        output_dir=DEFAULT_OUTPUT_DIR_MLP,
        # epochs=100, # Uses default from parser_utils unless overridden here
        # learning_rate=1e-4, # Uses default from parser_utils
        # batch_size=32, # Uses default from parser_utils
        # vis_every_n_epochs = 20, # Uses default from parser_utils
        # vis_num_samples = 5, # Uses default from parser_utils
        # vis_num_samples_test = 5 # Uses default from parser_utils
    )

    # Add model-specific arguments for KeypointConverterMLP
    parser.add_argument("--hidden_dim1", type=int, default=DEFAULT_HIDDEN_DIM1, help="Size of the first hidden layer for MLP.")
    parser.add_argument("--hidden_dim2", type=int, default=DEFAULT_HIDDEN_DIM2, help="Size of the second hidden layer for MLP.")
    # Note: num_source_kpt and num_target_kpt (+slice_idx) are common args now

    args = parser.parse_args()

    # Validate common arguments
    validate_common_args(args, parser) # parser instance for error reporting
    # Add any script-specific validation if needed

    main(args) 