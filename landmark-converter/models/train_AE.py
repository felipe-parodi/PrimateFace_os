"""Training script for Autoencoder-based landmark converter.

This script trains an autoencoder model that learns to convert keypoints
by encoding them into a latent representation and decoding to the target format.

Example:
    $ python train_AE.py --coco_json data.json --epochs 200 --latent_dim 64
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
from src.models import KeypointConverterAE
from src.training_pipeline import ModelTrainer

# --- Configuration & Hyperparameters (Defaults for this specific script) ---
DEFAULT_OUTPUT_DIR_AE = './outputs/ae_converter_training'
DEFAULT_LATENT_DIM = 64
DEFAULT_HIDDEN_DIM_ENC = 128
DEFAULT_HIDDEN_DIM_DEC = 128

def main(args):
    print("Starting Autoencoder (AE) Converter Training (using ModelTrainer)...")
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
        CocoPairedKeypointDataset # Pass the dataset class
    )

    if not train_loader or not val_loader:
        print("Error: Training or Validation DataLoader is None. Cannot proceed.")
        return

    print("Initializing KeypointConverterAE model...")
    num_keypoints_to_predict = args.num_target_kpt - args.target_kpt_slice_idx
    model = KeypointConverterAE(
        num_source_kpts=args.num_source_kpt,
        num_target_kpts=num_keypoints_to_predict,
        latent_dim=args.latent_dim,
        hidden_dim_enc=args.hidden_dim_enc,
        hidden_dim_dec=args.hidden_dim_dec
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
    )

    print("Starting training via ModelTrainer...")
    trainer.train()

    if test_loader:
        print("Starting test set evaluation via ModelTrainer...")
        trainer.evaluate_on_test_set()
    else:
        print("Skipping test set evaluation as no test data is available or test split was zero.")

    print("Keypoint Converter AE Training Process (using ModelTrainer) completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Keypoint Converter Autoencoder (AE) using ModelTrainer.")
    
    parser = add_common_training_args(parser)

    parser.set_defaults(
        output_dir=DEFAULT_OUTPUT_DIR_AE,
    )

    # Add model-specific arguments for KeypointConverterAE
    parser.add_argument("--latent_dim", type=int, default=DEFAULT_LATENT_DIM, help="Size of the latent dimension for AE.")
    parser.add_argument("--hidden_dim_enc", type=int, default=DEFAULT_HIDDEN_DIM_ENC, help="Size of the encoder hidden layer for AE.")
    parser.add_argument("--hidden_dim_dec", type=int, default=DEFAULT_HIDDEN_DIM_DEC, help="Size of the decoder hidden layer for AE.")

    args = parser.parse_args()
    
    validate_common_args(args, parser)
    
    main(args) 