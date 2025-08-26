"""Training script for Simple Linear landmark converter.

This script trains a direct linear transformation model for baseline
keypoint conversion with minimal parameters.

Example:
    $ python train_SimplerLinearConverter.py --coco_json data.json --epochs 50
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
from src.models import SimpleLinearConverter
from src.training_pipeline import ModelTrainer

# --- Configuration & Hyperparameters (Defaults for this specific script) ---
DEFAULT_OUTPUT_DIR_SIMPLE = './outputs/simple_linear_converter_training' 

def main(args):
    print("Starting Simple Linear Converter Training (using ModelTrainer)...")
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
        transforms.Resize((256, 256)), 
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

    print("Initializing SimpleLinearConverter model...")
    num_keypoints_to_predict = args.num_target_kpt - args.target_kpt_slice_idx
    model = SimpleLinearConverter(
        num_source_kpts=args.num_source_kpt,
        num_target_kpts=num_keypoints_to_predict
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

    print("Simple Linear Converter Training Process (using ModelTrainer) completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Simple Linear Converter for keypoint conversion using ModelTrainer.")
    
    parser = add_common_training_args(parser)

    parser.set_defaults(
        output_dir=DEFAULT_OUTPUT_DIR_SIMPLE,
        learning_rate=1e-3, 
        epochs=50,          
        vis_every_n_epochs=10,
        vis_num_samples=3,
        vis_num_samples_test=3
    )
    
    args = parser.parse_args()
    
    validate_common_args(args, parser)
    
    main(args) 