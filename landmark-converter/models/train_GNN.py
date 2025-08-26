"""Training script for Graph Neural Network landmark converter.

This script trains a GNN model that treats keypoints as nodes in a graph,
learning structural relationships for conversion. Requires torch_geometric.

Note: This model is experimental and not fully tested.

Example:
    $ python train_GNN.py --coco_json data.json --num_gcn_layers 3 --gcn_hidden_channels 128
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms 

# --- Utility Imports ---
from utils.data_utils import CocoPairedKeypointGraphDataset, prepare_data 
from utils.parser_utils import add_common_training_args, validate_common_args, DEFAULT_CONVERSION_MODES
from src.models import KeypointConverterGNN
from src.training_pipeline import ModelTrainer

# --- Configuration & Hyperparameters (Defaults for this specific script) ---
DEFAULT_OUTPUT_DIR_GNN = './outputs/gnn_converter_training'
DEFAULT_GCN_HIDDEN_CHANNELS = 128
DEFAULT_NUM_GCN_LAYERS = 2

def main(args):
    print("Starting GNN Converter Training (using ModelTrainer)...")
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

    print("Preparing data loaders using CocoPairedKeypointGraphDataset...")
    train_loader, val_loader, test_loader, dataset_details = prepare_data(
        args,
        img_transforms,
        CocoPairedKeypointGraphDataset # Pass the graph dataset class
    )

    if not train_loader or not val_loader:
        print("Error: Training or Validation DataLoader is None. Cannot proceed.")
        return

    print("Initializing KeypointConverterGNN model...")
    # num_source_kpt is not directly passed to GNN model constructor anymore.
    # node_input_features is fixed at 2 (x,y) for this setup.
    # num_target_kpt - args.target_kpt_slice_idx determines the output size.
    num_keypoints_to_predict = args.num_target_kpt - args.target_kpt_slice_idx
    model = KeypointConverterGNN(
        num_target_kpts=num_keypoints_to_predict,
        node_input_features=2, # (x,y) for each source keypoint node
        gcn_hidden_channels=args.gcn_hidden_channels, # New arg name
        num_gcn_layers=args.num_gcn_layers # New arg
    ).to(device)
    print(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    print("Initializing ModelTrainer...")
    # We need to inform ModelTrainer that it will receive PyG Batch objects
    # and how to handle them for model input and loss calculation.
    # Adding a flag like `model_expects_graph_data=True`
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
        model_expects_graph_data=True # Key addition for ModelTrainer
    )

    print("Starting training via ModelTrainer...")
    trainer.train()

    if test_loader:
        print("Starting test set evaluation via ModelTrainer...")
        trainer.evaluate_on_test_set()
    else:
        print("Skipping test set evaluation as no test data is available or test split was zero.")

    print("Keypoint Converter GNN Training Process (using ModelTrainer) completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Keypoint Converter GNN using ModelTrainer.")
    
    parser = add_common_training_args(parser)

    parser.set_defaults(
        output_dir=DEFAULT_OUTPUT_DIR_GNN,
        # Consider if GNNs need different default LR, epochs, etc.
    )

    parser.add_argument("--gcn_hidden_channels", type=int, default=DEFAULT_GCN_HIDDEN_CHANNELS, 
                        help="Number of hidden features in GCN layers.")
    parser.add_argument("--num_gcn_layers", type=int, default=DEFAULT_NUM_GCN_LAYERS, 
                        help="Number of GCN layers in the GNN model.")

    args = parser.parse_args()
    
    # --- Apply conversion_mode --- # This block is problematic when called by train_all_models.py
    if args.conversion_mode:      # train_all_models.py already sets all necessary fields directly.
        print(f"Applying conversion mode: {args.conversion_mode}")
        if args.conversion_mode == "68_to_49":
            args.source_field = "keypoints_68"
            args.target_field = "keypoints" # Adjusted for your 49 kpt field name
            args.source_num_field = "num_keypoints_68"
            args.target_num_field = "num_keypoints" # Adjusted for your 49 kpt count field name
            args.num_source_kpt = 68
            args.num_target_kpt = 48
            args.target_kpt_slice_idx = 1 # Common default for 68->49 (predicting 48)
        elif args.conversion_mode == "49_to_68":
            args.source_field = "keypoints" # Adjusted for your 49 kpt field name
            args.target_field = "keypoints_68"
            args.source_num_field = "num_keypoints" # Adjusted for your 49 kpt count field name
            args.target_num_field = "num_keypoints_68"
            args.num_source_kpt = 49
            args.num_target_kpt = 68
            args.target_kpt_slice_idx = 0 # Common default for 49->68 (predicting all 68)
        else:
            # This case should ideally not be hit if choices in argparse are set correctly
            # and default is None. But as a safeguard:
            if args.conversion_mode not in DEFAULT_CONVERSION_MODES: # Check against defined modes
                 parser.error(f"Invalid conversion_mode: {args.conversion_mode}. Must be one of {DEFAULT_CONVERSION_MODES} or None.")
    # --- End Apply conversion_mode ---

    validate_common_args(args, parser)
    # Specific GNN validation could be added here if needed.
    
    # Note on ModelTrainer adaptation for GNNs:
    # The ModelTrainer class (in src/training_pipeline.py) will need to be updated
    # to correctly handle the `model_expects_graph_data=True` flag.
    # If True, it should:
    # 1. Expect batches from the DataLoader to be PyG Batch objects.
    # 2. Pass the entire Batch object to model.forward(batch_obj).
    # 3. Calculate loss using `batch_obj.y` as the target, e.g., criterion(predictions, batch_obj.y.view_as(predictions)).
    # Ensure the output shape of the GNN (num_target_kpts, 2) matches how batch_obj.y is compared after view_as.

    main(args) 