"""Command-line argument parsing utilities for landmark converter.

This module provides shared argument parsing functions used across
training scripts to ensure consistency.
"""

import argparse

# Default configurations that might be shared or overridden by specific scripts
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 100
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.15
DEFAULT_SOURCE_KPT_FIELD = 'keypoints_68'
DEFAULT_TARGET_KPT_FIELD = 'keypoints'
DEFAULT_NUM_SOURCE_KPT = 68
DEFAULT_NUM_TARGET_KPT = 49
DEFAULT_TARGET_KPT_SLICE_IDX = 1
DEFAULT_WEIGHT_DECAY = 1e-4

DEFAULT_CONVERSION_MODES = ["68_to_49", "49_to_68"]

def add_common_training_args(parser):
    """Adds common arguments to a an argparse.ArgumentParser instance."""
    # Data and Paths
    parser.add_argument("--coco_json", type=str, required=True, help="Path to COCO JSON annotations file.")
    parser.add_argument("--image_base_dir", type=str, default=None, help="Base directory for resolving image paths in COCO JSON. If None, paths are assumed to be absolute or relative to CWD.")
    # Output directory default will be set by individual scripts
    parser.add_argument("--output_dir", type=str, help="Directory to save outputs.")
    parser.add_argument("--source_field", type=str, default=DEFAULT_SOURCE_KPT_FIELD, help="Source keypoints field in COCO JSON.")
    parser.add_argument("--target_field", type=str, default=DEFAULT_TARGET_KPT_FIELD, help="Target keypoints field in COCO JSON.")
    parser.add_argument("--source_num_field", type=str, default="num_keypoints_68", help="Field name for number of source keypoints in COCO annotations.")
    parser.add_argument("--target_num_field", type=str, default="num_keypoints", help="Field name for number of target keypoints in COCO annotations.")

    # Model Structure (Keypoint numbers)
    parser.add_argument("--num_source_kpt", type=int, default=DEFAULT_NUM_SOURCE_KPT, help="Number of source keypoints.")
    parser.add_argument("--num_target_kpt", type=int, default=DEFAULT_NUM_TARGET_KPT, help="Actual total number of target keypoints available in the data.")
    parser.add_argument("--target_kpt_slice_idx", type=int, default=DEFAULT_TARGET_KPT_SLICE_IDX,
                        help="Index to slice target keypoints for prediction (e.g., 1 if model predicts N-1 points from N total, 0 if model predicts all N points).")

    # Training Hyperparameters
    # Defaults for epochs, batch_size, lr will be set by individual scripts if they differ from these
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size.")
    parser.add_argument("--learning_rate", "--lr", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="Weight decay for AdamW optimizer.")

    # Data Splitting & Loading
    parser.add_argument("--train_ratio", type=float, default=DEFAULT_TRAIN_RATIO, help="Proportion of data for training.")
    parser.add_argument("--val_ratio", type=float, default=DEFAULT_VAL_RATIO, help="Proportion of data for validation.")
    # Test ratio is inferred: 1.0 - train_ratio - val_ratio
    parser.add_argument("--split_strategy", type=str, default="random", choices=["random", "pca_kmeans_stratified"],
                        help="Strategy for data splitting: 'random' or 'pca_kmeans_stratified'.")
    parser.add_argument("--pca_variance_threshold", type=float, default=0.95,
                        help="Minimum ratio of variance to be explained by PCA for 'pca_kmeans_stratified' split (e.g., 0.95 for 95%%).")
    parser.add_argument("--kmeans_clusters", type=int, default=10, # Changed from 15 in train_ConvertorMLP, 10 in train_SimplerLinear. Standardizing to 10.
                        help="Number of K-Means clusters for 'pca_kmeans_stratified' split.")
    parser.add_argument("--split_random_seed", type=int, default=42, help="Random seed for data splitting.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader workers.")

    # Hardware
    parser.add_argument("--gpu_id", type=int, default=None, help="Specific GPU ID to use if CUDA is available. If None, uses default CUDA device.")

    # Visualization & Logging
    parser.add_argument("--vis_every_n_epochs", type=int, default=20,
                        help="Frequency (in epochs) of visualizing validation predictions. 0 to disable.")
    parser.add_argument("--vis_num_samples", type=int, default=5,
                        help="Number of validation samples to visualize during training.")
    parser.add_argument("--vis_num_samples_test", type=int, default=5,
                        help="Number of test samples to visualize after training.")
    parser.add_argument("--save_plots_as_pdf", action='store_true', help="Save plots as PDF in addition to PNG.")
    parser.add_argument("--visualize_raw_data", action='store_true',
                        help="Visualize a few raw dataset samples (image, source kpts, target kpts) before training starts.")

    # Checkpointing & Resuming
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Path to a checkpoint file (.pth) to resume training from (model state, optimizer, epoch).")
    parser.add_argument("--resume_if_best_exists", action='store_true',
                        help="Automatically resume from 'best_model.pth' in output_dir if it exists and --resume_checkpoint is not set.")

    # Conversion Mode
    parser.add_argument("--conversion_mode", type=str, default=None, choices=DEFAULT_CONVERSION_MODES + [None],
                        help="Predefined conversion mode (e.g., '68_to_49', '49_to_68'). If set, overrides related source/target field and kpt_num arguments.")

    # Enforced Overlap / Keypoint Mapping
    parser.add_argument("--enforce_overlap", action='store_true',
                        help="If set, enforces a 1-to-1 mapping for specified keypoints using --overlap_mapping.")
    parser.add_argument("--overlap_mapping", type=str, default=None,
                        help="String representation of keypoint overlap mapping. E.g., '[[0,0],[1,2],[3,5]]\' for src_idx 0 to pred_tgt_idx 0, etc. Indices are for the predicted target keypoints (after slice_idx).")
    parser.add_argument("--use_fixed_mapping_68_to_49", action='store_true',
                        help="If set along with --enforce_overlap, uses a predefined 68-to-49 keypoint mapping. Ignored if --overlap_mapping string is provided.")
    parser.add_argument("--enforce_overlap_epochs", type=int, default=1,
                        help="Number of initial epochs to enforce the overlap mapping if --enforce_overlap is set. Defaults to 1 (first epoch only).")
    return parser

def validate_common_args(args, parser_instance):
    """Validates common argument constraints."""
    if not (0 < args.train_ratio <= 1 and 0 < args.val_ratio <= 1 and (args.train_ratio + args.val_ratio) <= 1):
        parser_instance.error("Train/Val ratios must be > 0 and <= 1. Their sum must be <= 1.")
    if (args.train_ratio + args.val_ratio) > 1: # This condition is covered by the one above but kept for explicitness
         parser_instance.error("Sum of train_ratio and val_ratio cannot exceed 1.")
    
    if args.epochs <= 0:
        parser_instance.error("Number of epochs must be positive.")
    if args.batch_size <= 0:
        parser_instance.error("Batch size must be positive.")

    # Note: Some args like output_dir being None before default is set by script is fine here.
    # Specific scripts might add more validation. 