"""Evaluate DeepLabCut models using Normalized Mean Error (NME).

This script computes NME metrics for DeepLabCut model predictions on test data,
supporting both single-animal and multi-animal models.

Example:
    $ python evaluate_nme.py --model_dir path/to/dlc_model --test_data test.json
"""

import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import argparse
from typing import Optional, Tuple, Dict, Any

from train_with_coco import (
    PrimateFaceCOCOLoader,
    PrimateFacePoseDataset,
    filter_coco_data,
    list_available_models,
)

from deeplabcut.pose_estimation_pytorch.config import read_config_as_dict
from deeplabcut.pose_estimation_pytorch.data.transforms import build_transforms
from deeplabcut.pose_estimation_pytorch.models import PoseModel
from deeplabcut.pose_estimation_pytorch.task import Task


# --- Configuration ---
# Path to your test JSON file
TEST_JSON_PATH = "path/to/test.json"

# Keypoint indices for normalization (outer eye corners for COCO WholeBody Face)
# These are 0-indexed.
NORM_KEYPOINT_INDICES = [36, 45]

# Batch size for evaluation
BATCH_SIZE = 32


def get_latest_config(model_dir: Path) -> Path:
    """Finds the path to the latest pose config YAML in the directory.
    
    Args:
        model_dir: Directory containing model config files
        
    Returns:
        Path to the latest config file
        
    Raises:
        FileNotFoundError: If no config files found
    """
    configs = list(model_dir.glob("pose_cfg*.yaml"))
    if not configs:
        raise FileNotFoundError(f"No pose_cfg*.yaml files found in {model_dir}")
    latest_config = max(configs, key=lambda p: p.stat().st_mtime)
    print(f"✅ Found latest model config: {latest_config.name}")
    return latest_config


def get_latest_snapshot(model_dir: Path) -> Path:
    """Finds the path to the latest model snapshot in the directory.
    
    Args:
        model_dir: Directory containing model snapshots
        
    Returns:
        Path to the latest snapshot file
        
    Raises:
        FileNotFoundError: If no snapshots found
    """
    snapshots = list(model_dir.glob("*.pt"))
    if not snapshots:
        raise FileNotFoundError(f"No model snapshots found in {model_dir}")
    latest_snapshot = max(snapshots, key=lambda p: p.stat().st_mtime)
    print(f"✅ Found latest model snapshot: {latest_snapshot.name}")
    return latest_snapshot


def keypoint_nme(
    pred_coords: np.ndarray,
    gt_coords: np.ndarray,
    mask: np.ndarray,
    normalize_factor: np.ndarray,
) -> float:
    """Calculate the Normalized Mean Error.

    Args:
        pred_coords (np.ndarray): Predicted keypoint coordinates [N, K, 2].
        gt_coords (np.ndarray): Ground-truth keypoint coordinates [N, K, 2].
        mask (np.ndarray): Visibility mask for keypoints [N, K].
        normalize_factor (np.ndarray): Normalization factor for each instance [N, 2].

    Returns:
        float: The final NME score.
    """
    # Calculate Euclidean distance for each keypoint
    distances = np.linalg.norm(pred_coords - gt_coords, axis=2)

    # Apply mask to consider only visible keypoints
    masked_distances = distances * mask

    # Normalize the distances
    # The normalization factor is [N, 2], but distances are [N, K].
    # We need to reshape the normalization factor for broadcasting.
    # We only use the first column since the factor is the same for x and y.
    norm_distances = masked_distances / normalize_factor[:, np.newaxis, 0]

    # Calculate the sum of errors and the number of visible keypoints
    total_error = np.sum(norm_distances)
    total_visible_keypoints = np.sum(mask)

    if total_visible_keypoints == 0:
        return 0.0

    return total_error / total_visible_keypoints


def main() -> None:
    """Main function to run NME evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a trained DLC model with NME.")
    parser.add_argument(
        "--model_dir",
        type=Path,
        help="Path to the directory containing trained model snapshots and config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="The device to run evaluation on (e.g., 'cuda', 'cuda:0', 'cpu').",
    )
    args = parser.parse_args()

    list_available_models()  # Shows available models for context

    # --- 1. Load Model Configuration and Snapshot ---
    print(f"⏳ Loading model from {args.model_dir}...")
    try:
        model_config_path = get_latest_config(args.model_dir)
    except FileNotFoundError as e:
        print(f"🛑 Error: {e}")
        return

    model_cfg = read_config_as_dict(str(model_config_path))
    snapshot_path = get_latest_snapshot(args.model_dir)

    device = torch.device(
        args.device if "cuda" in args.device and torch.cuda.is_available() else "cpu"
    )
    model = PoseModel.build(model_cfg["model"]).to(device)
    # The saved object is a dictionary; we need the 'model' state_dict.
    snapshot_data = torch.load(snapshot_path, map_location=device)
    model.load_state_dict(snapshot_data.get("model", snapshot_data))
    model.eval()
    print(f"✅ Model loaded and set to evaluation mode on {device}.")

    # --- 2. Load and Prepare Test Dataset ---
    print("⏳ Loading and preparing test dataset...")
    # Use the PrimateFaceCOCOLoader to get parameters and correctly processed data.
    data_loader = PrimateFaceCOCOLoader(
        model_config_path=model_config_path,
        train_json_path=TEST_JSON_PATH,  # Not used for test, but path needed for init
        val_json_path=TEST_JSON_PATH,    # Not used for test, but path needed for init
        test_json_path=TEST_JSON_PATH,
    )
    params = data_loader.get_dataset_parameters()
    task = Task(model_cfg.get("method", "BU").upper())
    
    # Load the test data using our custom loader; it handles NumPy conversion
    # and adds the default "individual" key.
    test_images, test_annotations = data_loader.load_data(mode="test")
    
    # Filter out data without keypoints to prevent errors
    test_images, test_annotations = filter_coco_data(test_images, test_annotations)

    if not test_annotations:
        print("🛑 Error: No valid annotations with keypoints found in the test set.")
        return

    test_transform = build_transforms(model_cfg["data"]["inference"])
    test_dataset = PrimateFacePoseDataset(
        images=test_images,
        annotations=test_annotations,
        parameters=params,
        transform=test_transform,
        task=task,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )
    print(
        f"✅ Test dataset ready: {len(test_dataset)} instances in "
        f"{len(test_images)} images."
    )

    # --- 3. Run Inference and Collect Results ---
    print("⏳ Running inference on the test set...")
    all_pred_coords = []
    all_gt_coords = []
    all_masks = []
    all_norm_factors = []

    kpt_idx1, kpt_idx2 = NORM_KEYPOINT_INDICES

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch["image"].to(device)
            gt_kpts = batch["annotations"]["keypoints"]  # [B, 1, K, 3]

            # Get model predictions
            outputs = model(images)
            predictions = model.get_predictions(outputs)
            pred_kpts = predictions["bodypart"]["poses"]  # [B, 1, K, 3]

            # Extract ground truth data and apply mask
            gt_coords_batch = gt_kpts[:, 0, :, :2].cpu().numpy()
            mask_batch = gt_kpts[:, 0, :, 2].cpu().numpy().astype(bool)
            
            # The model predicts on the resized/padded image. We need the original
            # keypoints for NME calculation. We assume here that the `PoseDataset`
            # returns GT keypoints in the original image space, which is correct.
            pred_coords_batch = pred_kpts[:, 0, :, :2].cpu().numpy()

            # Calculate the normalization factor (inter-ocular distance)
            # from the ground truth coordinates
            interocular = np.linalg.norm(
                gt_coords_batch[:, kpt_idx1, :] - gt_coords_batch[:, kpt_idx2, :], axis=1, keepdims=True
            )
            # Tile to match the [N, 2] shape expected by the metric function
            norm_factor_batch = np.tile(interocular, (1, 2))
            
            # Append batch results to the master lists
            all_pred_coords.append(pred_coords_batch)
            all_gt_coords.append(gt_coords_batch)
            all_masks.append(mask_batch)
            all_norm_factors.append(norm_factor_batch)

    # Concatenate all results into single NumPy arrays
    final_pred_coords = np.concatenate(all_pred_coords, axis=0)
    final_gt_coords = np.concatenate(all_gt_coords, axis=0)
    final_mask = np.concatenate(all_masks, axis=0)
    final_norm_factor = np.concatenate(all_norm_factors, axis=0)
    
    # --- 4. Calculate and Report NME ---
    print("⏳ Calculating NME...")
    nme_score = keypoint_nme(
        final_pred_coords, final_gt_coords, final_mask, final_norm_factor
    )
    
    print("\n" + "=" * 80)
    print(f"🎉 Evaluation Finished!")
    print(f"Normalized Mean Error (NME): {nme_score:.6f}")
    print(f"Normalization indices used: {NORM_KEYPOINT_INDICES}")
    print("=" * 80)


if __name__ == "__main__":
    main() 