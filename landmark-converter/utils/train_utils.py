"""Training utilities for landmark converter.

This module provides visualization and evaluation functions used during
model training including plotting predictions, training curves, and metrics.
"""

import json
import os
import random
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import matplotlib.patches as patches # For bounding boxes, if needed later
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import argparse
import numpy as np

from utils.data_utils import normalize_keypoints_bbox, denormalize_keypoints_bbox, plot_keypoints_on_ax

def visualize_predictions(model, dataset_subset, device, num_samples=5, output_dir=".", epoch_num="final", save_pdf=False, target_kpt_slice_idx=1, args_config=None, model_expects_graph_data=False):
    """Visualizes model predictions on a few samples from the dataset subset.
       NOTE: This function now always visualizes the raw output of the model.
    """
    # 'dataset' parameter renamed to 'dataset_subset' to reflect it's often a Subset
    if len(dataset_subset) == 0:
        print("Dataset subset is empty, cannot visualize predictions.")
        return

    model.eval()
    actual_num_samples = min(num_samples, len(dataset_subset))
    
    os.makedirs(output_dir, exist_ok=True)
    
    # sample_indices = random.sample(range(len(dataset_subset)), actual_num_samples)
    # The dataset_subset is already a subset with specific indices. We iterate through it or its indices.
    # If dataset_subset is a Subset, dataset_subset.indices gives original indices
    # If dataset_subset is a Dataset, range(len(dataset_subset)) is fine.
    # For simplicity, ModelTrainer passes a Subset, so we can just iterate actual_num_samples if it's already sampled,
    # or sample from it if it's larger than num_samples (which ModelTrainer already handles).
    # Here, we assume dataset_subset contains the exact samples to visualize.

    # Determine the original dataset to access properties like source/target_kpt_field if dataset_subset is a Subset
    # This can be nested if Subset is applied multiple times. We need the root CocoPairedKeypointDataset.
    root_dataset = dataset_subset
    while hasattr(root_dataset, 'dataset'):
        root_dataset = root_dataset.dataset
    
    source_kpt_field_name = getattr(root_dataset, 'source_kpt_field', 'source_kpts')
    target_kpt_field_name = getattr(root_dataset, 'target_kpt_field', 'target_kpts')
    num_target_kpts_in_data = getattr(root_dataset, 'target_num_kpt_field', 'N') # Field name for total number


    fig, axes = plt.subplots(actual_num_samples, 2, figsize=(12, 6 * actual_num_samples), squeeze=False)
    # Updated title_str to reflect GT slicing
    gt_trg_label = f"N-{target_kpt_slice_idx} pts" if target_kpt_slice_idx > 0 else "All pts"
    title_str = f"Model Predictions (Epoch {epoch_num}) - Src ({source_kpt_field_name}, Lime), " \
                f"GT Trg ({target_kpt_field_name}, Cyan, {gt_trg_label}), " \
                f"Pred Trg (Magenta, N-{target_kpt_slice_idx} pts)"
    fig.suptitle(title_str, fontsize=14)

    with torch.no_grad():
        for i in range(actual_num_samples):
            # dataset_subset should yield the items directly if it's a list of items or a small dataset.
            # If it's a Subset, direct indexing dataset_subset[i] is correct.
            
            # Adapt data extraction based on model_expects_graph_data
            if model_expects_graph_data:
                # --- GNN Data Handling ---
                data_sample_pyg = dataset_subset[i].to(device) # PyG Data object moved to device
                
                # Model expects the PyG Data/Batch object directly
                pred_target_kpts_norm_xy_structured_batch = model(data_sample_pyg) # Output: [1, K_pred, 2] (for single sample in eval)
                                                                               # PyG model might output [K_pred, 2] if not batching for single sample
                if pred_target_kpts_norm_xy_structured_batch.ndim == 2: # if model returns [K_pred, 2] for single graph
                    pred_target_kpts_norm_xy_structured_batch = pred_target_kpts_norm_xy_structured_batch.unsqueeze(0) # Add batch dim: [1, K_pred, 2]


                bbox_batch = data_sample_pyg.bbox.unsqueeze(0).to(device).float() # (1, 4)
                source_kpts_orig = data_sample_pyg.source_kpts_with_vis.cpu() # (K_src, 3) - for plotting
                target_kpts_orig = data_sample_pyg.target_kpts_with_vis.cpu() # (K_trg_total, 3) - for plotting
                img_path = data_sample_pyg.image_path
                
                # If image tensor is stored in data_sample_pyg.image (e.g. [C, H, W])
                # Convert to PIL for plotting if necessary
                if isinstance(data_sample_pyg.image, torch.Tensor):
                    pil_image_for_plot = transforms.ToPILImage()(data_sample_pyg.image.cpu())
                else: # Assuming image_path is primary way if not tensor
                    try:
                        pil_image_for_plot = PILImage.open(img_path).convert("RGB")
                    except FileNotFoundError:
                        pil_image_for_plot = PILImage.new('RGB', (256, 256), color = 'grey')
                
                K_pred = args_config.num_target_kpt - target_kpt_slice_idx # GNN model internally knows its target kpt num

            else:
                # --- Standard Data Handling ---
                image_tensor, source_kpts_orig_cpu, target_kpts_orig_cpu, bbox_orig_cpu, img_path = dataset_subset[i]
                
                source_kpts_batch = source_kpts_orig_cpu.unsqueeze(0).to(device).float() # (1, K_src, 3)
                bbox_batch = bbox_orig_cpu.unsqueeze(0).to(device).float() # (1, 4)

                norm_source_kpts_xy_batch = normalize_keypoints_bbox(source_kpts_batch[..., :2], bbox_batch) # (1, K_src, 2)
                source_kpts_flat_batch = norm_source_kpts_xy_batch.view(norm_source_kpts_xy_batch.size(0), -1) # (1, K_src*2)
                
                pred_target_kpts_norm_xy_flat_batch = model(source_kpts_flat_batch)
                
                if args_config is None:
                    print("Warning: args_config not provided to visualize_predictions. Inferring K_pred from tensor shape.")
                    num_predicted_features = pred_target_kpts_norm_xy_flat_batch.shape[1]
                    if num_predicted_features % 2 != 0:
                        print(f"Error: Number of predicted features ({num_predicted_features}) is not even.")
                        return 
                    K_pred = num_predicted_features // 2
                else:
                    K_pred = args_config.num_target_kpt - target_kpt_slice_idx
                
                pred_target_kpts_norm_xy_structured_batch = pred_target_kpts_norm_xy_flat_batch.view(1, K_pred, 2) # [1, K_pred, 2]

                # --- REMOVED Enforce Overlap Logic for Visualization ---
                # Visualization should show the model's raw learned output.
                # --- End REMOVED Enforce Overlap Logic ---

                source_kpts_orig = source_kpts_orig_cpu # for plotting (already on CPU)
                target_kpts_orig = target_kpts_orig_cpu # for plotting (already on CPU)
                try:
                    pil_image_for_plot = PILImage.open(img_path).convert("RGB")
                except FileNotFoundError:
                    pil_image_for_plot = PILImage.new('RGB', (256, 256), color = 'grey')


            # Denormalize predictions for visualization
            # pred_target_kpts_norm_xy_structured_batch is (1, K_pred, 2) on device
            # bbox_batch is (1, 4) on device
            denormalized_kpts_batch = denormalize_keypoints_bbox(
                pred_target_kpts_norm_xy_structured_batch.cpu(), # Move to CPU for denormalization
                bbox_batch.cpu()                      # Move to CPU for denormalization
            )
            pred_target_kpts_pixel_xy = denormalized_kpts_batch.squeeze(0) # Shape (K_pred, 2) on CPU

            pred_target_kpts_vis = torch.cat(
                [pred_target_kpts_pixel_xy, torch.ones(pred_target_kpts_pixel_xy.shape[0], 1) * 2], dim=1
            )

            # Slice the original target keypoints for visualization
            target_kpts_orig_sliced = target_kpts_orig[target_kpt_slice_idx:]

            base_img_name = os.path.basename(img_path)
            print(f"  Plotting prediction for sample {i+1}/{actual_num_samples}: {base_img_name}")

            ax_left = axes[i][0]
            plot_keypoints_on_ax(ax_left, pil_image_for_plot, source_kpts_orig, f"Src ({source_kpt_field_name})", color='lime', show_image=True)
            # Use sliced GT for plotting on the left
            plot_keypoints_on_ax(ax_left, pil_image_for_plot, target_kpts_orig_sliced, f"GT Trg ({target_kpt_field_name})", color='cyan', scatter_size=15, show_image=False)
            ax_left.set_title(f"{base_img_name}\nSrc (Lime), GT Trg (Cyan, {gt_trg_label})", fontsize=8)

            ax_right = axes[i][1]
            # Plot sliced GT target keypoints for reference on the right
            plot_keypoints_on_ax(ax_right, pil_image_for_plot, target_kpts_orig_sliced, f"GT Trg ({target_kpt_field_name})", color='cyan', scatter_size=15, show_image=True)
            # Plot predicted target keypoints (N-slice_idx of them)
            plot_keypoints_on_ax(ax_right, pil_image_for_plot, pred_target_kpts_vis, f"Predicted Target", color='magenta', scatter_size=10, show_image=False)
            ax_right.set_title(f"{base_img_name}\nGT Trg (Cyan, {gt_trg_label}), Pred Trg (Magenta, N-{target_kpt_slice_idx})", fontsize=8)

            # Scatter plot for this sample: Predicted vs GT coordinates
            # GT keypoints need to be sliced to match the N-slice_idx predicted points
            # This part already correctly uses target_kpts_orig[target_kpt_slice_idx:, :2]
            gt_kpts_xy_for_scatter = target_kpts_orig[target_kpt_slice_idx:, :2].cpu().numpy()
            pred_kpts_xy_for_scatter = pred_target_kpts_pixel_xy.cpu().numpy()
            
            # Ensure dimensions match for scatter plot if slicing results in different numbers
            if gt_kpts_xy_for_scatter.shape[0] != pred_kpts_xy_for_scatter.shape[0]:
                print(f"Warning: Mismatch in keypoint numbers for scatter plot. GT: {gt_kpts_xy_for_scatter.shape[0]}, Pred: {pred_kpts_xy_for_scatter.shape[0]}. Skipping scatter for this sample.")
            else:
                fig_scatter, (ax_scatter_x, ax_scatter_y) = plt.subplots(1, 2, figsize=(12, 5))
                fig_scatter.suptitle(f"Predicted vs. GT Coordinates (Sample: {base_img_name}, Epoch {epoch_num})", fontsize=12)

                ax_scatter_x.scatter(gt_kpts_xy_for_scatter[:, 0], pred_kpts_xy_for_scatter[:, 0], alpha=0.6, edgecolors='k', s=50)
                min_val_x = min(gt_kpts_xy_for_scatter[:, 0].min(), pred_kpts_xy_for_scatter[:, 0].min()) - 5
                max_val_x = max(gt_kpts_xy_for_scatter[:, 0].max(), pred_kpts_xy_for_scatter[:, 0].max()) + 5
                ax_scatter_x.plot([min_val_x, max_val_x], [min_val_x, max_val_x], 'r--', lw=2, label='Ideal (y=x)')
                ax_scatter_x.set_xlabel("Ground Truth X-coordinate (pixels)"); ax_scatter_x.set_ylabel("Predicted X-coordinate (pixels)")
                ax_scatter_x.set_title("X-coordinates"); ax_scatter_x.legend(); ax_scatter_x.grid(True); ax_scatter_x.axis('equal')

                ax_scatter_y.scatter(gt_kpts_xy_for_scatter[:, 1], pred_kpts_xy_for_scatter[:, 1], alpha=0.6, edgecolors='k', s=50)
                min_val_y = min(gt_kpts_xy_for_scatter[:, 1].min(), pred_kpts_xy_for_scatter[:, 1].min()) - 5
                max_val_y = max(gt_kpts_xy_for_scatter[:, 1].max(), pred_kpts_xy_for_scatter[:, 1].max()) + 5
                ax_scatter_y.plot([min_val_y, max_val_y], [min_val_y, max_val_y], 'r--', lw=2, label='Ideal (y=x)')
                ax_scatter_y.set_xlabel("Ground Truth Y-coordinate (pixels)"); ax_scatter_y.set_ylabel("Predicted Y-coordinate (pixels)")
                ax_scatter_y.set_title("Y-coordinates"); ax_scatter_y.legend(); ax_scatter_y.grid(True); ax_scatter_y.axis('equal')

                fig_scatter.tight_layout(rect=[0, 0.03, 1, 0.95])
                scatter_filename_png = os.path.join(output_dir, f"predictions_epoch_{epoch_num}_sample_{i}_coords_scatter.png")
                try:
                    plt.savefig(scatter_filename_png, dpi=100)
                    if save_pdf:
                        plt.savefig(os.path.join(output_dir, f"predictions_epoch_{epoch_num}_sample_{i}_coords_scatter.pdf"))
                    plt.close(fig_scatter)
                except Exception as e:
                    print(f"Error saving coordinate scatter plot for sample {i}: {e}"); plt.close(fig_scatter)

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    vis_filename_png = os.path.join(output_dir, f"predictions_epoch_{epoch_num}.png")
    try:
        print(f"Saving prediction visualization to {vis_filename_png}")
        plt.savefig(vis_filename_png, dpi=150)
        if save_pdf:
            plt.savefig(os.path.join(output_dir, f"predictions_epoch_{epoch_num}.pdf"))
        plt.close(fig)
        print("Visualization saved.")
    except Exception as e:
        print(f"Error saving prediction visualization: {e}"); plt.close(fig)

# --- Plotting Utilities (train_epoch and evaluate removed) ---

def plot_training_curve(epochs, train_losses, val_losses, output_dir, base_filename="training_curve", save_pdf=False):
    """Plots and saves the training and validation loss curves."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training and Validation Loss Curve'); plt.legend(); plt.grid(True)
    
    curve_path_png = os.path.join(output_dir, f"{base_filename}.png")
    plt.savefig(curve_path_png); print(f"Training curve saved to {curve_path_png}")
    if save_pdf:
        curve_path_pdf = os.path.join(output_dir, f"{base_filename}.pdf")
        plt.savefig(curve_path_pdf); print(f"Training curve also saved to {curve_path_pdf}")
    plt.close()

def plot_mpjpe_per_keypoint_bar(mpjpe_values, output_dir, base_filename="mpjpe_per_keypoint_bar", save_pdf=False, num_target_kpts_bar_label="N-1"):
    """Plots and saves a bar chart of MPJPE per keypoint."""
    os.makedirs(output_dir, exist_ok=True)
    if mpjpe_values is None or len(mpjpe_values) == 0:
        print("No MPJPE per keypoint values to plot.")
        return

    num_keypoints = len(mpjpe_values)
    keypoint_indices = np.arange(num_keypoints)

    plt.figure(figsize=(max(10, num_keypoints * 0.5), 6)) # Dynamic width
    plt.bar(keypoint_indices, mpjpe_values, color='skyblue')
    plt.xlabel("Keypoint Index")
    plt.ylabel("Mean Pixel Joint Error (MPJPE) in pixels")
    plt.title(f"Average MPJPE per Keypoint (for {num_target_kpts_bar_label} predicted keypoints)")
    plt.xticks(keypoint_indices)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()

    bar_chart_path_png = os.path.join(output_dir, f"{base_filename}.png")
    plt.savefig(bar_chart_path_png); print(f"MPJPE per keypoint bar chart saved to {bar_chart_path_png}")
    if save_pdf:
        bar_chart_path_pdf = os.path.join(output_dir, f"{base_filename}.pdf")
        plt.savefig(bar_chart_path_pdf); print(f"MPJPE bar chart also saved to {bar_chart_path_pdf}")
    plt.close()

def plot_mpjpe_heatmap_on_sample(image_path_sample, target_keypoints_original_scale, mpjpe_per_keypoint, output_dir, base_filename="keypoint_error_heatmap", save_pdf=False):
    """Plots keypoints on a sample image, colored by their MPJPE values."""
    os.makedirs(output_dir, exist_ok=True)
    
    if target_keypoints_original_scale is None or mpjpe_per_keypoint is None:
        print("Missing data for MPJPE heatmap plot.")
        return
    if len(target_keypoints_original_scale) != len(mpjpe_per_keypoint):
        print(f"Mismatch between number of target keypoints ({len(target_keypoints_original_scale)}) and MPJPE values ({len(mpjpe_per_keypoint)}). Cannot plot heatmap.")
        return

    try:
        img = PILImage.open(image_path_sample).convert("RGB")
    except FileNotFoundError:
        print(f"Image file not found for heatmap: {image_path_sample}")
        # Create a placeholder image if not found
        img = PILImage.new('RGB', (256, 256), color = 'lightgrey')
    except Exception as e:
        print(f"Error loading image {image_path_sample} for heatmap: {e}")
        return

    print(f"Heatmap: Loaded image {image_path_sample} - Mode: {img.mode}, Size: {img.size}") # Diagnostic print
    fig, ax = plt.subplots(figsize=(10, 10)) # Create figure and an axes
    ax.imshow(img) # Display the image on the axes
    
    # Normalize MPJPE values for colormap (e.g., 0 to 1)
    norm_mpjpe = (mpjpe_per_keypoint - np.min(mpjpe_per_keypoint)) / (np.max(mpjpe_per_keypoint) - np.min(mpjpe_per_keypoint) + 1e-6) # Add epsilon for stability
    cmap = plt.get_cmap('viridis') # Or 'coolwarm', 'plasma', etc.

    # target_keypoints_original_scale should be (N_kpts-slice, 2) or similar (x,y pairs)
    # mpjpe_per_keypoint should be (N_kpts-slice,)
    for i in range(len(target_keypoints_original_scale)):
        x, y = target_keypoints_original_scale[i, 0], target_keypoints_original_scale[i, 1]
        ax.scatter(x, y, color=cmap(norm_mpjpe[i]), s=100, edgecolors='black', linewidth=1)
        ax.text(x + 5, y + 5, f'{mpjpe_per_keypoint[i]:.1f}', color='white', backgroundcolor='black', fontsize=8)

    ax.set_title(f"Keypoint Errors (MPJPE) on Sample: {os.path.basename(image_path_sample)}")
    ax.axis('off')
    
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=np.min(mpjpe_per_keypoint), vmax=np.max(mpjpe_per_keypoint)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('MPJPE (pixels)')

    heatmap_path_png = os.path.join(output_dir, f"{base_filename}.png")
    fig.savefig(heatmap_path_png); print(f"Keypoint error heatmap saved to {heatmap_path_png}")
    if save_pdf:
        heatmap_path_pdf = os.path.join(output_dir, f"{base_filename}.pdf")
        fig.savefig(heatmap_path_pdf); print(f"Keypoint error heatmap also saved to {heatmap_path_pdf}")
    plt.close(fig)

def save_test_evaluation_results(test_metrics, model_config, training_args, output_dir, base_filename="test_evaluation_results"):
    """Saves test set evaluation metrics, model config, and training args to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    results = {
        "test_metrics": test_metrics,
        "model_configuration": model_config,
        "training_arguments": vars(training_args) if isinstance(training_args, argparse.Namespace) else training_args
    }
    results_path = os.path.join(output_dir, f"{base_filename}.json")
    try:
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4, default=lambda o: str(o) if isinstance(o, (torch.device)) or not isinstance(o, (dict, list, str, int, float, bool, type(None))) else o)
        print(f"Test evaluation results saved to {results_path}")
    except Exception as e:
        print(f"Error saving test evaluation results: {e}")

# Example of how to get root dataset for properties, if needed inside these utils
# def get_root_dataset(dataset_obj):
#     root = dataset_obj
#     while hasattr(root, 'dataset'): # Recursively go to the base dataset if it's a Subset
#         root = root.dataset
#     return root

