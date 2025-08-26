"""Data loading and processing utilities for landmark converter.

This module provides dataset classes and utilities for loading COCO-format
annotations, normalizing keypoints, and preparing data for training.
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
from torch.utils.data import DataLoader, random_split, Subset
# Optional: Import torch_geometric for GNN support
try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader as PyGDataLoader # Renamed to avoid conflict
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    Data = None
    PyGDataLoader = None

class CocoPairedKeypointDataset(Dataset):
    """
    PyTorch Dataset for loading images and paired keypoints (e.g., 68 and 49) 
    from a merged COCO JSON file.
    Assumes annotations in the COCO file have a primary 'keypoints' field 
    and a secondary field like 'keypoints_49'.
    """
    def __init__(self, coco_json_path, image_base_dir=None, transform=None,
                 source_kpt_field='keypoints', target_kpt_field='keypoints_49',
                 source_num_kpt_field='num_keypoints', target_num_kpt_field='num_keypoints_49'):
        """
        Args:
            coco_json_path (str): Path to the merged COCO JSON file.
            image_base_dir (str, optional): Base directory for image paths if they are relative in JSON.
                                           If None, assumes image paths in JSON are absolute.
            transform (callable, optional): Optional transform to be applied on a sample.
            source_kpt_field (str): Key for source keypoints in annotation dict (e.g., 'keypoints').
            target_kpt_field (str): Key for target keypoints in annotation dict (e.g., 'keypoints_49').
            source_num_kpt_field (str): Key for number of source keypoints.
            target_num_kpt_field (str): Key for number of target keypoints.
        """
        self.coco_json_path = coco_json_path
        self.image_base_dir = image_base_dir
        self.transform = transform
        self.source_kpt_field = source_kpt_field
        self.target_kpt_field = target_kpt_field
        self.source_num_kpt_field = source_num_kpt_field
        self.target_num_kpt_field = target_num_kpt_field

        self.samples = []
        self._load_data()

    def _load_data(self):
        print(f"Loading COCO data from: {self.coco_json_path}")
        with open(self.coco_json_path, 'r') as f:
            coco_data = json.load(f)

        images_dict = {img['id']: img for img in coco_data.get('images', [])}
        
        num_anns_total = 0
        num_anns_valid_pair = 0
        num_anns_skipped_no_bbox = 0
        num_anns_skipped_missing_kpts = 0
        num_anns_skipped_zero_kpts = 0
        num_anns_skipped_invalid_img_id = 0

        for ann in coco_data.get('annotations', []):
            num_anns_total += 1
            img_id = ann.get('image_id')
            
            source_kpts = ann.get(self.source_kpt_field)
            num_source_kpts = ann.get(self.source_num_kpt_field, 0)
            target_kpts = ann.get(self.target_kpt_field)
            num_target_kpts = ann.get(self.target_num_kpt_field, 0)
            bbox = ann.get('bbox') # [x, y, width, height]

            valid_bbox = bbox and isinstance(bbox, list) and len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0
            valid_img_id = img_id in images_dict
            valid_source_kpts_data = source_kpts and isinstance(source_kpts, list)
            valid_target_kpts_data = target_kpts and isinstance(target_kpts, list)
            positive_source_kpts_count = num_source_kpts > 0
            positive_target_kpts_count = num_target_kpts > 0

            # Ensure both keypoint sets are present, have positive num_keypoints,
            # image_id is valid, and a valid bbox is present.
            if (valid_source_kpts_data and positive_source_kpts_count and
                valid_target_kpts_data and positive_target_kpts_count and 
                valid_img_id and
                valid_bbox):
                
                img_info = images_dict[img_id]
                img_path = img_info.get('file_name')
                if self.image_base_dir and not os.path.isabs(img_path):
                    img_path = os.path.join(self.image_base_dir, img_path)
                
                # Reshape keypoints: list of (x,y,v) to N_kpts x 3 tensor
                # COCO format: [x1, y1, v1, x2, y2, v2, ...]
                source_kpts_arr = np.array(source_kpts).reshape(-1, 3)
                target_kpts_arr = np.array(target_kpts).reshape(-1, 3)

                self.samples.append({
                    'image_path': img_path,
                    'source_keypoints': torch.tensor(source_kpts_arr, dtype=torch.float32),
                    'target_keypoints': torch.tensor(target_kpts_arr, dtype=torch.float32),
                    'bbox': bbox # Store raw bbox list; will be converted to tensor in __getitem__
                })
                num_anns_valid_pair +=1
            else:
                ann_id_for_log = ann.get('id', 'N/A')
                if not valid_bbox:
                    num_anns_skipped_no_bbox += 1
                    # print(f"Skipping ann_id {ann_id_for_log} for img_id {img_id}: Invalid or missing bbox: {bbox}")
                elif not valid_img_id:
                    num_anns_skipped_invalid_img_id += 1
                    # print(f"Skipping ann_id {ann_id_for_log}: Invalid img_id {img_id}")
                elif not valid_source_kpts_data or not valid_target_kpts_data:
                    num_anns_skipped_missing_kpts +=1
                    # print(f"Skipping ann_id {ann_id_for_log} for img_id {img_id}: Missing source or target kpt data. Src present: {bool(source_kpts)}, Tgt present: {bool(target_kpts)}")
                elif not positive_source_kpts_count or not positive_target_kpts_count:
                    num_anns_skipped_zero_kpts +=1
                    # print(f"Skipping ann_id {ann_id_for_log} for img_id {img_id}: Zero kpts count. Num src: {num_source_kpts}, Num tgt: {num_target_kpts}")
                # Generic fallback if none of the above specific counters were incremented, though one should have been.
                # print(f"Skipping ann_id {ann.get('id')} for img_id {img_id}: missing required fields, num_keypoints is 0, or invalid bbox.")


        print(f"Processed {num_anns_total} annotations. Found {len(self.samples)} valid paired keypoint samples.")
        if num_anns_skipped_no_bbox > 0:
            print(f"Skipped {num_anns_skipped_no_bbox} annotations due to missing or invalid bounding box.")
        if num_anns_skipped_invalid_img_id > 0:
            print(f"Skipped {num_anns_skipped_invalid_img_id} annotations due to invalid image_id.")
        if num_anns_skipped_missing_kpts > 0:
            print(f"Skipped {num_anns_skipped_missing_kpts} annotations due to missing source or target keypoint lists (fields '{self.source_kpt_field}' or '{self.target_kpt_field}' might be absent or not lists).")
        if num_anns_skipped_zero_kpts > 0:
            print(f"Skipped {num_anns_skipped_zero_kpts} annotations due to zero count for source or target keypoints (fields '{self.source_num_kpt_field}' or '{self.target_num_kpt_field}' might be 0).")
        if not self.samples:
            print("Warning: No valid samples found. Check COCO file and field names being used for keypoints and their counts.")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx >= len(self.samples):
            raise IndexError("Index out of bounds")
            
        sample = self.samples[idx]
        image_path = sample['image_path']
        
        try:
            image = PILImage.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"ERROR: Image not found at {image_path} for sample {idx}. Returning dummy data.")
            # Return dummy data or handle appropriately
            dummy_image = torch.zeros((3, 256, 256), dtype=torch.float32) # Example size
            dummy_kpts_shape = sample['source_keypoints'].shape # Preserve number of keypoints
            dummy_kpts = torch.zeros_like(sample['source_keypoints'])
            # Ensure a dummy bbox tensor of correct shape is returned if original sample had one
            dummy_bbox = torch.zeros(4, dtype=torch.float32) if sample.get('bbox') else None
            return dummy_image, dummy_kpts, dummy_kpts, dummy_bbox, image_path

        source_kpts = sample['source_keypoints'].clone()
        target_kpts = sample['target_keypoints'].clone()
        bbox_list = sample['bbox'] # This should always be present due to filtering in _load_data
        
        # Convert bbox list to tensor. Assumes bbox is always present and valid.
        bbox_tensor = torch.tensor(bbox_list, dtype=torch.float32)

        # Keypoints are returned in original pixel coordinates. Normalization will be done in the training script.
        
        if self.transform:
            # Note: Standard torchvision transforms for images might not handle keypoints.
            # Custom transform would be needed if keypoints need to be augmented along with image.
            image = self.transform(image)

        return image, source_kpts, target_kpts, bbox_tensor, image_path


def plot_keypoints_on_ax(ax, image_or_path, keypoints_tensor, title, color='lime', scatter_size=10, show_image=True):
    """
    Plots keypoints on a given image axis.
    Args:
        ax (matplotlib.axes.Axes): The axes to plot on.
        image_or_path (str or PIL.Image or torch.Tensor): Path to image, PIL image, or image tensor.
        keypoints_tensor (torch.Tensor): Nx3 tensor of (x, y, visibility) keypoints.
        title (str): Title for the subplot.
        color (str): Color for the keypoints.
        scatter_size (int): Size of the keypoint markers.
        show_image (bool): If True, displays the image on the axes. Otherwise, only plots keypoints.
    """
    img = None
    if show_image:
        if isinstance(image_or_path, str):
            try:
                img = PILImage.open(image_or_path).convert("RGB")
            except FileNotFoundError:
                ax.text(0.5, 0.5, f"Image not found:\n{os.path.basename(image_or_path)}", 
                        ha='center', va='center', fontsize=8, color='red')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(title, fontsize=8)
                return
        elif isinstance(image_or_path, PILImage.Image):
            img = image_or_path
        elif isinstance(image_or_path, torch.Tensor):
            img = transforms.ToPILImage()(image_or_path.cpu())
        else:
            ax.text(0.5, 0.5, "Invalid image format for display", ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=8)
            return
        
        ax.imshow(img)
    
    if keypoints_tensor is not None and len(keypoints_tensor) > 0:
        kpts = keypoints_tensor.numpy()
        visible_kpts = kpts[kpts[:, 2] > 0] # COCO: v=0 not labeled, v=1 labeled but not visible, v=2 labeled and visible
                                         # Here we plot if v > 0 (labeled)
        if len(visible_kpts) > 0:
            ax.scatter(visible_kpts[:, 0], visible_kpts[:, 1], s=scatter_size, marker='.', c=color, edgecolors='black', linewidths=0.5)
        title += f" ({len(visible_kpts)} vis kpts)"
    else:
        title += " (no kpts)"

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8)


def visualize_dataset_samples(dataset, num_samples=3, output_basename='keypoint_conversion_samples'):
    """
    Visualizes a few samples from the CocoPairedKeypointDataset.
    Creates a grid of num_samples rows and 2 columns.
    Left: Image + Source Keypoints, Right: Image + Target Keypoints.
    """
    if len(dataset) == 0:
        print("Dataset is empty. Cannot visualize samples.")
        return

    actual_num_samples = min(num_samples, len(dataset))
    if actual_num_samples == 0:
        print("No samples to visualize.")
        return
        
    print(f"\nVisualizing {actual_num_samples} random samples from the dataset...")
    
    # Ensure indices are within the valid range of the dataset
    if len(dataset) < actual_num_samples:
        print(f"Warning: Requested {actual_num_samples} samples, but dataset only has {len(dataset)}. Visualizing all.")
        sample_indices = list(range(len(dataset)))
    else:
        sample_indices = random.sample(range(len(dataset)), actual_num_samples)

    fig, axes = plt.subplots(actual_num_samples, 2, figsize=(10, 5 * actual_num_samples), squeeze=False)

    for i, data_idx in enumerate(sample_indices):
        try:
            # image_tensor is potentially transformed (e.g., resized, ToTensor)
            image_tensor, source_kpts, target_kpts, _bbox, img_path = dataset[data_idx]
        except Exception as e:
            print(f"Error loading sample at index {data_idx}: {e}")
            # Plot placeholders if loading fails
            if actual_num_samples > 0: # Ensure axes exist
                 ax_source = axes[i][0]
                 ax_target = axes[i][1]
                 ax_source.text(0.5, 0.5, f"Error loading sample {data_idx}", ha='center', va='center', color='red')
                 ax_target.text(0.5, 0.5, f"Error loading sample {data_idx}", ha='center', va='center', color='red')
                 ax_source.set_title(f"Sample {i+1} Source Error", fontsize=8)
                 ax_target.set_title(f"Sample {i+1} Target Error", fontsize=8)
                 ax_source.set_xticks([]); ax_source.set_yticks([])
                 ax_target.set_xticks([]); ax_target.set_yticks([])
            continue

        base_img_name = os.path.basename(img_path)
        print(f"  Plotting sample {i+1}/{actual_num_samples}: {base_img_name}")

        # Load original image for visualization to ensure keypoints align correctly
        try:
            original_pil_image = PILImage.open(img_path).convert("RGB")
        except FileNotFoundError:
            # Handle error if original image not found for some reason, though dataset should have caught this.
            # Fallback or error indication on plot can be done in plot_keypoints_on_ax
            original_pil_image = img_path # Pass path to let plot_keypoints_on_ax handle display of error

        # Left column: Image + Source Keypoints
        ax_source = axes[i][0]
        plot_keypoints_on_ax(ax_source, original_pil_image, source_kpts, f"Source ({dataset.source_kpt_field})", color='lime', show_image=True)
        
        # Right column: Image + Target Keypoints (excluding the first keypoint)
        ax_target = axes[i][1]
        if target_kpts is not None and target_kpts.shape[0] > 0:
            print(f"  Visualizing target keypoints for {base_img_name}, excluding the first keypoint (index 0).")
            effective_target_kpts = target_kpts[1:, :] 
            plot_title = f"Target ({dataset.target_kpt_field} - No Idx 0)"
        else:
            effective_target_kpts = target_kpts # Pass as is if None or empty
            plot_title = f"Target ({dataset.target_kpt_field})"
        
        plot_keypoints_on_ax(ax_target, original_pil_image, effective_target_kpts, plot_title, color='cyan', show_image=True)

    fig.tight_layout(pad=2.0)
    
    output_png = output_basename + ".png"
    output_pdf = output_basename + ".pdf"

    try:
        print(f"Saving visualization to {output_png} and {output_pdf}")
        plt.savefig(output_png, dpi=300)
        plt.savefig(output_pdf)
        print("Visualization saved.")
    except Exception as e:
        print(f"Error saving visualization: {e}")
    finally:
        plt.close(fig)

# --- Keypoint Normalization / Denormalization ---
def normalize_keypoints_bbox(keypoints_tensor, bbox_tensor):
    """
    Normalizes keypoints (N, K, 3 or N, K, 2) using bounding box [x, y, w, h].
    Only x, y coordinates are normalized. Visibility (if present) is preserved.
    Args:
        keypoints_tensor (torch.Tensor): Batch of keypoints, shape (B, NumKeypoints, 2 or 3).
                                         Assumes x, y are in pixel coordinates.
        bbox_tensor (torch.Tensor): Batch of bounding boxes, shape (B, 4).
                                     Each bbox is [x_min, y_min, width, height].
    Returns:
        torch.Tensor: Normalized keypoints (x', y') in range [0, 1] relative to bbox.
                      Shape (B, NumKeypoints, 2 or 3).
    """
    if keypoints_tensor is None or bbox_tensor is None:
        raise ValueError("Keypoints or bbox tensor cannot be None for normalization.")
    if keypoints_tensor.ndim != 3 or bbox_tensor.ndim != 2 or bbox_tensor.shape[1] != 4:
        raise ValueError(f"Unexpected shapes: kpts {keypoints_tensor.shape}, bbox {bbox_tensor.shape}")

    kpts_normalized = keypoints_tensor.clone()
    bbox_x, bbox_y, bbox_w, bbox_h = bbox_tensor[:, 0:1], bbox_tensor[:, 1:2], bbox_tensor[:, 2:3], bbox_tensor[:, 3:4]

    epsilon = 1e-6
    valid_w = torch.clamp(bbox_w, min=epsilon)
    valid_h = torch.clamp(bbox_h, min=epsilon)

    kpts_normalized[..., 0] = (keypoints_tensor[..., 0] - bbox_x) / valid_w
    kpts_normalized[..., 1] = (keypoints_tensor[..., 1] - bbox_y) / valid_h
    
    return kpts_normalized

def denormalize_keypoints_bbox(normalized_keypoints_tensor, bbox_tensor):
    """
    Denormalizes keypoints (N, K, 2 or N, K, 3) from bbox-relative [0,1] to original pixel coords.
    Only x, y coordinates are denormalized. Visibility (if present) is preserved.
    Args:
        normalized_keypoints_tensor (torch.Tensor): Batch of normalized keypoints, shape (B, NumKeypoints, 2 or 3).
        bbox_tensor (torch.Tensor): Batch of bounding boxes, shape (B, 4).
                                     Each bbox is [x_min, y_min, width, height].
    Returns:
        torch.Tensor: Keypoints in original pixel coordinates. Shape (B, NumKeypoints, 2 or 3).
    """
    if normalized_keypoints_tensor is None or bbox_tensor is None:
        raise ValueError("Keypoints or bbox tensor cannot be None for denormalization.")
    if normalized_keypoints_tensor.ndim != 3 or bbox_tensor.ndim != 2 or bbox_tensor.shape[1] != 4:
        raise ValueError(f"Unexpected shapes: kpts {normalized_keypoints_tensor.shape}, bbox {bbox_tensor.shape}")

    kpts_denormalized = normalized_keypoints_tensor.clone()
    bbox_x, bbox_y, bbox_w, bbox_h = bbox_tensor[:, 0:1], bbox_tensor[:, 1:2], bbox_tensor[:, 2:3], bbox_tensor[:, 3:4]

    kpts_denormalized[..., 0] = normalized_keypoints_tensor[..., 0] * bbox_w + bbox_x
    kpts_denormalized[..., 1] = normalized_keypoints_tensor[..., 1] * bbox_h + bbox_y
    
    return kpts_denormalized

def _create_fully_connected_edge_index(num_nodes):
    """Helper function to create edge_index for a fully connected graph."""
    if num_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long)
    
    # Create edges for all pairs of nodes (i, j) where i != j
    adj = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    adj.fill_diagonal_(0) # No self-loops
    edge_index = adj.nonzero(as_tuple=False).t().contiguous()
    return edge_index

class CocoPairedKeypointGraphDataset(CocoPairedKeypointDataset):
    """
    PyTorch Dataset for loading images and paired keypoints, returning
    PyTorch Geometric Data objects. Inherits from CocoPairedKeypointDataset
    to reuse COCO loading logic.
    
    NOTE: Requires torch_geometric to be installed.
    """
    def __init__(self, coco_json_path, image_base_dir=None, transform=None,
                 source_kpt_field='keypoints', target_kpt_field='keypoints_49',
                 source_num_kpt_field='num_keypoints', target_num_kpt_field='num_keypoints_49',
                 target_kpt_slice_idx=0): # Added target_kpt_slice_idx for consistency
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError(
                "torch_geometric is required for Graph datasets. "
                "Install it with: pip install torch-geometric"
            )
        super().__init__(coco_json_path, image_base_dir, transform,
                         source_kpt_field, target_kpt_field,
                         source_num_kpt_field, target_num_kpt_field)
        self.target_kpt_slice_idx = target_kpt_slice_idx # Store for slicing target keypoints

    def __getitem__(self, idx):
        # Get the raw data using the parent class's __getitem__
        # This returns: image, source_kpts (N,3), target_kpts (M,3), bbox_tensor (4,), img_path
        img_pil, source_kpts_with_vis, target_kpts_with_vis, bbox_tensor, img_path = super().__getitem__(idx)

        # For GNN, we typically use only x, y coordinates as node features.
        # source_kpts: [num_source_kpts, 2]
        # node_features_source = source_kpts_with_vis[:, :2] 

        # Normalize source features (x,y)
        source_kpts_xy_raw = source_kpts_with_vis[:, :2]
        # normalize_keypoints_bbox expects batched input: (B, K, 2 or 3) and (B, 4)
        norm_node_features_source_xy = normalize_keypoints_bbox(
            source_kpts_xy_raw.unsqueeze(0), # Add batch dim: [1, K_src, 2]
            bbox_tensor.unsqueeze(0)         # Add batch dim: [1, 4]
        ).squeeze(0) # Remove batch dim: [K_src, 2]
        
        # Create edge_index for a fully connected graph of source keypoints
        num_source_nodes = norm_node_features_source_xy.shape[0]
        edge_index_source = _create_fully_connected_edge_index(num_source_nodes)

        # Prepare and normalize target keypoints (y for the graph)
        # Apply slicing similar to other models: target_kpts are M-slice_idx
        # Flatten target keypoints: (M-slice_idx)*2
        effective_target_kpts_xy_raw = target_kpts_with_vis[self.target_kpt_slice_idx:, :2]
        norm_effective_target_kpts_xy = normalize_keypoints_bbox(
            effective_target_kpts_xy_raw.unsqueeze(0), # Add batch dim: [1, K_trg_pred, 2]
            bbox_tensor.unsqueeze(0)                  # Add batch dim: [1, 4]
        ).squeeze(0) # Remove batch dim: [K_trg_pred, 2]
        target_kpts_flat_normalized = norm_effective_target_kpts_xy.reshape(-1) # Shape: [K_trg_pred * 2]
        
        # If an image transform is provided (e.g., ToTensor, Resize), apply it
        # This was already handled by super().__getitem__ if self.transform was set.
        # Let's assume img_pil is the PIL image before any tensor conversion if transform is None
        # or a transformed tensor if self.transform was applied in parent.
        # For Data object, we should ensure image is a tensor.
        if isinstance(img_pil, PILImage.Image): # If parent didn't apply ToTensor
            img_tensor = transforms.ToTensor()(img_pil) # Basic ToTensor
        else: # Parent already returned a tensor (e.g. if self.transform included ToTensor)
            img_tensor = img_pil


        # Create PyG Data object
        # We store the transformed image tensor.
        # We store the raw bbox_tensor for potential normalization/denormalization later.
        # img_path can be useful for debugging.
        data = Data(
            x=norm_node_features_source_xy,      # Node features: [num_source_kpts, 2] - NORMALIZED
            edge_index=edge_index_source,        # Edge connectivity: [2, num_edges]
            y=target_kpts_flat_normalized,       # Target output: [(M-target_kpt_slice_idx)*2] - NORMALIZED
            image=img_tensor,                    # Image tensor (transformed)
            bbox=bbox_tensor,                    # Bounding box tensor [4] - RAW, for denormalization
            image_path=img_path,                 # Original image path for reference
            source_kpts_with_vis=source_kpts_with_vis, # Store original source with vis for visualization - RAW
            target_kpts_with_vis=target_kpts_with_vis  # Store original target with vis for visualization - RAW
        )
        return data

def prepare_data(args, image_transforms, dataset_class):
    """
    Loads the dataset, optionally visualizes raw samples, splits the data, 
    and creates DataLoaders.

    Args:
        args: Argparse Namespace containing all necessary arguments 
              (coco_json, output_dir, split_strategy, ratios, visualize_raw_data, etc.).
        image_transforms: torchvision transforms to be applied to images.
        dataset_class: The class of the dataset to instantiate (e.g., CocoPairedKeypointDataset).

    Returns:
        tuple: (train_loader, val_loader, test_loader, full_dataset_instance)
               DataLoaders can be None if their corresponding split size is 0.
               full_dataset_instance is returned for potential metadata access.
    """
    print("Loading dataset...")
    # Add target_kpt_slice_idx to dataset instantiation if it's CocoPairedKeypointGraphDataset
    if dataset_class == CocoPairedKeypointGraphDataset:
        full_dataset = dataset_class(
            coco_json_path=args.coco_json,
            image_base_dir=args.image_base_dir,
            transform=image_transforms, # This transform is for the image
            source_kpt_field=args.source_field,
            target_kpt_field=args.target_field,
            source_num_kpt_field=args.source_num_field,
            target_num_kpt_field=args.target_num_field,
            target_kpt_slice_idx=args.target_kpt_slice_idx # Pass slice_idx
        )
    else: # Original behavior for other datasets
        full_dataset = dataset_class(
            coco_json_path=args.coco_json,
            image_base_dir=args.image_base_dir, 
            transform=image_transforms,
            source_kpt_field=args.source_field,
            target_kpt_field=args.target_field,
            source_num_kpt_field=args.source_num_field,
            target_num_kpt_field=args.target_num_field
        )


    if len(full_dataset) == 0:
        print("Dataset is empty. Cannot proceed.")
        return None, None, None, full_dataset # Or raise an error
    
    print(f"Dataset loaded. Total samples: {len(full_dataset)}")
    
    if args.visualize_raw_data:
        print("Visualizing a few raw dataset samples...")
        raw_samples_vis_path = os.path.join(args.output_dir, 'raw_dataset_samples')
        # Ensure visualize_dataset_samples can handle the dataset_class instance correctly
        visualize_dataset_samples(
            full_dataset, 
            num_samples=min(3, len(full_dataset)), 
            output_basename=raw_samples_vis_path, 
            save_pdf=args.save_plots_as_pdf
        )

    total_len = len(full_dataset)
    train_dataset, val_dataset, test_dataset = None, None, None
    train_len, val_len, test_len = 0, 0, 0

    print(f"Using data splitting strategy: {args.split_strategy}")
    if args.split_strategy == "random":
        if total_len == 0:
            print("Error: Cannot split an empty dataset.")
            return None, None, None, full_dataset

        train_len = int(args.train_ratio * total_len)
        val_len = int(args.val_ratio * total_len)
        # Ensure test_len is non-negative and accounts for all data
        if (args.train_ratio + args.val_ratio) > 1.0:
            # This should be caught by argparse validation, but as a safeguard:
            print(f"Error: train_ratio ({args.train_ratio}) + val_ratio ({args.val_ratio}) > 1.0. Adjust ratios.")
            # Attempt to salvage by prioritizing train and val if sum > 1 but individual ratios are valid
            val_len = total_len - train_len if (train_len < total_len) else 0
            test_len = 0
        elif (args.train_ratio + args.val_ratio) == 1.0:
            test_len = 0
        else: # train_ratio + val_ratio < 1.0
            test_len = total_len - train_len - val_len
        
        # Safety check for small datasets where int conversion might lead to sum < total_len
        if train_len + val_len + test_len < total_len and test_len >= 0:
             # print(f"Adjusting test_len to account for rounding in small dataset. Before: {test_len}")
             test_len = total_len - train_len - val_len
             # print(f"After adjustment: {test_len}")

        if not (train_len > 0 and val_len > 0) and total_len > 0 : # Test set can be 0.
            print(f"Warning: Calculated split sizes are not ideal for training/validation. Train: {train_len}, Val: {val_len}, Test: {test_len}. Total: {total_len}")
            if train_len == 0 or val_len == 0:
                print("Train or Validation set is zero. Training cannot proceed effectively. Please check data ratios or dataset size.")
                # Return None for loaders if critical sets are empty
                return (DataLoader(full_dataset) if train_len > 0 else None), None, None, full_dataset
        
        print(f"Splitting dataset randomly: Train ({train_len}), Validation ({val_len}), Test ({test_len})")
        try:
            train_dataset, val_dataset, test_dataset_maybe = random_split(full_dataset, [train_len, val_len, test_len],
                                                                    generator=torch.Generator().manual_seed(args.split_random_seed))
            if test_len > 0:
                test_dataset = test_dataset_maybe
            elif test_len == 0 and len(test_dataset_maybe) == 0:
                 test_dataset = None # Explicitly None if test_len is 0
            else: # Should not happen if logic above is correct
                print(f"Warning: Mismatch in random_split for test set. Expected len {test_len}, got {len(test_dataset_maybe)}")
                test_dataset = test_dataset_maybe # Assign anyway

        except ValueError as e:
            print(f"Error during random_split: {e}. This might be due to sum of lengths not matching dataset size. Lengths: {train_len, val_len, test_len}, Total: {total_len}")
            return None, None, None, full_dataset

    elif args.split_strategy == "pca_kmeans_stratified":
        # Assuming extract_keypoint_features and split_by_pca_kmeans are in scope (e.g. imported in this file or passed)
        # For now, directly import them if they are in data_splitting_utils.py
        from .data_splitting_utils import extract_keypoint_features, split_by_pca_kmeans
        
        features, original_indices = extract_keypoint_features(full_dataset, 
                                                               args.source_field, 
                                                               args.target_field,
                                                               args.source_num_field,
                                                               args.target_num_field)
        if features.shape[0] == 0:
            print("Error: No features extracted for PCA/K-Means split. Check dataset or field names.")
            return None, None, None, full_dataset
        
        if features.shape[0] < args.kmeans_clusters:
            print(f"Warning: Number of samples ({features.shape[0]}) is less than k_means_clusters ({args.kmeans_clusters}). Adjusting k_means_clusters to {features.shape[0]}.")
            actual_kmeans_clusters = features.shape[0]
        else:
            actual_kmeans_clusters = args.kmeans_clusters

        try:
            train_indices, val_indices, test_indices = split_by_pca_kmeans(
                features, original_indices,
                train_ratio=args.train_ratio, val_ratio=args.val_ratio,
                pca_variance_threshold=args.pca_variance_threshold,
                n_clusters=actual_kmeans_clusters, 
                random_seed=args.split_random_seed
            )
        except ValueError as e:
            print(f"Error during PCA/K-Means split: {e}. Check ratios or data compatibility.")
            return None, None, None, full_dataset
            
        train_dataset = Subset(full_dataset, train_indices) if len(train_indices) > 0 else None
        val_dataset = Subset(full_dataset, val_indices) if len(val_indices) > 0 else None
        test_dataset = Subset(full_dataset, test_indices) if len(test_indices) > 0 else None
        
        train_len = len(train_dataset) if train_dataset else 0
        val_len = len(val_dataset) if val_dataset else 0
        test_len = len(test_dataset) if test_dataset else 0
        print(f"PCA/K-Means split counts: Train ({train_len}), Val ({val_len}), Test ({test_len})")
        
        if not (train_len > 0 and val_len > 0):
            print(f"Error: PCA/K-Means split resulted in empty train ({train_len}) or val ({val_len}) set. Cannot proceed.")
            # Return None for loaders if critical sets are empty
            return (train_dataset if train_len > 0 else None), None, (test_dataset if test_len > 0 else None), full_dataset

    else:
        raise ValueError(f"Unknown split_strategy: {args.split_strategy}")

    # Final check on split lengths
    if total_len > 0 and (train_len == 0 or val_len == 0):
        print(f"Critical Error: After splitting, train_len ({train_len}) or val_len ({val_len}) is zero. Cannot create DataLoaders for training.")
        # Pass through whatever datasets were formed, loaders will be None if dataset is None

    # Determine DataLoader type based on dataset_class
    if dataset_class == CocoPairedKeypointGraphDataset:
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError(
                "torch_geometric is required for Graph datasets. "
                "Install it with: pip install torch-geometric"
            )
        LoaderClass = PyGDataLoader # Use PyG's DataLoader for graph data
    else:
        LoaderClass = DataLoader    # Use standard PyTorch DataLoader

    train_loader = LoaderClass(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True) if train_dataset and train_len > 0 else None
    val_loader = LoaderClass(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True) if val_dataset and val_len > 0 else None
    test_loader = LoaderClass(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True) if test_dataset and test_len > 0 else None

    if not train_loader or not val_loader:
        print("Warning: Training or Validation DataLoader could not be created (likely due to zero-length split).")
        # No explicit return here, allow ModelTrainer to handle None loaders if necessary, or scripts to check

    return train_loader, val_loader, test_loader, full_dataset
