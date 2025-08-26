"""Data splitting utilities for landmark converter.

This module provides functions for intelligent data splitting strategies
including PCA-based and cluster-based splitting for better train/val/test sets.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import torch # For tensor operations if keypoints are tensors

def normalize_kpts_for_splitting(keypoints_tensor, bbox_tensor):
    """
    Normalizes keypoints (K, 3 or K, 2) using a single bounding box [x, y, w, h].
    Only x, y coordinates are normalized.
    Input keypoints are expected to be a tensor for a single instance.
    Bbox is expected to be a 1D tensor or list/numpy array [x,y,w,h].
    """
    if keypoints_tensor is None or bbox_tensor is None:
        return None
    
    kpts_xy = keypoints_tensor[..., :2] # Work with X,Y coordinates

    # Ensure bbox_tensor is a tensor for consistent operations
    if not isinstance(bbox_tensor, torch.Tensor):
        bbox_tensor = torch.tensor(bbox_tensor, dtype=kpts_xy.dtype, device=kpts_xy.device if isinstance(kpts_xy, torch.Tensor) else None)

    bbox_x, bbox_y, bbox_w, bbox_h = bbox_tensor[0], bbox_tensor[1], bbox_tensor[2], bbox_tensor[3]

    epsilon = 1e-6
    valid_w = torch.clamp(bbox_w, min=epsilon)
    valid_h = torch.clamp(bbox_h, min=epsilon)

    normalized_kpts_xy = kpts_xy.clone()
    normalized_kpts_xy[..., 0] = (kpts_xy[..., 0] - bbox_x) / valid_w
    normalized_kpts_xy[..., 1] = (kpts_xy[..., 1] - bbox_y) / valid_h
    return normalized_kpts_xy.flatten().cpu().numpy()


def extract_keypoint_features(dataset, source_kpt_field_name, target_kpt_field_name, source_num_kpt_field_name, target_num_kpt_field_name):
    """
    Extracts concatenated, normalized source and target keypoint features from a dataset.
    Args:
        dataset (CocoPairedKeypointDataset): The full dataset instance.
        source_kpt_field_name (str): The key in the dataset item for source keypoints.
        target_kpt_field_name (str): The key for target keypoints.
        source_num_kpt_field_name (str): Key for number of source keypoints in annotation.
        target_num_kpt_field_name (str): Key for number of target keypoints in annotation.
    Returns:
        tuple: (np.array of feature_vectors, list of original_indices_with_features)
    """
    print("Extracting keypoint features for PCA/K-Means splitting...")
    feature_vectors = []
    original_indices_with_features = []

    # Correct approach: Iterate through the dataset as it would be used by DataLoader
    for original_idx in range(len(dataset)):
        try:
            # This relies on __getitem__ returning a tuple that includes keypoints and bbox
            # (img_tensor, source_kpts, target_kpts, bbox, img_path)
            # _, source_kpts_tensor, target_kpts_tensor, bbox_tensor, _ = dataset[original_idx]
            item = dataset[original_idx]

            # Check if the item is a PyG Data object (heuristic check for expected attributes)
            # These attributes were defined in CocoPairedKeypointGraphDataset.__getitem__
            if hasattr(item, 'x') and hasattr(item, 'edge_index') and hasattr(item, 'source_kpts_with_vis') and hasattr(item, 'target_kpts_with_vis') and hasattr(item, 'bbox'):
                source_kpts_tensor = item.source_kpts_with_vis # Shape (K_src, 3)
                target_kpts_tensor = item.target_kpts_with_vis # Shape (K_trg_total, 3)
                bbox_tensor = item.bbox                       # Shape (4,)
                # Image tensor (item.image) and path (item.image_path) are also available but not needed here.
            elif isinstance(item, tuple) and len(item) == 5: # Standard dataset item
                _, source_kpts_tensor, target_kpts_tensor, bbox_tensor, _ = item
            else:
                # print(f"Skipping sample {original_idx} due to unexpected item format: {type(item)}")
                continue

            if source_kpts_tensor is None or target_kpts_tensor is None or bbox_tensor is None:
                continue
            if bbox_tensor[2] <=0 or bbox_tensor[3] <=0: # width or height is zero/negative
                continue


            norm_source_flat = normalize_kpts_for_splitting(source_kpts_tensor, bbox_tensor)
            # Slice target_kpts_tensor to exclude the first keypoint before normalization
            norm_target_flat = normalize_kpts_for_splitting(target_kpts_tensor[1:, :], bbox_tensor)

            if norm_source_flat is not None and norm_target_flat is not None:
                combined_features = np.concatenate((norm_source_flat, norm_target_flat))
                feature_vectors.append(combined_features)
                original_indices_with_features.append(original_idx)
            # else:
                # print(f"Warning: Could not normalize keypoints for sample index {original_idx}")


        except Exception as e:
            # print(f"Skipping sample {original_idx} during feature extraction due to error: {e}")
            continue # Correctly indented continue for the exception case


    if not feature_vectors:
        print("Warning: No features extracted. PCA/K-Means splitting cannot proceed.")
        return np.array([]), []
        
    print(f"Successfully extracted features for {len(feature_vectors)} samples.")
    return np.array(feature_vectors), original_indices_with_features


def split_by_pca_kmeans(feature_vectors, original_indices,
                        train_ratio, val_ratio,
                        pca_variance_threshold, n_clusters, random_seed):
    """
    Splits data indices based on PCA embeddings and K-Means clustering.
    Args:
        feature_vectors (np.array): Array of shape (num_samples, num_features).
        original_indices (list): List of original dataset indices corresponding to feature_vectors.
        train_ratio (float): Proportion for training set.
        val_ratio (float): Proportion for validation set.
        pca_variance_threshold (float): The minimum ratio of variance to be explained by PCA components (e.g., 0.95).
        n_clusters (int): Number of clusters for K-Means.
        random_seed (int): Seed for reproducibility.
    Returns:
        tuple: (train_indices, val_indices, test_indices) - lists of original dataset indices.
    """
    if feature_vectors.shape[0] == 0:
        print("Warning: No feature vectors provided to split_by_pca_kmeans. Returning empty splits.")
        return [], [], []
    
    # Ensure pca_variance_threshold is valid for PCA's n_components
    # It should be > 0 and <= 1.0 for variance ratio, or an int for number of components.
    # We are now expecting a float for variance ratio.
    if not (0.0 < pca_variance_threshold <= 1.0):
        print(f"Error: pca_variance_threshold ({pca_variance_threshold}) must be between 0.0 (exclusive) and 1.0 (inclusive). Returning empty splits.")
        return [], [], []

    # Determine the maximum possible number of components if we were to use an integer.
    # This is used as an upper bound if pca_variance_threshold=1.0, or for very small datasets.
    max_possible_components = min(feature_vectors.shape[0], feature_vectors.shape[1])

    if max_possible_components == 0:
        print("Error: Cannot perform PCA with 0 samples or 0 features. Returning empty splits.")
        return [], [], []

    # If pca_variance_threshold is 1.0, it might select all components.
    # If the number of samples or features is very small, PCA might behave unexpectedly
    # or select fewer components than what 1.0 might imply if not capped.
    # sklearn's PCA handles n_components as a float (0 to 1) correctly.
    # No explicit cap needed here for pca_variance_threshold as a float,
    # but we'll check the number of components selected AFTER fit.

    print(f"Performing PCA to capture at least {pca_variance_threshold*100:.2f}% of variance...")
    pca = PCA(n_components=pca_variance_threshold, random_state=random_seed)
    pca_embeddings = pca.fit_transform(feature_vectors)
    
    actual_pca_components = pca.n_components_
    print(f"PCA complete. Selected {actual_pca_components} components, explaining {np.sum(pca.explained_variance_ratio_):.4f} of variance.")

    actual_n_clusters = min(n_clusters, pca_embeddings.shape[0])
    if actual_n_clusters < n_clusters:
         print(f"Warning: n_clusters ({n_clusters}) was greater than number of samples after PCA. Using {actual_n_clusters} instead.")
    if actual_n_clusters == 0: 
        print("Error: actual_n_clusters is 0 (possibly due to no samples after PCA or very few PCA components). Cannot perform K-Means. Returning empty splits.")
        return [], [], []


    print(f"Performing K-Means clustering with {actual_n_clusters} clusters...")
    kmeans = KMeans(n_clusters=actual_n_clusters, random_state=random_seed, n_init='auto')
    cluster_labels = kmeans.fit_predict(pca_embeddings)

    train_indices, val_indices, test_indices = [], [], []
    
    # Ensure original_indices is a numpy array for advanced indexing
    original_indices_np = np.array(original_indices)

    print("Performing stratified split based on clusters...")
    for cluster_id in range(actual_n_clusters):
        # Get original dataset indices for samples in the current cluster
        indices_in_cluster = original_indices_np[cluster_labels == cluster_id]
        
        n_cluster_samples = len(indices_in_cluster)
        if n_cluster_samples == 0:
            continue

        # Calculate split sizes for this cluster
        n_train_cluster = int(np.round(train_ratio * n_cluster_samples))
        n_val_cluster = int(np.round(val_ratio * n_cluster_samples))
        
        # Ensure test set gets at least what's left, adjust if sum > n_cluster_samples due to rounding
        n_test_cluster = n_cluster_samples - n_train_cluster - n_val_cluster
        if n_test_cluster < 0: # If rounding caused over-allocation
            n_test_cluster = 0
            if n_train_cluster + n_val_cluster > n_cluster_samples:
                 # Prioritize train, then val, if rounding caused an issue
                if n_train_cluster > n_cluster_samples:
                    n_train_cluster = n_cluster_samples
                    n_val_cluster = 0
                elif n_train_cluster + n_val_cluster > n_cluster_samples:
                    n_val_cluster = n_cluster_samples - n_train_cluster


        # Shuffle indices within the cluster for random assignment
        np.random.seed(random_seed) # Ensure this shuffle is reproducible
        shuffled_cluster_indices = np.random.permutation(indices_in_cluster)

        train_indices.extend(shuffled_cluster_indices[:n_train_cluster])
        val_indices.extend(shuffled_cluster_indices[n_train_cluster : n_train_cluster + n_val_cluster])
        test_indices.extend(shuffled_cluster_indices[n_train_cluster + n_val_cluster : n_train_cluster + n_val_cluster + n_test_cluster])

    print(f"Splitting complete. Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)} samples.")
    return train_indices, val_indices, test_indices 