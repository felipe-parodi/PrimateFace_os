"""Evaluate SLEAP models using Normalized Mean Error (NME).

This script computes NME metrics for SLEAP model predictions, supporting
both top-down and bottom-up models with various normalization strategies.

Example:
    $ python evaluate_sleap_nme.py --model_dir path/to/sleap_model --test_data test.json
"""

# --- Encoding Hotfix for Windows ---
# On Windows, Python's default encoding may not be UTF-8, causing errors when
# a library like SLEAP tries to load a JSON config file that contains UTF-8
# characters. This patch forces text files to be opened with UTF-8.
import functools
import io
import os # Add os import for the path hotfix

# Store the original open function
_original_open = io.open

@functools.wraps(_original_open)
def utf8_open(file, mode='r', *args, **kwargs):
    # If the mode is text-based, force encoding to utf-8
    if 'b' not in mode:
        kwargs['encoding'] = 'utf-8'
    return _original_open(file, mode, *args, **kwargs)

# Override the built-in open with our patched version
io.open = utf8_open
# --- End of Hotfix ---

import argparse
from pathlib import Path
import json
import contextlib
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import sleap
from tqdm.auto import tqdm
from sleap.io.format.coco import LabelsCocoAdaptor
from sleap.io.format.filehandle import FileHandle
from sleap.io.video import Video
from collections import defaultdict
import cv2
from sleap.nn.config import TrainingJobConfig
from tensorflow import keras
from sleap.nn.inference import (
    TopDownPredictor, SingleInstancePredictor, BottomUpPredictor, Predictor
)
from sleap.nn.model import Model as SleapModel

# --- Combined Hotfix for Video Loading ---
# This patch addresses two issues with loading videos from image sequences:
# 1. Grayscale Bug: Forces images to be loaded as 3-channel RGB.
# 2. Windows Paths: Prepends the long-path prefix ("\\?\") to absolute paths on
#    Windows to prevent file access errors with long or network paths.

# Get the original classmethod's underlying function before any patches are applied.
_original_video_from_filenames_func = Video.from_image_filenames.__func__

@classmethod
@functools.wraps(_original_video_from_filenames_func)
def _patched_video_loader(cls, filenames, **kwargs):
    """A single, robust patch for loading videos from image filenames."""
    # 1. Grayscale fix: Ensure all images are loaded as 3-channel RGB.
    kwargs['grayscale'] = False

    # 2. Windows Path fix: Prepend long-path prefix if on Windows.
    processed_filenames = []
    if os.name == 'nt':
        for f in filenames:
            # Check if it's an absolute path with a drive letter (e.g., "C:\...")
            if Path(f).is_absolute() and ":" in f:
                processed_filenames.append(f"\\\\?\\{f}")
            else:
                processed_filenames.append(f)
    else:
        # On non-Windows systems, use filenames as-is.
        processed_filenames = filenames
    
    # Call the original underlying function with the class and modified arguments.
    return _original_video_from_filenames_func(cls, processed_filenames, **kwargs)

# Apply the single, combined patch.
Video.from_image_filenames = _patched_video_loader
# --- End of Hotfix ---

# A mock file handle to pass in-memory data to the adapter
class MockFileHandle:
    def __init__(self, data):
        self.json = data

# --- Configuration ---
# Path to your test JSON file
TEST_JSON_PATH = "path/to/test.json"
# The COCO format requires a single root directory for all image paths specified in the JSON.
# If your JSON has absolute paths, this can be an empty string.
# If your JSON has relative paths, this should be the directory they are relative to.
IMAGE_DIR = r""

# Keypoint indices for normalization (e.g., outer eye corners for a face model)
# These are 0-indexed.
NORM_KEYPOINT_INDICES = [36, 45]

def validate_videos(labels: sleap.Labels, json_path: Path):
    """Checks if all videos in a Labels object have valid shapes."""
    if not labels.videos:
        print(f"⚠️ Warning: No videos found in '{json_path}'.")
        return True # Nothing to validate

    for video in labels.videos:
        if any(s is None for s in video.shape):
            print("=" * 80)
            print(f"🛑 Error: Could not determine dimensions for a video in '{json_path}'.")
            if video.backend and video.backend.filenames:
                print(f"Problematic video/image file: {video.backend.filenames[0]}")
            print("Please check that all image paths in the JSON are correct and files are readable.")
            print("Also ensure that image entries in the JSON have 'height' and 'width' fields if the image file is missing.")
            print("=" * 80)
            return False
    return True

def filter_and_validate_coco_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filters COCO data to include only annotations for 3-channel RGB images
    that have keypoints. This prevents errors from grayscale/RGBA images and
    annotations without keypoints.
    """
    if "annotations" not in data or not data["annotations"]:
        print("⚠️ No annotations found in JSON data.")
        return data

    # Create a map of image_id -> image_path for efficient lookup
    img_dir = Path(IMAGE_DIR)
    image_id_to_path = {
        img['id']: (
            str(img_dir / img['file_name'])
            if not Path(img['file_name']).is_absolute()
            else img['file_name']
        )
        for img in data['images']
    }

    # Create a map of image_id -> list of annotations for that image
    image_id_to_anns = defaultdict(list)
    for ann in data['annotations']:
        image_id_to_anns[ann['image_id']].append(ann)

    # Iterate through images that actually have annotations to find valid ones
    final_kept_image_ids = set()
    print("🔎 Filtering dataset to keep only 3-channel RGB images with keypoint annotations...")
    for image_id in tqdm(list(image_id_to_anns.keys()), desc="Checking images"):
        if image_id not in image_id_to_path:
            continue

        anns_for_image = image_id_to_anns[image_id]
        if not any("keypoints" in ann and ann["keypoints"] for ann in anns_for_image):
            continue

        img_path = image_id_to_path[image_id]
        if not Path(img_path).exists():
            continue

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is not None and img.ndim == 3 and img.shape[2] == 3:
            final_kept_image_ids.add(image_id)

    # Create the new filtered dataset
    original_image_count = len(data["images"])
    filtered_images = [img for img in data["images"] if img["id"] in final_kept_image_ids]
    filtered_annotations = [
        ann for ann in data["annotations"]
        if ann["image_id"] in final_kept_image_ids and "keypoints" in ann and ann["keypoints"]
    ]

    print(f"✅ Filtering complete. Kept {len(filtered_images)} / {original_image_count} images.")

    if len(filtered_images) == 0:
        print("🛑 Error: Filtering removed all images. Please check your dataset.")
        print("Common issues: all images are grayscale/RGBA, or no annotations have keypoints.")

    filtered_data = data.copy()
    filtered_data["annotations"] = filtered_annotations
    filtered_data["images"] = filtered_images
    
    return filtered_data

def get_model_path(model_dir: Path) -> Path:
    """Finds the path to the best_model.h5 file in the directory."""
    model_path = model_dir / "best_model.h5"
    if not model_path.exists():
        # Fallback for older SLEAP versions or different save formats
        models = list(model_dir.glob("*.h5"))
        if not models:
            raise FileNotFoundError(f"No '.h5' model files found in {model_dir}")
        model_path = max(models, key=lambda p: p.stat().st_mtime)

    print(f"✅ Found model: {model_path.name}")
    return model_path


def load_predictor_manually(model_dir: Path) -> Predictor:
    """
    Manually loads a SLEAP predictor to bypass file encoding issues on Windows.
    
    This function reads the config JSON with a specified encoding and then
    initializes the appropriate predictor class directly.
    """
    model_path = get_model_path(model_dir)
    config_path = model_dir / "initial_config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find 'initial_config.json' in {model_dir}")

    # 1. Load config and inspect the skeleton defined within it.
    job_config = TrainingJobConfig.load_json(str(config_path))
    
    print("\n--- SKELETON DIAGNOSTICS (from Model Config) ---")
    try:
        # For this model type, the skeleton nodes are stored in 'part_names'.
        if hasattr(job_config.model.heads.centered_instance, 'part_names'):
            part_names = job_config.model.heads.centered_instance.part_names
            print(f"Model expects {len(part_names)} nodes: {part_names}")
        else:
            print("⚠️ Could not find 'part_names' in the model's head config.")

    except Exception as e:
        print(f"Error extracting model skeleton: {e}")
    print("--- END MODEL SKELETON ---\n")


    # 2. Load the Keras model
    keras_model = keras.models.load_model(str(model_path), compile=False)

    # Wrap the loaded keras model in SLEAP's Model class, providing all required configs
    sleap_model = SleapModel(
        keras_model=keras_model,
        backbone=job_config.model.backbone,
        heads=job_config.model.heads,
    )

    # 3. Determine predictor type and initialize it
    predictor = None
    heads_config = job_config.model.heads

    if heads_config.centered_instance is not None:
        # This is a top-down model.
        # We assume this model is the 'confmap' part of the pipeline.
        predictor = TopDownPredictor(
            confmap_config=job_config,
            confmap_model=sleap_model,
        )
    elif heads_config.single_instance is not None:
        predictor = SingleInstancePredictor(
            confmap_config=job_config,
            confmap_model=sleap_model,
        )
    elif heads_config.multi_instance is not None:
         # This is a bottom-up model.
         predictor = BottomUpPredictor(
            bottomup_config=job_config,
            bottomup_model=sleap_model,
         )
    else:
        # Fallback to the original method if we can't determine the type
        model_type_str = f"{[f for f in attr.fields(type(heads_config)) if getattr(heads_config, f.name) is not None]}"
        print(f"⚠️ Could not determine predictor type from heads config: {model_type_str}. Falling back to sleap.load_model.")
        return sleap.load_model(str(model_dir))
    
    # Initialize the predictor's internal models. This is crucial.
    predictor._initialize_inference_model()
    
    print(f"✅ Manually loaded {type(predictor).__name__}.")
    return predictor


def keypoint_nme(
    pred_coords: np.ndarray,
    gt_coords: np.ndarray,
    mask: np.ndarray,
    normalize_factor: np.ndarray,
) -> float:
    """
    Calculate the Normalized Mean Error.

    Args:
        pred_coords (np.ndarray): Predicted keypoint coordinates [N, K, 2].
        gt_coords (np.ndarray): Ground-truth keypoint coordinates [N, K, 2].
        mask (np.ndarray): Visibility mask for keypoints [N, K].
        normalize_factor (np.ndarray): Normalization factor for each instance [N, 1].

    Returns:
        float: The final NME score.
    """
    # Calculate Euclidean distance for each keypoint
    distances = np.linalg.norm(pred_coords - gt_coords, axis=2)  # Shape: [N, K]

    # This check helps verify the inputs to the NME calculation.
    # If raw distances are huge, it points to a coordinate space mismatch.
    # If normalization factors are very small, they can inflate the NME.
    if distances.size > 0:
        print(f"Raw distance range: [{np.min(distances):.2f}, {np.max(distances):.2f}]")
        print(f"Normalization factor range: [{np.min(normalize_factor):.2f}, {np.max(normalize_factor):.2f}]")
    
    # Apply mask to consider only visible keypoints
    masked_distances = distances * mask

    # Normalize the distances
    # The normalization factor is [N, 1], so it broadcasts correctly.
    # Add a small epsilon to avoid division by zero.
    norm_distances = masked_distances / (normalize_factor + 1e-6)

    if np.sum(mask) > 0:
        sample_masked_distances = norm_distances[mask > 0]
        print(f"Sample masked normalized distances: {sample_masked_distances[:5]}")

    # Calculate the sum of errors and the number of visible keypoints
    total_error = np.sum(norm_distances)
    total_visible_keypoints = np.sum(mask)

    if total_visible_keypoints == 0:
        return 0.0

    return total_error / total_visible_keypoints


def draw_skeleton(image, points, skeleton, color, thickness=2):
    """Draws a skeleton on an image."""
    # Draw nodes
    for i, pt in enumerate(points):
        if not np.isnan(pt).any():
            cv2.circle(image, (int(pt[0]), int(pt[1])), thickness * 2, color, -1)
    
    # Draw edges
    for edge in skeleton.edges:
        # edge is a tuple of (source_node, destination_node)
        source_node, dest_node = edge
        pt1_idx = skeleton.node_names.index(source_node.name)
        pt2_idx = skeleton.node_names.index(dest_node.name)
        pt1, pt2 = points[pt1_idx], points[pt2_idx]
        if not np.isnan(pt1).any() and not np.isnan(pt2).any():
            cv2.line(image, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color, thickness)
    return image


def visualize_normalized_skeletons(gt_inst, pr_inst, scale_factor, norm_indices, output_path="normalized_skeleton_visualization.png"):
    """
    Creates a side-by-side visualization of the GT and PR skeletons,
    normalized for position and scale to compare their intrinsic geometry.
    """
    CANVAS_SIZE = (512, 512)
    PADDING = 0.1  # 10% padding

    # Apply the globally detected scale factor to the prediction first
    gt_pts_orig = gt_inst.numpy()
    pr_pts_orig = pr_inst.numpy() * scale_factor

    def process_points(points):
        """Helper to center and scale a set of keypoints to fit the canvas."""
        valid_mask = ~np.isnan(points).any(axis=1)
        if not np.any(valid_mask):
            return None

        pts = points[valid_mask].copy()

        # 1. Center points around their centroid
        centroid = np.mean(pts, axis=0)
        centered_pts = pts - centroid

        # 2. Calculate scale factor to fit canvas
        min_coords = np.min(centered_pts, axis=0)
        max_coords = np.max(centered_pts, axis=0)
        span = max_coords - min_coords
        if span[0] == 0 or span[1] == 0:
            return None # Cannot scale a 1D object

        canvas_fit_size = (CANVAS_SIZE[0] * (1 - 2 * PADDING), CANVAS_SIZE[1] * (1 - 2 * PADDING))
        scale = min(canvas_fit_size[0] / span[0], canvas_fit_size[1] / span[1])

        # 3. Scale and translate to final position
        final_pts = centered_pts * scale + np.array(CANVAS_SIZE) / 2

        # Re-insert NaNs to maintain original array shape for skeleton drawing
        final_points_full = np.full_like(points, np.nan)
        final_points_full[valid_mask] = final_pts
        return final_points_full

    gt_pts_norm = process_points(gt_pts_orig)
    pr_pts_norm = process_points(pr_pts_orig)

    # Create blank white canvases
    gt_canvas = np.full((CANVAS_SIZE[1], CANVAS_SIZE[0], 3), 255, dtype=np.uint8)
    pr_canvas = np.full((CANVAS_SIZE[1], CANVAS_SIZE[0], 3), 255, dtype=np.uint8)

    def draw_normalized(canvas, points, skeleton):
        """Draws the points and skeleton with the requested color scheme."""
        if points is None:
            return canvas
        # Draw edges first (as background)
        if skeleton:
            for edge in skeleton.edges:
                try:
                    # Correctly unpack the source and destination nodes from the edge tuple.
                    source_node, dest_node = edge
                    pt1_idx = skeleton.node_names.index(source_node.name)
                    pt2_idx = skeleton.node_names.index(dest_node.name)
                    pt1, pt2 = points[pt1_idx], points[pt2_idx]
                    if not np.isnan(pt1).any() and not np.isnan(pt2).any():
                        cv2.line(canvas, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (200, 200, 200), 1)
                except (ValueError, IndexError):
                    continue
        # Draw points
        for i, pt in enumerate(points):
            if not np.isnan(pt).any():
                color = (0, 0, 255) if i in norm_indices else (0, 0, 0)  # Red for norm, Black for others
                cv2.circle(canvas, (int(pt[0]), int(pt[1])), 4, color, -1)
        return canvas

    gt_canvas = draw_normalized(gt_canvas, gt_pts_norm, gt_inst.skeleton)
    pr_canvas = draw_normalized(pr_canvas, pr_pts_norm, pr_inst.skeleton)

    # Add titles and combine
    cv2.putText(gt_canvas, 'Ground Truth (Normalized)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    cv2.putText(pr_canvas, 'Prediction (Normalized)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    combined_img = np.hstack((gt_canvas, pr_canvas))
    cv2.imwrite(output_path, combined_img)
    print(f"✅ Normalized skeleton visualization saved to: {output_path}")


def main():
    """Main function to run NME evaluation on a trained SLEAP model."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SLEAP model with NME on a COCO dataset."
    )
    parser.add_argument(
        "model_dir",
        type=Path,
        help="Path to the directory containing the trained SLEAP model ('best_model.h5').",
    )
    args = parser.parse_args()

    # --- 1. Load Model and Test Data ---
    print(f"⏳ Loading model from {args.model_dir}...")
    try:
        predictor = load_predictor_manually(args.model_dir)
    except (FileNotFoundError, Exception) as e:
        print(f"🛑 Error loading model: {e}")
        return

    print("⏳ Loading and preparing test dataset...")
    test_json = Path(TEST_JSON_PATH)
    img_dir = Path(IMAGE_DIR)
    if not test_json.exists():
        print(f"🛑 Error: Test JSON not found at {test_json}")
        return

    img_dir_str = str(img_dir) if str(img_dir) != "." else ""
    
    print(f"Loading data from {test_json}...")
    with open(test_json, "r") as f:
        raw_data = json.load(f)
    filtered_data = filter_and_validate_coco_data(raw_data)

    if len(filtered_data["images"]) == 0:
        print("🛑 Error: No valid images found in test set after filtering.")
        return
    
    handle = MockFileHandle(filtered_data)
    labels_gt = LabelsCocoAdaptor.read(handle, img_dir=img_dir_str)
    
    print("\n--- SKELETON DIAGNOSTICS (from COCO Data) ---")
    if labels_gt.skeletons:
        data_skeleton = labels_gt.skeletons[0]
        print(f"Data provides {len(data_skeleton.nodes)} nodes: {[node.name for node in data_skeleton.nodes]}")
    else:
        print("⚠️ No skeleton found in the loaded COCO data.")
    print("--- END DATA SKELETON ---\n")

    labels_gt.videos = list({lf.video for lf in labels_gt.labeled_frames})
    
    if not validate_videos(labels_gt, test_json):
        return
        
    print(f"✅ Test data loaded: {len(labels_gt)} labeled frames.")

    # --- 2. Run Inference ---
    print("⏳ Running inference on the test set...")
    labels_pr = predictor.predict(labels_gt)

    # --- SCALE DETECTION ---
    print("🔬 Checking for global scale mismatch between GT and predictions...")
    
    frame_pairs = sleap.nn.evals.find_frame_pairs(labels_gt, labels_pr)[:10]
    scale_estimates = []

    for gt_frame, pr_frame in frame_pairs:
        if gt_frame.instances and pr_frame.instances:
            gt_pts = gt_frame.instances[0].numpy()
            pr_pts = pr_frame.instances[0].numpy()
            
            valid = ~np.isnan(gt_pts).any(axis=1) & ~np.isnan(pr_pts).any(axis=1)
            
            if np.sum(valid) > 5:
                gt_span = np.ptp(gt_pts[valid], axis=0)
                pr_span = np.ptp(pr_pts[valid], axis=0)
                
                if pr_span[0] > 1:
                    scale_estimates.append(gt_span[0] / pr_span[0])
                if pr_span[1] > 1:
                    scale_estimates.append(gt_span[1] / pr_span[1])

    SCALE_FACTOR = np.median(scale_estimates) if scale_estimates else 1.0
    
    if abs(SCALE_FACTOR - 1.0) > 0.1:
        print(f"🔍 Detected significant scale difference: {SCALE_FACTOR:.3f}")
        # Assuming original image width around 1500px for this estimation
        # This is a rough guide for the user to understand the training resolution.
        with contextlib.suppress(ZeroDivisionError):
             print(f"   Model was likely trained on images resized to ~{1500/SCALE_FACTOR:.0f}px")
    else:
        print("✅ No significant scale difference detected.")

    # --- 3. Match Instances and Collect Data for NME ---
    print("⏳ Matching ground truth and predicted instances...")
    
    total_gt_instances = len(labels_gt.all_instances)
    total_predicted_instances = len(labels_pr.all_instances)
    print(f"Total GT instances: {total_gt_instances}")
    print(f"Total predicted instances: {total_predicted_instances}")

    all_pred_coords, all_gt_coords, all_masks, all_norm_factors = [], [], [], []
    kpt_idx1, kpt_idx2 = NORM_KEYPOINT_INDICES
    conf_threshold = 0.1

    frame_pairs = sleap.nn.evals.find_frame_pairs(labels_gt, labels_pr)
    
    # Debug first pair with scale applied
    if frame_pairs:
        gt_frame, pr_frame = frame_pairs[0]
        if gt_frame.instances and pr_frame.instances:
            gt_inst = gt_frame.instances[0]
            pr_inst = pr_frame.instances[0]
            
            gt_pts = gt_inst.numpy()
            pr_pts = pr_inst.numpy() * SCALE_FACTOR  # Apply scale here
            
            print(f"\nPost-scale coordinate check:")
            print(f"  GT sample: {gt_pts[0:3]}")
            print(f"  PR sample (scaled): {pr_pts[0:3]}")
            
            gt_centroid = np.nanmean(gt_pts, axis=0)
            pr_centroid = np.nanmean(pr_pts, axis=0)
            if not np.isnan(gt_centroid).any() and not np.isnan(pr_centroid).any():
                distance = np.linalg.norm(gt_centroid - pr_centroid)
                print(f"  Centroid distance after scaling: {distance:.2f} pixels")

    # Try standard matching first with relaxed threshold
    matched_pairs, _ = sleap.nn.evals.match_frame_pairs(frame_pairs, threshold=0.5)
    
    # If not enough matches, use manual matching with scaled coordinates
    if len(matched_pairs) < total_gt_instances * 0.5:
        print("⚠️ Standard matching insufficient. Using manual matching with scaled coordinates...")
        
        matched_pairs = []
        for gt_frame, pr_frame in tqdm(frame_pairs, desc="Manual matching"):
            for gt_inst in gt_frame.instances:
                gt_pts = gt_inst.numpy()
                gt_centroid = np.nanmean(gt_pts[~np.isnan(gt_pts).any(axis=1)], axis=0)
                
                if np.isnan(gt_centroid).any():
                    continue
                
                best_match = None
                min_distance = float('inf')
                
                for pr_inst in pr_frame.instances:
                    # Apply scale during centroid calculation
                    pr_pts = pr_inst.numpy() * SCALE_FACTOR
                    pr_centroid = np.nanmean(pr_pts[~np.isnan(pr_pts).any(axis=1)], axis=0)
                    
                    if np.isnan(pr_centroid).any():
                        continue
                    
                    distance = np.linalg.norm(gt_centroid - pr_centroid)
                    if distance < min_distance:
                        min_distance = distance
                        best_match = pr_inst
                
                if best_match is not None and min_distance < 100:
                    matched_pairs.append((gt_inst, best_match, min_distance))

    print(f"Found {len(matched_pairs)} matched instances ({len(matched_pairs) / total_gt_instances * 100:.1f}% matching rate)")

    if not matched_pairs:
        print("🛑 No matches found!")
        return

    print("\n🔬 Generating normalized skeleton visualization for the first matched pair...")
    gt_inst, pr_inst, _ = matched_pairs[0]
    visualize_normalized_skeletons(gt_inst, pr_inst, SCALE_FACTOR, NORM_KEYPOINT_INDICES)
    print("Exiting after visualization.")
    return # Exit after creating the visualization


if __name__ == "__main__":
    main() 