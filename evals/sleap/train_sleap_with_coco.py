"""
Train a SLEAP model on a COCO-formatted dataset using the Python API.

This script provides a complete pipeline for training a SLEAP model, including:
1.  **Data Conversion**: Converts COCO JSON annotations into SLEAP's native .slp format.
2.  **Data Validation**: Filters the dataset to include only valid 3-channel RGB images
    with keypoint annotations, caching the results for faster subsequent runs.
3.  **Training**: Launches a training job using a specified SLEAP training profile.
4.  **Configuration**: Allows overriding key training parameters like epochs, input scaling,
    and data augmentation via command-line arguments.
5.  **Evaluation**: Generates loss curves and visualizes predictions on test data after
    training is complete.

SLEAP Training Profiles:
------------------------
SLEAP uses "training profiles" (JSON files) to define the model architecture,
data augmentation, and other hyperparameters. These profiles are located in the
`sleap/training_profiles` directory.

The profiles follow a naming convention:
- `rf`: Receptive field size (e.g., `large_rf`, `medium_rf`).
- `topdown`, `bottomup`, `single`, `centroid`: Model's head type, defining the
  pose estimation strategy.

Usage:
------
1.  **Configure Paths**:
    Update the `TRAIN_JSON_PATH`, `VAL_JSON_PATH`, `TEST_JSON_PATH`, and `IMAGE_DIR`
    variables at the top of the script to point to your dataset.

2.  **List Available Profiles**:
    Run the script without a `--profile` argument to see all built-in options.
    ```bash
    python train_sleap_with_coco.py
    ```

3.  **Run Training**:
    Specify a training profile and an output directory.
    ```bash
    python train_sleap_with_coco.py --profile baseline_large_rf.topdown.json --output_dir ./my_model
    ```

4.  **Advanced Usage**:
    - **Override epochs and input scale**:
      ```bash
      python train_sleap_with_coco.py --profile <profile> --output_dir <dir> --epochs 50 --input-scale 0.5
      ```
    - **Preload data into RAM for speed (requires significant memory)**:
      ```bash
      python train_sleap_with_coco.py --profile <profile> --output_dir <dir> --preload
      ```
    - **Run a quick test on a subset of images**:
      ```bash
      python train_sleap_with_coco.py --profile <profile> --output_dir <dir> --limit-images 100
      ```
    - **Force re-filtering of the source COCO files**:
      ```bash
      python train_sleap_with_coco.py --profile <profile> --output_dir <dir> --force-filter
      ```
"""
import argparse
from pathlib import Path
import numpy as np
import json
import tensorflow as tf
import cv2
from collections import Counter, defaultdict
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os
import time
import attr
import contextlib
import cattr
import functools
import builtins
import concurrent.futures
from typing import Optional

# --- Smart Encoding Hotfix for Windows ---
# The standard open() on Windows may use a system-specific encoding (like cp1252)
# by default, which fails on UTF-8 encoded files like SLEAP's configs.
# This patch intercepts open() calls for text files ending in .json and forces
# them to use UTF-8, while leaving all other file operations unchanged.

_original_open = builtins.open

@functools.wraps(_original_open)
def safe_open(file, mode='r', *args, **kwargs):
    if "b" not in mode and isinstance(file, (str, Path)) and str(file).endswith(".json"):
        kwargs['encoding'] = 'utf-8'
    return _original_open(file, mode, *args, **kwargs)

builtins.open = safe_open
# --- End of Hotfix ---

import sleap
from sleap.nn.config import TrainingJobConfig
from sleap.nn.training import Trainer
from sleap.io.format.coco import LabelsCocoAdaptor
from sleap.io.format.filehandle import FileHandle
from sleap.io.video import Video
from sleap.nn.inference import (
    Predictor, TopDownPredictor, SingleInstancePredictor, BottomUpPredictor
)
from tensorflow import keras

# =================================================================
# Hotfix for handling UNC paths on Windows
# =================================================================
# The standard path handling in Python and OpenCV can sometimes fail to
# resolve network paths (UNC paths, e.g., \\server\share\file.txt).
# This patch overrides the part of SLEAP that generates video objects
# from filenames to ensure UNC paths are correctly formatted for Windows.
_original_from_image_filenames = Video.from_image_filenames

def _patched_from_image_filenames(cls, filenames, height=None, width=None, **kwargs):
    """
    A patched version of Video.from_image_filenames that handles UNC paths.
    It intercepts the arguments, modifies the 'filenames' list to
    prepend the Windows UNC path prefix, and then calls the original method
    with the correct argument structure to avoid TypeErrors.
    """
    processed_filenames = []
    for f in filenames:
        # Check if the path is a UNC path (starts with \\)
        if f.startswith("\\\\"):
            # The \\?\UNC\ prefix allows Windows APIs to handle long paths
            # and network paths more reliably.
            processed_filenames.append("\\\\?\\UNC\\" + f[2:])
        else:
            processed_filenames.append(f)

    # Call the original underlying function directly to avoid classmethod issues.
    # Pass all arguments explicitly.
    return _original_from_image_filenames.__func__(
        cls, processed_filenames, height=height, width=width, **kwargs
    )

# Apply the monkey-patch
Video.from_image_filenames = classmethod(_patched_from_image_filenames)
# =================================================================


# =================================================================
# Step 1: Provide paths to your COCO JSON files and image directory
# =================================================================
# IMPORTANT: Update these paths to point to your actual data files
TRAIN_JSON_PATH = "./data/train.json"  # Update this path
VAL_JSON_PATH = "./data/val.json"      # Update this path
TEST_JSON_PATH = "./data/test.json"    # Update this path

# The COCO format requires a single root directory for all image paths specified in the JSON.
# If your JSON has absolute paths, this can be an empty string.
# If your JSON has relative paths, this should be the directory they are relative to.
IMAGE_DIR = r""
# =================================================================

# The mock file handle is still needed for in-memory data.
class MockFileHandle:
    def __init__(self, data):
        self.json = data


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

# =================================================================
# Helper for Parallelized Filtering
# =================================================================
def check_image_is_rgb(image_info: tuple) -> Optional[int]:
    """
    Worker function for multiprocessing. Checks if an image is a 3-channel
    RGB file.

    Args:
        image_info: A tuple containing (image_id, image_path).

    Returns:
        The image_id if the image is valid (3-channel RGB), otherwise None.
    """
    image_id, img_path = image_info
    try:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is not None and img.ndim == 3 and img.shape[2] == 3:
            return image_id
    except Exception:
        # Gracefully handle corrupted images or other reading errors
        return None
    return None

def filter_and_validate_coco_data(data: dict) -> dict:
    """
    Filters COCO data to include only annotations for 3-channel RGB images
    that have keypoints. This prevents errors from grayscale/RGBA images and
    annotations without keypoints. Uses multiprocessing for speed.
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

    # Pre-filter to get a list of image infos to check in parallel
    images_to_check = []
    image_ids_with_annotations = set(image_id_to_anns.keys())
    
    for image_id in image_ids_with_annotations:
        if image_id not in image_id_to_path:
            continue

        anns_for_image = image_id_to_anns[image_id]
        if not any("keypoints" in ann and ann["keypoints"] for ann in anns_for_image):
            continue

        img_path_str = image_id_to_path[image_id]
        if Path(img_path_str).exists():
            images_to_check.append((image_id, img_path_str))
            
    # Use a process pool to check images in parallel
    final_kept_image_ids = set()
    print(f"🔎 Filtering dataset across {len(images_to_check)} images using multiple cores...")
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Map the worker function to the images to check
        future_to_info = {executor.submit(check_image_is_rgb, info): info for info in images_to_check}
        
        # Use tqdm to show progress as futures complete
        for future in tqdm(concurrent.futures.as_completed(future_to_info), total=len(images_to_check), desc="Checking images"):
            result_image_id = future.result()
            if result_image_id is not None:
                final_kept_image_ids.add(result_image_id)

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

def check_image_formats(json_path):
    """Check color channels of all images in COCO dataset."""
    print("Running diagnostic check on source image files...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    image_paths = [img_info['file_name'] for img_info in data['images']]
    if not image_paths:
        print("No images found in JSON file.")
        return

    # If the main image directory is specified, prepend it to relative paths
    img_dir = Path(IMAGE_DIR)
    if str(img_dir) != ".":
        image_paths = [
            str(img_dir / path) if not Path(path).is_absolute() else path
            for path in image_paths
        ]

    channel_counts = Counter()
    problematic = []
    
    # Check first 50 images to avoid long waits
    for img_path in image_paths[:50]:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  - Warning: Could not read image at {img_path}")
            continue
        
        channels = 1 if img.ndim == 2 else img.shape[2]
        channel_counts[channels] += 1
        
        if channels == 1:
            problematic.append(img_path)
    
    print(f"  - Channel distribution (first 50 images): {dict(channel_counts)}")
    if problematic:
        print(f"  - Found {len(problematic)} grayscale images. Sample: {problematic[0]}")
    print("-" * 80)


def list_available_profiles():
    """Prints a list of available built-in SLEAP training profiles."""
    try:
        profile_dir = sleap.util.get_package_file("training_profiles")
        profiles = sorted([p.name for p in Path(profile_dir).glob("*.json")])
        print("✅ Available training profiles:")
        for profile in profiles:
            print(f"  - {profile}")
    except Exception as e:
        print(f"⚠️ Could not list training profiles: {e}")
    print("-" * 80)


def setup_gpu(device: str):
    """Configures GPU for training."""
    if device == "cpu" or not sleap.nn.system.is_gpu_system():
        sleap.nn.system.use_cpu_only()
        print("Running in CPU-only mode.")
        return None  # Return None for no strategy

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        print("No GPUs found, falling back to CPU-only mode.")
        sleap.nn.system.use_cpu_only()
        return None

    # By default, use all available GPUs
    print(f"Found {len(gpus)} available GPUs.")
    
    # The MirroredStrategy will handle device placement automatically.
    # We just need to ensure memory growth is enabled on all of them to prevent OOM errors.
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # Use ReductionToOneDevice as the cross-device communication method.
    # This is a fallback for when the default NCCL backend fails, which can
    # happen due to environment or installation issues on Windows.
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.ReductionToOneDevice()
    )
    print(f"✅ Using MirroredStrategy for {strategy.num_replicas_in_sync} GPUs with ReductionToOneDevice.")
    
    # Disable pre-allocation as a general good practice
    sleap.disable_preallocation()
    print("Disabled GPU memory pre-allocation.")
    
    sleap.nn.system.summary()
    return strategy


def plot_losses(history: dict, output_dir: Path):
    """Plots training and validation losses and saves the plot."""
    plt.figure()
    plt.plot(history["loss"], label="Train Loss")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    save_path = output_dir / "loss_curve.png"
    plt.savefig(str(save_path))
    plt.close()
    print(f"📉 Loss curve saved to: {save_path}")


def draw_instance(image: np.ndarray, instance: sleap.Instance):
    """Manually draws a SLEAP instance onto an image using OpenCV."""
    # Draw points
    for point in instance.points:
        if not np.isnan(point.x) and not np.isnan(point.y):
            # Use a default color, e.g., blue for predictions
            color = (255, 0, 0)
            cv2.circle(image, (int(point.x), int(point.y)), 3, color, -1)
    
    # Draw edges
    if instance.skeleton:
        for edge in instance.skeleton.edges:
            # Get points from the instance by their node names using dictionary-style lookup
            # In this version of SLEAP, edges are tuples of nodes.
            source_node, dest_node = edge
            p1 = instance[source_node.name]
            p2 = instance[dest_node.name]
            
            if all(p is not None and not np.isnan(p.x) and not np.isnan(p.y) for p in [p1, p2]):
                # Use a default color, e.g., green for edges
                color = (0, 255, 0)
                cv2.line(image, (int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), color, 1)


def visualize_predictions(
    model_path: str,
    labels_gt: sleap.Labels,
    output_dir: Path,
    num_images: int = 4,
):
    """Saves images with ground truth and predicted poses."""
    print(f"🎨 Visualizing predictions for {num_images} images...")
    
    if not labels_gt.labeled_frames:
        print("⚠️ No labeled frames in the test set to visualize.")
        return

    # Create a predictor from the trained model path
    print(f"  - Loading predictor from: {model_path}")
    predictor = sleap.load_model(model_path)

    # Select a few frames to visualize and create a new temporary Labels object from them.
    # This is the correct way to pass a subset of data to the predictor.
    gt_lfs_to_viz = labels_gt.labeled_frames[:num_images]
    labels_to_predict = sleap.Labels(labeled_frames=gt_lfs_to_viz)

    # Run prediction on the videos that contain our frames of interest.
    print("  - Running inference on sample frames...")
    labels_pr = predictor.predict(labels_to_predict)
    print("  - Inference complete.")

    for i, gt_lf in enumerate(gt_lfs_to_viz):
        # Find the corresponding predicted frame from the inference results
        pr_lf = labels_pr.find(gt_lf.video, gt_lf.frame_idx)
        if not pr_lf:
            print(f"  - Could not find prediction for frame {gt_lf.frame_idx} in {gt_lf.video.filename}")
            continue

        # Get the image and prepare for drawing (convert to BGR for OpenCV)
        image_np = gt_lf.image
        if image_np.shape[-1] == 1:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        else:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        img_gt = image_np.copy()
        img_pred = image_np.copy()

        # Draw ground truth and predicted instances using our manual function
        for instance in gt_lf.instances:
            draw_instance(img_gt, instance)
        
        # pr_lf can be a list of frames if there are duplicates; iterate through all of them
        pr_lfs = pr_lf if isinstance(pr_lf, list) else [pr_lf]
        for frame in pr_lfs:
            for instance in frame.instances:
                draw_instance(img_pred, instance)

        # Add text labels to the images
        cv2.putText(img_gt, 'Ground Truth', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img_pred, 'Prediction', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Combine images horizontally and save
        combined_img = np.hstack((img_gt, img_pred))
        save_path = output_dir / f"prediction_visualization_{i}.png"
        cv2.imwrite(str(save_path), combined_img)
        print(f"🖼️ Saved comparison to {save_path}")


def main():
    """
    Main function to convert COCO data to SLEAP format and launch training.
    """
    # Set TF log level to suppress verbose messages.
    # '1' filters INFO, '2' filters WARNING, '3' filters ERROR.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    parser = argparse.ArgumentParser(
        description="Train a SLEAP model on a COCO dataset using the Python API.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Name of the training profile to use (e.g., 'baseline_large_rf.topdown.json'). "
             "If not provided, lists available profiles and exits.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./sleap_trained_model"),
        help="Directory to save converted SLP files and training outputs.",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="auto",
        help="Specify which GPU to use, e.g., 'cuda:0', '0', 'cpu', or 'auto'.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Per-GPU batch size. Overrides the value in the training profile."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs to train for. Overrides the value in the training profile."
    )
    parser.add_argument(
        "--input-scale",
        type=float,
        default=None,
        help="Factor to scale input images by (e.g., 0.5 for half size). "
             "Overrides the value in the training profile."
    )
    parser.add_argument(
        "--force-filter",
        action="store_true",
        help="Force re-filtering of the dataset, ignoring any cached files."
    )
    parser.add_argument(
        "--no-augmentations",
        action="store_true",
        help="Disable all data augmentations for debugging training speed."
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Preload the entire dataset into memory for faster training. Requires significant RAM."
    )
    parser.add_argument(
        "--limit-images",
        type=int,
        default=None,
        help="Limit the dataset to a specific number of images for a quick test run."
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a saved model (.h5 file) to resume training from."
    )

    args = parser.parse_args()

    # --- 1. List available profiles and set up device ---
    list_available_profiles()

    if args.profile is None:
        print("No training profile specified. Exiting.")
        return
        
    strategy = setup_gpu(args.gpu)

    # --- 2. Verify paths and create output directory ---
    train_json = Path(TRAIN_JSON_PATH)
    val_json = Path(VAL_JSON_PATH)
    test_json = Path(TEST_JSON_PATH)
    img_dir = Path(IMAGE_DIR)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not all([train_json.exists(), val_json.exists(), test_json.exists()]):
        print("=" * 80)
        print("🛑 Error: Please update TRAIN_JSON_PATH, VAL_JSON_PATH, and TEST_JSON_PATH.")
        print(f"Current train path: {train_json} (Exists: {train_json.exists()})")
        print(f"Current val path: {val_json} (Exists: {val_json.exists()})")
        print(f"Current test path: {test_json} (Exists: {test_json.exists()})")
        print("=" * 80)
        return

    # --- 3. Convert COCO JSON to SLEAP format (.slp) ---
    print("⏳ Converting COCO annotations to SLEAP format...")
    try:
        # Define paths for the converted .slp files
        train_slp_path = args.output_dir / "train.slp"
        val_slp_path = args.output_dir / "val.slp"
        test_slp_path = args.output_dir / "test.slp"

        # The COCO adapter needs an explicit image directory.
        img_dir_str = str(img_dir) if img_dir else ""

        def load_and_fix_coco(json_path: Path) -> sleap.Labels:
            """
            Load COCO file, filter it, and apply patch for the 'videos' list.
            Caches the filtered results to a '.filtered.json' file to speed up
            subsequent runs.
            """
            print(f"Processing data from {json_path}...")
            
            # Define path for the cached (filtered) JSON file
            cached_json_path = json_path.with_suffix(".filtered.json")

            # Check for cache, but ignore it if --force-filter is used
            if not args.force_filter and cached_json_path.exists():
                print(f"  -> Found cached filtered data. Loading from {cached_json_path}")
                with open(cached_json_path, "r") as f:
                    filtered_data = json.load(f)
            else:
                if args.force_filter and cached_json_path.exists():
                    print("  -> --force-filter specified. Ignoring existing cache.")
                
                print(f"  -> No cache found or filtering is forced. Processing from scratch...")
                with open(json_path, "r") as f:
                    raw_data = json.load(f)
                filtered_data = filter_and_validate_coco_data(raw_data)

                # Save the filtered data to the cache file for next time
                print(f"  -> Saving filtered data to {cached_json_path} for future runs.")
                with open(cached_json_path, "w") as f:
                    json.dump(filtered_data, f)
            
            # --- Limit images if specified ---
            if args.limit_images is not None and args.limit_images > 0:
                print(f"⚠️  Limiting dataset to the first {args.limit_images} images.")
                
                # Get the IDs of the first N images
                limited_image_ids = {img['id'] for img in filtered_data['images'][:args.limit_images]}
                
                # Filter images and annotations to match these IDs
                filtered_data['images'] = [img for img in filtered_data['images'] if img['id'] in limited_image_ids]
                filtered_data['annotations'] = [ann for ann in filtered_data['annotations'] if ann['image_id'] in limited_image_ids]
            # --- End of limit ---

            # Use a mock handle to pass the filtered data to the adapter
            handle = MockFileHandle(filtered_data)
            # The monkey-patch above will handle the grayscale issue, so we don't
            # need to pass any special video_kwargs here.
            labels = LabelsCocoAdaptor.read(handle, img_dir=img_dir_str)
            
            # HOTFIX: The COCO adapter does not populate the .videos list.
            # We reconstruct it manually from the labeled frames.
            labels.videos = list({lf.video for lf in labels.labeled_frames})
            return labels

        # Convert and save each dataset
        train_labels = load_and_fix_coco(train_json)
        if not validate_videos(train_labels, train_json):
            return
        train_labels.save(str(train_slp_path))
        print(f"✅ Training data saved to {train_slp_path}")

        # --- User-requested check on saved SLP file ---
        print("Running diagnostic check on saved SLP file...")
        try:
            lbl = sleap.load_file(str(train_slp_path))
            # Loading the full LabeledFrame can be slow, let's just check one.
            if lbl.labeled_frames:
                sample_lf = lbl.labeled_frames[0]
                shape = sample_lf.image.shape
                print(f"  - Sample image shape in SLP file: {shape}")
                if shape[-1] != 3:
                    print(f"  - WARNING: Image in SLP has {shape[-1]} channels, not 3.")
            else:
                print("  - SLP file contains no labeled frames to check.")
        except Exception as e:
            print(f"  - Could not perform SLP check: {e}")
        print("-" * 80)
        # --- End of check ---

        val_labels = load_and_fix_coco(val_json)
        if not validate_videos(val_labels, val_json):
            return
        val_labels.save(str(val_slp_path))
        print(f"✅ Validation data saved to {val_slp_path}")

        test_labels = load_and_fix_coco(test_json)
        if not validate_videos(test_labels, test_json):
            return
        test_labels.save(str(test_slp_path))
        print(f"✅ Test data saved to {test_slp_path}")

    except Exception as e:
        print(f"🛑 Error during data conversion: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 4. Configure and run training via the Python API ---
    print("\n" + "=" * 80)
    print("🚀 Launching SLEAP training via Python API...")
    print("=" * 80)

    try:
        # If we're using a distribution strategy, all model building and training
        # must happen within its scope.
        strategy_scope = strategy.scope() if strategy else contextlib.suppress()

        with strategy_scope:
            # Load the base training profile
            profile_dir = sleap.util.get_package_file("training_profiles")
            profile_path = Path(profile_dir) / args.profile

            if not profile_path.exists():
                print(f"🛑 Error: Training profile '{args.profile}' not found in {profile_dir}")
                return
                
            job_config = TrainingJobConfig.load_json(str(profile_path))

            # Set initial weights if resuming
            if args.resume_from:
                resume_path = Path(args.resume_from)
                if resume_path.exists():
                    job_config.model.initial_weights_path = str(resume_path.resolve())
                    print(f"✅ Resuming training from checkpoint: {resume_path}")
                else:
                    print(f"🛑 Warning: Checkpoint not found at {args.resume_from}. Starting from scratch.")

            # Override batch size if specified
            if args.batch_size is not None:
                job_config.optimization.batch_size = args.batch_size
                print(f"✅ Overriding per-GPU batch size to {args.batch_size}.")
                if strategy:
                    global_batch_size = args.batch_size * strategy.num_replicas_in_sync
                    print(f"✅ Global batch size across {strategy.num_replicas_in_sync} GPUs will be {global_batch_size}.")

            # Override epochs if specified on the command line
            if args.epochs is not None:
                job_config.optimization.epochs = args.epochs
                print(f"✅ Overriding training epochs to {args.epochs}.")

            # Override input scaling if specified
            if args.input_scale is not None:
                job_config.data.preprocessing.input_scaling = args.input_scale
                print(f"✅ Overriding input scaling to {args.input_scale}.")

            # Disable augmentations if specified
            if args.no_augmentations:
                # The augmentation settings are in the 'optimization' config, not 'data'.
                # Also, there is no single 'enabled' flag. We must turn off each type.
                aug_cfg = job_config.optimization.augmentation_config
                for field in attr.fields(type(aug_cfg)):
                    if field.type is bool:
                        setattr(aug_cfg, field.name, False)
                print("⚠️ Disabling all data augmentations.")

            # Set the preloading flag based on the command-line argument.
            # This is the correct, built-in way to cache the dataset in memory.
            job_config.optimization.preload_data = args.preload
            if args.preload:
                print("✅ Preloading data into RAM is enabled. The first epoch may be slow.")
            else:
                print("ℹ️ Preloading is disabled. Data will be read from disk each epoch.")

            # --- Ensure RGB conversion in the data pipeline ---
            # This flag tells the SLEAP data pipeline to automatically convert
            # any grayscale images to 3-channel RGB.
            job_config.data.preprocessing.ensure_rgb = True
            print("✅ Configured data pipeline to ensure RGB conversion.")
            # --- End of fix ---

            # Direct outputs to our specified folder
            job_config.outputs.runs_folder = str(args.output_dir)
            job_config.outputs.save_outputs = True # Ensure outputs are saved

            # Create the trainer instance
            trainer = Trainer.from_config(
                job_config,
                training_labels=str(train_slp_path),
                validation_labels=str(val_slp_path),
                test_labels=str(test_slp_path),
            )
            
            # Start the training
            trainer.train()

        print("\n" + "=" * 80)
        print("🎉 Training finished!")
        print(f"Model and logs saved in: {trainer.run_path}")

        # --- Post-training visualizations and evaluation ---
        run_dir = Path(trainer.run_path)
        plot_losses(trainer.keras_model.history.history, run_dir)
        
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"🛑 Training failed with an error: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 80)


if __name__ == "__main__":
    main() 