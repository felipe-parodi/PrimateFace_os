"""
train_with_coco.py

This script trains a DeepLabCut model on a COCO dataset.

Possible DLC Top-Down Models: 
    top_down_resnet_50, top_down_resnet_101, top_down_hrnet_w32, 
    top_down_hrnet_w48, rtmpose_s, rtmpose_m, rtmpose_x

Usage:

Single-GPU Training:
  python train_with_coco.py --model_type <model_type> --output_dir <output_dir> --num_epochs <epochs> --batch_size <size>

  Example (note: 'top_down' is not needed in the model type):
  python train_with_coco.py --model_type hrnet_w32 --output_dir ./my_trained_model --num_epochs 100 --batch_size 64

Resuming Training:
  python train_with_coco.py --resume <path_to_snapshot.pt> --output_dir <dir_to_save_new_snapshots>
"""


import json
import yaml
from pathlib import Path
import torch
import warnings
from collections import Counter
import numpy as np
import types
from collections import defaultdict
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import cv2
import argparse
import os
import random
from typing import Dict, List, Tuple, Optional, Any, Union
from deeplabcut.pose_estimation_pytorch.data.cocoloader import COCOLoader
from deeplabcut.pose_estimation_pytorch.data.dataset import PoseDataset
from deeplabcut.pose_estimation_pytorch.data.utils import apply_transform
from deeplabcut.pose_estimation_pytorch.data.transforms import build_transforms
from deeplabcut.pose_estimation_pytorch.models import PoseModel
from deeplabcut.pose_estimation_pytorch.runners import build_training_runner
from deeplabcut.pose_estimation_pytorch.config import (
    make_pytorch_pose_config,
    make_basic_project_config,
    update_config,
    read_config_as_dict,
)
from deeplabcut.pose_estimation_pytorch.task import Task
from torch.utils.data import DataLoader

KPT_THRESHOLD = 0.05

# Default paths (can be overridden with command-line arguments)
TRAIN_JSON_PATH = None
VAL_JSON_PATH = None
TEST_JSON_PATH = None


def list_available_models():
    """Prints a list of available model backbones."""
    from deeplabcut.pose_estimation_pytorch.config.utils import get_config_folder_path
    
    configs_dir = get_config_folder_path()
    backbones_dir = configs_dir / "backbones"
    
    print("✅ Available model backbones:")
    for path in sorted(backbones_dir.glob("*.yaml")):
        if path.stem != "dlcrnet_ms":  # Not a standalone backbone
            print(f"  - {path.stem}")
    print("-" * 80)


def _verbose_epoch(
    self,
    loader: torch.utils.data.DataLoader,
    mode: str = "train",
    display_iters: int = 500,
) -> float:
    """
    An overridden version of the TrainingRunner._epoch method
    that includes a tqdm progress bar.
    """
    self.model.train(mode=mode == "train")

    epoch_loss = []
    loss_metrics = defaultdict(list)

    is_main_process = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

    pbar = tqdm(loader, desc=f"Epoch {self.current_epoch} - {mode}", disable=not is_main_process)
    for i, batch in enumerate(pbar):
        losses_dict = self.step(batch, mode)
        if "total_loss" in losses_dict:
            total_loss_val = losses_dict["total_loss"]
            epoch_loss.append(total_loss_val)
            pbar.set_postfix_str(f"loss: {total_loss_val:.5f}")

        for key in losses_dict.keys():
            loss_metrics[key].append(losses_dict[key])

    perf_metrics = None
    if mode == "eval":
        perf_metrics = self._compute_epoch_metrics()
        self._metadata["metrics"] = perf_metrics
        self._epoch_predictions = {}
        self._epoch_ground_truth = {}

    if len(epoch_loss) > 0:
        epoch_loss_val = np.mean(epoch_loss).item()
    else:
        epoch_loss_val = 0

    self.history[f"{mode}_loss"].append(epoch_loss_val)

    metrics_to_log = {}
    if perf_metrics:
        for name, score in perf_metrics.items():
            if not isinstance(score, (int, float)):
                score = 0.0
            metrics_to_log[name] = score

    for key in loss_metrics:
        name = f"{mode}.{key}"
        val = float("nan")
        if np.sum(~np.isnan(loss_metrics[key])) > 0:
            val = np.nanmean(loss_metrics[key]).item()
        self._metadata["losses"][name] = val
        metrics_to_log[f"losses/{name}"] = val

    self.csv_logger.log(metrics_to_log, step=self.current_epoch)
    if self.logger:
        self.logger.log(metrics_to_log, step=self.current_epoch)

    return epoch_loss_val


def plot_losses(history: dict, output_dir: Path):
    """Plots training and validation losses and saves the plot."""
    plt.figure()
    plt.plot(history["train_loss"], label="Train Loss")
    if "eval_loss" in history and history["eval_loss"]:
        plt.plot(history["eval_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    save_path = output_dir / "loss_curve.png"
    plt.savefig(save_path)
    plt.close()
    print(f"📉 Loss curve saved to: {save_path}")


def visualize_predictions(
    model: PoseModel,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    num_images: int = 4,
):
    """Saves images with ground truth and predicted keypoints."""
    print(f"🎨 Visualizing predictions for {num_images} images...")
    model.eval()
    try:
        batch = next(iter(loader))
    except StopIteration:
        print("️️️⚠️ Could not get a batch from the test loader, skipping visualization.")
        return

    images = batch["image"][:num_images].to(device)

    with torch.no_grad():
        outputs = model(images)
        predictions = model.get_predictions(outputs)

    # Un-normalize images for viewing
    images_np = images.cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    images_unnormalized = []
    for img_arr in images_np:
        img = np.transpose(img_arr, (1, 2, 0))
        img = std * img + mean
        img = np.clip(img, 0, 1)
        images_unnormalized.append((img * 255).astype(np.uint8))

    keypoints_gt = batch["annotations"]["keypoints"][:num_images].numpy()
    scales = batch["scales"][:num_images].numpy()
    offsets = batch["offsets"][:num_images].numpy()

    # Convert predictions tensor to numpy array
    predictions_np = predictions["bodypart"]["poses"].cpu().numpy()

    for i in range(len(images_unnormalized)):
        # Create two copies for side-by-side comparison
        img_gt = images_unnormalized[i].copy()
        img_pred = images_unnormalized[i].copy()

        # Convert to BGR for OpenCV
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_RGB2BGR)
        img_pred = cv2.cvtColor(img_pred, cv2.COLOR_RGB2BGR)
        h, w, _ = img_gt.shape

        # Draw ground truth on the left image
        kpts_gt_img = keypoints_gt[i]
        for kpt in kpts_gt_img.reshape(-1, 3):
            if kpt[2] > 0:  # visibility flag
                cv2.circle(img_gt, (int(kpt[0]), int(kpt[1])), 5, (0, 255, 0), -1)

        # Draw predicted keypoints on the right image
        preds = predictions_np[i]

        # For debugging, print the max confidence to see what the model is predicting
        max_confidence = np.max(preds[..., 2]) if preds.size > 0 else 0
        print(f"Max prediction confidence for image {i}: {max_confidence:.4f}")

        for kpt in preds.reshape(-1, 3):
            if kpt[2] > KPT_THRESHOLD:
                pred_x = int(kpt[0])
                pred_y = int(kpt[1])
                cv2.circle(img_pred, (pred_x, pred_y), 5, (0, 0, 255), -1)

        # Add text labels
        cv2.putText(img_gt, 'Ground Truth', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img_pred, 'Prediction', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Combine images horizontally
        combined_img = np.hstack((img_gt, img_pred))

        save_path = output_dir / f"evaluation_comparison_{i}.png"
        cv2.imwrite(str(save_path), combined_img)
        print(f"🖼️ Saved comparison to {save_path}")


def filter_coco_data(
    images: list[dict], annotations: list[dict]
) -> tuple[list[dict], list[dict]]:
    """
    Filters COCO data to include only images and annotations that have keypoints.
    This prevents KeyErrors when an annotation has a bounding box but no keypoints.
    """
    if not annotations:
        return [], []

    # Filter annotations to only keep those that have a non-empty keypoints array.
    annotations_with_keypoints = [
        ann
        for ann in annotations
        if "keypoints" in ann and hasattr(ann["keypoints"], "size") and ann["keypoints"].size > 0
    ]

    # Now, get a set of image_ids from these valid annotations.
    image_ids_with_keypoints = {ann["image_id"] for ann in annotations_with_keypoints}

    # Filter the images to keep only those that have at least one valid annotation.
    filtered_images = [img for img in images if img["id"] in image_ids_with_keypoints]

    return filtered_images, annotations_with_keypoints


def _patched_snapshot_manager_update(self, epoch: int, state_dict: Dict[str, Any], last: bool = False):
    """A patched version of the snapshot manager update that handles FileExistsError on Windows."""
    metrics = state_dict["metadata"]["metrics"]
    if (
        self._key in metrics
        and not np.isnan(metrics[self._key])
        and (
            self._best_metric is None
            or (self.key_metric_asc and self._best_metric < metrics[self._key])
            or (not self.key_metric_asc and self._best_metric > metrics[self._key])
        )
    ):
        current_best = self.best()
        self._best_metric = metrics[self._key]

        # Save the new best model
        save_path = self.snapshot_path(epoch, best=True)
        parsed_state_dict = {
            k: v
            for k, v in state_dict.items()
            if self.save_optimizer_state or k != "optimizer"
        }
        torch.save(parsed_state_dict, save_path)

        # Handle previous best model
        if current_best is not None:
            if current_best.epochs % self.save_epochs == 0:
                new_name = self.snapshot_path(epoch=current_best.epochs)
                try:
                    current_best.path.rename(new_name)
                except FileExistsError:
                    # If the regular snapshot already exists, the "best" one is redundant.
                    current_best.path.unlink(missing_ok=True)
            else:
                current_best.path.unlink(missing_ok=False)
    elif last or epoch % self.save_epochs == 0:
        # Save regular snapshot if needed
        save_path = self.snapshot_path(epoch=epoch)
        parsed_state_dict = {
            k: v
            for k, v in state_dict.items()
            if self.save_optimizer_state or k != "optimizer"
        }
        torch.save(parsed_state_dict, save_path)

    # Clean up old snapshots if needed
    existing_snapshots = [s for s in self.snapshots() if not s.best]
    if len(existing_snapshots) >= self.max_snapshots:
        num_to_delete = len(existing_snapshots) - self.max_snapshots
        to_delete = existing_snapshots[:num_to_delete]
        for snapshot in to_delete:
            snapshot.path.unlink(missing_ok=False)


class PrimateFaceCOCOLoader(COCOLoader):
    """
    A custom COCO loader that accepts direct paths to JSON files,
    bypassing the need for a rigid project directory structure.
    """

    def __init__(
        self,
        model_config_path: str | Path,
        train_json_path: str | Path,
        val_json_path: str | Path,
        test_json_path: str | Path | None = None,
        image_dir: Optional[str] = None,
    ):
        if image_dir:
            image_root = Path(image_dir)
        else:
            # Fallback for absolute paths or paths relative to the JSON file
            image_root = Path(train_json_path).parent

        super(COCOLoader, self).__init__(
            project_root=Path(train_json_path).parent,
            image_root=image_root,
            model_config_path=Path(model_config_path),
        )

        self.train_json_filename = str(train_json_path)
        self.val_json_filename = str(val_json_path)
        self.test_json_filename = str(test_json_path) if test_json_path else None
        self._dataset_parameters = None

        with open(self.train_json_filename, "r") as f:
            self.train_json = json.load(f)
            if not isinstance(self.train_json, dict):
                raise ValueError("COCO datasets must be JSON objects (dicts).")
            self.train_json = self.validate_images(self.train_json)
            if "annotations" in self.train_json:
                for ann in self.train_json["annotations"]:
                    if "keypoints" in ann:
                        ann["keypoints"] = np.array(ann["keypoints"])
                    if "individual" not in ann:
                        ann["individual"] = "individual_0"  # Add default individual

        with open(self.val_json_filename, "r") as f:
            self.val_json = json.load(f)
            if not isinstance(self.val_json, dict):
                raise ValueError("COCO datasets must be JSON objects (dicts).")
            self.val_json = self.validate_images(self.val_json)
            if "annotations" in self.val_json:
                for ann in self.val_json["annotations"]:
                    if "keypoints" in ann:
                        ann["keypoints"] = np.array(ann["keypoints"])
                    if "individual" not in ann:
                        ann["individual"] = "individual_0"  # Add default individual

        self.test_json = None
        if self.test_json_filename:
            with open(self.test_json_filename, "r") as f:
                self.test_json = json.load(f)
                if not isinstance(self.test_json, dict):
                    raise ValueError("COCO datasets must be JSON objects (dicts).")
                self.test_json = self.validate_images(self.test_json)
                if "annotations" in self.test_json:
                    for ann in self.test_json["annotations"]:
                        if "keypoints" in ann:
                            ann["keypoints"] = np.array(ann["keypoints"])
                        if "individual" not in ann:
                            ann["individual"] = "individual_0"  # Add default individual

    def load_data(self, mode: str = "train") -> tuple[list, dict]:
        if mode == "train":
            data = self.train_json
        elif mode == "val":
            data = self.val_json
        elif mode == "test":    
            data = self.test_json
        else:
            raise AttributeError(f"Unknown mode: {mode}")

        if data is None:
            return [], {}

        return data["images"], data["annotations"]


class PrimateFacePoseDataset(PoseDataset):
    def apply_transform_all_keypoints(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        keypoints_unique: np.ndarray,
        bboxes: np.ndarray,
    ) -> dict[str, np.ndarray]:
        class_labels = [
            f"individual{i}_{bpt}"
            for i in range(len(keypoints))
            for bpt in self.parameters.bodyparts
        ] + [f"unique_{bpt}" for bpt in self.parameters.unique_bpts]

        all_keypoints = keypoints.reshape(-1, 3)
        if self.parameters.num_unique_bpts > 0:
            all_keypoints = np.concatenate([all_keypoints, keypoints_unique], axis=0)

        transformed = apply_transform(
            self.transform, image, all_keypoints, bboxes, class_labels=class_labels
        )
        if self.parameters.num_unique_bpts > 0:
            keypoints = transformed["keypoints"][
                : -self.parameters.num_unique_bpts
            ].reshape(*keypoints.shape)
            keypoints_unique = transformed["keypoints"][
                -self.parameters.num_unique_bpts :
            ]
            keypoints_unique = keypoints_unique.reshape(
                self.parameters.num_unique_bpts, 3
            )
        else:
            keypoints = transformed["keypoints"].reshape(*keypoints.shape)
            keypoints_unique = np.zeros((0,))

        transformed["keypoints"] = keypoints
        transformed["keypoints_unique"] = keypoints_unique
        transformed["bboxes"] = np.array(transformed.get("bboxes", []))
        if len(transformed["bboxes"]) == 0:
            transformed["bboxes"] = np.zeros((0, 4))

        return transformed


def create_model_config(
    train_json_path: str | Path,
    max_individuals: int,
    output_dir: Path,
    model_type: str,
    num_epochs: int,
) -> str:
    """
    Dynamically creates a pose_cfg.yaml file for training.
    It extracts necessary metadata (like keypoints) from the training JSON.
    """
    with open(train_json_path, "r") as f:
        train_data = json.load(f)

    # Extract metadata from COCO file
    try:
        category = train_data["categories"][0]
        keypoints = category["keypoints"]
        num_keypoints = len(keypoints)
    except (IndexError, KeyError) as e:
        raise ValueError(
            f"Could not extract categories or keypoints from {train_json_path}. "
            f"Ensure it is a valid COCO format. Original error: {e}"
        )

    # Create a dummy project config which is needed to create the model config
    project_config = make_basic_project_config(
        dataset_path=Path(train_json_path).parent,
        bodyparts=keypoints,
        max_individuals=max_individuals,
        multi_animal=max_individuals > 1,
    )

    # Create a permanent path for the config in the model's output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    # Using a more descriptive name to avoid conflicts if MODEL_TYPE changes
    pose_config_path = output_dir / f"pose_cfg_{model_type}.yaml"

    net_type = model_type
    if max_individuals > 1:
        # Use a top-down model for multi-animal projects, or a bottom-up like dlcrnet
        # For simplicity we choose top-down here.
        top_down = True
    else:
        top_down = False

    # Generate the full pose config
    pose_config = make_pytorch_pose_config(
        project_config=project_config,
        pose_config_path=pose_config_path,
        net_type=net_type,
        top_down=top_down,
        save=False,  # We will save it ourselves after updates
    )

    # The base config from `make_pytorch_pose_config` has runner-related params under `runner`,
    # but the runner builder expects them at the top level. So we flatten it.
    if "runner" in pose_config:
        runner_params = pose_config.pop("runner")
        pose_config.update(runner_params)

    # By default, DLC's make_pytorch_pose_config sets a very high number of epochs.
    # We override it here with the user-provided value. Other hyperparameters
    # will use the defaults from the model's config file.
    update_dict = {
        "train_settings": {
            "epochs": num_epochs,
        },
    }
    pose_config = update_config(pose_config, update_dict)

    # Write to the permanent file
    with open(pose_config_path, "w") as f:
        yaml.dump(pose_config, f)

    return str(pose_config_path)


def main():
    """
    Main function to load COCO data and start the training process.
    """
    parser = argparse.ArgumentParser(
        description="Train a DeepLabCut model on a COCO dataset."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="resnet_50",
        help="The model backbone to use for training.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./my_trained_model"),
        help="Directory to save trained models and logs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="The device to train on (e.g., 'cuda', 'cuda:0', 'cpu').",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a model snapshot (.pt) to resume training from.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of epochs to train for.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--train_json",
        type=str,
        default=None,
        help="Path to training COCO JSON file. If not provided, uses all data from --coco_json.",
    )
    parser.add_argument(
        "--val_json",
        type=str,
        default=None,
        help="Path to validation COCO JSON file. If not provided, uses all data from --coco_json.",
    )
    parser.add_argument(
        "--test_json",
        type=str,
        default=None,
        help="Path to test COCO JSON file. If not provided, uses all data from --coco_json.",
    )
    parser.add_argument(
        "--coco_json",
        type=str,
        default=None,
        help="Single COCO JSON file to use for train/val/test (will be split automatically).",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.15,
        help="Validation split ratio when using single COCO JSON file.",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.15,
        help="Test split ratio when using single COCO JSON file.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Path to the image directory. If not set, paths in COCO file are assumed to be absolute.",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device if "cuda" in args.device and torch.cuda.is_available() else "cpu"
    )

    list_available_models()

    # Determine JSON paths based on arguments
    if args.coco_json:
        if not Path(args.coco_json).exists():
            print(f"🛑 Error: COCO JSON file not found: {args.coco_json}")
            return

        print("⏳ Loading and splitting single COCO file...")
        with open(args.coco_json, 'r') as f:
            data = json.load(f)

        images = data['images']
        random.seed(42) # for reproducibility
        random.shuffle(images)

        # Calculate split points
        n_images = len(images)
        n_val = int(n_images * args.val_split)
        n_test = int(n_images * args.test_split)
        n_train = n_images - n_val - n_test

        # Split images
        train_images = images[:n_train]
        val_images = images[n_train : n_train + n_val]
        test_images = images[n_train + n_val :]

        # Get image IDs for each split
        train_ids = {img['id'] for img in train_images}
        val_ids = {img['id'] for img in val_images}
        test_ids = {img['id'] for img in test_images}

        # Split annotations based on image IDs
        train_anns = [ann for ann in data['annotations'] if ann['image_id'] in train_ids]
        val_anns = [ann for ann in data['annotations'] if ann['image_id'] in val_ids]
        test_anns = [ann for ann in data['annotations'] if ann['image_id'] in test_ids]

        # Create new COCO dicts for each split
        base_coco = {
            'info': data.get('info', {}),
            'licenses': data.get('licenses', []),
            'categories': data.get('categories', []),
        }
        
        train_coco = {**base_coco, 'images': train_images, 'annotations': train_anns}
        val_coco = {**base_coco, 'images': val_images, 'annotations': val_anns}
        test_coco = {**base_coco, 'images': test_images, 'annotations': test_anns}

        # Save split JSONs to output directory
        args.output_dir.mkdir(parents=True, exist_ok=True)
        train_json = args.output_dir / 'train_split.json'
        val_json = args.output_dir / 'val_split.json'
        test_json = args.output_dir / 'test_split.json'

        with open(train_json, 'w') as f:
            json.dump(train_coco, f)
        with open(val_json, 'w') as f:
            json.dump(val_coco, f)
        with open(test_json, 'w') as f:
            json.dump(test_coco, f)

        print(f"✅ Data split into:")
        print(f"  - Train: {len(train_images)} images, {len(train_anns)} annotations ({train_json})")
        print(f"  - Val:   {len(val_images)} images, {len(val_anns)} annotations ({val_json})")
        print(f"  - Test:  {len(test_images)} images, {len(test_anns)} annotations ({test_json})")

    elif args.train_json and args.val_json and args.test_json:
        # Use separate files for each split
        train_json = Path(args.train_json)
        val_json = Path(args.val_json)
        test_json = Path(args.test_json)
        
        if not train_json.exists():
            print(f"🛑 Error: Training JSON not found: {args.train_json}")
            return
        if not val_json.exists():
            print(f"🛑 Error: Validation JSON not found: {args.val_json}")
            return
        if not test_json.exists():
            print(f"🛑 Error: Test JSON not found: {args.test_json}")
            return
        print("✅ Using separate COCO files for train/val/test")
        
    elif TRAIN_JSON_PATH and VAL_JSON_PATH and TEST_JSON_PATH:
        # Fall back to module-level paths if they were set
        train_json = Path(TRAIN_JSON_PATH)
        val_json = Path(VAL_JSON_PATH)
        test_json = Path(TEST_JSON_PATH)
        
        if not train_json.exists() or not val_json.exists() or not test_json.exists():
            print("=" * 80)
            print("🛑 Error: Please provide COCO JSON paths using one of these methods:")
            print("   1. --coco_json path/to/file.json (uses same file for train/val/test)")
            print("   2. --train_json, --val_json, and --test_json (separate files)")
            print("=" * 80)
            return
    else:
        print("=" * 80)
        print("🛑 Error: Please provide COCO JSON paths using one of these methods:")
        print("   1. --coco_json path/to/file.json (uses same file for train/val/test)")
        print("   2. --train_json, --val_json, and --test_json (separate files)")
        print("=" * 80)
        return
    
    print("✅ Paths verified.")

    # Step 2: Create a model configuration file from your data
    print("⏳ Creating dynamic model configuration...")

    with open(train_json, "r") as f:
        train_data = json.load(f)

    # Determine max_individuals from training data annotations
    if "annotations" in train_data and train_data["annotations"]:
        image_ids = [ann["image_id"] for ann in train_data["annotations"]]
        counts = Counter(image_ids)
        max_individuals = max(counts.values()) if counts else 1
    else:
        max_individuals = 1

    model_config_file = create_model_config(
        train_json, max_individuals, args.output_dir, args.model_type, args.num_epochs
    )
    print(f"📄 Model config created at: {model_config_file}")

    # Step 3: Initialize the custom data loader
    print("⏳ Initializing CustomCOCOLoader...")
    loader = PrimateFaceCOCOLoader(
        model_config_path=model_config_file,
        train_json_path=train_json,
        val_json_path=val_json,
        test_json_path=test_json,
        image_dir=args.image_dir,
    )
    print("✅ Loader initialized.")

    # Step 4: Load and prepare datasets
    print("⏳ Loading and preparing datasets...")
    params = loader.get_dataset_parameters()
    model_cfg = read_config_as_dict(model_config_file)
    task = Task(model_cfg.get("method", "BU").upper())

    train_images, train_data = loader.load_data(mode="train")
    val_images, val_data = loader.load_data(mode="val")
    test_images, test_data = loader.load_data(mode="test")

    print("⏳ Filtering datasets to include only annotated images...")
    train_images, train_data = filter_coco_data(train_images, train_data)
    val_images, val_data = filter_coco_data(val_images, val_data)
    test_images, test_data = filter_coco_data(test_images, test_data)
    print("✅ Filtering complete.")

    train_transform = build_transforms(model_cfg["data"]["train"])
    val_transform = build_transforms(model_cfg["data"]["inference"])
    test_transform = build_transforms(model_cfg["data"]["inference"])

    train_dataset = PrimateFacePoseDataset(
        images=train_images,
        annotations=train_data,
        parameters=params,
        transform=train_transform,
        task=task,
    )
    val_dataset = PrimateFacePoseDataset(
        images=val_images,
        annotations=val_data,
        parameters=params,
        transform=val_transform,
        task=task,
    )
    test_dataset = PrimateFacePoseDataset(
        images=test_images,
        annotations=test_data,
        parameters=params,
        transform=test_transform,
        task=task,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,  # This is per-GPU batch size
        shuffle=True,
        pin_memory=True,
        num_workers=4,  # Increase workers to parallelize data loading
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,  # Increase workers to parallelize data loading
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,  # Increase workers to parallelize data loading
    )
    print(
        f"✅ Datasets ready. "
        f"Train: {len(train_dataset)} instances in {len(train_images)} images. "
        f"Validation: {len(val_dataset)} instances in {len(val_images)} images. "
        f"Test: {len(test_dataset)} instances in {len(test_images)} images."
    )

    # Step 5: Build the model, optimizer, and trainer
    print("⏳ Building model and trainer...")

    print(f"🧠 Using device: {device}")

    model = PoseModel.build(model_cfg["model"]).to(device)

    # Ensure the output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"💾 Model snapshots will be saved to: {args.output_dir.resolve()}")

    runner = build_training_runner(
        runner_config=model_cfg,
        model_folder=args.output_dir,
        task=task,
        model=model,
        device=device,
    )
    runner._epoch = types.MethodType(_verbose_epoch, runner)
    runner.snapshot_manager.update = types.MethodType(
        _patched_snapshot_manager_update, runner.snapshot_manager
    )

    # --- Resume from Checkpoint ---
    if args.resume:
        if not Path(args.resume).exists():
            print(f"🛑 Error: Checkpoint file not found at {args.resume}")
            return
        else:
            print(f"✅ Resuming training from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)

            # Load model state
            model.load_state_dict(checkpoint.get("model", checkpoint))

            # Load optimizer and scheduler states if they exist
            if "optimizer" in checkpoint and hasattr(runner, "optimizer"):
                runner.optimizer.load_state_dict(checkpoint["optimizer"])

            if "scheduler" in checkpoint and hasattr(runner, "scheduler"):
                runner.scheduler.load_state_dict(checkpoint["scheduler"])

            # Load training history and epoch
            if "history" in checkpoint:
                runner.history = checkpoint["history"]

            if "metadata" in checkpoint and "epochs" in checkpoint["metadata"]:
                start_epoch = checkpoint["metadata"]["epochs"]
                runner.current_epoch = start_epoch + 1
                print(f"▶️ Resuming from epoch {runner.current_epoch}")


    # Step 6: Start training
    total_epochs = model_cfg["train_settings"]["epochs"]
    start_epoch = runner.current_epoch  # This is 1 by default, or updated by resume logic

    print("\n" + "=" * 80)
    if start_epoch > 1:
        epochs_remaining = total_epochs - (start_epoch - 1)
        print(
            f"🚀 Resuming training from epoch {start_epoch}, running for {epochs_remaining} more epochs to reach a total of {total_epochs}."
        )
    else:
        print(f"🚀 Starting fresh training for {total_epochs} epochs.")
    print("=" * 80)
    runner.fit(
        train_loader=train_loader,
        valid_loader=val_loader,
        epochs=total_epochs,
        display_iters=model_cfg["train_settings"]["display_iters"],
    )

    print("\n" + "=" * 80)
    print("🎉 Training finished!")
    print("=" * 80)

    # Step 7: Post-training evaluation and visualization
    plot_losses(runner.history, args.output_dir)
    visualize_predictions(runner.model, test_loader, device, args.output_dir)


if __name__ == "__main__":
    main() 