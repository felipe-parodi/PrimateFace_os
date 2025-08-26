"""
Compare primate face detection across frameworks. 

This script compares the performance of 
    Primate face detection with MMDetection, 
    Human face detection with MMDetection, and 
    GroundingDINO - prompted with "face" - on a set of images.

Note: Inference with GroundingDINO requires a separate python=3.10 environment.

"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Union

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from tqdm import tqdm

# Default paths for models
MMDET_PRIMATE_CONFIG_DEFAULT = "./path/to/mmdet-face-detection-config.py"
MMDET_PRIMATE_CHECKPOINT_DEFAULT = "./path/to/mmdet-face-detection-checkpoint.pth"
MMDET_HUMAN_CONFIG_DEFAULT = "./path/to/mmdet-human-detection-config.py"
MMDET_HUMAN_CHECKPOINT_DEFAULT = "./path/to/mmdet-human-detection-checkpoint.pth"

# Define GroundingDINO ontology
DINO_ONTOLOGY_DICT = {
    "face": "face, head",
}
DINO_COLOR_MAP = {
    "face": (0, 0, 255),          # Red
}
MMDET_CLASS_NAMES = ['face']
MMDET_PRIMATE_COLOR_MAP = {'face': (255, 0, 0)}  # Blue
MMDET_HUMAN_COLOR_MAP = {'face': (0, 255, 0)}  # Green
DEFAULT_COLOR = (128, 128, 128) # Gray

def draw_mpl_boxes(
    ax: plt.Axes,
    detections: object,
    model_type: str,
    color_map: dict,
    threshold: float,
    ontology_keys: List[str] = [],
    mmdet_class_names: List[str] = [],
):
    """Helper function to draw bounding boxes on a matplotlib Axes.
    
    Args:
        ax: matplotlib Axes object to draw on
        detections: detection results from the model
        model_type: type of model ('dino' or 'mmdet')
        color_map: dictionary mapping class names to colors
        threshold: confidence threshold for drawing boxes
        ontology_keys: list of class names to use for GroundingDINO
        mmdet_class_names: list of class names to use for MMDetection
        
    Returns:
        None
    """
    if model_type == 'dino' and hasattr(detections, 'xyxy'):
        # Process GroundingDINO sv.Detections object
        for i in range(len(detections.xyxy)):
            confidence = detections.confidence[i] if detections.confidence is not None else 1.0
            if confidence >= threshold:
                box = detections.xyxy[i].astype(int)
                class_id = detections.class_id[i] if detections.class_id is not None else -1

                label = "Unknown"
                if 0 <= class_id < len(ontology_keys):
                    label = ontology_keys[class_id]
                elif detections.data and 'class_name' in detections.data and i < len(detections.data['class_name']):
                    label = detections.data['class_name'][i]

                color_bgr = color_map.get(label, DEFAULT_COLOR)
                color_mpl = (color_bgr[2]/255, color_bgr[1]/255, color_bgr[0]/255)
                
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                rect = patches.Rectangle((x1, y1), w, h, linewidth=1.5, edgecolor=color_mpl, facecolor='none')
                ax.add_patch(rect)
                
                label_text = f"{label}: {confidence:.2f}"
                ax.text(x1, y1 - 5, label_text, color='black', fontsize=8,
                        bbox=dict(facecolor=color_mpl, alpha=0.6, pad=1, edgecolor='none'))

    elif model_type == 'mmdet':
        # Handle DetDataSample format
        if 'DetDataSample' in str(type(detections)):
            pred_instances = detections.pred_instances
            if len(pred_instances) == 0:
                return

            bboxes = pred_instances.bboxes.cpu().numpy()
            scores = pred_instances.scores.cpu().numpy()
            labels_idx = pred_instances.labels.cpu().numpy()

            for i in range(len(bboxes)):
                if scores[i] < threshold:
                    continue
                box = bboxes[i].astype(int)
                label_index = labels_idx[i]

                if label_index >= len(mmdet_class_names):
                    continue

                label = mmdet_class_names[label_index]
                color_bgr = color_map.get(label, DEFAULT_COLOR)
                color_mpl = (color_bgr[2]/255, color_bgr[1]/255, color_bgr[0]/255)
                
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                rect = patches.Rectangle((x1, y1), w, h, linewidth=1.5, edgecolor=color_mpl, facecolor='none')
                ax.add_patch(rect)

                label_text = f"{label}: {scores[i]:.2f}"
                ax.text(x1, y1 - 5, label_text, color='black', fontsize=8,
                        bbox=dict(facecolor=color_mpl, alpha=0.6, pad=1, edgecolor='none'))
        else:
            print(f"  Warning: MMDetection result type ({type(detections)}) not handled in draw_mpl_boxes. Skipping draw.")


def get_image_paths(args: argparse.Namespace) -> List[Path]:
    """Gets a list of image paths based on input arguments.
    
    Args:
        args: argparse.Namespace containing command line arguments
    
    Returns:
        List[Path]: List of image paths
    """
    image_paths = []
    if args.seed is not None:
        random.seed(args.seed)

    if args.image_dir:
        image_dir = Path(args.image_dir)
        if not image_dir.is_dir():
            raise ValueError(f"Image directory not found: {args.image_dir}")
        all_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpeg'))
        if not all_files:
            raise ValueError(f"No image files found in: {args.image_dir}")
        num_to_sample = min(10, len(all_files))
        image_paths = random.sample(all_files, num_to_sample)
        print(f"Selected {len(image_paths)} random images from {args.image_dir}")

    elif args.coco:
        coco_json_path = Path(args.coco)
        if not coco_json_path.is_file():
            raise ValueError(f"COCO JSON not found: {args.coco}")
        coco_dir = coco_json_path.parent
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)

        if 'images' not in coco_data or not coco_data['images']:
            raise ValueError(f"No 'images' found in {args.coco}")

        all_image_info = coco_data['images']
        num_to_sample = min(20, len(all_image_info))
        selected_image_info = random.sample(all_image_info, num_to_sample)

        image_paths = []
        missing_files = []
        for img_info in selected_image_info:
            filename = img_info.get('file_name')
            if filename:
                filename_path = Path(filename)
                if filename_path.is_absolute():
                    img_path = filename_path
                else:
                    img_path = coco_dir / filename_path
                
                if img_path.exists() and img_path.is_file():
                    image_paths.append(img_path)
                else:
                    missing_files.append(str(img_path))
            else:
                print("Warning: Image info in COCO JSON missing 'file_name'.")
        
        if missing_files:
            print(f"Warning: Could not find the following image files specified in {args.coco}: {', '.join(missing_files)}")

        if not image_paths:
            raise ValueError(f"Could not find any valid and existing image files based on {args.coco}")
        print(f"Selected {len(image_paths)} images based on {args.coco}")

    else:
        raise ValueError("Either --image-dir or --coco-json must be provided.")

    return image_paths


def process_mmdet_results(
    results_raw: Union[list, 'DetDataSample'],
    score_thr: float,
    nms_thr: float,
    device: str
) -> 'DetDataSample':
    """Applies confidence thresholding and NMS to raw MMDetection results.
    
    Args:
        results_raw: raw MMDetection results
        score_thr: confidence threshold for filtering
        nms_thr: NMS IoU threshold
        device: device to run inference on
        
    Returns:
        DetDataSample: processed MMDetection results
    """
    from mmengine.structures import InstanceData
    from mmdet.structures import DetDataSample
    from mmpose.evaluation.functional import nms

    if 'DetDataSample' in str(type(results_raw)):
        pred_instances = results_raw.pred_instances
        keep_conf = pred_instances.scores >= score_thr
        pred_instances_conf_filtered = pred_instances[keep_conf]

        if len(pred_instances_conf_filtered) > 0:
            bboxes_np = pred_instances_conf_filtered.bboxes.cpu().numpy()
            scores_np = pred_instances_conf_filtered.scores.cpu().numpy()
            nms_input = np.hstack((bboxes_np, scores_np[:, None]))
            
            if isinstance(nms_input, np.ndarray) and nms_input.ndim == 2 and nms_input.shape[1] == 5:
                keep_nms_indices = nms(nms_input, nms_thr)
                keep_nms_tensor = torch.from_numpy(np.array(keep_nms_indices)).long().to(pred_instances_conf_filtered.bboxes.device)
                pred_instances_final = pred_instances_conf_filtered[keep_nms_tensor]
                results_raw.pred_instances = pred_instances_final
            else: # Empty after NMS or bad input
                results_raw.pred_instances = InstanceData(
                    bboxes=torch.empty((0, 4), device=device),
                    scores=torch.empty((0,), device=device),
                    labels=torch.empty((0,), dtype=torch.long, device=device)
                )
        else: # Empty after confidence filter
            results_raw.pred_instances = InstanceData(
                bboxes=torch.empty((0, 4), device=device),
                scores=torch.empty((0,), device=device),
                labels=torch.empty((0,), dtype=torch.long, device=device)
            )
        return results_raw
    
    print(f"    Warning: Unexpected MMDetection result type ({type(results_raw)}). Returning empty results.")
    empty_instances = InstanceData(
        bboxes=torch.empty((0, 4), device=device),
        scores=torch.empty((0,), device=device),
        labels=torch.empty((0,), dtype=torch.long, device=device)
    )
    results_out = DetDataSample()
    results_out.pred_instances = empty_instances
    return results_out


def main():
    parser = argparse.ArgumentParser(description="Compare Primate MMDetection, Human MMDetection, and GroundingDINO models.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image-dir", type=str, help="Path to the directory containing images.")
    group.add_argument("--coco", type=str, help="Path to the COCO-formatted JSON annotation file.")

    # Primate MMDetection Model
    parser.add_argument("--mmdet-primate-config", type=str, default=MMDET_PRIMATE_CONFIG_DEFAULT, help="Path to Primate MMDetection config file.")
    parser.add_argument("--mmdet-primate-checkpoint", type=str, default=MMDET_PRIMATE_CHECKPOINT_DEFAULT, help="Path to Primate MMDetection checkpoint file.")

    # Human MMDetection Model
    parser.add_argument("--mmdet-human-config", type=str, default=MMDET_HUMAN_CONFIG_DEFAULT, help="Path to Human MMDetection config file.")
    parser.add_argument("--mmdet-human-checkpoint", type=str, default=MMDET_HUMAN_CHECKPOINT_DEFAULT, help="Path to Human MMDetection checkpoint file.")

    # General arguments
    parser.add_argument("--threshold", type=float, default=0.75, help="Confidence threshold for all models.")
    parser.add_argument("--nms-thr", type=float, default=0.3, help="NMS IoU threshold for MMDetection models.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for inference (e.g., 'cuda:0', 'cpu').")
    parser.add_argument("--output", type=str, default=None, help="Optional output directory for the plot. Defaults to the input image/JSON directory.")
    parser.add_argument("--output-format", type=str, default="svg", choices=['svg', 'jpg', 'png'], help="Output image format.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for image selection to ensure reproducibility.")
    parser.add_argument("--model-type", type=str, default="all", choices=['all', 'mmdet', 'dino'], help="Specify which type of models to run.")

    args = parser.parse_args()
    
    if args.model_type in ['all', 'mmdet']:
        from mmdet.apis import inference_detector, init_detector
        from mmpose.utils import adapt_mmdet_pipeline
        
    # TODO: Add environment support for GroundingDINO
    if args.model_type in ['all', 'dino']:
        from autodistill_grounding_dino import GroundingDINO
        from autodistill.detection import CaptionOntology

    if args.output:
        output_dir = Path(args.output)
    else:
        input_path = Path(args.image_dir if args.image_dir else args.coco)
        output_dir = input_path.parent

    output_filename = f"{args.model_type}_{args.seed}.{args.output_format}"
    args.output = str(output_dir / output_filename)

    print(f"Output will be saved to: {args.output}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu'
    print(f"Using device: {device}")

    try:
        image_paths = get_image_paths(args)
        if not image_paths:
            print("No images found or selected for processing.")
            return
    except (ValueError, FileNotFoundError) as e:
        print(f"Error selecting images: {e}")
        return

    print("Initializing models...")
    mm_primate_detector, mm_human_detector, dino_model = None, None, None
    try:
        if args.model_type in ['all', 'mmdet']:
            # Primate MMDetection
            mm_primate_detector = init_detector(args.mmdet_primate_config, args.mmdet_primate_checkpoint, device=device)
            mm_primate_detector.cfg = adapt_mmdet_pipeline(mm_primate_detector.cfg)
            print("Primate MMDetection model initialized.")
            
            # Human MMDetection
            mm_human_detector = init_detector(args.mmdet_human_config, args.mmdet_human_checkpoint, device=device)
            mm_human_detector.cfg = adapt_mmdet_pipeline(mm_human_detector.cfg)
            print("Human MMDetection model initialized.")

        if args.model_type in ['all', 'dino']:
            # GroundingDINO
            dino_ontology = CaptionOntology(DINO_ONTOLOGY_DICT)
            dino_model = GroundingDINO(ontology=dino_ontology)
            print("GroundingDINO model initialized.")
    except Exception as e:
        print(f"Error initializing models: {e}")
        return

    num_images = len(image_paths)
    if num_images == 0: return

    if args.output_format == 'svg':
        plt.rcParams['svg.fonttype'] = 'none'

    column_setups = {
        'all': {'count': 4, 'figsize_multiplier': 4},
        'mmdet': {'count': 3, 'figsize_multiplier': 3.5},
        'dino': {'count': 2, 'figsize_multiplier': 3}
    }
    setup = column_setups[args.model_type]
    fig, axes = plt.subplots(num_images, setup['count'], figsize=(setup['figsize_multiplier'] * 6, num_images * 4.5))
    axes = np.atleast_2d(axes)

    print(f"\nProcessing {num_images} images for '{args.model_type}' models...")

    for i, img_path in enumerate(tqdm(image_paths, desc="Processing Images")):
        print(f"\n[{i+1}/{num_images}] Processing: {img_path.name}")
        
        current_axes = { 'orig': axes[i, 0] }
        if args.model_type == 'all':
            current_axes.update({'primate': axes[i, 1], 'dino': axes[i, 2], 'human': axes[i, 3]})
        elif args.model_type == 'mmdet':
            current_axes.update({'primate': axes[i, 1], 'human': axes[i, 2]})
        elif args.model_type == 'dino':
            current_axes.update({'dino': axes[i, 1]})

        try:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                raise IOError(f"Failed to load image {img_path.name}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"  Error loading image: {e}")
            for ax in current_axes.values():
                ax.text(0.5, 0.5, 'Image Load Error', ha='center', va='center')
                ax.axis('off')
            continue

        # Column 1: Original Image
        current_axes['orig'].imshow(img_rgb)
        current_axes['orig'].set_title(f"{img_path.name}", fontsize=9)
        current_axes['orig'].axis('off')

        # Primate MMDetection
        if mm_primate_detector:
            try:
                ax = current_axes['primate']
                print("  Running Primate MMDet...")
                results_raw = inference_detector(mm_primate_detector, img_bgr)
                processed_results = process_mmdet_results(results_raw, args.threshold, args.nms_thr, device)
                ax.imshow(img_rgb.copy())
                draw_mpl_boxes(ax, processed_results, 'mmdet', MMDET_PRIMATE_COLOR_MAP, args.threshold, mmdet_class_names=MMDET_CLASS_NAMES)
                ax.set_title(f"Primate MMDet\n(Thr:{args.threshold}, NMS:{args.nms_thr})", fontsize=9)
            except Exception as e:
                print(f"    Error during Primate MMDet inference: {e}")
            finally:
                ax.axis('off')

        # GroundingDINO
        if dino_model:
            try:
                ax = current_axes['dino']
                print("  Running GroundingDINO...")
                dino_results = dino_model.predict(str(img_path))
                ax.imshow(img_rgb.copy())
                draw_mpl_boxes(ax, dino_results, 'dino', DINO_COLOR_MAP, args.threshold, ontology_keys=list(DINO_ONTOLOGY_DICT.keys()))
                ax.set_title(f"GroundingDINO\n(Thr:{args.threshold})", fontsize=9)
            except Exception as e:
                print(f"    Error during GroundingDINO inference: {e}")
                ax.imshow(img_rgb)
            finally:
                ax.axis('off')

        # Human MMDetection
        if mm_human_detector:
            try:
                ax = current_axes['human']
                print("  Running Human MMDet...")
                results_raw = inference_detector(mm_human_detector, img_bgr)
                processed_results = process_mmdet_results(results_raw, args.threshold, args.nms_thr, device)
                ax.imshow(img_rgb.copy())
                draw_mpl_boxes(ax, processed_results, 'mmdet', MMDET_HUMAN_COLOR_MAP, args.threshold, mmdet_class_names=MMDET_CLASS_NAMES)
                ax.set_title(f"Human MMDet\n(Thr:{args.threshold}, NMS:{args.nms_thr})", fontsize=9)
            except Exception as e:
                print(f"    Error during Human MMDet inference: {e}")
            finally:
                ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.suptitle(f"Detection Model Comparison: {args.model_type.upper()} (Seed: {args.seed})", fontsize=16)
    print(f"\nSaving plot to {args.output}...")
    plt.savefig(args.output, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print("Done.")

if __name__ == "__main__":
    main()
