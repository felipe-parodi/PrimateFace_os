"""Evaluate model performance across different primate genera.

This script analyzes model predictions by genus, computing per-genus metrics
to identify strengths and weaknesses across different primate groups.

Example:
    python eval_genera.py predictions.json annotations.json --output genus_metrics.json
"""

import json
import os
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import cv2
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import polars as pl
import torch
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_topdown, init_model as init_pose_estimator
from mmpose.evaluation.metrics import NME
from mmpose.utils import adapt_mmdet_pipeline
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

# --- Configuration --
LABEL_CSV = "path/to/genera.csv"
COCO_JSON = "path/to/coco.json"
IMAGES_DIR = "path/to/images"
OUTPUT_DIR = "path/to/output"

# Model Paths
DET_CONFIG = "path/to/det_config.py"
DET_CHECKPOINT = "path/to/det_checkpoint.pth"
POSE_CONFIG = "path/to/pose_config.py"
POSE_CHECKPOINT = "path/to/pose_checkpoint.pth"

# Evaluation Parameters
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BBOX_THR = 0.3
NUM_KEYPOINTS = 68

# Plotting parameters
PLT_FIGSIZE = (12, 7)
DEFAULT_COLOR = "grey"


def load_data(label_csv_path: str, coco_json_path: str) -> Tuple[Dict[str, str], Dict[str, Any], Dict[int, List[Dict[str, Any]]], List[str]]:
    """Loads genus-superfamily mapping, COCO annotations, and superfamily list.
    
    Args:
        label_csv_path: Path to CSV file with genus-superfamily mapping
        coco_json_path: Path to COCO annotations JSON file
        
    Returns:
        Tuple containing:
            - genus_map: Mapping from genus to superfamily
            - coco_data: COCO annotations dictionary
            - gt_annos_map: Mapping from image ID to annotations
            - superfamilies: List of unique superfamily names
    """
    print("Loading data...")
    df = pl.read_csv(label_csv_path)
    genus_map = {
        row["Genus"].lower(): row["Superfamily"].lower()
        for row in df.to_dicts() if row["Genus"] and row["Superfamily"]
    }

    superfamilies = df["Superfamily"].unique().drop_nulls().to_list()

    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    gt_annos_map = {img["id"]: [] for img in coco_data["images"]}
    for ann in coco_data.get("annotations", []):
        gt_annos_map[ann["image_id"]].append(ann)

    print("Data loaded successfully.")
    return genus_map, coco_data, gt_annos_map, superfamilies


def filter_and_group_images(coco_data: Dict[str, Any], genus_map: Dict[str, str], images_dir_path: str) -> Dict[str, Dict[str, List[int]]]:
    """Filters images and groups them by superfamily and genus.
    
    Args:
        coco_data: COCO annotations dictionary
        genus_map: Mapping from genus to superfamily
        images_dir_path: Path to directory containing image folders
        
    Returns:
        Nested dictionary: superfamily -> genus -> list of image IDs
    """
    print("Filtering and grouping images...")
    images_dir = Path(images_dir_path)
    valid_genera_dirs = {p.name.lower() for p in images_dir.iterdir() if p.is_dir()}

    grouped_images = {}
    for img_info in tqdm(coco_data["images"], desc="Filtering Images"):
        try:
            genus_name = Path(img_info["file_name"]).parent.name.lower()
            if genus_name in valid_genera_dirs:
                superfamily = genus_map.get(genus_name, "unknown")
                grouped_images.setdefault(superfamily, {}).setdefault(genus_name, []).append(img_info["id"])
        except IndexError:
            continue

    print(f"Found {sum(len(ids) for g in grouped_images.values() for ids in g.values())} images across {len(grouped_images)} superfamilies.")
    return grouped_images


def initialize_models(det_config: str, det_ckpt: str, pose_config: str, pose_ckpt: str, device: str) -> Tuple[Any, Any]:
    """Initializes MMDetection and MMPose models.
    
    Args:
        det_config: Path to detection model config
        det_ckpt: Path to detection model checkpoint
        pose_config: Path to pose model config
        pose_ckpt: Path to pose model checkpoint
        device: Device to load models on (e.g., 'cuda:0', 'cpu')
        
    Returns:
        Tuple of (detector model, pose estimator model)
    """
    print("Initializing models...")
    detector = init_detector(det_config, det_ckpt, device=device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    pose_estimator = init_pose_estimator(pose_config, pose_ckpt, device=device)
    print("Models initialized.")
    return detector, pose_estimator


def run_evaluation(
    grouped_images: Dict[str, Dict[str, List[int]]],
    coco_data: Dict[str, Any],
    gt_annos_map: Dict[int, List[Dict[str, Any]]],
    detector: Any,
    pose_estimator: Any
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Runs inference and evaluates mAP and NME for each group.
    
    Args:
        grouped_images: Images grouped by superfamily and genus
        coco_data: COCO annotations dictionary
        gt_annos_map: Mapping from image ID to annotations
        detector: Initialized detection model
        pose_estimator: Initialized pose estimation model
        
    Returns:
        Nested dictionary: superfamily -> genus -> metrics dict
    """
    print("Starting evaluation...")
    results = {}
    id2img = {img["id"]: img for img in coco_data["images"]}
    dataset_meta = pose_estimator.dataset_meta

    for superfamily, genera in grouped_images.items():
        results[superfamily] = {}
        for genus, image_ids in genera.items():
            print(f"\nEvaluating {superfamily} -> {genus} ({len(image_ids)} images)")
            
            # Prepare data for the current genus
            gt_subset = {
                'images': [img for img in coco_data['images'] if img['id'] in image_ids],
                'annotations': [ann for ann in coco_data['annotations'] if ann['image_id'] in image_ids],
                'categories': coco_data['categories']
            }
            det_predictions = []
            pose_data_samples = []

            # Initialize NME Metric
            nme_metric = NME(norm_mode='use_norm_item', norm_item='bbox_size', prefix='pose')
            nme_metric.dataset_meta = dataset_meta

            for img_id in tqdm(image_ids, desc=f"Genus {genus}"):
                img_info = id2img[img_id]
                if not Path(img_info["file_name"]).exists():
                    warnings.warn(f"Image not found, skipping: {img_info['file_name']}")
                    continue

                image = mmcv.imread(img_info["file_name"])
                
                # Detection Inference
                det_result = inference_detector(detector, image).pred_instances
                valid_idx = det_result.scores > BBOX_THR
                
                for bbox, score, label in zip(det_result.bboxes[valid_idx], det_result.scores[valid_idx], det_result.labels[valid_idx]):
                    x1, y1, x2, y2 = bbox.cpu().numpy()
                    det_predictions.append({
                        "image_id": img_id,
                        "category_id": coco_data['categories'][label.item()]['id'],
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": score.item(),
                    })

                # Pose Inference
                gt_anns_with_kpts = [ann for ann in gt_annos_map.get(img_id, []) if sum(ann.get("keypoints", [])) > 0]
                if not gt_anns_with_kpts:
                    continue
                
                gt_bboxes_for_pose = np.array([ann["bbox"] for ann in gt_anns_with_kpts])
                gt_bboxes_for_pose[:, 2:] += gt_bboxes_for_pose[:, :2] # to xyxy
                
                pose_results = inference_topdown(pose_estimator, image, gt_bboxes_for_pose)
                
                for i, pose_result in enumerate(pose_results):
                    pred_kpts_data = pose_result.pred_instances.keypoints
                    if isinstance(pred_kpts_data, torch.Tensor):
                        pred_kpts_np = pred_kpts_data.cpu().numpy()
                    else:
                        pred_kpts_np = np.array(pred_kpts_data)

                    gt_kpts = np.array(gt_anns_with_kpts[i]['keypoints']).reshape(1, NUM_KEYPOINTS, 3)
                    pose_data_samples.append({
                        'pred_instances': {'keypoints': pred_kpts_np},
                        'gt_instances': {
                            'keypoints': gt_kpts[:, :, :2],
                            'keypoints_visible': gt_kpts[:, :, 2],
                            'bboxes': np.array([gt_anns_with_kpts[i]['bbox']])
                        }
                    })

            # Compute Metrics for Genus
            genus_results = {}
            with tempfile.TemporaryDirectory() as tmpdir:
                gt_path = Path(tmpdir) / "gt.json"
                pred_path = Path(tmpdir) / "pred.json"
                
                with open(gt_path, 'w') as f: json.dump(gt_subset, f)
                with open(pred_path, 'w') as f: json.dump(det_predictions, f)
                
                if det_predictions:
                    coco_gt = COCO(str(gt_path))
                    coco_dt = coco_gt.loadRes(str(pred_path))
                    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
                    coco_eval.evaluate()
                    coco_eval.accumulate()
                    coco_eval.summarize()
                    genus_results['mAP'] = coco_eval.stats[0]
                    genus_results['mAP_50'] = coco_eval.stats[1]

            if pose_data_samples:
                nme_metric.process(data_batch=None, data_samples=pose_data_samples)
                eval_results = nme_metric.compute_metrics(nme_metric.results)
                genus_results['NME'] = eval_results.get('NME', float('nan'))

            results[superfamily][genus] = genus_results

    return results

def plot_results(results: Dict[str, Dict[str, Dict[str, float]]], output_dir: str, superfamily_colors: Dict[str, np.ndarray]) -> None:
    """Generates and saves bar plots for the evaluation results.
    
    Args:
        results: Nested dictionary with evaluation metrics
        output_dir: Directory to save plots
        superfamily_colors: Mapping from superfamily to color
    """
    print("Generating plots...")
    Path(output_dir).mkdir(exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['svg.fonttype'] = 'none'

    # Flatten results for plotting
    flat_results = []
    for sf, genera in results.items():
        for genus, metrics in genera.items():
            flat_results.append({'genus': genus, 'superfamily': sf, **metrics})
    if not flat_results:
        print("No results to plot.")
        return
        
    df = pl.DataFrame(flat_results)

    # mAP Plot
    df_map = df.filter(pl.col('mAP').is_not_null()).sort('mAP', descending=True)
    if not df_map.is_empty():
        fig_map, ax_map = plt.subplots(figsize=PLT_FIGSIZE)
        colors = [superfamily_colors.get(sf, DEFAULT_COLOR) for sf in df_map['superfamily']]
        ax_map.bar(df_map['genus'], df_map['mAP'], color=colors)
        ax_map.set_ylabel('mAP (bbox)')
        ax_map.set_title('Detection Performance (mAP) by Genus')
        ax_map.tick_params(axis='x', labelrotation=90)
        ax_map.set_ylim(0, 1)
        
        sorted_colors = sorted(superfamily_colors.items())
        handles = [plt.Rectangle((0,0),1,1, color=c) for sf, c in sorted_colors if sf != "unknown"]
        labels = [sf.capitalize() for sf, c in sorted_colors if sf != "unknown"]
        ax_map.legend(handles, labels, title="Superfamily")

        fig_map.tight_layout()
        fig_map.savefig(Path(output_dir) / "mAP_by_genus.pdf", bbox_inches='tight')
        print(f"Saved mAP plot to {Path(output_dir) / 'mAP_by_genus.pdf'}")

    # NME Plot
    df_nme = df.filter(pl.col('NME').is_not_nan()).sort('NME', descending=False)
    if not df_nme.is_empty():
        fig_nme, ax_nme = plt.subplots(figsize=PLT_FIGSIZE)
        colors = [superfamily_colors.get(sf, DEFAULT_COLOR) for sf in df_nme['superfamily']]
        ax_nme.bar(df_nme['genus'], df_nme['NME'], color=colors)
        ax_nme.set_ylabel('NME (lower is better)')
        ax_nme.set_title('Pose Estimation Performance (NME) by Genus')
        ax_nme.tick_params(axis='x', labelrotation=90)

        sorted_colors = sorted(superfamily_colors.items())
        handles = [plt.Rectangle((0,0),1,1, color=c) for sf, c in sorted_colors if sf != "unknown"]
        labels = [sf.capitalize() for sf, c in sorted_colors if sf != "unknown"]
        ax_nme.legend(handles, labels, title="Superfamily")

        fig_nme.tight_layout()
        fig_nme.savefig(Path(output_dir) / "NME_by_genus.pdf", bbox_inches='tight')
        print(f"Saved NME plot to {Path(output_dir) / 'NME_by_genus.pdf'}")
        
    plt.close('all')


def main() -> None:
    """Main execution pipeline."""
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    genus_map, coco_data, gt_annos_map, superfamilies = load_data(LABEL_CSV, COCO_JSON)
    
    # Generate color map for plotting
    cmap = plt.get_cmap('tab20')
    colors = cmap(np.linspace(0, 1, len(superfamilies)))
    superfamily_colors = {sf.lower(): color for sf, color in zip(superfamilies, colors)}
    superfamily_colors["unknown"] = DEFAULT_COLOR

    grouped_images = filter_and_group_images(coco_data, genus_map, IMAGES_DIR)
    detector, pose_estimator = initialize_models(DET_CONFIG, DET_CHECKPOINT, POSE_CONFIG, POSE_CHECKPOINT, DEVICE)
    results = run_evaluation(grouped_images, coco_data, gt_annos_map, detector, pose_estimator)

    results_path = Path(OUTPUT_DIR) / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved detailed results to {results_path}")

    plot_results(results, OUTPUT_DIR, superfamily_colors)
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
