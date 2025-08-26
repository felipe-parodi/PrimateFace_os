"""
Compare multiple pose estimation models on the same video frames.

This script:
1. Extracts N evenly spaced frames from a video
2. Runs multiple pose models on each frame
3. Creates a comparison grid showing all results
4. Saves the visualization to a single image

Usage:
    python compare_pose_models.py --input VIDEO_PATH [--output OUTPUT_PATH] [--num-frames N]
"""

import os
import cv2
import numpy as np
import argparse
import json
import random
from typing import List, Tuple, Dict, Optional, Any, Union
import mmcv
from mmdet.apis import inference_detector, init_detector
from mmdet.structures import DetDataSample
from mmpose.apis import inference_topdown, init_model
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples, PoseDataSample
from mmpose.utils import adapt_mmdet_pipeline
from viz_utils import FastPoseVisualizer
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


# Matplotlib specific drawing parameters
BBOX_COLOR_MPL = '#66B2FF'          # Sky Blue
KEYPOINT_COLOR_MPL = '#FFFF00'      # Yellow
SKELETON_COLOR_MPL = '#FF00FF'      # Magenta
KEYPOINT_SIZE_MPL = 5               # Size for scatter plot keypoints (points^2)
LINE_THICKNESS_MPL = 1.0
BBOX_LINE_THICKNESS_MPL = 1.0
DEFAULT_FIG_DPI = 300


# Common detection model config
DET_CONFIG = "./path/to/det_config.py"
DET_CHECKPOINT = "./path/to/det_checkpoint.pth"

# Pose models configs
POSE_MODELS = {
    "pf_hrnet": {
        "config": "./path/to/primate_pose_config.py",
        "checkpoint": "./path/to/primate_pose_checkpoint.pth"
    },
    "human_hrnet": {
        "config": "./path/to/human_pose_config.py",
        "checkpoint": "./path/to/human_pose_checkpoint.pth"
    } 
}


def extract_frames(video_path: str, num_frames: int = 3) -> List[np.ndarray]:
    """Extract evenly spaced frames from video.
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract

    Returns:
        List[np.ndarray]: List of extracted frames
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame indices
    if num_frames == 1:
        # For single frame, take middle frame
        indices = [total_frames // 2]
    else:
        # For multiple frames, space evenly
        indices = [i * (total_frames - 1) // (num_frames - 1) for i in range(num_frames)]
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    cap.release()
    return frames


def process_single_frame(
    frame: np.ndarray,
    detector,
    pose_estimator,
    bbox_thr: float = 0.05,
    nms_thr: float = 0.3,
    max_detections: Optional[int] = None,
) -> Optional[PoseDataSample]:
    """Process a single frame with detection and pose estimation.
    
    Args:
        frame: The input frame to process
        detector: The detection model
        pose_estimator: The pose estimation model
        bbox_thr: The threshold for bounding box filtering
        nms_thr: The threshold for non-maximum suppression
        max_detections: The maximum number of detections to keep

    Returns:
        Optional[PoseDataSample]: The pose estimation results
    """
    # Detection
    det_result = inference_detector(detector, frame)

    # Apply max_detections limit (Top-K by score from raw detections)
    if max_detections is not None and len(det_result.pred_instances) > 0 and len(det_result.pred_instances) > max_detections:
        if hasattr(det_result.pred_instances, 'scores'):
            scores = det_result.pred_instances.scores
            if scores.numel() > 0: # Check if scores tensor is not empty
                num_to_keep = min(max_detections, len(scores))
                topk_indices = scores.topk(num_to_keep).indices
                det_result.pred_instances = det_result.pred_instances[topk_indices]
            else: # No scores to sort by, just truncate if necessary
                det_result.pred_instances = det_result.pred_instances[:max_detections]
        else: # No scores, just truncate
            det_result.pred_instances = det_result.pred_instances[:max_detections]


    pred_instances = det_result.pred_instances[
        det_result.pred_instances.scores > bbox_thr
    ]
    
    # Filter for monkey class (label 0)
    if hasattr(pred_instances, 'labels') and pred_instances.labels.numel() > 0:
        monkey_indices = (pred_instances.labels == 0).nonzero(as_tuple=True)[0]
        pred_instances = pred_instances[monkey_indices]
    
    bboxes_cpu = pred_instances.bboxes.cpu().numpy()
    scores_cpu = pred_instances.scores.cpu().numpy()

    if not scores_cpu.size:
        return None

    # Prepare for NMS: requires bboxes with scores, e.g., (x1, y1, x2, y2, score)
    if bboxes_cpu.shape[0] > 0:
        nms_input = np.hstack((bboxes_cpu, scores_cpu[:, np.newaxis]))
        keep_indices = nms(nms_input, nms_thr)

        bboxes_after_nms = nms_input[keep_indices, :4]
        scores_after_nms = nms_input[keep_indices, 4]
    else:
        bboxes_after_nms = np.empty((0,4))
        scores_after_nms = np.empty((0,))


    if not bboxes_after_nms.size:
        # print("No detections after NMS!")
        return None

    # Pose estimation
    pose_results_list = inference_topdown(pose_estimator, frame, bboxes_after_nms)
    if not pose_results_list:
        return None

    final_pose_results = []
    for i, pose_data_sample in enumerate(pose_results_list):
        if hasattr(pose_data_sample, 'pred_instances'):
            # Ensure fields are set correctly. Scores and bboxes should be tensors.
            # instance_id is typically a list of ints, but mmpose can handle tensor as well.
            # Get device from the model parameters
            model_device = next(pose_estimator.parameters()).device
            current_score_tensor = torch.tensor([scores_after_nms[i]], dtype=torch.float32, device=model_device)
            current_bbox_tensor = torch.from_numpy(bboxes_after_nms[i:i+1]).to(dtype=torch.float32, device=model_device)
            current_id_tensor = torch.tensor([i], dtype=torch.long, device=model_device) # IDs are usually integer types

            pose_data_sample.pred_instances.set_field(current_score_tensor, 'bbox_scores')
            pose_data_sample.pred_instances.set_field(current_id_tensor, 'instance_id')
            pose_data_sample.pred_instances.set_field(current_bbox_tensor, 'bboxes')
            final_pose_results.append(pose_data_sample)

    if not final_pose_results:
        return None

    # Merge results
    data_samples = merge_data_samples(final_pose_results)
    return data_samples


def extract_frames_multiview(
    video_paths: List[str], 
    num_frames: int = 3,
    random_window: int = 30
) -> List[List[np.ndarray]]:
    """Extract same frames from multiple videos with random offset.
    
    Args:
        video_paths: List of video paths
        num_frames: Number of frames to extract
        random_window: Window size for random frame selection around target frame
    
    Returns:
        List[List[np.ndarray]]: First list is frames, second list is views
    """
    frames = []
    np.random.seed()
    
    # Get total frames for each video
    caps = [cv2.VideoCapture(path) for path in video_paths]
    total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps)
    
    # Calculate base frame indices
    if num_frames == 1:
        base_indices = [total_frames // 2]
    else:
        base_indices = [i * (total_frames - 1) // (num_frames - 1) for i in range(num_frames)]
    
    # Add random offset within window
    indices = []
    for base_idx in base_indices:
        # Ensure window doesn't go out of bounds
        window_start = max(0, base_idx - random_window // 2)
        window_end = min(total_frames - 1, base_idx + random_window // 2)
        random_idx = np.random.randint(window_start, window_end + 1)
        indices.append(random_idx)
    
    print(f"Selected frame indices: {indices}")
    
    # Extract frames from each video
    for idx in indices:
        views = []
        for cap in caps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                views.append(frame)
        frames.append(views)
    
    for cap in caps:
        cap.release()
        
    return frames


def save_comparison_frames(
    frames: List[np.ndarray],
    results: Dict[str, List[object]],
    pose_estimator,
    visualizer: FastPoseVisualizer,
    output_dir: str,
) -> None:
    """Save individual comparison frames using same logic as video_inference.py.
    
    Args:
        frames: List of frames
        results: Dictionary of model results
        pose_estimator: Pose estimator model
        visualizer: Visualizer for drawing poses
        output_dir: Output directory for saving frames
    
    Returns:
        None
    """
    
    # Get skeleton links from pose estimator
    skeleton_links = None
    if hasattr(pose_estimator, 'dataset_meta'):
        skeleton_links = pose_estimator.dataset_meta.get('skeleton_links')
        print(f"Found skeleton links: {skeleton_links}")
    
    # For each frame
    for frame_idx, frame in enumerate(frames):
        # Create a subdirectory for this frame
        frame_dir = os.path.join(output_dir, f"frame_{frame_idx}")
        os.makedirs(frame_dir, exist_ok=True)
        
        # Process each model's result
        for model_name, model_results in results.items():
            data_samples = model_results[frame_idx]
            
            if data_samples is not None:
                print(f"Drawing poses for {model_name}, frame {frame_idx}")
                
                frame_viz = visualizer.draw_poses(
                    frame=frame.copy(),  # Use original frame size
                    pred_instances=data_samples.pred_instances,
                    skeleton_links=skeleton_links
                )
                
                cv2.putText(
                    frame_viz,
                    model_name,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
                
                output_path = os.path.join(frame_dir, f"{model_name}.jpg")
                cv2.imwrite(output_path, frame_viz)
                print(f"Saved {output_path}")
            else:
                print(f"No pose result for {model_name}, frame {frame_idx}")


def load_coco_data(coco_path: str, num_images: int) -> List[Dict[str, Any]]:
    """Loads COCO data, selects images, and prepares bboxes.
    
    Args:
        coco_path: Path to the COCO JSON annotation file
        num_images: Number of images to select

    Returns:
        List[Dict[str, Any]]: List of image data
    """
    print(f"Loading COCO data from: {coco_path}")
    with open(coco_path, 'r') as f:
        coco_data = json.load(f)

    images = {img['id']: img for img in coco_data['images']}
    annotations = coco_data['annotations']
    
    target_category_id = 1 
    
    img_to_anns = {}
    for ann in annotations:
        img_id = ann['image_id']
        if ann['category_id'] == target_category_id:
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)

    valid_image_ids = list(img_to_anns.keys())
    
    if not valid_image_ids:
        raise ValueError(f"No annotations found for category ID {target_category_id} in {coco_path}")

    if num_images >= len(valid_image_ids):
        selected_image_ids = valid_image_ids
        print(f"Requested {num_images}, but only {len(valid_image_ids)} valid images found. Using all valid images.")
    else:
        selected_image_ids = random.sample(valid_image_ids, num_images)
        
    print(f"Selected {len(selected_image_ids)} images randomly.")

    selected_data = []
    for img_id in selected_image_ids:
        img_info = images[img_id]
        anns = img_to_anns[img_id]
        
        image_path = img_info['file_name']
        if not os.path.exists(image_path):
            print(f"Warning: Image path not found: {image_path}. Skipping this image.")
            continue

        bboxes = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            score = 1.0 # Assign default score
            bboxes.append([x1, y1, x2, y2, score])
        
        if bboxes: # Only add if there are valid bboxes for the image
            selected_data.append({
                'image_path': image_path,
                'bboxes': np.array(bboxes, dtype=np.float32)
            })
        else:
            print(f"Warning: No valid bounding boxes found for image ID {img_id} after processing.")


    if not selected_data:
        raise RuntimeError("Failed to load any valid image data from the COCO file.")
        
    return selected_data


def create_comparison_grid_mpl(
    processed_data: Union[List[List[np.ndarray]], List[Dict[str, Any]]],
    results: Dict[str, Union[List[List[Optional[PoseDataSample]]], List[Optional[PoseDataSample]]]],
    pose_estimator,
    mode: str,
    kpt_thr: float,
    draw_bbox: bool,
    draw_skeleton: bool,
    dpi: int = 300
) -> plt.Figure:
    """Create a comparison grid visualization using Matplotlib.
    
    Args:
        processed_data: List of frames or images
        results: Dictionary of model results
        pose_estimator: Pose estimator model
        mode: Mode of operation (coco or video)
        kpt_thr: Keypoint score threshold
        draw_bbox: Whether to draw bounding boxes
        draw_skeleton: Whether to draw skeleton links
        dpi: DPI for the output figure

    Returns:
        plt.Figure: The comparison grid figure
    """
    
    n_models = len(results)
    if n_models == 0:
        raise ValueError("No model results provided for visualization.")

    skeleton_links = None
    if hasattr(pose_estimator, 'dataset_meta'):
        skeleton_links = pose_estimator.dataset_meta.get('skeleton_links')
        
    model_names = list(results.keys())

    n_display_rows = len(processed_data)
    if n_display_rows == 0:
        raise ValueError("No data (frames or images) to display.")

    ref_img_path = None
    if mode == 'coco':
        ref_img_path = processed_data[0]['image_path']
    elif mode == 'video' and processed_data[0]:
        ref_img_for_aspect = processed_data[0][0]       # First view of first frame
        h, w = ref_img_for_aspect.shape[:2]
    
    if ref_img_path:
        with Image.open(ref_img_path) as img:
            w, h = img.size

    aspect_ratio = w / h
    fig_w = 20 
    fig_h = (fig_w / n_models * n_display_rows) / aspect_ratio
    
    fig, axes = plt.subplots(
        nrows=n_display_rows, 
        ncols=n_models, 
        figsize=(fig_w, fig_h), 
        dpi=dpi
    )
    if n_display_rows == 1 and n_models == 1:
        axes = np.array([[axes]])
    elif n_display_rows == 1:
        axes = np.array([axes])
    elif n_models == 1:
        axes = np.array([[ax] for ax in axes])

    fig.tight_layout(pad=0.5, h_pad=1.5, w_pad=0.5)

    for row_idx in range(n_display_rows):
        for col_idx, model_name in enumerate(model_names):
            ax = axes[row_idx, col_idx]
            ax.axis('off')

            img_to_draw = None
            data_samples_for_cell = None

            if mode == 'coco':
                img_path = processed_data[row_idx]['image_path']
                try:
                    img_to_draw = mmcv.imread(img_path)
                except Exception as e:
                    print(f"Error reading {img_path}: {e}")
                
                model_results = results.get(model_name)
                if model_results and row_idx < len(model_results):
                    data_samples_for_cell = model_results[row_idx]

                img_name_short = os.path.basename(img_path)
                img_name_short = img_name_short[:15] + '...' if len(img_name_short) > 18 else img_name_short
                ax.set_title(f"{model_name}\n{img_name_short}", fontsize=8)

            elif mode == 'video':
                if processed_data[row_idx]:
                    img_to_draw = processed_data[row_idx][0] # Use first view's frame
                
                model_results = results.get(model_name)
                if model_results and row_idx < len(model_results) and model_results[row_idx]:
                    data_samples_for_cell = model_results[row_idx][0] # Use first view's results

                ax.set_title(f"{model_name}\nFrame Index {row_idx}", fontsize=8)

            if img_to_draw is not None:
                ax.imshow(cv2.cvtColor(img_to_draw, cv2.COLOR_BGR2RGB))
            else:
                ax.text(0.5, 0.5, "Image not found", ha='center', va='center', transform=ax.transAxes)
                continue

            if data_samples_for_cell and hasattr(data_samples_for_cell, 'pred_instances'):
                pred_instances = data_samples_for_cell.pred_instances
                
                for i in range(len(pred_instances)):
                    instance = pred_instances[i]
                    
                    # Draw BBox
                    if draw_bbox and 'bboxes' in instance:
                        bbox_data = instance.bboxes
                        if hasattr(bbox_data, 'cpu'):
                            bbox = bbox_data.cpu().numpy()
                        else:
                            bbox = np.array(bbox_data)
                        
                        x1, y1, x2, y2 = bbox.flatten()[:4]
                        w, h = x2 - x1, y2 - y1
                        rect = patches.Rectangle((x1, y1), w, h, linewidth=BBOX_LINE_THICKNESS_MPL, edgecolor=BBOX_COLOR_MPL, facecolor='none', zorder=10)
                        ax.add_patch(rect)

                    # Draw Keypoints and Skeleton
                    if 'keypoints' in instance:
                        keypoints_data = instance.keypoints
                        if hasattr(keypoints_data, 'cpu'):
                            keypoints = keypoints_data.cpu().numpy().squeeze()
                        else:
                            keypoints = np.array(keypoints_data).squeeze()

                        scores_data = instance.keypoint_scores
                        if hasattr(scores_data, 'cpu'):
                            keypoint_scores = scores_data.cpu().numpy().flatten()
                        else:
                            keypoint_scores = np.array(scores_data).flatten()

                        if keypoints.ndim != 2 or keypoint_scores.ndim != 1:
                            print(f"DEBUG: Skipping instance due to unexpected array shapes. Kpts shape: {keypoints.shape}, Scores shape: {keypoint_scores.shape}")
                            continue

                        visible_kpts_x = [px for px, score in zip(keypoints[:, 0], keypoint_scores) if score >= kpt_thr]
                        visible_kpts_y = [py for py, score in zip(keypoints[:, 1], keypoint_scores) if score >= kpt_thr]

                        if visible_kpts_x:
                            ax.scatter(visible_kpts_x, visible_kpts_y, s=KEYPOINT_SIZE_MPL, color=KEYPOINT_COLOR_MPL, zorder=20, alpha=0.9)
                        
                        if draw_skeleton and skeleton_links:
                            for link in skeleton_links:
                                idx1, idx2 = link[0] - 1, link[1] - 1
                                if keypoint_scores[idx1] >= kpt_thr and keypoint_scores[idx2] >= kpt_thr:
                                    pt1_x, pt1_y = keypoints[idx1]
                                    pt2_x, pt2_y = keypoints[idx2]
                                    ax.plot([pt1_x, pt2_x], [pt1_y, pt2_y], color=SKELETON_COLOR_MPL, linewidth=LINE_THICKNESS_MPL, solid_capstyle='round', zorder=15, alpha=0.85)
            else:
                ax.text(0.5, 0.1, "No Prediction", ha='center', va='center', transform=ax.transAxes, color='red', fontsize=8)

    return fig


def main():
    parser = argparse.ArgumentParser(description="Compare multiple pose estimation models on video frames or COCO images.")
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--inputs",
        nargs='+',
        help="Input video paths (space separated). Used for video mode."
    )
    input_group.add_argument(
        "--coco",
        type=str,
        help="Path to COCO format JSON annotation file. Used for COCO mode."
    )

    parser.add_argument("--output", help="Output image path for the comparison grid (e.g., 'comparison.svg').")
    parser.add_argument(
        "--num-items",
        type=int,
        default=3,
        help="Number of video frames or COCO images to process."
    )
    parser.add_argument(
        "--random-window",
        type=int,
        default=30,
        help="[Video Mode] Window size (frames) for random frame selection around evenly spaced points."
    )
    parser.add_argument(
        "--bbox-thr",
        type=float,
        default=0.3,
        help="[Video Mode] Bounding box score threshold for detection."
    )
    parser.add_argument(
        "--nms-thr",
        type=float,
        default=0.3,
        help="[Video Mode] IoU threshold for bounding box NMS."
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=None,
        help="[Video Mode] Maximum number of detections to process per frame, after initial score thresholding."
    )
    parser.add_argument(
        "--kpt-thr",
        type=float,
        default=0.3,
        help="Keypoint score threshold for visualization."
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=3,
        help="Keypoint radius for visualization."
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=1,
        help="Link thickness for visualization."
    )
    parser.add_argument(
        "--draw-bbox",
        action="store_true",
        help="Draw bounding boxes on the visualization."
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=2160,
        help="Target total height for the output grid image."
    )
    parser.add_argument(
        "--no-skeleton",
        action="store_true",
        help="Do not draw skeleton links, only keypoints."
    )

    args = parser.parse_args()

    mode = 'coco' if args.coco else 'video'

    if not args.output:
        if mode == 'video':
            output_dir = os.path.dirname(args.inputs[0])
            base_name = os.path.splitext(os.path.basename(args.inputs[0]))[0]
            args.output = os.path.join(output_dir, f"{base_name}_comparison.svg")
        else: # mode == 'coco'
            output_dir = os.path.dirname(args.coco)
            base_name = os.path.splitext(os.path.basename(args.coco))[0]
            args.output = os.path.join(output_dir, f"{base_name}_comparison.svg")
    else:
        output_base, output_ext = os.path.splitext(args.output)
        if not output_ext or output_ext.lower() not in ['.svg', '.pdf']:
            args.output = output_base + '.svg'
            print(f"Output format not specified or not vector-based, saving as SVG: {args.output}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    processed_data: Union[List[List[np.ndarray]], List[Dict[str, Any]]]
    
    print("Initializing detector...")
    detector = init_detector(
        DET_CONFIG,
        DET_CHECKPOINT,
        device="cuda:0"
    )
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
        
    if mode == 'video':
        print("Mode: Video")
        processed_data = extract_frames_multiview(
            args.inputs,
            args.num_items,
            args.random_window
        )
    else: # mode == 'coco'
        print("Mode: COCO")
        processed_data = load_coco_data(args.coco, args.num_items)

    results: Dict[str, Union[List[List[Optional[PoseDataSample]]], List[Optional[PoseDataSample]]]] = {}
    current_pose_estimator = None # To get skeleton links later

    for model_name, model_config in POSE_MODELS.items():
        print(f"\nProcessing with {model_name}...")

        try:
            pose_estimator = init_model(
                model_config["config"],
                model_config["checkpoint"],
                device="cuda:0",
                cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))), 
            )
            current_pose_estimator = pose_estimator # Keep the last one for skeleton info
        except Exception as e:
            print(f"Error initializing pose model {model_name}: {e}")
            print("Skipping this model.")
            continue # Skip to the next model


        print("Running inference...")
        if mode == 'video':
            model_results_video: List[List[Optional[PoseDataSample]]] = []
            for frame_views_list in processed_data: # processed_data is List[List[np.ndarray]]
                view_results: List[Optional[PoseDataSample]] = []
                frames_for_current_item = frame_views_list 

                for frame_arr in frames_for_current_item: # Iterate through views of the current item
                    result = process_single_frame(
                        frame_arr, # This is the actual np.ndarray for a single view
                        detector,
                        pose_estimator,
                        bbox_thr=args.bbox_thr,
                        nms_thr=args.nms_thr,
                        max_detections=args.max_detections
                    )
                    view_results.append(result)
                model_results_video.append(view_results)
            results[model_name] = model_results_video

        else: # mode == 'coco'
            model_results_coco: List[Optional[PoseDataSample]] = []
            for item in processed_data: # processed_data is List[Dict[str, Any]], item is {'image_path': ..., 'bboxes': ...}
                image_path = item['image_path']
                
                try:
                    img = mmcv.imread(image_path)
                    if img is None: 
                        print(f"Error: mmcv.imread returned None for {image_path}. Skipping.")
                        model_results_coco.append(None)
                        continue
                except Exception as e:
                    print(f"Error reading image {image_path}: {e}. Skipping inference for this image.")
                    model_results_coco.append(None)
                    continue
                
                result = process_single_frame(
                    img,
                    detector,
                    pose_estimator,
                    bbox_thr=args.bbox_thr,
                    nms_thr=args.nms_thr,
                    max_detections=args.max_detections
                )
                model_results_coco.append(result)
            results[model_name] = model_results_coco

    if not results:
        print("No models were successfully processed. Exiting.")
        return
        
    if current_pose_estimator is None:
        print("Warning: Could not get skeleton links from any pose estimator.")

    print("\nCreating comparison grid with Matplotlib...")
    
    try:
        fig = create_comparison_grid_mpl(
            processed_data,
            results,
            current_pose_estimator,
            mode=mode,
            kpt_thr=args.kpt_thr,
            draw_bbox=args.draw_bbox,
            draw_skeleton=not args.no_skeleton
        )
        
        print(f"Saving comparison grid to {args.output}...")
        fig.savefig(args.output, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        print(f"Successfully saved {args.output}")

    except Exception as e:
        print(f"Error during grid creation or saving: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 