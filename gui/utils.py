"""Shared utilities for GUI module.

This module provides utility functions for parallel processing,
file handling, and other common operations.
"""

import json
import multiprocessing as mp
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm


def split_videos_for_parallel(
    video_paths: List[Path],
    num_workers: int
) -> List[List[Path]]:
    """Split video paths evenly across workers.
    
    Args:
        video_paths: List of video file paths.
        num_workers: Number of parallel workers.
        
    Returns:
        List of video path chunks for each worker.
    """
    chunks = []
    chunk_size = len(video_paths) // num_workers
    remainder = len(video_paths) % num_workers
    
    start = 0
    for i in range(num_workers):
        end = start + chunk_size + (1 if i < remainder else 0)
        if start < len(video_paths):
            chunks.append(video_paths[start:end])
        else:
            chunks.append([])
        start = end
    
    return chunks


def worker_process(
    args_namespace: Any,
    video_chunk: List[Path],
    gpu_id: int,
    process_func: Callable
) -> Optional[str]:
    """Worker function for parallel video processing.
    
    Args:
        args_namespace: Arguments namespace to copy.
        video_chunk: List of videos for this worker.
        gpu_id: GPU ID for this worker.
        process_func: Function to process videos.
        
    Returns:
        Path to output file or None if failed.
    """
    worker_args = deepcopy(args_namespace)
    worker_args.device = f"cuda:{gpu_id}"
    
    if not video_chunk:
        return None
    
    worker_basename = video_chunk[0].stem
    worker_output_path = Path(worker_args.output_dir) / f"temp_{worker_basename}_{gpu_id}.json"
    
    try:
        results = process_func(worker_args, video_files=video_chunk)
        with open(worker_output_path, 'w') as f:
            json.dump(results, f)
        return str(worker_output_path)
    except Exception as e:
        print(f"ERROR in worker on GPU {gpu_id} for {worker_basename}: {e}")
        return None


def merge_coco_files(
    file_paths: List[str],
    final_path: Union[str, Path],
    cleanup: bool = True
) -> None:
    """Merge multiple COCO annotation files into one.
    
    Args:
        file_paths: List of partial annotation files.
        final_path: Path for merged output.
        cleanup: Whether to delete partial files after merging.
    """
    if not file_paths:
        print("Warning: No partial annotation files to merge.")
        return
    
    merged_coco = None
    global_img_id, global_ann_id = 0, 0
    
    print(f"Merging {len(file_paths)} annotation files...")
    for file_path in tqdm(file_paths, desc="Merging Files"):
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if merged_coco is None:
            merged_coco = data
            merged_coco['images'] = []
            merged_coco['annotations'] = []
        
        img_id_map = {}
        for image in data['images']:
            old_img_id = image['id']
            new_img_id = global_img_id
            img_id_map[old_img_id] = new_img_id
            image['id'] = new_img_id
            merged_coco['images'].append(image)
            global_img_id += 1
        
        for ann in data['annotations']:
            ann['id'] = global_ann_id
            ann['image_id'] = img_id_map.get(ann['image_id'], -1)
            if ann['image_id'] != -1:
                merged_coco['annotations'].append(ann)
            global_ann_id += 1
    
    with open(final_path, 'w') as f:
        json.dump(merged_coco, f, indent=4)
    print(f"Merged annotations saved to {final_path}")
    
    if cleanup:
        for file_path in file_paths:
            os.remove(file_path)


def parallel_process_videos(
    video_dir: Union[str, Path],
    output_dir: Union[str, Path],
    process_func: Callable,
    args: Any,
    gpus: List[int],
    jobs_per_gpu: int = 1
) -> str:
    """Process videos in parallel across multiple GPUs.
    
    Args:
        video_dir: Directory containing videos.
        output_dir: Output directory for results.
        process_func: Function to process videos.
        args: Arguments for processing function.
        gpus: List of GPU IDs to use.
        jobs_per_gpu: Number of parallel jobs per GPU.
        
    Returns:
        Path to final merged annotation file.
    """
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_paths = list(video_dir.glob("*.mp4"))
    video_paths.extend(video_dir.glob("*.avi"))
    video_paths.extend(video_dir.glob("*.mov"))
    
    if not video_paths:
        raise ValueError(f"No videos found in {video_dir}")
    
    num_workers = len(gpus) * jobs_per_gpu
    video_chunks = split_videos_for_parallel(video_paths, num_workers)
    
    print(f"Processing {len(video_paths)} videos across {num_workers} workers")
    
    with mp.Pool(processes=num_workers) as pool:
        worker_args = []
        for i, chunk in enumerate(video_chunks):
            gpu_id = gpus[i % len(gpus)]
            worker_args.append((args, chunk, gpu_id, process_func))
        
        results = pool.starmap(worker_process, worker_args)
    
    partial_files = [r for r in results if r is not None]
    
    final_path = output_dir / "annotations.json"
    merge_coco_files(partial_files, final_path)
    
    return str(final_path)


def extract_frames_from_video(
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    frame_interval: int = 30,
    max_frames: Optional[int] = None
) -> List[str]:
    """Extract frames from video at specified interval.
    
    Args:
        video_path: Path to video file.
        output_dir: Directory to save extracted frames.
        frame_interval: Extract every Nth frame.
        max_frames: Maximum frames to extract.
        
    Returns:
        List of saved frame paths.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    saved_frames = []
    frame_idx = 0
    
    pbar = tqdm(total=total_frames, desc=f"Extracting frames from {video_path.name}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            frame_name = f"{video_path.stem}_frame_{frame_idx:06d}.jpg"
            frame_path = output_dir / frame_name
            cv2.imwrite(str(frame_path), frame)
            saved_frames.append(str(frame_path))
        
        frame_idx += 1
        pbar.update(1)
        
        if max_frames and frame_idx >= max_frames:
            break
    
    cap.release()
    pbar.close()
    
    return saved_frames


def resize_image_maintaining_aspect(
    image: np.ndarray,
    max_size: int = 1280
) -> Tuple[np.ndarray, float]:
    """Resize image maintaining aspect ratio.
    
    Args:
        image: Input image.
        max_size: Maximum dimension size.
        
    Returns:
        Tuple of (resized_image, scale_factor).
    """
    h, w = image.shape[:2]
    
    if max(h, w) <= max_size:
        return image, 1.0
    
    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return resized, scale


def scale_bboxes(
    bboxes: List[List[float]],
    scale: float
) -> List[List[float]]:
    """Scale bounding boxes by a factor.
    
    Args:
        bboxes: List of bounding boxes [x1, y1, x2, y2].
        scale: Scale factor.
        
    Returns:
        Scaled bounding boxes.
    """
    if scale == 1.0:
        return bboxes
    
    scaled = []
    for bbox in bboxes:
        scaled.append([
            bbox[0] * scale,
            bbox[1] * scale,
            bbox[2] * scale,
            bbox[3] * scale
        ])
    
    return scaled


def scale_keypoints(
    keypoints: np.ndarray,
    scale: float
) -> np.ndarray:
    """Scale keypoint coordinates.
    
    Args:
        keypoints: Keypoints array (N, 3) with (x, y, visibility).
        scale: Scale factor.
        
    Returns:
        Scaled keypoints.
    """
    if scale == 1.0:
        return keypoints
    
    scaled = keypoints.copy()
    scaled[:, 0] *= scale
    scaled[:, 1] *= scale
    
    return scaled


def calculate_iou(
    bbox1: List[float],
    bbox2: List[float]
) -> float:
    """Calculate IoU between two bounding boxes.
    
    Args:
        bbox1: First bbox [x1, y1, x2, y2].
        bbox2: Second bbox [x1, y1, x2, y2].
        
    Returns:
        IoU value between 0 and 1.
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def filter_overlapping_bboxes(
    bboxes: List[List[float]],
    scores: List[float],
    iou_threshold: float = 0.5
) -> Tuple[List[List[float]], List[float]]:
    """Filter overlapping bboxes keeping highest scoring ones.
    
    Args:
        bboxes: List of bounding boxes.
        scores: Confidence scores for each bbox.
        iou_threshold: IoU threshold for filtering.
        
    Returns:
        Tuple of (filtered_bboxes, filtered_scores).
    """
    if not bboxes:
        return [], []
    
    indices = np.argsort(scores)[::-1]
    keep = []
    
    for i in indices:
        keep_box = True
        for j in keep:
            if calculate_iou(bboxes[i], bboxes[j]) > iou_threshold:
                keep_box = False
                break
        
        if keep_box:
            keep.append(i)
    
    filtered_bboxes = [bboxes[i] for i in keep]
    filtered_scores = [scores[i] for i in keep]
    
    return filtered_bboxes, filtered_scores


def visualize_skeleton(
    image: np.ndarray,
    keypoints: np.ndarray,
    skeleton_links: List[List[int]],
    keypoint_names: Optional[List[str]] = None,
    thickness: int = 2
) -> np.ndarray:
    """Visualize skeleton on image.
    
    Args:
        image: Input image.
        keypoints: Keypoints array (N, 3).
        skeleton_links: List of [start_idx, end_idx] connections.
        keypoint_names: Optional keypoint names for coloring.
        thickness: Line thickness.
        
    Returns:
        Image with skeleton drawn.
    """
    vis_image = image.copy()
    
    for link in skeleton_links:
        if len(link) != 2:
            continue
        
        idx1, idx2 = link
        if idx1 >= len(keypoints) or idx2 >= len(keypoints):
            continue
        
        if keypoints[idx1, 2] > 0 and keypoints[idx2, 2] > 0:
            pt1 = (int(keypoints[idx1, 0]), int(keypoints[idx1, 1]))
            pt2 = (int(keypoints[idx2, 0]), int(keypoints[idx2, 1]))
            
            color = (0, 255, 0)
            if keypoint_names:
                if idx1 < len(keypoint_names) and idx2 < len(keypoint_names):
                    name1, name2 = keypoint_names[idx1], keypoint_names[idx2]
                    if "left" in name1.lower() or "left" in name2.lower():
                        color = (255, 0, 0)
                    elif "right" in name1.lower() or "right" in name2.lower():
                        color = (0, 0, 255)
            
            cv2.line(vis_image, pt1, pt2, color, thickness)
    
    for i, (x, y, v) in enumerate(keypoints):
        if v > 0:
            color = (0, 255, 255)
            if keypoint_names and i < len(keypoint_names):
                if "left" in keypoint_names[i].lower():
                    color = (255, 0, 0)
                elif "right" in keypoint_names[i].lower():
                    color = (0, 0, 255)
            
            cv2.circle(vis_image, (int(x), int(y)), 3, color, -1)
    
    return vis_image


def load_coco_categories(
    coco_json_path: Union[str, Path]
) -> Tuple[List[str], List[List[int]]]:
    """Load keypoint names and skeleton from COCO JSON.
    
    Args:
        coco_json_path: Path to COCO JSON file.
        
    Returns:
        Tuple of (keypoint_names, skeleton_links).
    """
    with open(coco_json_path, 'r') as f:
        data = json.load(f)
    
    if 'categories' not in data or not data['categories']:
        return [], []
    
    category = data['categories'][0]
    keypoint_names = category.get('keypoints', [])
    skeleton_links = category.get('skeleton', [])
    
    return keypoint_names, skeleton_links