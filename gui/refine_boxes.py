# refine_boxes.py

#TODO:
# - add docstring
# - why does the gui look low-res? (Often due to lack of DPI awareness in Tkinter on Windows, can be complex to fix fully)

# Enhanced GUI Features (2024):
# - Reorganized UI with menu bar, collapsible sidebar, and expanded canvas
# - Dropdown menus for better organization (File, Edit, View, Tools)
# - Delete confirmation with toggle option (can be disabled in Edit menu)
# - Undo functionality for instance deletions (Ctrl+Z)
# - Enhanced canvas dragging and positioning
# - Bbox zoom on double-click top-right corner with padding
# - Keypoint width slider control (1-20 pixels)
# - Improved status bar with zoom percentage
# - Keyboard shortcuts preserved and enhanced


import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import copy
import argparse
import random
import sys # For platform check

import cv2
import numpy as np
from PIL import Image, ImageTk

# Import the shrinking utility
try:
    from coco_utils.coco_labels_utils import calculate_shrunk_bboxes
except ImportError:
    print("Warning: Could not import calculate_shrunk_bboxes. Shrinking functionality disabled.")
    def calculate_shrunk_bboxes(data, percentage): # Dummy function
        print("Shrinking is disabled due to import error.")
        return copy.deepcopy(data), {}

# Define the fixed shrink percentage (if used via arg)
# SHRINK_PERCENTAGE = 5.0 # Now handled by args

# --- Skeleton Visualization Colors ---
# Color constants for keypoint and skeleton visualization
KEYPOINT_COLOR_LEFT = (0, 255, 0)      # Green for left keypoints
KEYPOINT_COLOR_RIGHT = (255, 0, 0)     # Blue for right keypoints  
KEYPOINT_COLOR_CENTER = (0, 255, 255)  # Yellow for center keypoints (nose, eyes, ears)
SKELETON_COLOR_LEFT = (0, 255, 0)      # Green for left-side skeleton links
SKELETON_COLOR_RIGHT = (255, 0, 0)     # Blue for right-side skeleton links
SKELETON_COLOR_CENTER = (0, 255, 255)  # Yellow for center skeleton links
SKELETON_COLOR_MIXED = (255, 0, 255)   # Magenta for mixed left/right links

def get_keypoint_color(keypoint_name: str) -> tuple:
    """Get color for keypoint based on name."""
    keypoint_name_lower = keypoint_name.lower()
    
    # Center keypoints (nose, eyes, ears)
    if any(center_kpt in keypoint_name_lower for center_kpt in ['nose', 'eye', 'ear']):
        return KEYPOINT_COLOR_CENTER
    # Left keypoints
    elif 'left' in keypoint_name_lower or 'l_' in keypoint_name_lower:
        return KEYPOINT_COLOR_LEFT
    # Right keypoints  
    elif 'right' in keypoint_name_lower or 'r_' in keypoint_name_lower:
        return KEYPOINT_COLOR_RIGHT
    else:
        return KEYPOINT_COLOR_CENTER  # Default fallback

def get_skeleton_color(keypoint1_name: str, keypoint2_name: str) -> tuple:
    """Get color for skeleton link based on connected keypoint names."""
    color1 = get_keypoint_color(keypoint1_name)
    color2 = get_keypoint_color(keypoint2_name)
    
    # If both keypoints are the same type (left/right/center), use that color
    if color1 == color2:
        return color1
    # Otherwise use mixed color for connections between different types
    else:
        return SKELETON_COLOR_MIXED

# --- COCO Data Loading ---

#TODO:
# add area, center, scale, iscrowd, category_id, id, etc to final JSON (area is added, others depend on source json)


def _load_and_organize_coco_json(
    coco_json_path: str,
    valid_ids: Optional[Set[int]] = None, # Set of valid image/frame IDs
    is_video_mode: bool = False # Flag to indicate if IDs are frame numbers
) -> Tuple[Optional[Dict[int, List[Dict]]], Optional[List[Dict]], int]:
    """Loads and organizes annotations from a COCO JSON file. Internal helper."""
    print(f"Loading COCO annotations from: {coco_json_path}")
    cleaned_coco_path = coco_json_path.strip('\"\'')
    try:
        with open(cleaned_coco_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        messagebox.showerror("Error", f"COCO JSON file not found: {cleaned_coco_path}")
        return None, None, -1
    except json.JSONDecodeError:
        messagebox.showerror(
            "Error", f"Error decoding COCO JSON file: {cleaned_coco_path}"
        )
        return None, None, -1
    except OSError as e:
        messagebox.showerror(
            "Error", f"Error opening COCO JSON file: {cleaned_coco_path}\n{e}"
        )
        return None, None, -1

    annotation_map = {}
    if valid_ids is not None: # Initialize map only for valid IDs if provided
        annotation_map = {img_id: [] for img_id in valid_ids}

    max_ann_id = 0
    print("Mapping annotations...")
    annotations_in_file = data.get("annotations", [])
    processed_count = 0
    skipped_count = 0

    for ann in annotations_in_file:
        key_id = ann["image_id"] # Use image_id as the key (frame num in video)

        # If valid_ids are provided (video mode), only process if the frame exists.
        # If valid_ids is None (image mode), we rely on later checks.
        if valid_ids is not None and key_id not in valid_ids:
            skipped_count += 1
            continue

        # Ensure basic fields exist & preprocess
        ann_copy = ann.copy()
        if "bbox" not in ann_copy: ann_copy["bbox"] = []
        if "keypoints" not in ann_copy: ann_copy["keypoints"] = []
        if "num_keypoints" not in ann_copy: ann_copy["num_keypoints"] = 0
        if "area" not in ann_copy and ann_copy["bbox"] and len(ann_copy["bbox"]) == 4:
            ann_copy["area"] = ann_copy["bbox"][2] * ann_copy["bbox"][3]
        elif "area" not in ann_copy:
             ann_copy["area"] = 0
        if "iscrowd" not in ann_copy: ann_copy["iscrowd"] = 0
        if "category_id" not in ann_copy: ann_copy["category_id"] = 1 # Default
        # Instance ID (track ID) might exist, keep it if present
        # if "instance_id" not in ann_copy: ann_copy["instance_id"] = None # Or some default

        # --- Preprocess Keypoints: Randomize only if (0,0) ---
        bbox = ann_copy.get("bbox")
        kpts_flat = ann_copy.get("keypoints")
        if bbox and len(bbox) == 4 and kpts_flat and len(kpts_flat) % 3 == 0:
            x, y, w, h = bbox
            if w > 0 and h > 0: # Ensure valid bbox dimensions
                modified_kpts = False
                for i in range(0, len(kpts_flat), 3):
                    kx, ky, kv = kpts_flat[i], kpts_flat[i+1], kpts_flat[i+2]
                    # Only randomize fully missing (0,0) points; keep provided coords for v=0
                    if (kx == 0 and ky == 0):
                        new_kx = max(x, min(random.uniform(x, x + w), x + w))
                        new_ky = max(y, min(random.uniform(y, y + h), y + h))
                        kpts_flat[i] = new_kx
                        kpts_flat[i+1] = new_ky
                        kpts_flat[i+2] = 0 # Ensure visibility is 0
                        modified_kpts = True

                if modified_kpts:
                    ann_copy["keypoints"] = kpts_flat
                    ann_copy['num_keypoints'] = sum(1 for j in range(2, len(kpts_flat), 3) if kpts_flat[j] == 2)
        # --- End Keypoint Preprocessing ---

        annotation_map.setdefault(key_id, []).append(ann_copy)
        max_ann_id = max(max_ann_id, ann.get("id", 0))
        processed_count += 1

    if skipped_count > 0:
        print(f"Warning: Skipped {skipped_count} annotations because their 'image_id' (frame number) was not found in the video.")

    categories = data.get("categories", [])
    print(f"Mapped {processed_count} annotations.")
    if not annotation_map and processed_count == 0 and len(annotations_in_file) > 0:
         prefix = "frame numbers" if is_video_mode else "image IDs"
         messagebox.showwarning(
             "Annotation Warning",
             f"Found {len(annotations_in_file)} annotations in the JSON, but none matched the {prefix} from the source."
         )


    next_new_ann_id = max_ann_id + 1

    return annotation_map, categories, next_new_ann_id


def load_coco_data(
    coco_json_path: str, img_dir: str
) -> Tuple[Optional[Dict[int, Dict]], Optional[Dict[int, List[Dict]]], Optional[List[Dict]], int]:
    """Loads COCO data for IMAGE mode and organizes it for the GUI."""
    print(f"--- Loading IMAGE data ---")
    print(f"COCO JSON: {coco_json_path}")
    print(f"Image Dir: {img_dir}")
    # Strip potential quotes from the image directory path
    cleaned_img_dir = img_dir.strip('\"\'')
    print(f"Using cleaned image directory: {cleaned_img_dir}")

    # First, load image info to know which image IDs are valid
    cleaned_coco_path = coco_json_path.strip('\"\'')
    try:
        with open(cleaned_coco_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        messagebox.showerror("Error", f"Error opening/reading COCO JSON file: {cleaned_coco_path}\n{e}")
        return None, None, None, -1

    image_map = {}
    valid_image_ids = set()
    print("Mapping images...")
    for img in data.get("images", []):
        img_id = img["id"]
        img_path = os.path.join(cleaned_img_dir, img["file_name"])
        if not os.path.exists(img_path):
            print(f"Warning: Image file not found, skipping: {img_path}")
            continue
        image_map[img_id] = {
            "path": img_path,
            "width": img["width"],
            "height": img["height"],
            "file_name": img["file_name"],
            "original_width": img["width"],
            "original_height": img["height"],
            "id": img_id, # Store id for consistency
        }
        valid_image_ids.add(img_id)

    if not image_map:
        messagebox.showerror(
            "Error",
            f"No valid image paths found based on COCO file and image directory.",
        )
        return None, None, None, -1

    # Now load annotations, filtering by valid image IDs
    annotation_map, categories, next_new_ann_id = _load_and_organize_coco_json(
        coco_json_path, valid_ids=valid_image_ids, is_video_mode=False
    )

    if annotation_map is None: # Check if annotation loading failed
        return None, None, None, -1

    print(f"Loaded info for {len(image_map)} images and associated annotations.")
    return image_map, annotation_map, categories, next_new_ann_id


def load_video_data(
    video_path: str, coco_json_path: str
) -> Tuple[Optional[Dict[int, Dict]], Optional[Dict[int, List[Dict]]], Optional[List[Dict]], int, Optional[cv2.VideoCapture]]:
    """Loads VIDEO data, COCO annotations, and organizes by frame number."""
    print(f"--- Loading VIDEO data ---")
    print(f"Video Path: {video_path}")
    print(f"COCO JSON: {coco_json_path}")

    cleaned_video_path = video_path.strip('\"\'')
    cap = cv2.VideoCapture(cleaned_video_path)

    if not cap.isOpened():
        messagebox.showerror("Error", f"Could not open video file: {cleaned_video_path}")
        return None, None, None, -1, None

    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if frame_count <= 0 or frame_width <= 0 or frame_height <= 0:
         messagebox.showerror("Error", f"Video file has invalid properties (frames/width/height): {cleaned_video_path}")
         cap.release()
         return None, None, None, -1, None

    print(f"Video properties: {frame_width}x{frame_height}, {frame_count} frames, {fps:.2f} FPS")

    # Create frame map (equivalent to image_map but without file paths)
    # Use 1-based indexing for frames to match common convention and COCO image_id
    frame_map = {}
    valid_frame_ids = set()
    for i in range(1, frame_count + 1):
        frame_id = i
        frame_map[frame_id] = {
            "width": frame_width,
            "height": frame_height,
            "original_width": frame_width, # Store original dims
            "original_height": frame_height,
            "id": frame_id,
            "file_name": f"frame_{frame_id:06d}", # Create a dummy filename for display
        }
        valid_frame_ids.add(frame_id)

    # Load annotations using the helper, filtering by valid frame IDs
    annotation_map, categories, next_new_ann_id = _load_and_organize_coco_json(
        coco_json_path, valid_ids=valid_frame_ids, is_video_mode=True
    )

    if annotation_map is None: # Check if annotation loading failed
        cap.release()
        return None, None, None, -1, None

    print(f"Loaded annotations for {len(annotation_map)} frames out of {frame_count} total frames.")

    # Important: Keep the video capture object open! Return it.
    return frame_map, annotation_map, categories, next_new_ann_id, cap


# --- NEW FUNCTION: Load data from custom results format ---
def load_results_video_data(
    video_path: str, results_json_path: str
) -> Tuple[Optional[Dict[int, Dict]], Optional[Dict[int, List[Dict]]], Optional[List[Dict]], int, Optional[cv2.VideoCapture]]:
    """Loads VIDEO data and custom format RESULTS annotations, organizes by frame number."""
    print(f"--- Loading VIDEO data with RESULTS format annotations ---")
    print(f"Video Path: {video_path}")
    print(f"Results JSON: {results_json_path}")

    # --- Load Video (Same as load_video_data) ---
    cleaned_video_path = video_path.strip('\"\'')
    cap = cv2.VideoCapture(cleaned_video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", f"Could not open video file: {cleaned_video_path}")
        return None, None, None, -1, None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if frame_count <= 0 or frame_width <= 0 or frame_height <= 0:
         messagebox.showerror("Error", f"Video file has invalid properties: {cleaned_video_path}")
         cap.release()
         return None, None, None, -1, None
    print(f"Video properties: {frame_width}x{frame_height}, {frame_count} frames, {fps:.2f} FPS")

    # Create frame map (1-based index)
    frame_map = {}
    valid_frame_ids = set()
    for i in range(1, frame_count + 1):
        frame_id = i
        frame_map[frame_id] = {
            "width": frame_width, "height": frame_height,
            "original_width": frame_width, "original_height": frame_height,
            "id": frame_id, "file_name": f"frame_{frame_id:06d}",
        }
        valid_frame_ids.add(frame_id)
    # --- End Video Loading ---

    # --- Load and Parse Results JSON ---
    print(f"Loading results annotations from: {results_json_path}")
    cleaned_results_path = results_json_path.strip('\"\'')
    try:
        with open(cleaned_results_path, "r") as f:
            results_data = json.load(f)
    except Exception as e:
        messagebox.showerror("Error", f"Error opening/reading results JSON file: {cleaned_results_path}\n{e}")
        cap.release()
        return None, None, None, -1, None

    annotation_map = {frame_id: [] for frame_id in valid_frame_ids} # Initialize for all frames
    max_internal_ann_id = 0
    processed_count = 0
    skipped_frame_count = 0

    instance_info_list = results_data.get("instance_info", [])
    print("Mapping results instances to internal annotation format...")

    for frame_entry in instance_info_list:
        frame_id = frame_entry.get("frame_id")
        if frame_id is None or frame_id not in valid_frame_ids:
            skipped_frame_count += 1
            continue

        instances = frame_entry.get("instances", [])
        for inst_idx, instance in enumerate(instances):
            max_internal_ann_id += 1 # Generate a unique internal ID
            internal_ann_id = max_internal_ann_id

            # --- Transform results instance to internal COCO-like format ---
            keypoints_list = instance.get("keypoints", []) # List of [x, y] pairs
            keypoint_scores = instance.get("keypoint_scores", []) # List of scores
            bbox_list = instance.get("bbox", []) # List containing one [x1, y1, x2, y2] list
            bbox_score = instance.get("bbox_score")
            instance_id_track = instance.get("instance_id") # The track ID (1 or 2)

            # Convert bbox [x1, y1, x2, y2] to [x, y, w, h]
            coco_bbox = []
            if bbox_list and len(bbox_list[0]) == 4:
                x1, y1, x2, y2 = bbox_list[0]
                coco_bbox = [x1, y1, x2 - x1, y2 - y1]
                area = (x2 - x1) * (y2 - y1)
            else:
                coco_bbox = [0, 0, 0, 0] # Default if missing/invalid
                area = 0

            # Convert keypoints [[x,y], [x,y]...] to [x,y,v, x,y,v...]
            # Assume v=2 if score exists and is > 0? Or maybe just load as v=2? Let's default to v=2 if present.
            coco_keypoints = []
            num_keypoints_visible = 0
            num_expected_kpts = len(keypoint_scores) if keypoint_scores else len(keypoints_list)

            for i in range(num_expected_kpts):
                if i < len(keypoints_list) and len(keypoints_list[i]) == 2:
                    kx, ky = keypoints_list[i]
                    # Determine visibility (e.g., v=2 if score > threshold, else v=1? Or just v=2 always?)
                    # For now, let's assume v=2 if the point exists. Handle v=0 later if needed.
                    kv = 2
                    num_keypoints_visible += 1
                else:
                    # Missing keypoint - use (0,0,0)
                    kx, ky, kv = 0, 0, 0

                coco_keypoints.extend([kx, ky, kv])

            # Pad with 0s if keypoints_list was shorter than keypoint_scores
            while len(coco_keypoints) < num_expected_kpts * 3:
                coco_keypoints.extend([0,0,0])


            # Create the internal annotation dictionary
            internal_ann = {
                "id": internal_ann_id, # Unique ID for GUI interaction
                "image_id": frame_id,
                "category_id": 1, # Assuming single category
                "bbox": coco_bbox,
                "area": area,
                "iscrowd": 0,
                "keypoints": coco_keypoints,
                "num_keypoints": num_keypoints_visible,
                "instance_id": instance_id_track, # Store the track ID

                # Store original results data for potential reconstruction during save
                "_original_results_instance": copy.deepcopy(instance)
            }

            # --- Preprocess Keypoints (randomize only (0,0)) ---
            # Apply the same preprocessing as in _load_and_organize_coco_json
            bbox_proc = internal_ann.get("bbox")
            kpts_flat_proc = internal_ann.get("keypoints")
            if bbox_proc and len(bbox_proc) == 4 and kpts_flat_proc and len(kpts_flat_proc) % 3 == 0:
                x_proc, y_proc, w_proc, h_proc = bbox_proc
                if w_proc > 0 and h_proc > 0:
                    modified_kpts_proc = False
                    for i_proc in range(0, len(kpts_flat_proc), 3):
                        kx_proc, ky_proc, kv_proc = kpts_flat_proc[i_proc], kpts_flat_proc[i_proc+1], kpts_flat_proc[i_proc+2]
                        if (kx_proc == 0 and ky_proc == 0):
                            new_kx_proc = max(x_proc, min(random.uniform(x_proc, x_proc + w_proc), x_proc + w_proc))
                            new_ky_proc = max(y_proc, min(random.uniform(y_proc, y_proc + h_proc), y_proc + h_proc))
                            kpts_flat_proc[i_proc] = new_kx_proc
                            kpts_flat_proc[i_proc+1] = new_ky_proc
                            kpts_flat_proc[i_proc+2] = 0 # Ensure visibility is 0
                            modified_kpts_proc = True
                    if modified_kpts_proc:
                        internal_ann["keypoints"] = kpts_flat_proc
                        internal_ann['num_keypoints'] = sum(1 for j in range(2, len(kpts_flat_proc), 3) if kpts_flat_proc[j] == 2)
            # --- End Keypoint Preprocessing ---

            annotation_map.setdefault(frame_id, []).append(internal_ann)
            processed_count += 1
            # --- End Transformation ---

    print(f"Mapped {processed_count} instances across {len(annotation_map)} frames.")
    if skipped_frame_count > 0:
        print(f"Warning: Skipped data for {skipped_frame_count} frame IDs not found in the video.")
    if not annotation_map and processed_count == 0 and len(instance_info_list) > 0:
         messagebox.showwarning("Annotation Warning", "Found instance data in JSON, but none matched the frame numbers from the video.")

    # TODO: Extract or define categories if possible/needed
    # For now, use a default category
    categories = [{"id": 1, "name": "object", "supercategory": "object"}]
    if instance_info_list:
         first_instance = instance_info_list[0].get("instances", [{}])[0]
         if first_instance and "keypoints" in first_instance:
              num_kpts = len(first_instance.get("keypoints",[]))
              # Try to create dummy keypoint names/skeleton if needed by GUI
              categories[0]["keypoints"] = [f"kpt_{i+1}" for i in range(num_kpts)]
              # No skeleton info in results format, so leave skeleton empty
              categories[0]["skeleton"] = []


    next_new_ann_id = max_internal_ann_id + 1

    return frame_map, annotation_map, categories, next_new_ann_id, cap


# --- Main Refinement GUI ---


class COCORefinementGUI:
    """Tkinter GUI for refining COCO annotations (Images or Video Frames)."""

    # Define a color palette for instances
    INSTANCE_COLORS = [
        "red", "blue", "lime", "magenta", "cyan", "yellow", "orange",
        "#FF7F50", "#DC143C", "#00FFFF", "#0000FF", "#8A2BE2", "#A52A2A",
        "#DEB887", "#5F9EA0", "#7FFF00", "#D2691E", "#FF7F50", "#6495ED",
        "#FFF8DC", "#DC143C", "#00FFFF", "#00008B", "#008B8B", "#B8860B",
        "#A9A9A9", "#006400", "#BDB76B", "#8B008B", "#556B2F", "#FF8C00",
        "#9932CC", "#8B0000", "#E9967A", "#8FBC8F", "#483D8B", "#2F4F4F",
        "#00CED1", "#9400D3", "#FF1493", "#00BFFF", "#696969", "#1E90FF",
        "#B22222", "#FFFAF0", "#228B22", "#FF00FF", "#DCDCDC", "#F8F8FF",
        "#FFD700", "#DAA520", "#808080", "#008000", "#ADFF2F", "#F0FFF0",
        "#FF69B4", "#CD5C5C", "#4B0082", "#FFFFF0", "#F0E68C", "#E6E6FA",
        "#FFF0F5", "#7CFC00", "#FFFACD", "#ADD8E6", "#F08080", "#E0FFFF",
        "#FAFAD2", "#D3D3D3", "#90EE90", "#FFB6C1", "#FFA07A", "#20B2AA",
        "#87CEFA", "#778899", "#B0C4DE", "#FFFFE0", "#00FF00", "#32CD32",
        "#FAF0E6", "#FF00FF", "#800000", "#66CDAA", "#0000CD", "#BA55D3",
        "#9370DB", "#3CB371", "#7B68EE", "#00FA9A", "#48D1CC", "#C71585",
        "#191970", "#F5FFFA", "#FFE4E1", "#FFE4B5", "#FFDEAD", "#000080",
        "#FDF5E6", "#808000", "#6B8E23", "#FFA500", "#FF4500", "#DA70D6",
        "#EEE8AA", "#98FB98", "#AFEEEE", "#DB7093", "#FFEFD5", "#FFDAB9",
        "#CD853F", "#FFC0CB", "#DDA0DD", "#B0E0E6", "#800080", "#FF0000",
        "#BC8F8F", "#4169E1", "#8B4513", "#FA8072", "#F4A460", "#2E8B57",
        "#FFF5EE", "#A0522D", "#C0C0C0", "#87CEEB", "#6A5ACD", "#708090",
        "#FFFAFA", "#00FF7F", "#4682B4", "#D2B48C", "#008080", "#D8BFD8",
        "#FF6347", "#40E0D0", "#EE82EE", "#F5DEB3", "#FFFFFF", "#F5F5F5",
        "#FFFF00", "#9ACD32"
    ]

    def __init__(
        self,
        source_data_map: Dict[int, Dict], # Combined map for images/frames
        annotation_map: Dict[int, List[Dict]], # Internal COCO-like format
        shrunk_annotation_map: Dict[int, List[Dict]], # Used only in image mode
        categories: List[Dict],
        initial_view_mode: str,
        input_json_path: str, # Original path (COCO or Results)
        next_start_ann_id: int,
        is_video_mode: bool = False,
        video_capture: Optional[cv2.VideoCapture] = None,
        loaded_from_results: bool = False, # NEW Flag
    ):
        # Common Initialization
        self.source_map = source_data_map
        self.categories = categories
        self.input_json_path = input_json_path # Store original input path
        # Generate output path based on original input path
        self.output_path = input_json_path.replace(".json", "_refined.json")
        self.is_video_mode = is_video_mode
        self.video_capture = video_capture
        self.loaded_from_results = loaded_from_results # Store flag

        # Store original and possibly shrunk annotations (internal format)
        self.original_annotations = copy.deepcopy(annotation_map)
        self.shrunk_annotations = copy.deepcopy(shrunk_annotation_map)

        # Holds the currently active annotations (internal format)
        self.start_in_comparison_mode = (not is_video_mode) and bool(shrunk_annotation_map)
        if self.start_in_comparison_mode:
            self.modified_annotations = {}
        else:
            self.modified_annotations = copy.deepcopy(self.original_annotations)

        self.source_ids = sorted(list(source_data_map.keys()))

        if not self.source_ids:
            messagebox.showerror("Error", f"No valid {'frames' if is_video_mode else 'images'} loaded. Exiting.")
            self.root = None
            if self.video_capture: self.video_capture.release() # Release video if init fails
            return

        self.current_idx = 0
        self.view_mode = initial_view_mode
        self.next_start_ann_id = next_start_ann_id


        # Initialize main window
        self.root = tk.Tk()
        # Determine mode string for title
        mode_str = "Video" if is_video_mode else "Image"
        if loaded_from_results: mode_str += " (Results Format)"
        self.root.title(f"COCO Annotation Refinement ({mode_str} Mode)")
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
        self.root.geometry("1000x750")

        # --- Zoom/Pan State ---
        self.zoom_level = 1.0
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0
        self.is_panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.base_scale = 1.0
        self.current_cv_image = None # Holds the loaded image/frame
        self.photo = None
        self.flip_mode = 'none'

        # --- Mouse Interaction State ---
        self.dragging = False
        self.drag_start_canvas = None
        self.selected_ann_id = None
        self.selected_element_type = None
        self.drag_type = None
        self.selected_corner = None
        self.selected_kpt_index = None
        self.new_box_start = None
        self.creating_new_box = False
        self.handle_size = 8
        self.kpt_click_radius = 5  # Will be synced with keypoint_radius

        # --- GUI State ---
        # In video mode, all frames start as 'editable' (equivalent to 'original' chosen)
        # In image mode, use comparison or 'original' based on shrink presence
        self.source_decision_state: Dict[int, str] = {
            src_id: 'undecided' if self.start_in_comparison_mode else 'original'
            for src_id in self.source_ids
        }
        self.deleted_source_ids = set()
        self.modifications_made = False
        self.current_frame_number_for_video = -1
        
        # --- Enhanced GUI State ---
        self.sidebar_visible = True
        self.delete_confirmation_enabled = True
        self.keypoint_radius = 5  # Default keypoint radius
        self.undo_stack = []  # For instance deletion undo
        self.sidebar_width = 250

        # --- Annotation Info ---
        self.skeleton_links = []
        self.keypoint_names = []
        # Use the potentially derived categories from loading
        if self.categories and self.categories[0].get("keypoints"):
            self.keypoint_names = self.categories[0].get("keypoints", [])
            skeleton_raw = self.categories[0].get("skeleton", [])
            print(f"DEBUG: Loaded {len(self.keypoint_names)} keypoint names: {self.keypoint_names[:5]}...")
            print(f"DEBUG: Loaded {len(skeleton_raw)} skeleton links: {skeleton_raw[:5]}...")
            if skeleton_raw:
                # Detect index base: 0-based vs 1-based
                try:
                    flat = [idx for pair in skeleton_raw for idx in pair if isinstance(idx, (int, np.integer))]
                    num_kpts = len(self.keypoint_names)
                    max_idx = max(flat) if flat else -1
                    min_idx = min(flat) if flat else 0
                    # If any index is 0, treat as 0-based; if any index equals num_kpts, treat as 1-based
                    if min_idx == 0 or (max_idx <= num_kpts - 1):
                        self.skeleton_links = [[int(a), int(b)] for a, b in skeleton_raw if len((a, b)) == 2]
                    else:
                        self.skeleton_links = [[int(a)-1, int(b)-1] for a, b in skeleton_raw if len((a, b)) == 2]
                    print(f"DEBUG: Processed skeleton links: {self.skeleton_links[:5]}...")
                except Exception:
                    self.skeleton_links = []
            # Fallback to COCO 17 skeleton if missing
            if not self.skeleton_links and len(self.keypoint_names) in (17,):
                default_coco17 = [
                    [16,14],[14,12],[17,15],[15,13],[12,13],[6,7],
                    [6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5]
                ]
                self.skeleton_links = [[a-1,b-1] for a,b in default_coco17]
                print(f"DEBUG: Using fallback COCO 17 skeleton: {self.skeleton_links[:5]}...")

        # If no keypoint names from categories, infer from first annotation
        if not self.keypoint_names:
            try:
                # Search original or modified annotations for first keypoints array
                all_maps = [self.original_annotations, self.modified_annotations]
                inferred_K = 0
                for amap in all_maps:
                    for _, anns in amap.items():
                        for ann in anns:
                            k = ann.get("keypoints")
                            if k and len(k) % 3 == 0 and len(k) >= 3:
                                inferred_K = len(k) // 3
                                break
                        if inferred_K:
                            break
                    if inferred_K:
                        break
                if inferred_K:
                    # Use standard COCO-17 keypoint names if we have 17 keypoints
                    if inferred_K == 17:
                        self.keypoint_names = [
                            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                            "left_wrist", "right_wrist", "left_hip", "right_hip",
                            "left_knee", "right_knee", "left_ankle", "right_ankle"
                        ]
                        print(f"Using standard COCO-17 keypoint names: {self.keypoint_names[:5]}...")
                    else:
                        self.keypoint_names = [f"kpt_{i+1}" for i in range(inferred_K)]
                    
                    # Use standard COCO-17 skeleton if we have 17 keypoints
                    if not self.skeleton_links and inferred_K == 17:
                        default_coco17 = [
                            [16,14],[14,12],[17,15],[15,13],[12,13],[6,7],
                            [6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5]
                        ]
                        self.skeleton_links = [[a-1,b-1] for a,b in default_coco17]
                        print(f"Using standard COCO-17 skeleton: {self.skeleton_links[:5]}...")
                    
                    print(f"DEBUG: Inferred {inferred_K} keypoints: {self.keypoint_names[:5]}...")
            except Exception:
                pass
        
        # Add additional skeleton links from hips to shoulders if we have keypoint names
        if self.keypoint_names:
            additional_links = []
            # Find hip and shoulder keypoint indices with their side information
            hip_indices = []
            shoulder_indices = []
            
            for i, kpt_name in enumerate(self.keypoint_names):
                if 'hip' in kpt_name.lower():
                    hip_indices.append((i, kpt_name))
                elif 'shoulder' in kpt_name.lower():
                    shoulder_indices.append((i, kpt_name))
            
            # Add hip-to-shoulder connections only for matching sides
            for hip_idx, hip_name in hip_indices:
                for shoulder_idx, shoulder_name in shoulder_indices:
                    # Only connect if both are left or both are right
                    if ('left' in hip_name.lower() and 'left' in shoulder_name.lower()) or \
                       ('right' in hip_name.lower() and 'right' in shoulder_name.lower()):
                        additional_links.append([hip_idx, shoulder_idx])
            
            # Combine with existing skeleton links
            if additional_links:
                self.skeleton_links.extend(additional_links)
                print(f"Added {len(additional_links)} hip-to-shoulder skeleton connections")
                print(f"DEBUG: Final skeleton links: {self.skeleton_links[:10]}...")
        
        print(f"DEBUG: Final keypoint names count: {len(self.keypoint_names)}")
        print(f"DEBUG: Final skeleton links count: {len(self.skeleton_links)}")

        # Mask overlay toggle and storage for PhotoImages
        self.show_masks = False
        self.photo_masks: List[ImageTk.PhotoImage] = []

        self._setup_gui()
        if self.source_ids:
            self._load_current_source()
        else:
             messagebox.showinfo("Info", f"No {'frames' if is_video_mode else 'images'} found to display.")


    def _setup_gui(self):
        """Setup enhanced GUI layout with menu bar, collapsible sidebar, and improved controls"""
        # Create menu bar
        self._create_menu_bar()
        
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Top toolbar frame
        toolbar_frame = ttk.Frame(main_frame)
        toolbar_frame.pack(fill=tk.X, pady=(0, 5))
        self._create_toolbar(toolbar_frame)
        
        # Content frame (canvas + sidebar)
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas frame with larger size
        canvas_frame = ttk.Frame(content_frame)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Collapsible sidebar
        self._create_sidebar(content_frame)
        
        # Bottom status bar
        self._create_status_bar(main_frame)


        # --- Bind mouse events ---
        self.canvas.bind("<Button-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)  # For bbox zoom
        self.canvas.bind("<Button-2>", self._on_pan_start)
        self.canvas.bind("<B2-Motion>", self._on_pan_move)
        self.canvas.bind("<ButtonRelease-2>", self._on_pan_end)
        if sys.platform == "darwin": self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        elif sys.platform.startswith("linux"):
            self.canvas.bind("<Button-4>", self._on_mouse_wheel)
            self.canvas.bind("<Button-5>", self._on_mouse_wheel)
        else: self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Motion>", self._on_mouse_hover)
        self.canvas.bind("<Configure>", self._on_canvas_resize)


        # --- Bind keyboard shortcuts ---
        self.root.bind("<Left>", lambda e: self._prev_source())
        self.root.bind("<Right>", lambda e: self._next_source())
        self.root.bind("<Delete>", lambda e: self._delete_current_source())
        self.root.bind("<Escape>", lambda e: self._complete())
        self.root.bind("<t>", lambda e: self._toggle_view_mode())
        self.root.bind("<n>", lambda e: self._toggle_new_box_mode())
        self.root.bind("<r>", lambda e: self._reset_view())
        self.root.bind("<Control-s>", self._save_temp_progress)
        self.root.bind("<Control-z>", lambda e: self._undo_last_deletion())  # Undo
        self.root.bind("f", self._toggle_flip_mode)
        self.root.bind("m", lambda e: self._toggle_masks())

        # Comparison mode keys (Only bind if not in video mode)
        if not self.is_video_mode:
            self.root.bind("<KeyPress-a>", self._accept_original)
            self.root.bind("<KeyPress-d>", self._accept_shrunk)

        # Track ID keys (Only bind if in video mode)
        if self.is_video_mode:
            self.root.bind("<KeyPress-1>", lambda e: self._assign_track_id(1))
            self.root.bind("<KeyPress-2>", lambda e: self._assign_track_id(2))

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close_window)
        
    def _create_menu_bar(self):
        """Create the main menu bar with dropdown menus"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save (Ctrl+S)", command=self._save_temp_progress)
        file_menu.add_command(label="Complete & Save (Esc)", command=self._complete)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close_window)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo Delete (Ctrl+Z)", command=self._undo_last_deletion)
        edit_menu.add_command(label="New Box (N)", command=self._toggle_new_box_mode)
        edit_menu.add_separator()
        # Create a persistent variable for the checkbutton
        self._delete_confirm_var = tk.BooleanVar(value=self.delete_confirmation_enabled)
        edit_menu.add_checkbutton(label="Delete Confirmation", variable=self._delete_confirm_var, command=self._toggle_delete_confirmation)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Toggle View Mode (T)", command=self._toggle_view_mode)
        view_menu.add_command(label="Reset View (R)", command=self._reset_view)
        view_menu.add_command(label="Toggle Masks (M)", command=self._toggle_masks)
        view_menu.add_command(label="Toggle Flip (F)", command=self._toggle_flip_mode)
        view_menu.add_separator()
        view_menu.add_command(label="Toggle Sidebar", command=self._toggle_sidebar)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Fit to Window", command=self._fit_to_window)
        tools_menu.add_command(label="Actual Size (100%)", command=self._actual_size)
        
    def _create_toolbar(self, parent):
        """Create the top toolbar with essential controls"""
        # Essential navigation buttons
        nav_frame = ttk.Frame(parent)
        nav_frame.pack(side=tk.LEFT, padx=5)
        
        prev_text = "◀ Prev" if self.is_video_mode else "◀ Previous"
        next_text = "Next ▶" if self.is_video_mode else "Next ▶"
        
        ttk.Button(nav_frame, text=prev_text, command=self._prev_source).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text=next_text, command=self._next_source).pack(side=tk.LEFT, padx=2)
        
        # Jumper entry
        jumper_frame = ttk.Frame(parent)
        jumper_frame.pack(side=tk.LEFT, padx=20)
        
        jumper_label_text = "Frame:" if self.is_video_mode else "Image:"
        ttk.Label(jumper_frame, text=jumper_label_text).pack(side=tk.LEFT, padx=(0, 5))
        self.source_jumper_var = tk.StringVar()
        self.source_jumper_entry = ttk.Entry(jumper_frame, textvariable=self.source_jumper_var, width=10)
        self.source_jumper_entry.pack(side=tk.LEFT)
        self.source_jumper_entry.bind("<Return>", self._jump_to_source)
        
        # Mode indicators
        indicators_frame = ttk.Frame(parent)
        indicators_frame.pack(side=tk.LEFT, padx=20)
        
        self.view_mode_label = tk.StringVar(value=f"View: {self.view_mode.capitalize()}")
        ttk.Label(indicators_frame, textvariable=self.view_mode_label).pack(side=tk.LEFT, padx=5)
        
        self.new_box_label = tk.StringVar(value="")
        ttk.Label(indicators_frame, textvariable=self.new_box_label, foreground="blue").pack(side=tk.LEFT, padx=5)
        
        # Track ID info (video mode only)
        if self.is_video_mode:
            self.track_id_info_label = tk.StringVar(value="Track ID: Select box, press '1' or '2'")
            ttk.Label(indicators_frame, textvariable=self.track_id_info_label, foreground="green").pack(side=tk.LEFT, padx=5)
        
    def _create_sidebar(self, parent):
        """Create the collapsible right sidebar with controls"""
        self.sidebar_frame = ttk.Frame(parent, width=self.sidebar_width)
        self.sidebar_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        self.sidebar_frame.pack_propagate(False)
        
        # Sidebar header with collapse button
        header_frame = ttk.Frame(self.sidebar_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(header_frame, text="Controls", font=('TkDefaultFont', 10, 'bold')).pack(side=tk.LEFT)
        self.collapse_btn = ttk.Button(header_frame, text="◀", width=3, command=self._toggle_sidebar)
        self.collapse_btn.pack(side=tk.RIGHT)
        
        # View controls section
        view_section = ttk.LabelFrame(self.sidebar_frame, text="View Controls", padding=10)
        view_section.pack(fill=tk.X, pady=5)
        
        ttk.Button(view_section, text="Toggle View (T)", command=self._toggle_view_mode).pack(fill=tk.X, pady=2)
        ttk.Button(view_section, text="Reset View (R)", command=self._reset_view).pack(fill=tk.X, pady=2)
        ttk.Button(view_section, text="Toggle Masks (M)", command=self._toggle_masks).pack(fill=tk.X, pady=2)
        
        # Keypoint controls section
        kpt_section = ttk.LabelFrame(self.sidebar_frame, text="Keypoint Controls", padding=10)
        kpt_section.pack(fill=tk.X, pady=5)
        
        ttk.Label(kpt_section, text="Keypoint Size:").pack(anchor=tk.W)
        self.keypoint_scale = tk.Scale(kpt_section, from_=1, to=20, orient=tk.HORIZONTAL, 
                                     command=self._on_keypoint_scale_change, length=180)
        self.keypoint_scale.set(self.keypoint_radius)
        self.keypoint_scale.pack(fill=tk.X, pady=2)
        
        # Annotation controls section
        ann_section = ttk.LabelFrame(self.sidebar_frame, text="Annotation Controls", padding=10)
        ann_section.pack(fill=tk.X, pady=5)
        
        ttk.Button(ann_section, text="New Box (N)", command=self._toggle_new_box_mode).pack(fill=tk.X, pady=2)
        del_text = "Delete Frame (Del)" if self.is_video_mode else "Delete Image (Del)"
        ttk.Button(ann_section, text=del_text, command=self._delete_current_source).pack(fill=tk.X, pady=2)
        ttk.Button(ann_section, text="Undo Delete (Ctrl+Z)", command=self._undo_last_deletion).pack(fill=tk.X, pady=2)
        
        # Status section
        status_section = ttk.LabelFrame(self.sidebar_frame, text="Status", padding=10)
        status_section.pack(fill=tk.X, pady=5)
        
        self.flip_mode_label = tk.StringVar(value="Flip: None")
        ttk.Label(status_section, textvariable=self.flip_mode_label).pack(anchor=tk.W)
        
    def _create_status_bar(self, parent):
        """Create the bottom status bar"""
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(5, 0))
        
        # Zoom info
        self.zoom_label = tk.StringVar(value="Zoom: 100%")
        ttk.Label(status_frame, textvariable=self.zoom_label).pack(side=tk.LEFT)
        
        # Action buttons on the right
        ttk.Button(status_frame, text="Save", command=self._save_temp_progress).pack(side=tk.RIGHT, padx=5)
        ttk.Button(status_frame, text="Complete", command=self._complete).pack(side=tk.RIGHT, padx=5)
        
    def _toggle_sidebar(self):
        """Toggle the visibility of the sidebar"""
        if self.sidebar_visible:
            self.sidebar_frame.pack_forget()
            self.collapse_btn.config(text="▶")
            self.sidebar_visible = False
        else:
            self.sidebar_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
            self.collapse_btn.config(text="◀")
            self.sidebar_visible = True
            
    def _toggle_delete_confirmation(self):
        """Toggle the delete confirmation setting"""
        self.delete_confirmation_enabled = self._delete_confirm_var.get()
        print(f"Delete confirmation {'enabled' if self.delete_confirmation_enabled else 'disabled'}")
        
    def _on_keypoint_scale_change(self, value):
        """Handle keypoint size slider change"""
        self.keypoint_radius = int(float(value))
        self.kpt_click_radius = self.keypoint_radius
        # Redraw annotations with new keypoint size
        self._draw_annotations()
        
    def _fit_to_window(self):
        """Fit the current image to the window"""
        self.zoom_level = 1.0
        self._calculate_initial_scale_and_pan()
        self._display_source()
        self._draw_annotations()
        self._update_zoom_label()
        
    def _actual_size(self):
        """Display image at actual size (100%)"""
        if self.current_cv_image is not None:
            self.zoom_level = 1.0
            self.base_scale = 1.0
            self.pan_offset_x = 0
            self.pan_offset_y = 0
            self._display_source()
            self._draw_annotations()
            self._update_zoom_label()
            
    def _undo_last_deletion(self):
        """Undo the last instance deletion"""
        if not self.undo_stack:
            print("No deletions to undo")
            return
            
        last_action = self.undo_stack.pop()
        if last_action['type'] == 'delete_instance':
            # Restore the deleted annotation
            src_id = last_action['source_id']
            annotation = last_action['annotation']
            
            if src_id not in self.modified_annotations:
                self.modified_annotations[src_id] = []
            self.modified_annotations[src_id].append(annotation)
            
            print(f"Undone deletion of annotation {annotation['id']}")
            self.modifications_made = True
            self._draw_annotations()
        elif last_action['type'] == 'delete_source':
            # Restore the deleted source
            src_id = last_action['source_id']
            annotations = last_action['annotations']
            
            # Remove from deleted set
            self.deleted_source_ids.discard(src_id)
            
            # Restore annotations if they were saved
            if annotations:
                self.modified_annotations[src_id] = annotations
            
            print(f"Undone deletion of {'frame' if self.is_video_mode else 'image'} {src_id}")
            self.modifications_made = True
            self._draw_annotations()
            
    def _on_double_click(self, event):
        """Handle double-click for bbox zoom functionality"""
        # Check if double-click is on a bbox in the top-right area
        current_src_id = self.source_ids[self.current_idx]
        state = self.source_decision_state.get(current_src_id, 'undecided')
        if state == 'undecided':
            return
            
        # Get annotations for current source
        annotations = self.modified_annotations.get(current_src_id, [])
        if not annotations:
            return
            
        # Check if click is near the top-right corner of any bbox
        click_x, click_y = event.x, event.y
        
        for ann in annotations:
            bbox = ann.get('bbox')
            if not bbox:
                continue
                
            x, y, w, h = bbox
            # Convert to canvas coordinates
            x1_canvas, y1_canvas = self._image_to_canvas_coords(x, y)
            x2_canvas, y2_canvas = self._image_to_canvas_coords(x + w, y + h)
            
            # Check if click is in top-right corner region (10% of bbox size)
            corner_size = min(abs(x2_canvas - x1_canvas), abs(y2_canvas - y1_canvas)) * 0.1
            corner_size = max(corner_size, 10)  # Minimum 10 pixels
            
            if (x2_canvas - corner_size <= click_x <= x2_canvas and 
                y1_canvas <= click_y <= y1_canvas + corner_size):
                # Zoom to this bbox
                self._zoom_to_bbox(bbox)
                return
                
    def _zoom_to_bbox(self, bbox, padding_factor=0.2):
        """Zoom to fit the given bbox in the canvas with padding"""
        x, y, w, h = bbox
        
        # Calculate canvas dimensions
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1:
            canvas_w, canvas_h = 1000, 700
            
        # Add padding
        padding_w = w * padding_factor
        padding_h = h * padding_factor
        bbox_w_padded = w + 2 * padding_w
        bbox_h_padded = h + 2 * padding_h
        
        # Calculate zoom level to fit bbox with padding
        zoom_w = canvas_w / bbox_w_padded if bbox_w_padded > 0 else 1.0
        zoom_h = canvas_h / bbox_h_padded if bbox_h_padded > 0 else 1.0
        target_zoom = min(zoom_w, zoom_h)
        
        # Set zoom level
        self.zoom_level = target_zoom
        
        # Calculate pan to center the bbox
        bbox_center_x = x + w / 2
        bbox_center_y = y + h / 2
        
        # Convert to canvas coordinates at new zoom
        eff_scale = self.base_scale * self.zoom_level
        target_canvas_x = canvas_w / 2
        target_canvas_y = canvas_h / 2
        
        self.pan_offset_x = target_canvas_x - bbox_center_x * eff_scale
        self.pan_offset_y = target_canvas_y - bbox_center_y * eff_scale
        
        # Update display
        self._display_source()
        self._draw_annotations()
        
        print(f"Zoomed to bbox at ({x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}) with zoom {target_zoom:.2f}x")
        self._update_zoom_label()
        
    def _update_zoom_label(self):
        """Update the zoom percentage in the status bar"""
        if hasattr(self, 'zoom_label'):
            # Use base_scale if available, otherwise default to 1.0
            base_scale = getattr(self, 'base_scale', 1.0)
            zoom_percent = int(self.zoom_level * base_scale * 100)
            self.zoom_label.set(f"Zoom: {zoom_percent}%")

    # --- Add Placeholder for _assign_track_id ---
    def _assign_track_id(self, track_id_to_assign: int):
        """Assigns the given track ID (1 or 2) to the selected annotation.
        If changing an existing ID (1->2 or 2->1), prompts to change all instances globally,
        swapping IDs within each affected frame to maintain exclusivity."""
        if not self.is_video_mode: return # Only active in video mode

        if self.selected_ann_id is None:
            print("No annotation selected to assign track ID.")
            messagebox.showinfo("Assign Track ID", "Please click inside a box first to select it.")
            return

        current_src_id = self.source_ids[self.current_idx]
        ann = self._find_annotation_by_id(self.selected_ann_id)

        if not ann:
            print(f"Error: Could not find selected annotation {self.selected_ann_id} to assign track ID.")
            return

        old_track_id = ann.get("instance_id") # Get current ID, might be None
        new_track_id = track_id_to_assign # The ID we want to assign

        # --- Global Change Logic ---
        apply_globally = False
        # Only prompt for global change if changing an *existing* valid ID (1 or 2) to the *other* valid ID.
        if old_track_id is not None and old_track_id in [1, 2] and new_track_id in [1, 2] and old_track_id != new_track_id:
             prompt_message = (
                 f"Globally SWAP Track ID {old_track_id} with {new_track_id} "
                 f"throughout the entire video?"
             )
             if messagebox.askyesno("Global Track ID Swap", prompt_message):
                 apply_globally = True
             else:
                 print("Global swap cancelled. Only changing current annotation.")
        # --- End Global Change Logic ---

        if apply_globally:
            print(f"Applying global swap: Track ID {old_track_id} <-> {new_track_id}")
            global_change_count_old_to_new = 0
            global_change_count_new_to_old = 0
            frames_affected_count = 0
            affected_frames = set()

            # Iterate through all source IDs (frames)
            for src_id in self.source_ids:
                frame_anns = self.modified_annotations.get(src_id, [])
                if not frame_anns: continue # Skip empty frames

                # Find annotations with the old and new IDs *within this frame*
                anns_with_old_id = [fa for fa in frame_anns if fa.get("instance_id") == old_track_id]
                anns_with_new_id = [fa for fa in frame_anns if fa.get("instance_id") == new_track_id]

                frame_changed = False
                # Perform the swap only if the old ID actually exists in this frame
                if anns_with_old_id:
                    for ann_to_change in anns_with_old_id:
                        ann_to_change["instance_id"] = new_track_id
                        global_change_count_old_to_new += 1
                        frame_changed = True

                    # Only swap the 'new' ones to 'old' if we actually changed some 'old' ones
                    for ann_to_swap in anns_with_new_id:
                        ann_to_swap["instance_id"] = old_track_id
                        global_change_count_new_to_old += 1
                        # frame_changed is already True if we got here

                if frame_changed:
                    affected_frames.add(src_id)

            frames_affected_count = len(affected_frames)
            print(f"Globally changed {global_change_count_old_to_new} annotations from {old_track_id} -> {new_track_id}.")
            print(f"Globally changed {global_change_count_new_to_old} annotations from {new_track_id} -> {old_track_id}.")
            print(f"Total frames affected: {frames_affected_count}.")
            self.modifications_made = True # Mark modification

        else:
            # Apply change only to the currently selected annotation
            # Check if the new ID is already taken by another box *in the current frame*
            current_frame_anns = self.modified_annotations.get(current_src_id, [])
            id_already_taken = any(
                fa.get("instance_id") == new_track_id and fa["id"] != self.selected_ann_id
                for fa in current_frame_anns
            )

            if id_already_taken:
                messagebox.showwarning("Track ID Conflict", f"Track ID {new_track_id} is already assigned to another box in this frame. Cannot assign duplicate ID.")
                print(f"Assignment cancelled: Track ID {new_track_id} already exists in frame {current_src_id}.")
            elif ann.get("instance_id") != new_track_id:
                ann["instance_id"] = new_track_id
                print(f"Assigned Track ID {new_track_id} to annotation {self.selected_ann_id} on frame {current_src_id}")
                self.modifications_made = True
            else:
                print(f"Annotation {self.selected_ann_id} already has Track ID {new_track_id}.")

        # Redraw the current frame's annotations
        self._draw_annotations()


    def _find_valid_index(self, direction=1):
        """Find the next/previous valid (not deleted) image/frame index."""
        if not self.source_ids: # Renamed
            return -1

        original_idx = self.current_idx
        new_idx = self.current_idx
        num_sources = len(self.source_ids) # Renamed
        checked_count = 0

        while checked_count < num_sources:
            new_idx = (new_idx + direction + num_sources) % num_sources
            if self.source_ids[new_idx] not in self.deleted_source_ids: # Renamed
                return new_idx
            if new_idx == original_idx:
                 break
            checked_count += 1

        if num_sources == 1 and self.source_ids[original_idx] not in self.deleted_source_ids: # Renamed
            return original_idx

        print(f"Info: All {'frames' if self.is_video_mode else 'images'} appear to be marked for deletion.")
        return -1

    # --- Coordinate Transformation Helpers ---
    # (Keep _canvas_to_image_coords and _image_to_canvas_coords as they are)
    def _canvas_to_image_coords(self, canvas_x: float, canvas_y: float) -> Tuple[float, float]:
        """Converts canvas coordinates to original image/frame coordinates."""
        eff_scale = self.base_scale * self.zoom_level
        if eff_scale == 0: return 0, 0
        img_x = (canvas_x - self.pan_offset_x) / eff_scale
        img_y = (canvas_y - self.pan_offset_y) / eff_scale

        if self.current_cv_image is not None:
            H, W = self.current_cv_image.shape[:2]
            if 'h' in self.flip_mode: img_x = W - 1 - img_x
            if 'v' in self.flip_mode: img_y = H - 1 - img_y

        return img_x, img_y

    def _image_to_canvas_coords(self, img_x: float, img_y: float) -> Tuple[float, float]:
        """Converts original image/frame coordinates to canvas coordinates."""
        if self.current_cv_image is not None:
            H, W = self.current_cv_image.shape[:2]
            if 'h' in self.flip_mode: img_x = W - 1 - img_x
            if 'v' in self.flip_mode: img_y = H - 1 - img_y

        eff_scale = self.base_scale * self.zoom_level
        canvas_x = img_x * eff_scale + self.pan_offset_x
        canvas_y = img_y * eff_scale + self.pan_offset_y
        return canvas_x, canvas_y


    # --- View Control Methods ---
    # (Keep _reset_view, _calculate_initial_scale_and_pan, _on_canvas_resize,
    #  _on_mouse_wheel, _on_pan_start, _on_pan_move, _on_pan_end as they are)

    def _reset_view(self, event=None):
        """Resets zoom and pan to default."""
        print("Resetting view.")
        self.zoom_level = 1.0
        self._calculate_initial_scale_and_pan()
        self._display_source() # Renamed
        self._draw_annotations()
        self._update_zoom_label()

    def _calculate_initial_scale_and_pan(self):
        """Calculates the initial scaling and centering pan offsets."""
        if self.current_cv_image is None: return

        h, w = self.current_cv_image.shape[:2]
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1:
            canvas_w, canvas_h = 1000, 700 # Use GUI size estimate

        self.base_scale = min(canvas_w / w, canvas_h / h) if w > 0 and h > 0 else 1.0

        display_w_at_base = w * self.base_scale
        display_h_at_base = h * self.base_scale
        self.pan_offset_x = (canvas_w - display_w_at_base) / 2
        self.pan_offset_y = (canvas_h - display_h_at_base) / 2

    def _on_canvas_resize(self, event):
         """Handles canvas resize events to keep image/frame centered."""
         self._calculate_initial_scale_and_pan()
         self._display_source() # Renamed
         self._draw_annotations()


    def _on_mouse_wheel(self, event):
        """Handles mouse wheel zooming."""
        zoom_factor = 1.1
        if sys.platform == "darwin": delta = event.delta
        elif sys.platform.startswith("linux"): delta = 120 if event.num == 4 else -120
        else: delta = event.delta

        anchor_canvas_x, anchor_canvas_y = event.x, event.y
        anchor_img_x, anchor_img_y = self._canvas_to_image_coords(anchor_canvas_x, anchor_canvas_y)

        if delta > 0: new_zoom_level = self.zoom_level * zoom_factor
        else: new_zoom_level = self.zoom_level / zoom_factor
        new_zoom_level = max(0.1, min(new_zoom_level, 10.0))

        scale_change = new_zoom_level / self.zoom_level
        self.pan_offset_x = anchor_canvas_x - (anchor_img_x * self.base_scale * new_zoom_level)
        self.pan_offset_y = anchor_canvas_y - (anchor_img_y * self.base_scale * new_zoom_level)
        self.zoom_level = new_zoom_level

        self._display_source() # Renamed
        self._draw_annotations()
        self._update_zoom_label()


    def _on_pan_start(self, event):
        """Starts panning when middle mouse button is pressed."""
        self.is_panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.canvas.config(cursor="fleur")

    def _on_pan_move(self, event):
        """Moves the image/frame based on middle mouse drag."""
        if self.is_panning:
            dx = event.x - self.pan_start_x
            dy = event.y - self.pan_start_y
            self.pan_offset_x += dx
            self.pan_offset_y += dy
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            self._display_source() # Renamed
            self._draw_annotations()

    def _on_pan_end(self, event):
        """Stops panning when middle mouse button is released."""
        self.is_panning = False
        self.canvas.config(cursor="")


    # --- Image/Frame and Annotation Display ---

    def _display_source(self): # Renamed from _display_image
         """Displays the current image/frame according to zoom/pan state."""
         # print(f"DEBUG: Entering _display_source: type={type(self.current_cv_image)}, is_none={self.current_cv_image is None}")
         if self.current_cv_image is None:
             self.canvas.delete("image") # Still use "image" tag for the display item
             return

         h, w = self.current_cv_image.shape[:2]
         effective_scale = self.base_scale * self.zoom_level
         display_w = int(w * effective_scale)
         display_h = int(h * effective_scale)

         if display_w <=0 or display_h <= 0: return

         img_to_display = self.current_cv_image
         if self.flip_mode == 'h': img_to_display = cv2.flip(self.current_cv_image, 1)
         elif self.flip_mode == 'v': img_to_display = cv2.flip(self.current_cv_image, 0)
         elif self.flip_mode == 'hv': img_to_display = cv2.flip(self.current_cv_image, -1)

         img_rgb = cv2.cvtColor(img_to_display, cv2.COLOR_BGR2RGB)
         image_pil = Image.fromarray(img_rgb)
         image_pil = image_pil.resize((display_w, display_h), Image.Resampling.LANCZOS)
         self.photo = ImageTk.PhotoImage(image_pil)

         self.canvas.delete("image")
         self.canvas.create_image(
             self.pan_offset_x, self.pan_offset_y, anchor=tk.NW, image=self.photo, tags="image"
         )
         self.canvas.tag_lower("image")


    def _load_current_source(self): # Renamed from _load_current_image
        """Load and display current image or video frame with annotations"""
        if not self.source_ids or self.current_idx < 0 or self.current_idx >= len(self.source_ids):
            self.canvas.delete("all")
            self.current_cv_image = None
            self.photo = None
            return

        current_src_id = self.source_ids[self.current_idx]
        src_info = self.source_map[current_src_id]
        error_occurred = False

        # --- Load based on mode ---
        if self.is_video_mode:
            if self.video_capture and self.video_capture.isOpened():
                # Video frames are 0-indexed for seeking, but our IDs are 1-indexed
                target_frame_zero_idx = current_src_id - 1

                # --- Simpler Seek Logic ---
                # Always set the position before reading, avoids complex state tracking issues.
                # print(f"Setting video position to frame index {target_frame_zero_idx}...") # Optional Debug Log
                set_success = self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame_zero_idx)
                if not set_success:
                     print(f"Warning: Failed to set video position to {target_frame_zero_idx}.")
                     # Attempting read anyway, might still work or fail gracefully

                ret, frame = self.video_capture.read()
                # --- End Simpler Seek Logic ---

                if ret:
                    self.current_cv_image = frame
                    # Update the tracked frame number *after* successfully reading it.
                    # Since set was to target_frame_zero_idx, and read gets that frame, this is correct.
                    self.current_frame_number_for_video = target_frame_zero_idx
                else:
                    messagebox.showerror("Error Reading Frame", f"Failed to read frame {current_src_id} (index {target_frame_zero_idx}) from video.")
                    error_occurred = True
                    self.current_cv_image = None
            else:
                 messagebox.showerror("Video Error", "Video capture is not open or available.")
                 error_occurred = True
                 self.current_cv_image = None
        else: # Image Mode
            img_path = src_info["path"]
            try:
                self.current_cv_image = cv2.imread(img_path)
                if self.current_cv_image is None:
                    raise IOError(f"cv2.imread returned None for: {img_path}")
            except Exception as e:
                messagebox.showerror(
                    "Error Loading Image", f"Error loading {img_path}:\n{e}"
                )
                error_occurred = True
                self.current_cv_image = None

        # --- Handle loading errors ---
        if error_occurred:
            self.deleted_source_ids.add(current_src_id) # Mark as deleted
            next_idx = self._find_valid_index(1) # Try to move to next
            if next_idx == -1 or next_idx == self.current_idx:
                if self.root: self._complete() # Close if nothing else to show
                return
            else:
                self.current_idx = next_idx
                self._load_current_source() # Recursively try next
                return

        # --- Reset view and display ---
        self.zoom_level = 1.0
        self._calculate_initial_scale_and_pan()
        # print(f"DEBUG: Before calling _display_source in _load_current_source: type={type(self.current_cv_image)}, is_none={self.current_cv_image is None}")
        self._display_source()

        # --- Display filename/frame number ---
        display_name = src_info.get("file_name", f"ID {current_src_id}")
        if self.is_video_mode: display_name = f"Frame {current_src_id}" # Override for video
        canvas_w = self.canvas.winfo_width()
        self.canvas.delete("filename_text")
        self.canvas.create_text(
            canvas_w - 10, 10,
            text=display_name,
            anchor=tk.NE,
            fill="yellow",
            font=("Arial", 10),
            tags="filename_text"
        )

        # --- Update counter/jumper ---
        self._update_jumper_text()

        # Draw annotations
        self._draw_annotations()


    def _draw_annotations(self):
        """Draw annotations based on current source decision state and view mode."""
        self.canvas.delete(
            "box", "handle", "pose", "keypoint", "skeleton", "comparison_box", "status_text", "hover_text", "track_id_text" # Clear specific tags + track ID
        )

        if not self.source_ids or self.current_idx < 0 or self.current_idx >= len(self.source_ids): return
        current_src_id = self.source_ids[self.current_idx]
        state = self.source_decision_state.get(current_src_id, 'undecided') # Use renamed state dict

        if state == 'undecided' and not self.is_video_mode: # Comparison only in image mode
            self._draw_comparison_boxes(current_src_id)
            self.canvas.create_text(
                self.canvas.winfo_width() / 2, 20,
                text="Compare: Original (Blue) vs Shrunk (Red). Press 'A' to keep Original, 'D' for Shrunk.",
                fill="yellow", font=("Arial", 12), tags="status_text"
            )
        else:
            # Active mode (video mode, or image mode after A/D/shrink)
            # Draw mask first (above image), then draw boxes/poses on top
            if self.show_masks:
                anns = self.modified_annotations.get(current_src_id, [])
                self._draw_active_masks(anns)
            self._draw_active_annotations(current_src_id)

    def _draw_comparison_boxes(self, img_id: int):
        """Draws original (blue) and different shrunk (red) boxes for comparison (IMAGE MODE ONLY)."""
        # (Keep existing logic - this won't be called in video mode)
        original_anns = self.original_annotations.get(img_id, [])
        shrunk_anns = self.shrunk_annotations.get(img_id, [])
        original_bboxes_by_id = {ann.get("id"): ann.get("bbox") for ann in original_anns if ann.get("id") is not None and ann.get("bbox")}
        shrunk_bboxes_by_id = {ann.get("id"): ann.get("bbox") for ann in shrunk_anns if ann.get("id") is not None and ann.get("bbox")}

        for ann_id, bbox in original_bboxes_by_id.items():
            x, y, w, h = bbox
            x1c, y1c = self._image_to_canvas_coords(x, y)
            x2c, y2c = self._image_to_canvas_coords(x + w, y + h)
            self.canvas.create_rectangle(x1c, y1c, x2c, y2c, outline="blue", width=2, tags=("comparison_box", f"ann{ann_id}_orig"))

        for ann_id, shrunk_bbox in shrunk_bboxes_by_id.items():
            original_bbox = original_bboxes_by_id.get(ann_id)
            if shrunk_bbox and (original_bbox is None or shrunk_bbox != original_bbox):
                x, y, w, h = shrunk_bbox
                x1c, y1c = self._image_to_canvas_coords(x, y)
                x2c, y2c = self._image_to_canvas_coords(x + w, y + h)
                self.canvas.create_rectangle(x1c, y1c, x2c, y2c, outline="red", width=2, tags=("comparison_box", f"ann{ann_id}_shrunk"))


    def _draw_active_annotations(self, src_id: int): # Renamed img_id -> src_id
        """Draw annotations from self.modified_annotations with editing handles/poses."""
        annotations = self.modified_annotations.get(src_id, [])
        if not annotations:
            return

        num_colors = len(self.INSTANCE_COLORS)
        for idx, ann in enumerate(annotations):
            instance_color = self.INSTANCE_COLORS[idx % num_colors]
            # Draw box and potentially track ID
            self._draw_active_boxes([ann], instance_color)
            if self.view_mode == "pose":
                self._draw_active_poses([ann], instance_color)

    def _draw_active_boxes(self, annotations: List[Dict], color: str = "red"):
        """Draw editable bounding boxes and handles from the active set. Adds Track ID text."""
        if not annotations: return
        ann = annotations[0]

        ann_id = ann["id"]
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            return

        x, y, w, h = bbox
        x1c, y1c = self._image_to_canvas_coords(x, y)
        x2c, y2c = self._image_to_canvas_coords(x + w, y + h)

        # Draw box
        box_tag = f"ann{ann_id}"
        self.canvas.create_rectangle(
            x1c, y1c, x2c, y2c, outline=color, width=2, tags=("box", box_tag, "editable")
        )

        # Draw handles
        handles_coords = [(x1c, y1c, "topleft"), (x2c, y1c, "topright"), (x1c, y2c, "bottomleft"), (x2c, y2c, "bottomright")]
        handle_draw_size = self.handle_size / 2
        for hx, hy, corner_tag in handles_coords:
            self.canvas.create_rectangle(
                hx - handle_draw_size, hy - handle_draw_size,
                hx + handle_draw_size, hy + handle_draw_size,
                fill="white", outline=color,
                tags=("handle", box_tag, corner_tag, "editable"),
            )

        # Draw Track ID if available (video mode)
        if self.is_video_mode and "instance_id" in ann and ann["instance_id"] is not None:
            track_id_text = f"ID: {ann['instance_id']}"
            # Position text slightly inside the top-left corner of the box
            text_x = x1c + 5
            text_y = y1c + 5
            self.canvas.create_text(
                text_x, text_y,
                text=track_id_text,
                anchor=tk.NW,
                fill=color, # Use instance color for text
                font=("Arial", 10, "bold"),
                tags=("track_id_text", box_tag, "editable"), # Tag it for deletion
            )

    def _toggle_masks(self):
        self.show_masks = not self.show_masks
        # Clear existing mask layer immediately to avoid stale overlay
        try:
            self.canvas.delete("mask_layer")
            self.photo_masks.clear()
        except Exception:
            pass
        # Force a redraw of current image and annotations
        self._display_source()
        self._draw_annotations()

    def _draw_active_masks(self, annotations: List[Dict]):
        """Draw segmentation masks (if present in annotations)."""
        if not annotations or self.current_cv_image is None:
            return
        try:
            import pycocotools.mask as mask_util
        except Exception:
            print("Note: pycocotools not available; mask overlay disabled.")
            return
        H, W = self.current_cv_image.shape[:2]
        effective_scale = self.base_scale * self.zoom_level
        # Clear previous mask layers
        self.canvas.delete("mask_layer")
        self.photo_masks.clear()
        for ann in annotations:
            seg = ann.get("segmentation")
            if not seg or not isinstance(seg, dict) or "counts" not in seg or "size" not in seg:
                continue
            rle = {"counts": seg["counts"], "size": seg["size"]}
            if isinstance(rle["counts"], str):
                rle["counts"] = rle["counts"].encode("utf-8")
            mask = mask_util.decode(rle)
            if mask is None:
                continue
            # Apply current flip mode so overlay matches the displayed image
            try:
                if self.flip_mode == 'h':
                    mask = cv2.flip(mask, 1)
                elif self.flip_mode == 'v':
                    mask = cv2.flip(mask, 0)
                elif self.flip_mode == 'hv':
                    mask = cv2.flip(mask, -1)
            except Exception:
                pass
            # Build semi-transparent overlay
            rgba = np.zeros((H, W, 4), dtype=np.uint8)
            rgba[mask > 0] = [0, 255, 0, 90]
            overlay = Image.fromarray(rgba, mode="RGBA")
            overlay = overlay.resize((int(W * effective_scale), int(H * effective_scale)), Image.Resampling.NEAREST)
            photo_mask = ImageTk.PhotoImage(overlay)
            self.photo_masks.append(photo_mask)  # Keep reference to prevent GC
            # Draw mask above the base image; annotations will be drawn after and appear on top
            self.canvas.create_image(self.pan_offset_x, self.pan_offset_y, anchor=tk.NW, image=photo_mask, tags=("mask_layer",))
            try:
                # Ensure mask sits above base image
                self.canvas.tag_raise("mask_layer", "image")
            except Exception:
                pass

        try:
            # Flush pending drawing ops for immediate visual update
            self.canvas.update_idletasks()
        except Exception:
            pass


    def _draw_active_poses(self, annotations: List[Dict], color: str = "cyan"):
        """Draw keypoints and skeletons from the active set with color-coded skeleton."""
        print(f"DEBUG: _draw_active_poses called with {len(annotations)} annotations")
        print(f"DEBUG: skeleton_links: {self.skeleton_links[:5]}...")
        print(f"DEBUG: keypoint_names: {self.keypoint_names[:5]}...")
        
        kpt_draw_radius = max(1, int(self.keypoint_radius * 0.8))
        line_width = 1

        for ann in annotations:
            ann_id = ann["id"]
            kpts_flat = ann.get("keypoints")
            if not kpts_flat or len(kpts_flat) % 3 != 0: continue

            kpts = np.array(kpts_flat).reshape(-1, 3)
            print(f"DEBUG: Processing annotation {ann_id} with {len(kpts)} keypoints")
            
            # Draw skeleton with color coding
            if self.skeleton_links:
                print(f"DEBUG: Drawing {len(self.skeleton_links)} skeleton links")
                for start_idx, end_idx in self.skeleton_links:
                    if start_idx < len(kpts) and end_idx < len(kpts):
                        pt_start = kpts[start_idx]
                        pt_end = kpts[end_idx]
                        
                        # Draw skeleton links for all keypoint pairs, but style differently based on visibility
                        x1c, y1c = self._image_to_canvas_coords(pt_start[0], pt_start[1])
                        x2c, y2c = self._image_to_canvas_coords(pt_end[0], pt_end[1])
                        
                        # Get keypoint names for color coding
                        start_kpt_name = self.keypoint_names[start_idx] if start_idx < len(self.keypoint_names) else f"kpt_{start_idx}"
                        end_kpt_name = self.keypoint_names[end_idx] if end_idx < len(self.keypoint_names) else f"kpt_{end_idx}"
                        
                        # Determine skeleton color based on keypoint types
                        skeleton_color = get_skeleton_color(start_kpt_name, end_kpt_name)
                        
                        # Convert BGR to hex for Tkinter
                        hex_color = f"#{skeleton_color[2]:02x}{skeleton_color[1]:02x}{skeleton_color[0]:02x}"
                        
                        # Style based on visibility: solid for both visible, dotted for others
                        if pt_start[2] == 2 and pt_end[2] == 2:
                            # Both visible - solid line
                            dash_pattern = None
                            line_width_adj = line_width
                        else:
                            # Any other combination (including v=0, v=0 or mixed v=0/v=2) - dotted line
                            dash_pattern = (2, 2)
                            line_width_adj = max(1, line_width - 1)
                        
                        self.canvas.create_line(
                            x1c, y1c, x2c, y2c, 
                            fill=hex_color, 
                            width=line_width_adj,
                            dash=dash_pattern,
                            tags=("skeleton", f"ann{ann_id}", "editable")
                        )

            # Draw keypoints with color coding
            print(f"DEBUG: Drawing {len(kpts)} keypoints")
            # Debug: count visibility values
            v_counts = {0: 0, 1: 0, 2: 0}
            for _, _, v in kpts:
                v_counts[v] = v_counts.get(v, 0) + 1
            print(f"DEBUG: Visibility counts - v=0: {v_counts[0]}, v=1: {v_counts[1]}, v=2: {v_counts[2]}")
            
            for i, (x, y, v) in enumerate(kpts):
                # Draw ALL keypoints regardless of visibility
                # Get keypoint name for color coding
                kpt_name = self.keypoint_names[i] if i < len(self.keypoint_names) else f"kpt_{i}"
                kpt_color = get_keypoint_color(kpt_name)
                print(f"DEBUG: Drawing keypoint {i} ({kpt_name}) at ({x}, {y}) with visibility {v}, color {kpt_color}")
                
                # Convert BGR to hex for Tkinter
                hex_fill_color = f"#{kpt_color[2]:02x}{kpt_color[1]:02x}{kpt_color[0]:02x}"
                
                # Use different fill colors based on visibility
                if v == 2:  # Visible
                    fill_color = hex_fill_color
                # elif v == 1:  # Occluded - commented out for now
                #     fill_color = "#CCCCCC"  # Light gray for occluded
                else:  # v == 0, invisible
                    fill_color = "#FF0000"  # Red for invisible keypoints
                
                outline_color = color  # Keep instance color for outline
                center_xc, center_yc = self._image_to_canvas_coords(x, y)
                visibility_tag = "visible" if v >= 1 else "invisible"
                
                self.canvas.create_oval(
                    center_xc - kpt_draw_radius, center_yc - kpt_draw_radius, 
                    center_xc + kpt_draw_radius, center_yc + kpt_draw_radius, 
                    fill=fill_color, outline=outline_color, 
                    tags=("keypoint", f"ann{ann_id}", f"kpt{i}", visibility_tag, "editable")
                )

        # Ensure skeleton lines are above mask and box but under keypoints
        try:
            self.canvas.tag_raise("skeleton")
            self.canvas.tag_raise("keypoint")
        except Exception:
            pass


    # --- Interaction Handlers ---
    # (Need updates for video mode constraints and track ID selection)

    def _on_mouse_hover(self, event):
        """Handles mouse movement for hover effects (e.g., showing keypoint names)."""
        # (Keep existing logic, maybe add hover for track ID?)
        self.canvas.delete("hover_text")
        if self.view_mode != "pose": return

        items = self.canvas.find_overlapping(event.x - 1, event.y - 1, event.x + 1, event.y + 1)
        hover_target_found = False
        for item in reversed(items):
            tags = self.canvas.gettags(item)
            if "keypoint" in tags and "editable" in tags:
                kpt_tag = next((t for t in tags if t.startswith("kpt")), None)
                if kpt_tag:
                    try:
                        kpt_idx = int(kpt_tag[3:])
                        if 0 <= kpt_idx < len(self.keypoint_names):
                            kpt_name = self.keypoint_names[kpt_idx]
                            self.canvas.create_text(event.x + 5, event.y - 5, text=kpt_name, anchor=tk.SW, fill="white", font=("Arial", 9), tags="hover_text")
                            hover_target_found = True
                            break
                    except (ValueError, IndexError): continue
            if hover_target_found: break


    def _on_mouse_down(self, event):
        """Handles left mouse button down: selecting elements or starting creation."""
        self.canvas.focus_set()

        self.drag_start_canvas = (event.x, event.y)
        self.selected_ann_id = None # Reset selection on new click
        self.drag_type = None
        self.selected_corner = None
        self.selected_kpt_index = None

        # Check if in comparison mode (only possible in image mode)
        current_src_id = self.source_ids[self.current_idx]
        state = self.source_decision_state.get(current_src_id, 'undecided')
        if state == 'undecided':
            print("In comparison mode. Choose 'A' or 'D' to enable editing.")
            return

        # --- Interaction Logic (Priority: Keypoint -> Handle -> Box -> Create) ---

        # 0. Keypoint Clicks (Pose Mode Only)
        if self.view_mode == "pose":
            items = self.canvas.find_overlapping(event.x - self.keypoint_radius, event.y - self.keypoint_radius, event.x + self.keypoint_radius, event.y + self.keypoint_radius)
            for item in reversed(items):
                tags = self.canvas.gettags(item)
                if "keypoint" in tags and "editable" in tags:
                    ann_tag = next((t for t in tags if t.startswith("ann")), None)
                    kpt_tag = next((t for t in tags if t.startswith("kpt")), None)
                    if ann_tag and kpt_tag:
                        self.selected_ann_id = int(ann_tag[3:])
                        self.selected_kpt_index = int(kpt_tag[3:])
                        self.drag_type = "keypoint"
                        print(f"Selected keypoint {self.selected_kpt_index} for ann {self.selected_ann_id}")
                        return # Stop checking

        # 1. Handle Clicks
        items = self.canvas.find_overlapping(event.x - self.handle_size / 2, event.y - self.handle_size / 2, event.x + self.handle_size / 2, event.y + self.handle_size / 2)
        for item in reversed(items):
            tags = self.canvas.gettags(item)
            if "handle" in tags and "editable" in tags:
                ann_tag = next((t for t in tags if t.startswith("ann")), None)
                corner = next((t for t in tags if t in ["topleft", "topright", "bottomleft", "bottomright"]), None)
                if ann_tag and corner:
                    self.selected_ann_id = int(ann_tag[3:])
                    self.selected_corner = corner
                    self.drag_type = "resize"
                    print(f"Selected handle for ann {self.selected_ann_id}, corner: {self.selected_corner}")
                    return # Stop checking

        # 2. Box Interior Clicks (for move or track ID assignment)
        # Need to check annotations explicitly, as find_overlapping might miss clicks near edges
        annotations = self.modified_annotations.get(current_src_id, [])
        clicked_on_box = False
        for ann in reversed(annotations): # Check topmost first
            ann_id = ann["id"]
            bbox = ann.get("bbox")
            if not bbox: continue

            x_img, y_img, w_img, h_img = bbox
            x1c, y1c = self._image_to_canvas_coords(x_img, y_img)
            x2c, y2c = self._image_to_canvas_coords(x_img + w_img, y_img + h_img)

            if min(x1c, x2c) < event.x < max(x1c, x2c) and min(y1c, y2c) < event.y < max(y1c, y2c):
                self.selected_ann_id = ann_id
                self.drag_type = "move" # Assume move initially
                self.drag_offset_canvas = (event.x - x1c, event.y - y1c)
                print(f"Selected box interior for ann {self.selected_ann_id} (potential move/ID assign)")
                clicked_on_box = True
                break # Stop after finding the topmost box clicked

        if clicked_on_box: return # Stop checking

        # 3. New Box Creation Mode
        if self.creating_new_box:
            self.drag_type = "create"
            self.new_box_start = (event.x, event.y)
            print("Starting new box creation")
            return # Stop checking

        # 4. Clicked on empty space - deselect
        print("Click on empty space, deselected.")
        self.selected_ann_id = None
        self.drag_type = None


    def _on_mouse_drag(self, event):
        """Handles left mouse button drag: moving/resizing elements or creating new box."""
        if not self.drag_start_canvas: return

        current_canvas_x, current_canvas_y = event.x, event.y

        if self.drag_type == "create" and self.new_box_start:
            self.canvas.delete("temp_box")
            self.canvas.create_rectangle(self.new_box_start[0], self.new_box_start[1], current_canvas_x, current_canvas_y, outline="blue", width=2, tags="temp_box")
        elif self.drag_type in ["move", "resize", "keypoint"] and self.selected_ann_id is not None:
            ann = self._find_annotation_by_id(self.selected_ann_id)
            if not ann:
                print(f"Error: Could not find annotation {self.selected_ann_id} to drag.")
                self.drag_type = None; return

            img_info = self.source_map[ann["image_id"]] # Use source_map
            img_width = img_info["original_width"]
            img_height = img_info["original_height"]

            # --- Move Logic ---
            if self.drag_type == "move" and "bbox" in ann and ann["bbox"]:
                new_x1_canvas = current_canvas_x - self.drag_offset_canvas[0]
                new_y1_canvas = current_canvas_y - self.drag_offset_canvas[1]
                new_x_img, new_y_img = self._canvas_to_image_coords(new_x1_canvas, new_y1_canvas)
                _, _, w_img, h_img = ann["bbox"]
                new_x_img = max(0, min(new_x_img, img_width - w_img))
                new_y_img = max(0, min(new_y_img, img_height - h_img))
                dx_img = new_x_img - ann["bbox"][0]
                dy_img = new_y_img - ann["bbox"][1]
                ann["bbox"][0] = new_x_img
                ann["bbox"][1] = new_y_img

                if ann.get("keypoints"):
                    kpts_flat = ann["keypoints"]
                    for i in range(0, len(kpts_flat), 3):
                        kpts_flat[i] += dx_img
                        kpts_flat[i+1] += dy_img
                        kpts_flat[i] = max(0, min(kpts_flat[i], img_width))
                        kpts_flat[i+1] = max(0, min(kpts_flat[i+1], img_height))
                self.modifications_made = True
                self._draw_annotations()

            # --- Resize Logic ---
            elif self.drag_type == "resize" and "bbox" in ann and ann["bbox"]:
                current_x_img, current_y_img = self._canvas_to_image_coords(current_canvas_x, current_canvas_y)
                x_img, y_img, w_img, h_img = ann["bbox"]
                orig_x1_img, orig_y1_img, orig_x2_img, orig_y2_img = x_img, y_img, x_img + w_img, y_img + h_img
                final_x1_img, final_y1_img, final_x2_img, final_y2_img = orig_x1_img, orig_y1_img, orig_x2_img, orig_y2_img

                if "left" in self.selected_corner: final_x1_img = current_x_img
                if "right" in self.selected_corner: final_x2_img = current_x_img
                if "top" in self.selected_corner: final_y1_img = current_y_img
                if "bottom" in self.selected_corner: final_y2_img = current_y_img

                final_x1_img = max(0, min(final_x1_img, img_width))
                final_y1_img = max(0, min(final_y1_img, img_height))
                final_x2_img = max(0, min(final_x2_img, img_width))
                final_y2_img = max(0, min(final_y2_img, img_height))

                new_x_img = min(final_x1_img, final_x2_img)
                new_y_img = min(final_y1_img, final_y2_img)
                new_w_img = abs(final_x1_img - final_x2_img)
                new_h_img = abs(final_y1_img - final_y2_img)

                if w_img > 0 and h_img > 0 and new_w_img > 0 and new_h_img > 0 and ann.get("keypoints"):
                    kpts_flat = ann["keypoints"]
                    scale_x = new_w_img / w_img
                    scale_y = new_h_img / h_img
                    for i in range(0, len(kpts_flat), 3):
                        # Scale relative to the *original* top-left corner being resized *from*
                        # This is complex. A simpler approach scales relative to the fixed corner.
                        # Let's try scaling relative to the new top-left.
                        rel_x = (kpts_flat[i] - x_img) / w_img
                        rel_y = (kpts_flat[i+1] - y_img) / h_img
                        kpts_flat[i] = new_x_img + rel_x * new_w_img
                        kpts_flat[i+1] = new_y_img + rel_y * new_h_img
                        kpts_flat[i] = max(0, min(kpts_flat[i], img_width))
                        kpts_flat[i+1] = max(0, min(kpts_flat[i+1], img_height))


                if final_x1_img > final_x2_img:
                    corner_map_h = {"topleft": "topright", "topright": "topleft", "bottomleft": "bottomright", "bottomright": "bottomleft"}
                    self.selected_corner = corner_map_h.get(self.selected_corner, self.selected_corner)
                if final_y1_img > final_y2_img:
                    corner_map_v = {"topleft": "bottomleft", "bottomleft": "topleft", "topright": "bottomright", "bottomright": "topright"}
                    self.selected_corner = corner_map_v.get(self.selected_corner, self.selected_corner)

                ann["bbox"] = [new_x_img, new_y_img, new_w_img, new_h_img]
                ann["area"] = new_w_img * new_h_img
                self.modifications_made = True
                self._draw_annotations()

            # --- Keypoint Drag Logic ---
            elif self.drag_type == 'keypoint' and self.selected_kpt_index is not None:
                 # (Keep existing keypoint drag logic)
                if not ann or "keypoints" not in ann or not ann["keypoints"]:
                    print(f"Error: Could not find annotation {self.selected_ann_id} or keypoints for drag.")
                    self.drag_type = None; return

                kpts_flat = ann["keypoints"]
                kpt_base_idx = self.selected_kpt_index * 3
                if kpt_base_idx + 1 >= len(kpts_flat):
                     print(f"Error: Invalid keypoint index {self.selected_kpt_index} for ann {self.selected_ann_id}.")
                     self.drag_type = None; return

                current_x_img, current_y_img = self._canvas_to_image_coords(current_canvas_x, current_canvas_y)
                clamped_x_img = max(0, min(current_x_img, img_width))
                clamped_y_img = max(0, min(current_y_img, img_height))
                kpts_flat[kpt_base_idx] = clamped_x_img
                kpts_flat[kpt_base_idx + 1] = clamped_y_img

                self.modifications_made = True
                self._draw_annotations()


    def _on_mouse_up(self, event):
        """Handles left mouse button release: finalizing creation or drag."""
        if self.drag_type == "create" and self.new_box_start:
            self.canvas.delete("temp_box")
            x_start_canvas, y_start_canvas = self.new_box_start
            x_end_canvas, y_end_canvas = event.x, event.y
            x_start_img, y_start_img = self._canvas_to_image_coords(x_start_canvas, y_start_canvas)
            x_end_img, y_end_img = self._canvas_to_image_coords(x_end_canvas, y_end_canvas)

            current_src_id = self.source_ids[self.current_idx]
            img_info = self.source_map[current_src_id] # Use source_map
            img_width = img_info["original_width"]
            img_height = img_info["original_height"]

            x_img = max(0, min(x_start_img, x_end_img, img_width))
            y_img = max(0, min(y_start_img, y_end_img, img_height))
            x2_img = max(0, min(max(x_start_img, x_end_img), img_width))
            y2_img = max(0, min(max(y_start_img, y_end_img), img_height))
            w_img = x2_img - x_img
            h_img = y2_img - y_img

            if w_img > 5 and h_img > 5:
                num_kpts = len(self.keypoint_names)
                initial_keypoints = []
                for _ in range(num_kpts):
                    kpx = max(x_img, min(random.uniform(x_img, x_img + w_img), x_img + w_img))
                    kpy = max(y_img, min(random.uniform(y_img, y_img + h_img), y_img + h_img))
                    initial_keypoints.extend([kpx, kpy, 0]) # v=0

                new_ann = {
                    "id": self.next_start_ann_id,
                    "image_id": current_src_id, # Use frame/image ID
                    "category_id": self.categories[0]["id"] if self.categories else 1,
                    "bbox": [x_img, y_img, w_img, h_img],
                    "area": w_img * h_img,
                    "iscrowd": 0,
                    "keypoints": initial_keypoints,
                    "num_keypoints": 0,
                    # "instance_id": None, # Initialize track ID as None
                }
                self.modified_annotations.setdefault(current_src_id, []).append(new_ann)
                self.next_start_ann_id += 1
                self.modifications_made = True
                self._draw_annotations()
            self._toggle_new_box_mode() # Exit creation mode

        # Reset drag state
        # print(f"Mouse Up at ({event.x}, {event.y}), drag_type was: {self.drag_type}, selected_ann: {self.selected_ann_id}")
        self.dragging = False
        self.drag_start_canvas = None
        # Keep self.selected_ann_id if just clicked (for track ID assignment)
        # self.selected_ann_id = None # Don't reset here
        self.drag_type = None # Reset drag type
        self.selected_corner = None
        self.selected_kpt_index = None
        self.new_box_start = None


    def _on_right_click(self, event):
        """Handles right-click: toggle keypoint visibility or delete instance."""
        current_src_id = self.source_ids[self.current_idx]
        state = self.source_decision_state.get(current_src_id, 'undecided')
        # Allow actions if not in comparison mode
        if state == 'undecided':
            print("In comparison mode. Choose 'A' or 'D' to enable deletion/toggle.")
            return

        print(f"Right Click at ({event.x}, {event.y})")
        toggled_keypoint = False
        deleted_something = False

        # --- Keypoint Visibility Toggle ---
        if self.view_mode == "pose":
            items = self.canvas.find_overlapping(event.x - self.keypoint_radius, event.y - self.keypoint_radius, event.x + self.keypoint_radius, event.y + self.keypoint_radius)
            for item in reversed(items):
                tags = self.canvas.gettags(item)
                if "keypoint" in tags and "editable" in tags:
                    ann_tag = next((t for t in tags if t.startswith("ann")), None)
                    kpt_tag = next((t for t in tags if t.startswith("kpt")), None)
                    if ann_tag and kpt_tag:
                        ann_id_to_toggle = int(ann_tag[3:])
                        kpt_idx_to_toggle = int(kpt_tag[3:])
                        ann = self._find_annotation_by_id(ann_id_to_toggle)
                        if ann and ann.get("keypoints"):
                            kpts_flat = ann["keypoints"]
                            kpt_base_idx = kpt_idx_to_toggle * 3
                            if kpt_base_idx + 2 < len(kpts_flat):
                                current_v = kpts_flat[kpt_base_idx + 2]
                                # Cycle through visibility states: 0 (invisible) -> 2 (visible) -> 0
                                new_v = 2 if current_v == 0 else 0
                                kpts_flat[kpt_base_idx + 2] = new_v
                                visibility_names = {0: "invisible", 2: "visible"}
                                print(f"Toggled visibility for keypoint {kpt_idx_to_toggle} ann {ann_id_to_toggle} from {visibility_names.get(current_v, current_v)} to {visibility_names.get(new_v, new_v)}")
                                ann['num_keypoints'] = sum(1 for j in range(2, len(kpts_flat), 3) if kpts_flat[j] == 2)
                                self.modifications_made = True
                            toggled_keypoint = True
                            self._draw_annotations()
                            return # Exit handler

        # --- Instance Deletion ---
        if not toggled_keypoint:
            # Use find_overlapping first for handles/kpts, then check boxes manually
            items = self.canvas.find_overlapping(event.x - self.handle_size, event.y - self.handle_size, event.x + self.handle_size, event.y + self.handle_size)
            ann_id_to_delete = None
            for item in reversed(items):
                tags = self.canvas.gettags(item)
                if ("handle" in tags or "keypoint" in tags) and "editable" in tags: # Also check kpts
                    ann_tag = next((t for t in tags if t.startswith("ann")), None)
                    if ann_tag:
                        ann_id_to_delete = int(ann_tag[3:])
                        break

            # If not found via small items, check box interiors
            if ann_id_to_delete is None:
                annotations = self.modified_annotations.get(current_src_id, [])
                for ann in reversed(annotations):
                    ann_id = ann["id"]
                    bbox = ann.get("bbox")
                    if not bbox: continue
                    x_img, y_img, w_img, h_img = bbox
                    x1c, y1c = self._image_to_canvas_coords(x_img, y_img)
                    x2c, y2c = self._image_to_canvas_coords(x_img + w_img, y_img + h_img)
                    if min(x1c, x2c) < event.x < max(x1c, x2c) and min(y1c, y2c) < event.y < max(y1c, y2c):
                        ann_id_to_delete = ann_id
                        break

            # Perform deletion if an ID was found
            if ann_id_to_delete is not None:
                annotations = self.modified_annotations.get(current_src_id, [])
                # Find and store the annotation for undo
                annotation_to_delete = None
                for ann in annotations:
                    if ann["id"] == ann_id_to_delete:
                        annotation_to_delete = copy.deepcopy(ann)
                        break
                
                initial_len = len(annotations)
                self.modified_annotations[current_src_id] = [
                    ann for ann in annotations if ann["id"] != ann_id_to_delete
                ]
                if len(self.modified_annotations.get(current_src_id, [])) < initial_len:
                    # Store for undo
                    if annotation_to_delete:
                        undo_action = {
                            'type': 'delete_instance',
                            'source_id': current_src_id,
                            'annotation': annotation_to_delete
                        }
                        self.undo_stack.append(undo_action)
                        # Keep only last 10 undo actions
                        if len(self.undo_stack) > 10:
                            self.undo_stack.pop(0)
                    
                    print(f"Deleted annotation instance {ann_id_to_delete}")
                    if self.selected_ann_id == ann_id_to_delete: # Deselect if deleted
                        self.selected_ann_id = None
                    self.modifications_made = True
                    deleted_something = True

            if deleted_something:
                self._draw_annotations()


    # --- Helper and Control Methods ---

    def _find_annotation_by_id(self, ann_id_to_find: int) -> Optional[Dict]:
        """Helper to find an annotation dict by its ID in the current source's list."""
        if not self.source_ids: return None
        current_src_id = self.source_ids[self.current_idx]
        for ann in self.modified_annotations.get(current_src_id, []):
            if ann["id"] == ann_id_to_find:
                return ann
        return None

    def _toggle_view_mode(self, event=None):
        """Toggle between bbox and pose view."""
        current_src_id = self.source_ids[self.current_idx]
        state = self.source_decision_state.get(current_src_id, 'undecided')
        # Allow toggle if not in comparison mode
        if state != 'undecided':
            self.view_mode = "pose" if self.view_mode == "bbox" else "bbox"
            self.view_mode_label.set(f"View: {self.view_mode.capitalize()}")
            self.selected_ann_id = None # Reset selection on view toggle
            self.drag_type = None
            self.selected_corner = None
            self.selected_kpt_index = None
            self._draw_annotations()
        else:
            messagebox.showinfo("Info", "Please choose Original ('A') or Shrunk ('D') before toggling view.")

    def _toggle_flip_mode(self, event=None):
        """Cycles through image flip modes: none, h, v, hv."""
        # (Keep existing logic)
        modes = ['none', 'h', 'v', 'hv']
        try: current_mode_idx = modes.index(self.flip_mode)
        except ValueError: current_mode_idx = 0
        next_mode_idx = (current_mode_idx + 1) % len(modes)
        self.flip_mode = modes[next_mode_idx]
        label_text = f"Flip: {self.flip_mode.upper() if self.flip_mode != 'none' else 'None'}"
        self.flip_mode_label.set(label_text)
        print(f"Set flip mode to: {self.flip_mode}")
        self._display_source() # Renamed
        self._draw_annotations()

    def _toggle_new_box_mode(self, event=None):
        """Toggle new box creation mode"""
        current_src_id = self.source_ids[self.current_idx]
        state = self.source_decision_state.get(current_src_id, 'undecided')
        # Prevent starting if in comparison mode
        if state == 'undecided' and not self.creating_new_box:
             messagebox.showinfo("Info", "Please choose Original ('A') or Shrunk ('D') before creating new boxes.")
             return

        self.creating_new_box = not self.creating_new_box
        if self.creating_new_box:
            self.canvas.config(cursor="cross")
            self.new_box_label.set("Mode: Create New Box")
        else:
            self.canvas.config(cursor="")
            if self.is_panning: self.canvas.config(cursor="fleur")
            self.new_box_label.set("")


    def _jump_to_source(self, event=None): # Renamed from _jump_to_image
        """Attempts to jump to the image/frame number entered in the jumper Entry."""
        try:
            entry_text = self.source_jumper_var.get() # Renamed
            target_num_str = entry_text.split('/')[0].strip()
            target_num = int(target_num_str) # This is the 1-based source ID
            total_sources = len(self.source_ids)

            if target_num in self.source_ids: # Check if the ID actually exists
                target_idx = self.source_ids.index(target_num) # Find index for this ID
                if target_idx != self.current_idx:
                    print(f"Jumping to {'frame' if self.is_video_mode else 'image'} {target_num} (index {target_idx})...")
                    # Reset selection when jumping
                    self.selected_ann_id = None
                    self.drag_type = None
                    self.current_idx = target_idx
                    self._load_current_source() # Renamed
                else:
                    self._update_jumper_text()
            else:
                messagebox.showwarning("Invalid ID", f"Please enter a valid {'frame' if self.is_video_mode else 'image'} number between {self.source_ids[0]} and {self.source_ids[-1]}.")
                self._update_jumper_text()
        except (ValueError, IndexError):
            messagebox.showwarning("Invalid Input", f"Please enter the {'frame' if self.is_video_mode else 'image'} number (e.g., '100').")
            self._update_jumper_text()
        except Exception as e:
            messagebox.showerror("Error Jumping", f"An unexpected error occurred: {e}")
            self._update_jumper_text()

    def _update_jumper_text(self):
        """Helper to reset the jumper text to the current source ID."""
        total_sources = len(self.source_ids)
        current_display_num = self.source_ids[self.current_idx] if total_sources > 0 and self.current_idx >= 0 else 0
        last_source_num = self.source_ids[-1] if total_sources > 0 else 0 # Use last ID for total display
        counter_text = f"{current_display_num} / {last_source_num}"
        self.source_jumper_var.set(counter_text) # Renamed

    def _prev_source(self): # Renamed from _prev_image
        if self.source_ids:
            next_idx = self._find_valid_index(-1)
            if next_idx != -1 and next_idx != self.current_idx:
                self.selected_ann_id = None # Reset selection
                self.current_idx = next_idx
                self._load_current_source()

    def _next_source(self): # Renamed from _next_image
        if self.source_ids:
            next_idx = self._find_valid_index(1)
            if next_idx != -1 and next_idx != self.current_idx:
                self.selected_ann_id = None # Reset selection
                self.current_idx = next_idx
                self._load_current_source()

    def _delete_current_source(self): # Renamed from _delete_current_image
        if not self.source_ids: return
        current_src_id = self.source_ids[self.current_idx]
        src_info = self.source_map.get(current_src_id, {})
        display_name = src_info.get('file_name', f'ID {current_src_id}')
        if self.is_video_mode: display_name = f"Frame {current_src_id}"
        type_name = "frame" if self.is_video_mode else "image"

        # Check if confirmation is enabled
        confirm_deletion = True
        if self.delete_confirmation_enabled:
            confirm_deletion = messagebox.askyesno(
                f"Confirm Deletion",
                f"Mark {type_name} '{display_name}' and all its annotations for deletion?",
            )
            
        if confirm_deletion:
            # Store for undo - save current annotations before deletion
            current_annotations = self.modified_annotations.get(current_src_id, [])
            if current_annotations:
                undo_action = {
                    'type': 'delete_source',
                    'source_id': current_src_id,
                    'annotations': copy.deepcopy(current_annotations)
                }
                self.undo_stack.append(undo_action)
                # Keep only last 10 undo actions
                if len(self.undo_stack) > 10:
                    self.undo_stack.pop(0)
            
            self.deleted_source_ids.add(current_src_id)
            self.modifications_made = True
            # Deselect if current item is deleted
            self.selected_ann_id = None
            self.drag_type = None

            next_idx = self._find_valid_index(1)
            current_idx_was_deleted = self.current_idx

            if next_idx == -1:
                 messagebox.showinfo("Info", f"Last {type_name} marked for deletion. Saving...")
                 self._complete()
            elif next_idx == current_idx_was_deleted: # Only one left, just deleted
                 messagebox.showinfo("Info", f"All {type_name}s marked for deletion. Saving...")
                 self._complete()
            else:
                 self.current_idx = next_idx
                 self._load_current_source()


    def _on_close_window(self):
        """Handle clicking the window's close button. Releases video capture."""
        if self.modifications_made:
            response = messagebox.askyesnocancel(
                "Unsaved Changes", "You have unsaved changes. Save before closing?"
            )
            if response is True: # Yes
                self._save_refined_annotations()
                if hasattr(self, "save_successful") and self.save_successful:
                    if self.video_capture: self.video_capture.release() # Release video
                    if self.root: self.root.destroy()
                # else: Don't close if save failed
            elif response is False: # No
                print("Discarding changes and closing.")
                if self.video_capture: self.video_capture.release() # Release video
                if self.root: self.root.destroy()
            # else: Cancel, do nothing
                print("Close operation cancelled.")
        else:
            if self.video_capture: self.video_capture.release() # Release video
            if self.root: self.root.destroy()

    def _complete(self):
        """Save and close the application. Releases video capture."""
        should_close = True
        if self.modifications_made:
            self._save_refined_annotations()
            if not (hasattr(self, "save_successful") and self.save_successful):
                should_close = False
        else:
            print("No modifications detected.")

        if should_close:
            if self.video_capture: self.video_capture.release() # Release video before closing
            if self.root: self.root.destroy()


    def _save_refined_annotations(self):
        """Save the modified annotations back to the appropriate format."""
        self.save_successful = False
        print(f"Saving refined annotations to: {self.output_path}")

        # --- Filter final source IDs ---
        final_source_ids = sorted(list(set(self.source_ids) - self.deleted_source_ids))
        type_name = "frame" if self.is_video_mode else "image"
        print(
            f"Keeping {len(final_source_ids)} {type_name}s out of {len(self.source_ids)} initial valid {type_name}s."
        )

        # --- Decide Output Format ---
        if self.loaded_from_results:
            # --- Save in Original Results Format ---
            print("Saving in original results format...")
            output_data = {"instance_info": []}
            processed_instances_count = 0

            for frame_id in final_source_ids:
                frame_instances = []
                annotations_for_frame = self.modified_annotations.get(frame_id, [])

                for internal_ann in annotations_for_frame:
                    # --- Transform internal COCO-like back to results instance ---

                    # Retrieve original data if stored, otherwise reconstruct
                    original_instance_data = internal_ann.get("_original_results_instance", {})

                    # Reconstruct keypoints list [[x,y], ...] from [x,y,v,...]
                    results_keypoints = []
                    coco_kpts_flat = internal_ann.get("keypoints", [])
                    for i in range(0, len(coco_kpts_flat), 3):
                        # Only include keypoint if visibility is > 0? Or always include?
                        # Let's include based on original structure or if modified.
                        # If the original had it, or if v > 0 now, include it.
                        # This is tricky. Let's just save the current [x,y] pairs for simplicity.
                        # We might lose the original score association if points were deleted/added.
                        # A safer bet might be to update the stored original data. Let's try that.

                        # Update the stored original data's keypoints
                        if "_original_results_instance" in internal_ann:
                            orig_kpts_list = internal_ann["_original_results_instance"].get("keypoints", [])
                            kpt_idx = i // 3
                            if kpt_idx < len(orig_kpts_list):
                                orig_kpts_list[kpt_idx] = [coco_kpts_flat[i], coco_kpts_flat[i+1]]
                            # Note: Scores are not updated here, kept original. Need score update logic if req.
                            # Bbox and instance_id are easier below.

                    # Use the updated original structure if available
                    if "_original_results_instance" in internal_ann:
                        results_instance = internal_ann["_original_results_instance"]
                        # Update bbox [x1,y1,x2,y2] from internal [x,y,w,h]
                        x, y, w, h = internal_ann.get("bbox", [0,0,0,0])
                        if results_instance.get("bbox"):
                             results_instance["bbox"][0] = [x, y, x + w, y + h]
                        # Update instance_id
                        results_instance["instance_id"] = internal_ann.get("instance_id")
                    else:
                        # Fallback: Reconstruct from scratch (might lose scores etc.)
                        print(f"Warning: Reconstructing results instance for ann {internal_ann['id']} - may lose original scores.")
                        results_kpts = [[coco_kpts_flat[i], coco_kpts_flat[i+1]] for i in range(0, len(coco_kpts_flat), 3)]
                        x, y, w, h = internal_ann.get("bbox", [0,0,0,0])
                        results_bbox = [[x, y, x + w, y + h]]
                        results_instance = {
                             "keypoints": results_kpts,
                             "keypoint_scores": [], # Scores lost
                             "bbox": results_bbox,
                             "bbox_score": None, # Score lost
                             "instance_id": internal_ann.get("instance_id")
                        }


                    frame_instances.append(results_instance)
                    processed_instances_count += 1
                    # --- End Transformation ---

                if frame_instances: # Only add frame entry if it has instances
                    output_data["instance_info"].append({
                        "frame_id": frame_id,
                        "instances": frame_instances
                    })

            print(f"Saving {processed_instances_count} instances across {len(output_data['instance_info'])} frames.")

        else:
            # --- Save in Standard COCO Format ---
            print("Saving in standard COCO format...")
            output_data = {"images": [], "annotations": [], "categories": self.categories}

            # Check for undecided images (only in image mode)
            if not self.is_video_mode:
                undecided_sources = []
                # Iterate through all source_ids that are not marked for deletion
                for src_id in self.source_ids:
                    if src_id not in self.deleted_source_ids and \
                       self.source_decision_state.get(src_id) == 'undecided':
                        undecided_sources.append(src_id)

                if undecided_sources:
                    num_undecided = len(undecided_sources)
                    # Define the message string 'msg'
                    msg = (
                        f"There are {num_undecided} image(s) for which you haven't chosen between "
                        f"original and shrunk annotations (using 'A' or 'D' keys).\n\n"
                        f"If you proceed, these images will default to their original annotations. "
                        f"Do you want to save anyway?"
                    )
                    if not messagebox.askyesno("Undecided Images", msg):
                        print("Save cancelled by user due to undecided images.")
                        self.save_successful = False
                        return
                    else:
                        # User chose to proceed. Default undecided images to original.
                        print(f"Proceeding to save. {num_undecided} undecided image(s) will use their ORIGINAL annotations.")
                        for src_id_undecided in undecided_sources:
                            # Ensure modified_annotations has the original data for these undecided images
                            # Check if key exists and if its value (list of annotations) is empty or not
                            current_modified_anns = self.modified_annotations.get(src_id_undecided)
                            if current_modified_anns is None or not current_modified_anns: # If key doesn't exist or list is empty
                                self.modified_annotations[src_id_undecided] = copy.deepcopy(
                                    self.original_annotations.get(src_id_undecided, [])
                                )
                                # Optionally update the state to reflect this default action
                                self.source_decision_state[src_id_undecided] = 'original_defaulted_on_save'
                                print(f"Image ID {src_id_undecided}: Defaulted to original annotations.")
                                # If we are defaulting, it implies a modification might be needed for saving state.
                                self.modifications_made = True # Ensure this is True if we are defaulting

            # Populate 'images' section
            for src_id in final_source_ids:
                # ... (image info saving remains the same) ...
                if src_id in self.source_map:
                    src_info = self.source_map[src_id]
                    file_name = src_info.get("file_name", f"{type_name}_{src_id:06d}")
                    output_data["images"].append(
                        {
                            "id": src_id,
                            "width": src_info["original_width"],
                            "height": src_info["original_height"],
                            "file_name": file_name,
                        }
                    )


            # Populate 'annotations' section
            current_ann_id = 1
            missing_track_id_count = 0
            ann_count = 0
            for src_id in final_source_ids:
                annotations_to_save = self.modified_annotations.get(src_id, [])
                for ann in annotations_to_save:
                    ann_copy = ann.copy()
                    # Remove internal helper data if it exists
                    ann_copy.pop("_original_results_instance", None)

                    ann_copy["id"] = current_ann_id # Re-ID sequentially
                    ann_copy["image_id"] = src_id
                    # Ensure essential fields
                    if "bbox" not in ann_copy: ann_copy["bbox"] = []
                    if "keypoints" not in ann_copy: ann_copy["keypoints"] = []
                    if "num_keypoints" not in ann_copy: ann_copy["num_keypoints"] = sum(1 for i in range(2, len(ann_copy.get("keypoints",[])), 3) if ann_copy.get("keypoints",[])[i] > 0)
                    if "area" not in ann_copy and ann_copy["bbox"] and len(ann_copy["bbox"])==4: ann_copy["area"] = ann_copy["bbox"][2] * ann_copy["bbox"][3]
                    elif "area" not in ann_copy: ann_copy["area"] = 0
                    if "iscrowd" not in ann_copy: ann_copy["iscrowd"] = 0
                    if "category_id" not in ann_copy: ann_copy["category_id"] = 1

                    if self.is_video_mode:
                        if "instance_id" not in ann_copy or ann_copy["instance_id"] is None:
                            ann_copy["instance_id"] = 0 # Assign default/invalid ID
                            missing_track_id_count += 1

                    output_data["annotations"].append(ann_copy)
                    current_ann_id += 1
                    ann_count += 1

            if self.is_video_mode and missing_track_id_count > 0:
                 print(f"Warning: {missing_track_id_count} annotations were saved with a default track ID (0) because none was assigned.")

            print(f"Saving {ann_count} annotations for {len(output_data['images'])} {type_name}s.")

        # --- Write JSON file ---
        try:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            with open(self.output_path, "w") as f:
                json.dump(output_data, f, indent=4) # Use indent for readability
            messagebox.showinfo(
                "Save Successful", f"Refined annotations saved to:\n{self.output_path}"
            )
            self.modifications_made = False
            self.save_successful = True
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save annotations:\n{e}")
            self.save_successful = False

    def _save_temp_progress(self, event=None):
        """Saves current progress to the output file without closing."""
        print("Attempting temporary save...")
        self._save_refined_annotations()
        if hasattr(self, "save_successful") and self.save_successful:
            print("Temporary save successful.")
        else:
            print("Temporary save failed.")

    # --- Hotkey Decision Methods (Image Mode Only) ---
    def _accept_original(self, event=None):
        """Handles the 'A' key press to keep the original annotations (IMAGE MODE)."""
        if self.is_video_mode: return
        current_src_id = self.source_ids[self.current_idx]
        if self.source_decision_state.get(current_src_id) == 'undecided':
            print(f"Image {current_src_id}: Keeping ORIGINAL annotations.")
            self.source_decision_state[current_src_id] = 'original'
            # Need to copy original annotations into modified map for this image
            self.modified_annotations[current_src_id] = copy.deepcopy(
                self.original_annotations.get(current_src_id, [])
            )
            self.modifications_made = True
            self._draw_annotations() # Redraw in editable mode
        else:
            print(f"Decision already made for image {current_src_id}.")

    def _accept_shrunk(self, event=None):
        """Handles the 'D' key press to keep the shrunk annotations (IMAGE MODE)."""
        if self.is_video_mode: return
        current_src_id = self.source_ids[self.current_idx]
        if self.source_decision_state.get(current_src_id) == 'undecided':
             if current_src_id in self.shrunk_annotations:
                 print(f"Image {current_src_id}: Keeping SHRUNK annotations.")
                 self.source_decision_state[current_src_id] = 'shrunk'
                 # Need to copy shrunk annotations into modified map
                 self.modified_annotations[current_src_id] = copy.deepcopy(
                     self.shrunk_annotations.get(current_src_id, [])
                 )
                 self.modifications_made = True
                 self._draw_annotations() # Redraw in editable mode
             else:
                  messagebox.showwarning("Shrink Data Missing", f"No shrunk annotation data available for image {current_src_id}.")
        else:
            print(f"Decision already made for image {current_src_id}.")

    def run(self):
        """Start the Tkinter main loop."""
        if not self.root:
             print("GUI initialization failed. Exiting.")
             # Ensure video released if init failed but capture object exists
             if self.video_capture and self.video_capture.isOpened():
                 self.video_capture.release()
             return
        if not self.source_ids:
            type_name = "frames" if self.is_video_mode else "images"
            print(f"No {type_name} loaded. GUI not started effectively.")
            # messagebox.showinfo(f"No {type_name.capitalize()}", f"No {type_name} found.")
            # self.root.destroy() # Keep window open but empty?
            return

        self.root.mainloop()
        # Ensure video is released when mainloop finishes (redundant if closed properly)
        if self.video_capture and self.video_capture.isOpened():
            print("Releasing video capture...")
            self.video_capture.release()


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactively refine COCO bounding boxes and poses, with optional video mode.")

    # Input Source Arguments (mutually exclusive groups might be better, but simple check works)
    parser.add_argument("--coco_path", help="Path to the COCO annotation JSON file (used for image mode OR video mode).")
    parser.add_argument("--results_file", help="Path to the custom results JSON file (used ONLY for video mode).")
    parser.add_argument("--img_dir", help="Path to the directory containing images (REQUIRED for image mode).")
    parser.add_argument("--video", help="Path to the video file (REQUIRED for video mode).")

    parser.add_argument("--view_mode", default="bbox", choices=["bbox", "pose"], help="Initial view mode ('bbox' or 'pose').")
    # Image mode specific option
    parser.add_argument("--shrink", type=float, default=None, help="Optional percentage (0-100) to shrink boxes in IMAGE mode. If provided, starts in comparison mode.")


    args = parser.parse_args()

    # --- Input Validation ---
    is_video_mode = args.video is not None
    is_image_mode = args.img_dir is not None

    if is_video_mode == is_image_mode: # Must be one or the other
        print("Error: Must specify EITHER --video (for video mode) OR --img_dir (for image mode).")
        exit(1)

    if is_video_mode:
        if not args.coco_path and not args.results_file:
            print("Error: Video mode requires either --coco_path OR --results_file for annotations.")
            exit(1)
        if args.coco_path and args.results_file:
            print("Error: Cannot specify both --coco_path and --results_file in video mode.")
            exit(1)
        if args.shrink is not None:
             print("Warning: --shrink is ignored in video mode.")
             args.shrink = None
        input_json_path = args.coco_path if args.coco_path else args.results_file
        loaded_from_results = args.results_file is not None
    else: # Image Mode
        if not args.coco_path:
            print("Error: Image mode requires --coco_path.")
            exit(1)
        if args.results_file:
            print("Warning: --results_file is ignored in image mode.")
        if args.shrink is not None and not (0 < args.shrink < 100):
            print(f"Error: --shrink value must be between 0 and 100 (exclusive). Got {args.shrink}")
            exit(1)
        input_json_path = args.coco_path
        loaded_from_results = False
    # --- End Input Validation ---

    view_mode = args.view_mode

    if is_video_mode:
        # --- Video Mode ---
        video_path = args.video
        print(f"--- Running in VIDEO mode {'(Results Format)' if loaded_from_results else '(COCO Format)'} ---")
        print(f"Video Path: {video_path}")
        print(f"Annotation Path: {input_json_path}")
        print(f"Initial View: {view_mode}")

        if loaded_from_results:
            # Load using the new results parser
            source_data, anno_data, category_data, next_ann_id, video_capture = load_results_video_data(
                video_path, input_json_path
            )
        else:
            # Load using the standard COCO video loader
            source_data, anno_data, category_data, next_ann_id, video_capture = load_video_data(
                video_path, input_json_path
            )

        if source_data and anno_data is not None and category_data is not None and video_capture:
            print("Launching Video Refinement GUI...")
            gui = COCORefinementGUI(
                source_data_map=source_data,
                annotation_map=anno_data,
                shrunk_annotation_map={}, # No shrinking in video mode
                categories=category_data,
                initial_view_mode=view_mode,
                input_json_path=input_json_path, # Pass original path
                next_start_ann_id=next_ann_id,
                is_video_mode=True,
                video_capture=video_capture,
                loaded_from_results=loaded_from_results # Pass flag
            )
            gui.run()
            print("Video GUI Closed.")
        else:
            print("Failed to load video data or annotations. Exiting.")
            if video_capture and video_capture.isOpened():
                 video_capture.release()

    else:
        # --- Image Mode ---
        img_dir = args.img_dir
        shrink_percentage = args.shrink
        print("--- Running in IMAGE mode ---")
        print(f"Image Dir: {img_dir}")
        print(f"COCO Path: {input_json_path}")
        print(f"Initial View: {view_mode}")
        if shrink_percentage is not None: print(f"Shrink Percentage: {shrink_percentage}%")
        else: print("Shrinking: Not requested.")

        # Load using standard COCO image loader
        source_data, anno_data, category_data, next_ann_id = load_coco_data(
            input_json_path, img_dir
        )

        if source_data is not None and anno_data is not None and category_data is not None:
            shrunk_anno_data = {}
            perform_shrink_comparison = shrink_percentage is not None

            if perform_shrink_comparison:
                # ... (shrinking logic remains the same) ...
                print(f"Calculating shrunk annotations ({shrink_percentage}%)...")
                temp_coco_data_for_shrinking = {
                    "images": [{"id": img_id, "width": info["original_width"], "height": info["original_height"], "file_name": info["file_name"]} for img_id, info in source_data.items()],
                    "annotations": [ann for anns_list in anno_data.values() for ann in anns_list],
                    "categories": category_data
                }
                try:
                    shrunk_full_coco_data, _ = calculate_shrunk_bboxes(temp_coco_data_for_shrinking, shrink_percentage)
                    for ann in shrunk_full_coco_data.get("annotations", []):
                        img_id = ann["image_id"]
                        if img_id in source_data: shrunk_anno_data.setdefault(img_id, []).append(ann)
                    print("Shrunk annotation calculation complete.")
                except ImportError:
                     print("Warning: Cannot calculate shrunk annotations - coco_labels_utils import failed. Proceeding without comparison.")
                     perform_shrink_comparison = False; shrunk_anno_data = {}
                except Exception as e:
                    try: tk.Tk().withdraw(); messagebox.showerror("Shrinking Error", f"Failed to calculate shrunk boxes: {e}")
                    except tk.TclError: print(f"Error: Tkinter not available for error dialog.")
                    print(f"Error during shrinking calculation: {e}")
                    print("Proceeding without shrinking comparison.")
                    perform_shrink_comparison = False; shrunk_anno_data = {}


            print("Launching Image Refinement GUI...")
            gui = COCORefinementGUI(
                source_data_map=source_data,
                annotation_map=anno_data,
                shrunk_annotation_map=shrunk_anno_data if perform_shrink_comparison else {},
                categories=category_data,
                initial_view_mode=view_mode,
                input_json_path=input_json_path, # Pass original path
                next_start_ann_id=next_ann_id,
                is_video_mode=False,
                # video_capture=None, # Default
                loaded_from_results=False # Explicitly false for image mode
            )
            gui.run()
            print("Image GUI Closed.")
        else:
            print("Failed to load COCO data or no valid images/annotations found. Exiting.")

# --- End Main Execution ---
