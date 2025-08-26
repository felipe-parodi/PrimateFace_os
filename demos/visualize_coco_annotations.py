#!/usr/bin/env python3
"""COCO annotation visualizer.

This module creates visualizations of COCO-formatted annotations including
bounding boxes, keypoints, and skeletons on images.

Usage:
    python visualize_coco_annotations.py --coco annotations.json \
        --img-dir ./images --output-dir ./visualizations \
        --num-samples 10 --format png
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

try:
    from .constants import (
        BBOX_COLOR_HEX,
        BBOX_LINE_THICKNESS_MPL,
        DEFAULT_FIG_DPI,
        KEYPOINT_COLOR_HEX,
        KEYPOINT_SIZE_MPL,
        LINE_THICKNESS_MPL,
        SKELETON_COLOR_HEX,
    )
except ImportError:
    from constants import (
        BBOX_COLOR_HEX,
        BBOX_LINE_THICKNESS_MPL,
        DEFAULT_FIG_DPI,
        KEYPOINT_COLOR_HEX,
        KEYPOINT_SIZE_MPL,
        LINE_THICKNESS_MPL,
        SKELETON_COLOR_HEX,
    )

def load_coco_data(
    coco_file_path: str
) -> Tuple[
    Optional[List[Dict[str, Any]]], 
    Optional[Dict[int, List[Dict[str, Any]]]], 
    Optional[Dict[int, Dict[str, Any]]]
]:
    """Load and organize COCO JSON data.

    Args:
        coco_file_path: Path to the COCO JSON file.

    Returns:
        Tuple containing:
            - List of image information dictionaries
            - Dictionary mapping image_id to its annotations
            - Dictionary mapping category_id to category information
        Returns (None, None, None) if loading fails.
    """
    try:
        with open(coco_file_path, 'r') as f:
            coco_data = json.load(f)
    except Exception as e:
        print(f"Error loading COCO JSON file {coco_file_path}: {e}")
        return None, None, None

    images_info = coco_data.get('images', [])
    annotations_list = coco_data.get('annotations', [])
    categories_list = coco_data.get('categories', [])

    image_id_to_annotations = {}
    for ann in annotations_list:
        img_id = ann['image_id']
        if img_id not in image_id_to_annotations:
            image_id_to_annotations[img_id] = []
        image_id_to_annotations[img_id].append(ann)

    category_id_to_info = {cat['id']: cat for cat in categories_list}

    return images_info, image_id_to_annotations, category_id_to_info

def draw_annotations_on_image_matplotlib(
    image_path_obj: Path,
    annotations_for_image: List[Dict],
    category_map: Dict[int, Dict],
    fig_dpi: int = DEFAULT_FIG_DPI
) -> Optional[plt.Figure]:
    """
    Draws bounding boxes, keypoints, and skeletons on an image using Matplotlib.

    Args:
        image_path_obj: Path object for the image file.
        annotations_for_image: List of annotation dicts for this image.
        category_map: Dictionary mapping category_id to category info (for skeleton).
        fig_dpi: DPI for the Matplotlib figure.

    Returns:
        Matplotlib Figure object with annotations drawn, or None if image loading fails.
    """
    try:
        # Load image using Pillow to get dimensions, then Matplotlib to display
        with Image.open(image_path_obj) as pil_img:
            img_width_px, img_height_px = pil_img.size
            # Load with plt.imread for display in matplotlib
            img_data = plt.imread(image_path_obj)

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path_obj}")
        return None
    except Exception as e:
        print(f"Error loading image {image_path_obj}: {e}")
        return None

    # Calculate figure size in inches to maintain aspect ratio
    fig_width_inches = img_width_px / fig_dpi
    fig_height_inches = img_height_px / fig_dpi

    fig, ax = plt.subplots(figsize=(fig_width_inches, fig_height_inches), dpi=fig_dpi)
    ax.imshow(img_data)
    ax.axis('off')  # Turn off axis numbers and ticks

    if not annotations_for_image:
        print(f"No annotations to draw for {image_path_obj.name}. Displaying raw image.")
        return fig # Return figure with raw image

    for ann in annotations_for_image:
        # Draw Bounding Box
        if 'bbox' in ann:
            bbox = ann['bbox']  # COCO format: [x, y, width, height]
            x, y, w, h = map(float, bbox) # Use float for Matplotlib patches
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=BBOX_LINE_THICKNESS_MPL,
                edgecolor=BBOX_COLOR_HEX,
                facecolor='none',
                zorder=10
            )
            ax.add_patch(rect)

        # Draw Keypoints and Skeleton
        if 'keypoints' in ann and 'category_id' in ann:
            keypoints_coco = ann['keypoints']
            num_keypoints = len(keypoints_coco) // 3
            
            parsed_keypoints_coords_x = []
            parsed_keypoints_coords_y = []
            parsed_keypoints_full = [] # List of (x, y, v) for skeleton

            for i in range(num_keypoints):
                px, py, pv = float(keypoints_coco[i*3]), float(keypoints_coco[i*3+1]), int(keypoints_coco[i*3+2])
                parsed_keypoints_full.append((px, py, pv))
                if pv > 0:  # 0: not labeled, 1: labeled but not visible, 2: labeled and visible
                    parsed_keypoints_coords_x.append(px)
                    parsed_keypoints_coords_y.append(py)

            # Draw keypoints using scatter (size is in points^2)
            if parsed_keypoints_coords_x:
                ax.scatter(parsed_keypoints_coords_x, parsed_keypoints_coords_y,
                        s=KEYPOINT_SIZE_MPL, color=KEYPOINT_COLOR_HEX, zorder=20, alpha=0.9)

            # Draw Skeleton
            category_id = ann['category_id']
            category_info = category_map.get(category_id)
            if category_info and 'skeleton' in category_info:
                skeleton_links_1_based = category_info['skeleton'] # Typically 1-based indices
                for link in skeleton_links_1_based:
                    idx1, idx2 = link[0] - 1, link[1] - 1 # Convert 1-based to 0-based
                    if 0 <= idx1 < len(parsed_keypoints_full) and 0 <= idx2 < len(parsed_keypoints_full):
                        pt1_x, pt1_y, pt1_v = parsed_keypoints_full[idx1]
                        pt2_x, pt2_y, pt2_v = parsed_keypoints_full[idx2]

                        if pt1_v > 0 and pt2_v > 0: # Draw if both points are at least labeled
                            ax.plot([pt1_x, pt2_x], [pt1_y, pt2_y],
                                    color=SKELETON_COLOR_HEX, linewidth=LINE_THICKNESS_MPL,
                                    solid_capstyle='round', zorder=15, alpha=0.85)
    return fig

def save_image_outputs_matplotlib(fig: Optional[plt.Figure], base_output_path: Path) -> None:
    """
    Saves the Matplotlib figure as PNG and SVG.

    Args:
        fig: Matplotlib Figure object.
        base_output_path: Base path for saving (without extension).
    """
    if fig is None:
        print(f"Skipping save for {base_output_path.name} as figure object is None.")
        return

    # Save PNG (raster)
    png_path = base_output_path.with_suffix(".png")
    try:
        # pad_inches=0.01 helps to prevent minor clipping with tight bbox
        fig.savefig(str(png_path), dpi=DEFAULT_FIG_DPI, bbox_inches='tight', pad_inches=0.01)
        print(f"Saved PNG: {png_path}")
    except Exception as e:
        print(f"Error saving PNG {png_path}: {e}")

    # Save as SVG (vector)
    svg_path = base_output_path.with_suffix(".svg")
    try:
        fig.savefig(str(svg_path), format='svg', bbox_inches='tight', pad_inches=0.01)
        print(f"Saved SVG: {svg_path}")
    except Exception as e:
        print(f"Error saving SVG {svg_path}: {e}")
    finally:
        plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sample visualizations (bounding box and pose) from a COCO JSON file using Matplotlib.")
    parser.add_argument("--coco-json", required=True, type=str, help="Path to the COCO JSON annotation file.")
    parser.add_argument("--output-dir", required=True, type=str, help="Directory to save the output images (PNG and SVG).")
    parser.add_argument("--num-images", type=int, default=30, help="Number of random images to select for visualization.")
    parser.add_argument("--image-root-dir", type=str, default=None,
                        help="Optional root directory if image paths in COCO JSON are relative. "
                            "If not provided, paths are assumed to be absolute or relative to where the script is run/COCO file location.")
    parser.add_argument("--dpi", type=int, default=DEFAULT_FIG_DPI, help="DPI for the output raster images and figure sizing.")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images_info, image_id_to_annotations, category_id_to_info = load_coco_data(args.coco_json)

    if not images_info or not image_id_to_annotations or not category_id_to_info:
        print("Failed to load or parse COCO data. Exiting.")
        return

    if not images_info:
        print("No images found in COCO data. Exiting.")
        return

    # TODO: Make this a command line argument
    target_prefixes = ["Daubentonia", "mandrillus", "nasalis", "Saguinus", "rhinopithecus"]

    filtered_images_info = [
        img_info for img_info in images_info
        if any(Path(img_info['file_name']).name.startswith(prefix) for prefix in target_prefixes)
    ]

    if not filtered_images_info:
        print(f"Error: No images found with the specified prefixes: {', '.join(target_prefixes)}. Exiting.")
        return
    
    print(f"Found {len(filtered_images_info)} images matching the specified prefixes.")

    num_to_select = min(args.num_images, len(filtered_images_info))
    if num_to_select == 0:
        print("No images to select (num_images is 0 or no matching images after filtering). Exiting.")
        return
        
    selected_images_info = random.sample(filtered_images_info, num_to_select)
    print(f"Selected {len(selected_images_info)} random images from the filtered set for visualization.")

    for i, img_info in enumerate(selected_images_info):
        image_id = img_info['id']
        file_name = img_info['file_name']

        if args.image_root_dir:
            image_path = Path(args.image_root_dir) / file_name
        else:
            coco_json_path_obj = Path(args.coco_json)
            image_path_candidate = Path(file_name)
            if not image_path_candidate.is_absolute():
                image_path = coco_json_path_obj.parent / file_name
            else:
                image_path = image_path_candidate
        
        image_path = image_path.resolve()

        print(f"\nProcessing image {i+1}/{num_to_select}: {image_path}")

        if not image_path.exists():
            print(f"Warning: Image file not found at {image_path}. Skipping.")
            continue

        annotations = image_id_to_annotations.get(image_id, [])
        
        # Generate base name for output files
        base_name_no_ext = Path(file_name).stem
        if not annotations:
            output_file_label = "_raw"
        else:
            output_file_label = "_annotated_mpl"
        base_output_path = output_dir / f"{base_name_no_ext}{output_file_label}"

        # Draw annotations using Matplotlib
        figure_object = draw_annotations_on_image_matplotlib(
            image_path,
            annotations,
            category_id_to_info,
            fig_dpi=args.dpi
        )

        if figure_object:
            save_image_outputs_matplotlib(figure_object, base_output_path)
        else:
            print(f"Skipping saving for {image_path.name} as figure generation failed.")

    print(f"\nFinished generating Matplotlib visualizations in {output_dir}")

if __name__ == "__main__":
    main()