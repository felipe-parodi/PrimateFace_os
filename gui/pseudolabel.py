#!/usr/bin/env python
"""Unified pseudo-labeling CLI for PrimateFace.

This script provides a single entry point for all pseudo-labeling workflows:
- Generate annotations from images/videos using pretrained models
- Refine existing COCO annotations with interactive GUI
- Convert between different annotation formats

Examples:
    Generate annotations from images:
        $ python pseudolabel.py generate --input ./images --type image \\
            --det-config det.py --det-checkpoint det.pth
            
    Generate with pose estimation:
        $ python pseudolabel.py generate --input ./images --type image \\
            --det-config det.py --det-checkpoint det.pth \\
            --pose-config pose.py --pose-checkpoint pose.pth
            
    Process videos with SAM:
        $ python pseudolabel.py generate --input ./videos --type video \\
            --det-config det.py --det-checkpoint det.pth \\
            --sam-checkpoint sam.pth
            
    Refine existing annotations:
        $ python pseudolabel.py refine --coco annotations.json --images ./images
        
    Parallel processing on GPUs:
        $ python pseudolabel.py generate --input ./videos --type video \\
            --det-config det.py --det-checkpoint det.pth \\
            --gpus 0 1 2 3 --jobs-per-gpu 2
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from gui.converters import ImageCOCOConverter, VideoCOCOConverter
from gui.core import Detector, ModelManager, PoseEstimator, SAMMasker
from gui.utils import parallel_process_videos


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with subcommands.
    
    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="PrimateFace Pseudo-labeling Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0"
    )
    
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands"
    )
    
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate annotations from images/videos"
    )
    add_generate_args(generate_parser)
    
    refine_parser = subparsers.add_parser(
        "refine",
        help="Refine existing COCO annotations with GUI"
    )
    add_refine_args(refine_parser)
    
    detect_parser = subparsers.add_parser(
        "detect",
        help="Run detection only (no pose)"
    )
    add_detect_args(detect_parser)
    
    pose_parser = subparsers.add_parser(
        "pose",
        help="Run pose estimation on existing detections"
    )
    add_pose_args(pose_parser)
    
    return parser


def add_generate_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for generate command."""
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Input directory (images or videos)"
    )
    
    parser.add_argument(
        "--type",
        required=True,
        choices=["image", "video"],
        help="Input type"
    )
    
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--framework",
        default="mmdet",
        choices=["mmdet", "ultralytics"],
        help="Detection framework"
    )
    
    parser.add_argument(
        "--det-config",
        type=str,
        help="Detection model config"
    )
    
    parser.add_argument(
        "--det-checkpoint",
        required=True,
        type=str,
        help="Detection model checkpoint"
    )
    
    parser.add_argument(
        "--pose-config",
        type=str,
        help="Pose model config"
    )
    
    parser.add_argument(
        "--pose-checkpoint",
        type=str,
        help="Pose model checkpoint"
    )
    
    parser.add_argument(
        "--pose-framework",
        default="mmpose",
        choices=["mmpose", "ultralytics"],
        help="Pose framework"
    )
    
    parser.add_argument(
        "--sam-checkpoint",
        type=str,
        help="SAM checkpoint for mask generation"
    )
    
    parser.add_argument(
        "--coco-template",
        type=str,
        help="COCO JSON to copy skeleton from"
    )
    
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
        help="Device to use"
    )
    
    parser.add_argument(
        "--bbox-thr",
        default=0.3,
        type=float,
        help="Detection confidence threshold"
    )
    
    parser.add_argument(
        "--nms-thr",
        default=0.9,
        type=float,
        help="NMS IoU threshold"
    )
    
    parser.add_argument(
        "--max-instances",
        default=3,
        type=int,
        help="Max instances per image"
    )
    
    parser.add_argument(
        "--kpt-thr",
        default=0.05,
        type=float,
        help="Keypoint confidence threshold"
    )
    
    parser.add_argument(
        "--frame-interval",
        default=30,
        type=int,
        help="Frame interval for video processing"
    )
    
    parser.add_argument(
        "--save-viz",
        action="store_true",
        help="Save visualizations"
    )
    
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        help="GPU IDs for parallel processing"
    )
    
    parser.add_argument(
        "--jobs-per-gpu",
        default=1,
        type=int,
        help="Parallel jobs per GPU"
    )
    
    parser.add_argument(
        "--max-images",
        type=int,
        help="Maximum images to process"
    )
    
    parser.add_argument(
        "--max-videos",
        type=int,
        help="Maximum videos to process"
    )
    
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Maximum frames per video"
    )


def add_refine_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for refine command."""
    parser.add_argument(
        "--coco",
        required=True,
        type=str,
        help="COCO JSON file to refine"
    )
    
    parser.add_argument(
        "--images",
        required=True,
        type=str,
        help="Directory containing images"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for refined annotations"
    )
    
    parser.add_argument(
        "--enable-keypoints",
        action="store_true",
        help="Enable keypoint editing"
    )
    
    parser.add_argument(
        "--enable-sam",
        action="store_true",
        help="Enable SAM for mask refinement"
    )
    
    parser.add_argument(
        "--sam-checkpoint",
        type=str,
        help="SAM checkpoint path"
    )


def add_detect_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for detect command."""
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Input directory"
    )
    
    parser.add_argument(
        "--type",
        required=True,
        choices=["image", "video"],
        help="Input type"
    )
    
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Output COCO JSON"
    )
    
    parser.add_argument(
        "--framework",
        default="mmdet",
        choices=["mmdet", "ultralytics"],
        help="Detection framework"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Model config"
    )
    
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Model checkpoint"
    )
    
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
        help="Device to use"
    )
    
    parser.add_argument(
        "--bbox-thr",
        default=0.3,
        type=float,
        help="Detection threshold"
    )


def add_pose_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for pose command."""
    parser.add_argument(
        "--coco",
        required=True,
        type=str,
        help="COCO JSON with detections"
    )
    
    parser.add_argument(
        "--images",
        required=True,
        type=str,
        help="Directory containing images"
    )
    
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Output COCO JSON with poses"
    )
    
    parser.add_argument(
        "--framework",
        default="mmpose",
        choices=["mmpose", "ultralytics"],
        help="Pose framework"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Model config"
    )
    
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Model checkpoint"
    )
    
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
        help="Device to use"
    )
    
    parser.add_argument(
        "--kpt-thr",
        default=0.05,
        type=float,
        help="Keypoint threshold"
    )


def cmd_generate(args: argparse.Namespace) -> None:
    """Execute generate command."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_manager = ModelManager()
    
    detector = Detector(
        framework=args.framework,
        config_path=args.det_config,
        checkpoint_path=args.det_checkpoint,
        device=args.device,
        bbox_thr=args.bbox_thr,
        nms_thr=args.nms_thr,
        max_instances=args.max_instances,
        model_manager=model_manager
    )
    
    pose_estimator = None
    if args.pose_checkpoint:
        pose_estimator = PoseEstimator(
            framework=args.pose_framework,
            config_path=args.pose_config,
            checkpoint_path=args.pose_checkpoint,
            device=args.device,
            kpt_thr=args.kpt_thr,
            model_manager=model_manager
        )
    
    sam_masker = None
    if args.sam_checkpoint:
        sam_masker = SAMMasker(
            checkpoint_path=args.sam_checkpoint,
            device=args.device,
            model_manager=model_manager
        )
    
    if args.type == "image":
        converter = ImageCOCOConverter(
            detector,
            pose_estimator,
            sam_masker,
            output_dir=output_dir,
            coco_template=args.coco_template,
            save_visualizations=args.save_viz
        )
        
        json_path = converter.process_directory(
            args.input,
            max_images=args.max_images,
            save_images=True,
            images_output_dir=output_dir / "images"
        )
    else:
        converter = VideoCOCOConverter(
            detector,
            pose_estimator,
            sam_masker,
            output_dir=output_dir,
            coco_template=args.coco_template,
            frame_interval=args.frame_interval
        )
        
        if args.gpus and len(args.gpus) > 1:
            json_path = converter.process_parallel(
                args.input,
                gpus=args.gpus,
                jobs_per_gpu=args.jobs_per_gpu
            )
        else:
            json_path = converter.process_directory(
                args.input,
                max_videos=args.max_videos,
                max_frames_per_video=args.max_frames
            )
    
    print(f"✓ Annotations saved to: {json_path}")


def cmd_refine(args: argparse.Namespace) -> None:
    """Execute refine command."""
    try:
        from gui.refinement.gui import launch_refinement_gui
    except ImportError:
        print("Error: Refinement GUI module not found.")
        print("This feature is under development.")
        sys.exit(1)
    
    launch_refinement_gui(
        coco_path=args.coco,
        images_dir=args.images,
        output_path=args.output,
        enable_keypoints=args.enable_keypoints,
        enable_sam=args.enable_sam,
        sam_checkpoint=args.sam_checkpoint
    )


def cmd_detect(args: argparse.Namespace) -> None:
    """Execute detect command."""
    model_manager = ModelManager()
    
    detector = Detector(
        framework=args.framework,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        bbox_thr=args.bbox_thr,
        model_manager=model_manager
    )
    
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.type == "image":
        converter = ImageCOCOConverter(
            detector,
            output_dir=output_dir
        )
        json_path = converter.process_directory(args.input)
    else:
        converter = VideoCOCOConverter(
            detector,
            output_dir=output_dir
        )
        json_path = converter.process_directory(args.input)
    
    if Path(json_path) != Path(args.output):
        Path(json_path).rename(args.output)
    
    print(f"✓ Detections saved to: {args.output}")


def cmd_pose(args: argparse.Namespace) -> None:
    """Execute pose command."""
    import json
    
    with open(args.coco, 'r') as f:
        coco_data = json.load(f)
    
    model_manager = ModelManager()
    
    pose_estimator = PoseEstimator(
        framework=args.framework,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        kpt_thr=args.kpt_thr,
        model_manager=model_manager,
        coco_metadata=coco_data.get("categories", [{}])[0]
    )
    
    images_dir = Path(args.images)
    detections_by_image = {}
    
    for ann in coco_data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in detections_by_image:
            detections_by_image[img_id] = []
        detections_by_image[img_id].append(ann["bbox"])
    
    image_id_to_name = {
        img["id"]: img["file_name"]
        for img in coco_data.get("images", [])
    }
    
    updated_annotations = []
    
    for img_id, bboxes in detections_by_image.items():
        img_name = image_id_to_name[img_id]
        img_path = images_dir / img_name
        
        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            continue
        
        import cv2
        image = cv2.imread(str(img_path))
        
        poses = pose_estimator.estimate_pose(image, bboxes)
        
        for bbox, pose in zip(bboxes, poses):
            ann = {
                "image_id": img_id,
                "bbox": bbox,
                "keypoints": pose["keypoints"],
                "num_keypoints": pose["num_keypoints"],
                "category_id": 1,
                "iscrowd": 0
            }
            updated_annotations.append(ann)
    
    for i, ann in enumerate(updated_annotations):
        ann["id"] = i
    
    coco_data["annotations"] = updated_annotations
    
    with open(args.output, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"✓ Poses saved to: {args.output}")


def main() -> None:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "refine":
        cmd_refine(args)
    elif args.command == "detect":
        cmd_detect(args)
    elif args.command == "pose":
        cmd_pose(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()