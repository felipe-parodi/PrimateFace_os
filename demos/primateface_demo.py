#!/usr/bin/env python3
"""PrimateFace Demo Interface.

Unified entry point for all PrimateFace demonstration scripts.
Provides a consistent command-line interface for different tasks.

Available Commands:
    process     Process videos or images for detection and pose estimation
    classify    Classify primate genus using Vision Language Models
    visualize   Create visualizations from COCO annotations

Examples:
    # Process a single image
    python primateface_demo.py process --input image.jpg --input-type image \
        --det-config config.py --det-checkpoint model.pth \
        --output-dir results/ --save-viz

    # Process a video
    python primateface_demo.py process --input video.mp4 --input-type video \
        --det-config config.py --det-checkpoint model.pth \
        --pose-config pose_config.py --pose-checkpoint pose_model.pth \
        --output-dir results/ --save-viz --viz-pose

    # Process image directory
    python primateface_demo.py process --input ./images/ --input-type images \
        --det-config config.py --det-checkpoint model.pth \
        --pose-config pose_config.py --pose-checkpoint pose_model.pth \
        --output-dir results/ --save-predictions

    # Classify genus
    python primateface_demo.py classify --input image.jpg \
        --model SmolVLM --output results.json

    # Create visualizations
    python primateface_demo.py visualize --coco annotations.json \
        --img-dir ./images --output-dir visualizations/ --num-samples 10
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Import PrimateFace modules (relative imports when run as module)
try:
    from . import process
    from . import classify_genus
    from . import visualize_coco_annotations
    from .constants import DEFAULT_BBOX_THR, DEFAULT_KPT_THR, DEFAULT_NMS_THR
except ImportError:
    import process
    import classify_genus
    import visualize_coco_annotations
    from constants import DEFAULT_BBOX_THR, DEFAULT_KPT_THR, DEFAULT_NMS_THR


def add_process_args(parser: argparse.ArgumentParser) -> None:
    """Add processing arguments to parser.
    
    Args:
        parser: ArgumentParser to add arguments to.
    """
    # Required arguments
    parser.add_argument('--input', required=True, 
                    help='Input video file or image directory')
    parser.add_argument('--input-type', required=True, choices=['video', 'image', 'images'],
                    help='Type of input: video file, single image, or image directory')
    parser.add_argument('--det-config', required=True, 
                    help='Detection model config file')
    parser.add_argument('--det-checkpoint', required=True, 
                    help='Detection model checkpoint')
    parser.add_argument('--pose-config', 
                    help='Pose model config file (optional, if not provided only detection is run)')
    parser.add_argument('--pose-checkpoint', 
                    help='Pose model checkpoint (optional, if not provided only detection is run)')
    
    # Optional arguments
    parser.add_argument('--output-dir', help='Output directory for results')
    parser.add_argument('--save-predictions', action='store_true',
                    help='Save predictions as JSON')
    parser.add_argument('--save-viz', action='store_true',
                    help='Save visualization output')
    parser.add_argument('--viz-pose', action='store_true',
                    help='Visualize pose keypoints instead of just bboxes')
    
    # Thresholds
    parser.add_argument('--bbox-thr', type=float, default=DEFAULT_BBOX_THR,
                    help=f'Detection confidence threshold (default: {DEFAULT_BBOX_THR})')
    parser.add_argument('--kpt-thr', type=float, default=DEFAULT_KPT_THR,
                    help=f'Keypoint confidence threshold (default: {DEFAULT_KPT_THR})')
    parser.add_argument('--nms-thr', type=float, default=DEFAULT_NMS_THR,
                    help=f'NMS threshold (default: {DEFAULT_NMS_THR})')
    
    # Smoothing (for videos)
    parser.add_argument('--smooth', action='store_true',
                    help='Apply temporal smoothing to keypoints (videos only)')
    parser.add_argument('--smooth-median-window', type=int, default=5,
                    help='Window size for median filter (default: 5)')
    parser.add_argument('--smooth-savgol-window', type=int, default=7,
                    help='Window size for Savitzky-Golay filter (default: 7)')
    parser.add_argument('--smooth-savgol-order', type=int, default=3,
                    help='Polynomial order for Savitzky-Golay filter (default: 3)')
    
    # Device
    parser.add_argument('--device', default='cuda:0',
                    help='Device to use for inference')


def add_classify_args(parser: argparse.ArgumentParser) -> None:
    """Add classification arguments to parser.
    
    Args:
        parser: ArgumentParser to add arguments to.
    """
    parser.add_argument('--input', required=True,
                    help='Input image file or directory')
    parser.add_argument('--output', help='Output JSON file')
    parser.add_argument('--model', choices=['SmolVLM', 'InternVL2-2B'], 
                    default='SmolVLM', help='VLM model to use')
    parser.add_argument('--device', default='cuda:0',
                    help='Device to use for inference')


def add_visualize_args(parser: argparse.ArgumentParser) -> None:
    """Add visualization arguments to parser.
    
    Args:
        parser: ArgumentParser to add arguments to.
    """
    parser.add_argument('--coco', required=True, 
                    help='COCO annotations file')
    parser.add_argument('--img-dir', required=True, 
                    help='Image directory')
    parser.add_argument('--output-dir', required=True, 
                    help='Output directory')
    parser.add_argument('--num-samples', type=int, 
                    help='Number of samples to visualize')
    parser.add_argument('--seed', type=int, 
                    help='Random seed for sampling')
    parser.add_argument('--dpi', type=int, default=300, 
                    help='Output DPI')
    parser.add_argument('--format', choices=['png', 'pdf', 'svg'], default='png',
                    help='Output format')


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='PrimateFace Demo Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        metavar='COMMAND'
    )
    
    # Process subcommand
    process_parser = subparsers.add_parser(
        'process',
        help='Process videos or images for detection and pose estimation'
    )
    add_process_args(process_parser)
    
    # Classify subcommand
    classify_parser = subparsers.add_parser(
        'classify',
        help='Classify primate genus using Vision Language Models'
    )
    add_classify_args(classify_parser)
    
    # Visualize subcommand
    viz_parser = subparsers.add_parser(
        'visualize',
        help='Create visualizations from COCO annotations'
    )
    add_visualize_args(viz_parser)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
    
    try:
        if args.command == 'process':
            # Call the process module's main function
            sys.argv = [
                'process.py',
                '--input', args.input,
                '--input-type', args.input_type,
                '--det-config', args.det_config,
                '--det-checkpoint', args.det_checkpoint,
            ]
            
            # Add pose arguments if provided
            if args.pose_config:
                sys.argv.extend(['--pose-config', args.pose_config])
            if args.pose_checkpoint:
                sys.argv.extend(['--pose-checkpoint', args.pose_checkpoint])
            
            # Add optional arguments
            if args.output_dir:
                sys.argv.extend(['--output-dir', args.output_dir])
            if args.save_predictions:
                sys.argv.append('--save-predictions')
            if args.save_viz:
                sys.argv.append('--save-viz')
            if args.viz_pose:
                sys.argv.append('--viz-pose')
            if args.smooth:
                sys.argv.append('--smooth')
            
            # Add thresholds and device
            sys.argv.extend(['--bbox-thr', str(args.bbox_thr)])
            sys.argv.extend(['--kpt-thr', str(args.kpt_thr)])
            sys.argv.extend(['--nms-thr', str(args.nms_thr)])
            sys.argv.extend(['--device', args.device])
            
            # Add smoothing parameters if smoothing is enabled
            if args.smooth:
                sys.argv.extend(['--smooth-median-window', str(args.smooth_median_window)])
                sys.argv.extend(['--smooth-savgol-window', str(args.smooth_savgol_window)])
                sys.argv.extend(['--smooth-savgol-order', str(args.smooth_savgol_order)])
            
            return process.main()
            
        elif args.command == 'classify':
            # Call the classify module's main function
            sys.argv = ['classify_genus.py', args.input]
            if args.output:
                sys.argv.append(args.output)
            sys.argv.extend(['--model', args.model])
            sys.argv.extend(['--device', args.device])
            
            return classify_genus.main()
            
        elif args.command == 'visualize':
            # Call the visualize module's main function
            sys.argv = [
                'visualize_coco_annotations.py',
                '--coco', args.coco,
                '--img-dir', args.img_dir,
                '--output-dir', args.output_dir,
            ]
            
            if args.num_samples is not None:
                sys.argv.extend(['--num-samples', str(args.num_samples)])
            if args.seed is not None:
                sys.argv.extend(['--seed', str(args.seed)])
            sys.argv.extend(['--dpi', str(args.dpi)])
            sys.argv.extend(['--format', args.format])
            
            return visualize_coco_annotations.main()
            
        else:
            print(f"Unknown command: {args.command}")
            
    except KeyboardInterrupt:
        print("\nInterrupted by user") 
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()