#!/usr/bin/env python3
"""Unified CLI for DINOv2 feature extraction and analysis.

This module provides a command-line interface for all DINOv2 operations
including feature extraction, visualization, and subset selection.

Usage:
    # Extract embeddings
    python -m dinov2.dinov2_cli extract --input images/ --output embeddings.pt
    
    # Visualize with UMAP
    python -m dinov2.dinov2_cli visualize --embeddings embeddings.pt --output plot.svg
    
    # Generate patch visualizations
    python -m dinov2.dinov2_cli patches --images images/ --output patches/
    
    # Select diverse subset
    python -m dinov2.dinov2_cli select --embeddings embeddings.pt --n 1000 --output subset.txt
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader

from .constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_DOT_SIZE,
    DEFAULT_MODEL,
    DEFAULT_NUM_CLUSTERS,
    DEFAULT_NUM_WORKERS,
    DEFAULT_PLOT_FORMAT,
    DEFAULT_TARGET_SUBSET_SIZE,
    DEFAULT_UMAP_MIN_DIST,
    DEFAULT_UMAP_N_NEIGHBORS,
    LOG_FORMAT,
    LOG_LEVEL,
)
from .core import DINOv2Extractor, ImageDataset, load_embeddings, save_embeddings
from .selection import DiverseImageSelector, save_selection
from .visualization import PatchVisualizer, UMAPVisualizer

# Setup logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def collate_pil(batch):
    """Collate function for PIL images in DataLoader.
    
    Must be defined at module level for Windows multiprocessing compatibility.
    """
    images = [item['images'] for item in batch]
    image_ids = [item['image_ids'] for item in batch]
    indices = [item['index'] for item in batch]
    return {'images': images, 'image_ids': image_ids, 'indices': indices}


def extract_embeddings(args: argparse.Namespace) -> int:
    """Extract DINOv2 embeddings from images.
    
    Args:
        args: Command line arguments.
        
    Returns:
        - A .pt file with the embeddings and the image ids.
    """
    try:
        # Initialize extractor
        extractor = DINOv2Extractor(
            model_name=args.model,
            device=args.device
        )
        
        # Create dataset without transforms (DINOv2 processor handles preprocessing)
        dataset = ImageDataset(args.input, transform=None)
        
        # On Windows, set num_workers to 0 to avoid multiprocessing issues
        import platform
        num_workers = 0 if platform.system() == 'Windows' else args.num_workers
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=num_workers,
            shuffle=False,
            collate_fn=collate_pil
        )
        
        embeddings, image_ids = extractor.extract_from_dataloader(
            dataloader,
            progress=not args.quiet
        )
        
        # Save embeddings
        metadata = {
            'model': args.model,
            'input': str(args.input),
            'num_samples': len(embeddings)
        }
        save_embeddings(embeddings, image_ids, args.output, metadata)
        
        print(f"Successfully extracted {len(embeddings)} embeddings")
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return 1


def visualize_embeddings(args: argparse.Namespace) -> int:
    """Create UMAP visualization of embeddings.
    
    Args:
        args: Command line arguments.
        
    Returns:
        - A .svg file with the UMAP visualization.
    """
    try:
        # Load embeddings
        embeddings, image_ids, metadata = load_embeddings(args.embeddings)
        
        # Initialize visualizer
        visualizer = UMAPVisualizer(
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            random_state=args.seed
        )
        
        # Perform UMAP
        visualizer.fit_transform(embeddings)
        
        # Perform clustering
        visualizer.cluster(n_clusters=args.n_clusters)
        
        # Create plot
        if args.interactive:
            output_path = args.output.with_suffix('.html')
            visualizer.plot_interactive(
                output_path,
                image_paths=image_ids if args.show_images else None,
                title=args.title or "DINOv2 UMAP Projection"
            )
        else:
            visualizer.plot_static(
                args.output,
                title=args.title or "DINOv2 UMAP Projection",
                figsize=(args.width, args.height),
                dpi=args.dpi,
                dot_size=args.dot_size,
                colormap=args.colormap,
                show_axes=not args.no_axes
            )
        
        print(f"Visualization saved to {args.output}")
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        return 1


def visualize_patches(args: argparse.Namespace) -> int:
    """Generate patch and attention visualizations.
    
    Args:
        args: Command line arguments.
        
    Returns:
        - A .svg file with the patch visualizations.
    """
    try:
        # Initialize extractor with attention
        extractor = DINOv2Extractor(
            model_name=args.model,
            device=args.device,
            return_attention=True
        )
        
        # Get image paths
        dataset = ImageDataset(args.images)
        image_paths = dataset.image_paths[:args.limit] if args.limit else dataset.image_paths
        
        # Initialize visualizer
        visualizer = PatchVisualizer(model_name=args.model)
        
        # Process each image
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in tqdm(image_paths, desc="Processing images"):
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Extract features
            features = extractor.extract_features(image, return_patches=True)
            
            # Create visualization
            output_path = output_dir / f"{img_path.stem}_patches.svg"
            visualizer.visualize_patches(
                image,
                features['patch_tokens'][0],
                output_path,
                features.get('attention')
            )
        
        print(f"Generated {len(image_paths)} patch visualizations in {output_dir}")
        
    except Exception as e:
        logger.error(f"Patch visualization failed: {e}")
        return 1


def select_subset(args: argparse.Namespace) -> int:
    """Select diverse subset of images.
    
    Args:
        args: Command line arguments.
        
    Returns:
        - A .txt file with the selected image ids.
    """
    try:
        # Load embeddings
        embeddings, image_ids, metadata = load_embeddings(args.embeddings)
        
        # Initialize selector
        selector = DiverseImageSelector(
            strategy=args.strategy,
            random_state=args.seed
        )
        
        # Perform selection
        indices, selected_ids = selector.select(
            embeddings,
            n_samples=args.n,
            n_clusters=args.n_clusters,
            image_ids=image_ids
        )
        
        # Save selection - pass image_ids (not selected_ids) so indices can be used properly
        save_selection(indices, args.output, image_ids)
        
        print(f"Selected {len(indices)} diverse samples")
        
    except Exception as e:
        logger.error(f"Selection failed: {e}")
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser.
    
    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description='DINOv2 Feature Extraction and Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        metavar='COMMAND'
    )
    
    # Extract command
    extract_parser = subparsers.add_parser(
        'extract',
        help='Extract DINOv2 embeddings from images'
    )
    extract_parser.add_argument('--input', required=True, type=Path,
                            help='Input directory, CSV file, or image list')
    extract_parser.add_argument('--output', required=True, type=Path,
                            help='Output embeddings file (.pt)')
    extract_parser.add_argument('--model', default=DEFAULT_MODEL,
                            help=f'DINOv2 model name (default: {DEFAULT_MODEL})')
    extract_parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                            help=f'Batch size (default: {DEFAULT_BATCH_SIZE})')
    extract_parser.add_argument('--num-workers', type=int, default=DEFAULT_NUM_WORKERS,
                            help=f'DataLoader workers (default: {DEFAULT_NUM_WORKERS})')
    extract_parser.add_argument('--device', default=None,
                            help='Device (cuda/cpu, auto-detect if not specified)')
    extract_parser.add_argument('--quiet', action='store_true',
                            help='Disable progress bar')
    
    # Visualize command
    viz_parser = subparsers.add_parser(
        'visualize',
        help='Create UMAP visualization of embeddings'
    )
    viz_parser.add_argument('--embeddings', required=True, type=Path,
                        help='Input embeddings file (.pt)')
    viz_parser.add_argument('--output', required=True, type=Path,
                        help='Output plot file')
    viz_parser.add_argument('--n-clusters', type=int, default=DEFAULT_NUM_CLUSTERS,
                        help=f'Number of clusters (default: {DEFAULT_NUM_CLUSTERS})')
    viz_parser.add_argument('--n-neighbors', type=int, default=DEFAULT_UMAP_N_NEIGHBORS,
                        help=f'UMAP n_neighbors (default: {DEFAULT_UMAP_N_NEIGHBORS})')
    viz_parser.add_argument('--min-dist', type=float, default=DEFAULT_UMAP_MIN_DIST,
                        help=f'UMAP min_dist (default: {DEFAULT_UMAP_MIN_DIST})')
    viz_parser.add_argument('--metric', default='cosine',
                        help='Distance metric (default: cosine)')
    viz_parser.add_argument('--interactive', action='store_true',
                        help='Create interactive HTML plot')
    viz_parser.add_argument('--show-images', action='store_true',
                        help='Show image thumbnails in interactive plot')
    viz_parser.add_argument('--title', help='Plot title')
    viz_parser.add_argument('--width', type=int, default=12,
                        help='Figure width in inches')
    viz_parser.add_argument('--height', type=int, default=10,
                        help='Figure height in inches')
    viz_parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for static plot')
    viz_parser.add_argument('--dot-size', type=float, default=DEFAULT_DOT_SIZE,
                        help=f'Size of scatter plot points (default: {DEFAULT_DOT_SIZE})')
    viz_parser.add_argument('--colormap', default='tab20',
                        help='Matplotlib colormap')
    viz_parser.add_argument('--no-axes', action='store_true',
                        help='Hide axes labels')
    viz_parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Patches command
    patches_parser = subparsers.add_parser(
        'patches',
        help='Generate patch and attention visualizations'
    )
    patches_parser.add_argument('--images', required=True, type=Path,
                            help='Input images directory or list')
    patches_parser.add_argument('--output', required=True, type=Path,
                            help='Output directory for visualizations')
    patches_parser.add_argument('--model', default=DEFAULT_MODEL,
                            help=f'DINOv2 model (default: {DEFAULT_MODEL})')
    patches_parser.add_argument('--limit', type=int,
                            help='Limit number of images to process')
    patches_parser.add_argument('--device', default=None,
                            help='Device (cuda/cpu)')
    
    # Select command
    select_parser = subparsers.add_parser(
        'select',
        help='Select diverse subset of images'
    )
    select_parser.add_argument('--embeddings', required=True, type=Path,
                            help='Input embeddings file (.pt)')
    select_parser.add_argument('--output', required=True, type=Path,
                            help='Output selection file')
    select_parser.add_argument('--n', type=int, default=DEFAULT_TARGET_SUBSET_SIZE,
                            help=f'Number to select (default: {DEFAULT_TARGET_SUBSET_SIZE})')
    select_parser.add_argument('--strategy', choices=['random', 'cluster', 'fps', 'hybrid'],
                            default='hybrid', help='Selection strategy')
    select_parser.add_argument('--n-clusters', type=int,
                            help='Number of clusters for cluster-based strategies')
    select_parser.add_argument('--seed', type=int, default=42,
                            help='Random seed')
    
    return parser


def main() -> int:
    """Main entry point for the CLI"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Dispatch to appropriate function
    if args.command == 'extract':
        return extract_embeddings(args)
    elif args.command == 'visualize':
        return visualize_embeddings(args)
    elif args.command == 'patches':
        return visualize_patches(args)
    elif args.command == 'select':
        return select_subset(args)
    else:
        logger.error(f"Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())