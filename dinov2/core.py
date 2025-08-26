"""Core DINOv2 functionality for feature extraction.

This module provides the main classes for DINOv2 model management and
feature extraction from images.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from .constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MODEL,
    DEFAULT_NUM_WORKERS,
    SUPPORTED_IMAGE_EXTENSIONS,
)

logger = logging.getLogger(__name__)


class DINOv2Extractor:
    """DINOv2 feature extractor for images.
    
    This class handles DINOv2 model initialization, image processing,
    and feature extraction with support for batch processing.
    
    Attributes:
        model_name: Name or path of the DINOv2 model.
        device: Device for computation (cuda/cpu).
        model: Loaded DINOv2 model.
        processor: Image processor for the model.
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        return_attention: bool = False
    ) -> None:
        """Initialize the DINOv2 extractor.
        
        Args:
            model_name: HuggingFace model name or path.
            device: Device for computation. Auto-detects if None.
            return_attention: Whether to return attention maps.
        """
        self.model_name = model_name
        self.return_attention = return_attention
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing {model_name} on {self.device}")
        
        # Load model and processor
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        # Use eager attention when we need attention outputs
        if return_attention:
            self.model = AutoModel.from_pretrained(model_name, attn_implementation="eager")
        else:
            self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded successfully on {self.device}")
    
    def extract_features(
        self,
        images: Union[Image.Image, List[Image.Image], torch.Tensor],
        return_patches: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Extract features from images.
        
        Args:
            images: Single image, list of images, or tensor batch.
            return_patches: Whether to return patch features.
            
        Returns:
            Dictionary containing:
                - 'cls_token': CLS token features [B, D]
                - 'patch_tokens': Patch features [B, N, D] (if requested)
                - 'attention': Attention maps (if initialized with return_attention)
        """
        # Ensure images is a list
        if isinstance(images, Image.Image):
            images = [images]
        
        # Process images
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract features
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=self.return_attention)
        
        results = {
            'cls_token': outputs.last_hidden_state[:, 0, :].cpu()
        }
        
        if return_patches:
            results['patch_tokens'] = outputs.last_hidden_state[:, 1:, :].cpu()
        
        if self.return_attention and hasattr(outputs, 'attentions'):
            # Stack attention from all layers
            results['attention'] = torch.stack(outputs.attentions).cpu()
        
        return results
    
    def extract_from_dataloader(
        self,
        dataloader: DataLoader,
        return_patches: bool = False,
        progress: bool = True
    ) -> Tuple[torch.Tensor, List[str]]:
        """Extract features from a DataLoader.
        
        Args:
            dataloader: PyTorch DataLoader with images.
            return_patches: Whether to return patch features.
            progress: Whether to show progress bar.
            
        Returns:
            Tuple of (embeddings, image_ids).
        """
        all_embeddings = []
        all_image_ids = []
        
        iterator = tqdm(dataloader, desc="Extracting features") if progress else dataloader
        
        for batch in iterator:
            if isinstance(batch, dict):
                images = batch['images']
                image_ids = batch.get('image_ids', [None] * len(images))
            else:
                images, image_ids = batch
            
            # Extract features
            features = self.extract_features(images, return_patches=return_patches)
            all_embeddings.append(features['cls_token'])
            all_image_ids.extend(image_ids)
        
        # Concatenate all embeddings
        embeddings = torch.cat(all_embeddings, dim=0)
        
        return embeddings, all_image_ids
    
    def extract_from_directory(
        self,
        directory: Union[str, Path],
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_workers: int = DEFAULT_NUM_WORKERS,
        return_patches: bool = False,
        progress: bool = True
    ) -> Tuple[torch.Tensor, List[str]]:
        """Extract features from all images in a directory.
        
        Args:
            directory: Path to directory containing images.
            batch_size: Batch size for processing.
            num_workers: Number of data loading workers.
            return_patches: Whether to return patch features.
            progress: Whether to show progress bar.
            
        Returns:
            Tuple of (embeddings, image_paths).
        """
        # Normalize Windows paths
        if isinstance(directory, str):
            directory = Path(directory.replace('\\', '/'))
        
        dataset = ImageDataset(directory)
        
        # Custom collate function for PIL images
        def collate_pil(batch):
            images = [item['images'] for item in batch]
            image_ids = [item['image_ids'] for item in batch]
            indices = [item['index'] for item in batch]
            return {'images': images, 'image_ids': image_ids, 'indices': indices}
        
        # On Windows, set num_workers to 0 to avoid multiprocessing issues
        import platform
        if platform.system() == 'Windows':
            num_workers = 0
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            collate_fn=collate_pil
        )
        
        return self.extract_from_dataloader(dataloader, return_patches, progress)


class ImageDataset(Dataset):
    """PyTorch dataset for loading images from various sources.
    
    This dataset supports loading from:
    - Directory of images
    - List of image paths
    - CSV file with image paths
    
    Attributes:
        image_paths: List of paths to images.
        transform: Optional transform to apply to images.
    """
    
    def __init__(
        self,
        source: Union[str, Path, List[str]],
        transform: Optional[transforms.Compose] = None,
        image_column: str = "image_path"
    ) -> None:
        """Initialize the dataset.
        
        Args:
            source: Directory path, list of paths, or CSV file.
            transform: Optional torchvision transforms. If None, no transforms applied.
            image_column: Column name for image paths in CSV.
        """
        # Normalize Windows paths
        if isinstance(source, str):
            source = source.replace('\\', '/')
        
        # Only use default transform if explicitly not set to None
        if transform == 'default':
            self.transform = self._get_default_transform()
        else:
            self.transform = transform
        self.image_paths = self._parse_source(source, image_column)
        
        if not self.image_paths:
            raise ValueError(f"No images found in {source}")
        
        logger.info(f"Created dataset with {len(self.image_paths)} images")
        
        # Check for duplicates
        unique_paths = set(self.image_paths)
        if len(unique_paths) != len(self.image_paths):
            logger.warning(f"Found {len(self.image_paths) - len(unique_paths)} duplicate image paths!")
            # Show duplicates
            from collections import Counter
            counts = Counter(self.image_paths)
            for path, count in counts.items():
                if count > 1:
                    logger.warning(f"  {path.name} appears {count} times")
    
    def _get_default_transform(self) -> transforms.Compose:
        """Get default image transformation pipeline."""
        return transforms.Compose([
            transforms.Resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _parse_source(
        self,
        source: Union[str, Path, List[str]],
        image_column: str
    ) -> List[Path]:
        """Parse various source types to get image paths."""
        if isinstance(source, list):
            return [Path(p) for p in source]
        
        source = Path(source)
        
        if source.is_dir():
            # Directory of images - use case-insensitive matching
            paths = set()  # Use set to avoid duplicates
            for ext in SUPPORTED_IMAGE_EXTENSIONS:
                # Use rglob with case-insensitive pattern
                for p in source.iterdir():
                    if p.is_file() and p.suffix.lower() == ext.lower():
                        paths.add(p)
            
            sorted_paths = sorted(paths)
            
            # Log the found files for debugging
            logger.debug(f"Found {len(sorted_paths)} images in {source}:")
            for p in sorted_paths[:10]:  # Show first 10
                logger.debug(f"  - {p.name}")
            if len(sorted_paths) > 10:
                logger.debug(f"  ... and {len(sorted_paths) - 10} more")
            
            return sorted_paths
        
        elif source.suffix == '.csv':
            # CSV file with image paths
            import polars as pl
            df = pl.read_csv(source)
            if image_column not in df.columns:
                raise ValueError(f"Column '{image_column}' not found in CSV")
            return [Path(p) for p in df[image_column].to_list()]
        
        elif source.suffix == '.txt':
            # Text file with one path per line
            with open(source, 'r') as f:
                return [Path(line.strip()) for line in f if line.strip()]
        
        else:
            # Single image file
            if source.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                return [source]
            else:
                raise ValueError(f"Unsupported source type: {source}")
    
    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load and return an image with its metadata.
        
        Args:
            idx: Index of the image to load.
            
        Returns:
            Dictionary containing:
                - 'image': Loaded and transformed image
                - 'image_id': Path to the image
                - 'index': Original index
        """
        image_path = self.image_paths[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
        
        return {
            'images': image,
            'image_ids': str(image_path),
            'index': idx
        }


def save_embeddings(
    embeddings: torch.Tensor,
    image_ids: List[str],
    output_path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Save embeddings to a file.
    
    Args:
        embeddings: Tensor of embeddings [N, D].
        image_ids: List of image identifiers.
        output_path: Path to save the embeddings.
        metadata: Optional metadata to include.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'embeddings': embeddings,
        'image_ids': image_ids,
    }
    
    if metadata:
        save_dict['metadata'] = metadata
    
    torch.save(save_dict, output_path)
    logger.info(f"Saved embeddings to {output_path}")


def load_embeddings(
    embeddings_path: Union[str, Path]
) -> Tuple[torch.Tensor, List[str], Optional[Dict[str, Any]]]:
    """Load embeddings from a file.
    
    Args:
        embeddings_path: Path to the embeddings file.
        
    Returns:
        Tuple of (embeddings, image_ids, metadata).
    """
    data = torch.load(embeddings_path)
    
    embeddings = data['embeddings']
    image_ids = data['image_ids']
    metadata = data.get('metadata', None)
    
    logger.info(f"Loaded {len(embeddings)} embeddings from {embeddings_path}")
    
    return embeddings, image_ids, metadata