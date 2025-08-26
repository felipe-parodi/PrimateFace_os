"""Image directory to COCO format converter.

This module provides functionality to convert directories of images
to COCO annotation format using detection and pose estimation models.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
from tqdm import tqdm

from .base import COCOConverter


class ImageCOCOConverter(COCOConverter):
    """Converter for image directories to COCO format.
    
    This class processes directories of images through detection
    and optional pose estimation pipelines to generate COCO annotations.
    
    Attributes:
        save_visualizations: Whether to save visualization images.
        image_extensions: Supported image file extensions.
    """
    
    def __init__(
        self,
        *args,
        save_visualizations: bool = False,
        image_extensions: Optional[List[str]] = None,
        **kwargs
    ) -> None:
        """Initialize ImageCOCOConverter.
        
        Args:
            *args: Arguments for COCOConverter.
            save_visualizations: Whether to save visualizations.
            image_extensions: List of image extensions to process.
            **kwargs: Keyword arguments for COCOConverter.
        """
        super().__init__(*args, **kwargs)
        self.save_visualizations = save_visualizations
        
        if image_extensions is None:
            self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        else:
            self.image_extensions = image_extensions
    
    def process_directory(
        self,
        image_dir: Union[str, Path],
        output_json: Optional[Union[str, Path]] = None,
        max_images: Optional[int] = None,
        save_images: bool = False,
        images_output_dir: Optional[Union[str, Path]] = None
    ) -> str:
        """Process all images in a directory.
        
        Args:
            image_dir: Directory containing images.
            output_json: Output path for COCO JSON.
            max_images: Maximum number of images to process.
            save_images: Whether to copy/save processed images.
            images_output_dir: Directory to save processed images.
            
        Returns:
            Path to saved COCO JSON file.
        """
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise ValueError(f"Image directory does not exist: {image_dir}")
        
        image_paths = self._collect_image_paths(image_dir)
        
        if not image_paths:
            raise ValueError(f"No images found in {image_dir}")
        
        if max_images:
            image_paths = image_paths[:max_images]
        
        print(f"Processing {len(image_paths)} images...")
        
        if save_images and images_output_dir:
            images_output_dir = Path(images_output_dir)
            images_output_dir.mkdir(parents=True, exist_ok=True)
        
        all_images = []
        all_annotations = []
        
        for image_path in tqdm(image_paths, desc="Processing images"):
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: Could not read {image_path}")
                continue
            
            relative_name = image_path.relative_to(image_dir)
            image_info, annotations, visualization = self.process_image(
                image,
                str(relative_name),
                return_visualizations=self.save_visualizations
            )
            
            if save_images and images_output_dir:
                output_image_path = images_output_dir / relative_name
                output_image_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_image_path), image)
                image_info["file_name"] = str(relative_name)
            
            if self.save_visualizations and visualization is not None:
                viz_dir = self.output_dir / "visualizations"
                viz_dir.mkdir(parents=True, exist_ok=True)
                viz_path = viz_dir / f"{image_path.stem}_viz.jpg"
                cv2.imwrite(str(viz_path), visualization)
            
            all_images.append(image_info)
            all_annotations.extend(annotations)
        
        coco_data = self.create_coco_dataset(all_images, all_annotations)
        
        if output_json is None:
            output_json = self.output_dir / "annotations.json"
        
        return self.save_coco_json(coco_data, output_json)
    
    def _collect_image_paths(
        self,
        directory: Path
    ) -> List[Path]:
        """Collect all image paths from directory.
        
        Args:
            directory: Directory to search.
            
        Returns:
            List of image file paths.
        """
        image_paths = []
        
        for ext in self.image_extensions:
            image_paths.extend(directory.glob(f"*{ext}"))
            image_paths.extend(directory.glob(f"*{ext.upper()}"))
            image_paths.extend(directory.glob(f"**/*{ext}"))
            image_paths.extend(directory.glob(f"**/*{ext.upper()}"))
        
        return sorted(list(set(image_paths)))
    
    def process_image_list(
        self,
        image_paths: List[Union[str, Path]],
        output_json: Optional[Union[str, Path]] = None
    ) -> str:
        """Process a specific list of images.
        
        Args:
            image_paths: List of image file paths.
            output_json: Output path for COCO JSON.
            
        Returns:
            Path to saved COCO JSON file.
        """
        all_images = []
        all_annotations = []
        
        for image_path in tqdm(image_paths, desc="Processing images"):
            image_path = Path(image_path)
            image = cv2.imread(str(image_path))
            
            if image is None:
                print(f"Warning: Could not read {image_path}")
                continue
            
            image_info, annotations, _ = self.process_image(
                image,
                image_path.name,
                return_visualizations=False
            )
            
            all_images.append(image_info)
            all_annotations.extend(annotations)
        
        coco_data = self.create_coco_dataset(all_images, all_annotations)
        
        if output_json is None:
            output_json = self.output_dir / "annotations.json"
        
        return self.save_coco_json(coco_data, output_json)