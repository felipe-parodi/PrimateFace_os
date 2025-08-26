#!/usr/bin/env python3
"""Primate genus classifier using Vision Language Models.

This module provides classification of primate genus using various VLMs from
HuggingFace. Supports single image or batch directory processing with
COCO-formatted output.

Usage:
    # Single image
    python classify_genus.py image.jpg [--model SmolVLM]
    
    # Directory with output
    python classify_genus.py ./images/ output.json [--model SmolVLM]

Available Models:
    - SmolVLM (default): Lightweight and fast
    - InternVL2-2B: More accurate but requires more resources
"""

import argparse
import json
import os
import pathlib
import sys
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image
from transformers import (
    AutoModel,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
)

try:
    # Try relative imports (when imported as module)
    from .constants import PRIMATE_GENERA, VLM_CONFIGS
except ImportError:
    # Fall back to absolute imports (when imported by standalone script)
    from constants import PRIMATE_GENERA, VLM_CONFIGS

# Suppress tokenizer warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


class PrimateClassifierVLM:
    """Vision Language Model-based primate genus classifier.
    
    This class provides optimized inference for primate genus classification
    using various pre-trained VLMs from HuggingFace.
    
    Attributes:
        model_name: Name of the VLM model to use.
        device: Device for inference (cuda/cpu).
        model: Loaded VLM model.
        processor: Model processor/tokenizer.
        config: Model configuration dictionary.
    """
    
    def __init__(self, model_name: str = "SmolVLM") -> None:
        """Initialize the classifier with specified model.
        
        Args:
            model_name: Name of the VLM to use ('SmolVLM' or 'InternVL2-2B').
            
        Raises:
            ValueError: If model_name is not supported.
        """
        if model_name not in VLM_CONFIGS:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available models: {list(VLM_CONFIGS.keys())}"
            )

        self.config = VLM_CONFIGS[model_name]
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = self.config['model_id']
        self.processor = None
        self.model = None
        
        print(f"Initializing {model_name} on {self.device.upper()}...")
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the VLM model and processor.
        
        Raises:
            ImportError: If required model dependencies are not installed.
            RuntimeError: If model loading fails.
        """
        try:
            # Load processor/tokenizer
            processor_class = self.config.get('processor_class', AutoProcessor)
            trust_remote = self.config.get('trust_remote_code', False)
            
            if processor_class == AutoProcessor:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_id,
                    trust_remote_code=trust_remote
                )
            else:
                # For models using AutoTokenizer instead
                self.processor = AutoTokenizer.from_pretrained(
                    self.model_id,
                    trust_remote_code=trust_remote
                )
            
            # Load model
            model_class = self.config.get('model_class', AutoModelForVision2Seq)
            
            print(f"Loading {self.model_name} model...")
            self.model = model_class.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=trust_remote
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            print(f"✓ {self.model_name} loaded successfully on {self.device.upper()}")
            
        except ImportError as e:
            if "flash" in str(e):
                print("Note: flash-attn not found, using standard attention")
            else:
                raise ImportError(
                    f"Failed to load {self.model_name}. "
                    f"Please install required dependencies: {e}"
                )
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {e}")
    
    def preprocess_image(
        self, 
        image_path: Union[str, pathlib.Path]
    ) -> Union[Image.Image, Dict[str, str]]:
        """Load and preprocess an image for classification.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Preprocessed PIL Image or error dictionary.
        """
        try:
            img_path = pathlib.Path(image_path)
            if not img_path.exists():
                return {"error": f"Image not found: {image_path}"}
            
            image = Image.open(img_path)
            
            # Convert to RGB (VLMs expect RGB)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Validate image dimensions
            w, h = image.size
            if w * h > 2048 * 2048:
                print(f"Warning: Large image ({w}x{h}), consider resizing")
            
            return image
            
        except Exception as e:
            return {"error": f"Image preprocessing failed: {e}"}
    
    def classify_genus(self, image: Image.Image) -> Dict[str, str]:
        """Perform genus classification on a preprocessed image.
        
        Args:
            image: PIL Image to classify.
            
        Returns:
            Dictionary with 'genus' key containing the prediction,
            or 'error' key if classification fails.
        """
        prompt_text = self.config['prompt_template'].format(
            genera_list=', '.join(PRIMATE_GENERA)
        )
        
        try:
            with torch.no_grad():
                if self.model_name == "InternVL2-2B":
                    # InternVL2 uses a custom .chat() method
                    response, _ = self.model.chat(
                        self.processor, image, prompt_text, 
                        history=[], temperature=0.0
                    )
                    generated_text = response
                else:
                    # Standard HuggingFace generation pipeline
                    inputs = self.processor(
                        text=prompt_text,
                        images=image,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=40,
                        do_sample=False,
                        temperature=0.0
                    )
                    
                    # Decode the generated tokens
                    generated_ids = output[0][inputs['input_ids'].shape[1]:]
                    generated_text = self.processor.decode(
                        generated_ids, 
                        skip_special_tokens=True
                    ).strip()
            
            # Extract genus from response
            genus = self._extract_genus(generated_text)
            return {"genus": genus}
            
        except Exception as e:
            return {"error": f"Classification failed: {e}"}
    
    def _extract_genus(self, response: str) -> str:
        """Extract genus name from model response.
        
        Args:
            response: Raw text response from the model.
            
        Returns:
            Extracted genus name or 'Unknown' if not found.
        """
        response = response.strip()
        
        # Direct match with genera list
        for genus in PRIMATE_GENERA:
            if genus.lower() in response.lower():
                return genus
        
        # If no match, return first word (often the genus)
        first_word = response.split()[0] if response else "Unknown"
        return first_word.capitalize()
    
    def classify_image(
        self, 
        image_path: Union[str, pathlib.Path]
    ) -> str:
        """Classify a single primate image.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Predicted primate genus name.
        """
        # Preprocess image
        image = self.preprocess_image(image_path)
        if isinstance(image, dict) and "error" in image:
            print(f"Error: {image['error']}")
            return "Unknown"
        
        # Classify
        result = self.classify_genus(image)
        if "error" in result:
            print(f"Error: {result['error']}")
            return "Unknown"
        
        return result["genus"]
    
    def process_directory(
        self, 
        directory_path: Union[str, pathlib.Path]
    ) -> Dict[str, str]:
        """Process all images in a directory.
        
        Args:
            directory_path: Path to directory containing images.
            
        Returns:
            Dictionary mapping image filenames to predicted genera.
        """
        dir_path = pathlib.Path(directory_path)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [
            f for f in dir_path.iterdir() 
            if f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            print(f"No images found in {directory_path}")
            return {}
        
        # Process each image
        results = {}
        for img_file in image_files:
            print(f"Processing: {img_file.name}")
            genus = self.classify_image(img_file)
            results[img_file.name] = genus
            print(f"  → Genus: {genus}")
        
        return results


def create_coco_output(
    classifications: Dict[str, str], 
    image_dir: pathlib.Path
) -> Dict[str, Any]:
    """Create COCO-formatted output from classifications.
    
    Args:
        classifications: Dictionary mapping image names to genera.
        image_dir: Path to the image directory.
        
    Returns:
        COCO-formatted dictionary with image annotations.
    """
    coco_output = {
        "info": {
            "description": "PrimateFace Genus Classification Results",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat()
        },
        "images": [],
        "annotations": [],
        "categories": [
            {"id": i, "name": genus, "supercategory": "primate"} 
            for i, genus in enumerate(PRIMATE_GENERA, 1)
        ]
    }
    
    # Create genus to category ID mapping
    genus_to_id = {genus: i for i, genus in enumerate(PRIMATE_GENERA, 1)}
    
    # Add images and annotations
    for img_id, (img_name, genus) in enumerate(classifications.items(), 1):
        # Add image entry
        img_path = image_dir / img_name
        if img_path.exists():
            img = Image.open(img_path)
            width, height = img.size
        else:
            width, height = 0, 0
        
        coco_output["images"].append({
            "id": img_id,
            "file_name": img_name,
            "width": width,
            "height": height
        })
        
        # Add annotation
        category_id = genus_to_id.get(genus, 0)
        coco_output["annotations"].append({
            "id": img_id,
            "image_id": img_id,
            "category_id": category_id,
            "genus": genus,
            "score": 1.0
        })
    
    return coco_output


def main() -> int:
    """Main CLI entry point.
    
    Usage:
        python classify_genus.py image.jpg [--model SmolVLM]
    """
    parser = argparse.ArgumentParser(
        description="Primate Genus Classifier using Vision Language Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'input',
        help='Path to image file or directory'
    )
    parser.add_argument(
        'output',
        nargs='?',
        help='Output JSON file (for directory processing)'
    )
    parser.add_argument(
        '--model',
        choices=['SmolVLM', 'InternVL2-2B'],
        default='SmolVLM',
        help='VLM model to use (default: SmolVLM)'
    )
    parser.add_argument(
        '--device',
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device for inference (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize classifier
        classifier = PrimateClassifierVLM(model_name=args.model)
        
        input_path = pathlib.Path(args.input)
        
        if input_path.is_file():
            # Single image processing
            genus = classifier.classify_image(input_path)
            print(f"\nClassification Result:")
            print(f"  Image: {input_path.name}")
            print(f"  Genus: {genus}")
            
            # Save if output specified
            if args.output:
                result = {
                    "image": str(input_path),
                    "genus": genus,
                    "model": args.model
                }
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\nResults saved to: {args.output}")
                
        elif input_path.is_dir():
            # Directory processing
            results = classifier.process_directory(input_path)
            
            if args.output:
                # Create COCO format output
                coco_output = create_coco_output(results, input_path)
                with open(args.output, 'w') as f:
                    json.dump(coco_output, f, indent=2)
                print(f"\nCOCO format results saved to: {args.output}")
            else:
                # Print summary
                print(f"\nProcessed {len(results)} images")
                print("Summary of classifications:")
                genus_counts = {}
                for genus in results.values():
                    genus_counts[genus] = genus_counts.get(genus, 0) + 1
                for genus, count in sorted(genus_counts.items()):
                    print(f"  {genus}: {count}")
        else:
            print(f"Error: {input_path} is neither a file nor directory")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()