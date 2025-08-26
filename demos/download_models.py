#!/usr/bin/env python
"""
Download PrimateFace models from Google Drive.
Models include detection (MMDetection) and pose estimation (MMPose) checkpoints.
"""

import os
import sys
import argparse
from pathlib import Path

try:
    import gdown
except ImportError:
    print("Installing gdown...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
    import gdown


def download_models(output_dir="."):
    """Download all PrimateFace models to the specified directory."""
    
    # Model IDs from Google Drive
    models = {
        "mmdet_config.py": "1Y_YFdIDRcWQLI-gRiCnOrDxCptzCiiNp",
        "mmdet_checkpoint.pth": "1zZ8S31zPHX5BWYKbnHxI1QOqP-fPnVFO",
        "mmpose_config.py": "1sG2lLybRkLwmC0IkEqtEGuT1OxwXomju",
        "mmpose_checkpoint.pth": "1Oa18Ty90bNE8fud0cuK3gZmPY_LAQo3Y",
    }
    
    # Create output directory if needed
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading models to: {output_path.absolute()}")
    print()
    
    # Download each model
    for filename, file_id in models.items():
        output_file = output_path / filename
        model_type = "MMDetection" if "mmdet" in filename else "MMPose"
        file_type = "config" if filename.endswith(".py") else "checkpoint"
        
        print(f"Downloading {model_type} {file_type}...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(output_file), quiet=False)
        print()
    
    # List downloaded files
    print("✓ All models downloaded successfully!")
    print("\nFiles downloaded:")
    for filename in models.keys():
        file_path = output_path / filename
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  - {filename} ({size_mb:.1f} MB)")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Download PrimateFace models from Google Drive")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=".",
        help="Directory to save models (default: current directory)"
    )
    
    args = parser.parse_args()
    
    try:
        download_models(args.output_dir)
    except Exception as e:
        print(f"Error downloading models: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())