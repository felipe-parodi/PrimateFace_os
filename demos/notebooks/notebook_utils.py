"""Shared utilities for PrimateFace tutorial notebooks.

Provides environment checking, device detection, model path resolution,
and demo asset downloading for local notebook execution.
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional


def check_environment(require_packages: Optional[List[str]] = None) -> Dict[str, str]:
    """Check local environment and return device info.

    Verifies Python version, PyTorch, CUDA availability, and optional
    package imports. Warns on missing GPU but never raises.

    Args:
        require_packages: Optional list of package names to verify
            (e.g., ["mmdet", "mmpose", "gazelle"]).

    Returns:
        Dict with keys: python, torch, cuda, device, gpu_name, and
        per-package availability.
    """
    info: Dict[str, str] = {}

    # Python
    info["python"] = sys.version.split()[0]

    # PyTorch + CUDA
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["device"] = "cuda:0"
            info["gpu_name"] = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            info["gpu_memory_gb"] = f"{gpu_mem:.1f}"
        else:
            info["device"] = "cpu"
            info["gpu_name"] = "N/A"
            warnings.warn(
                "CUDA not available. Running on CPU -- inference will be "
                "significantly slower. A GPU is strongly recommended.",
                stacklevel=2,
            )
    except ImportError:
        info["torch"] = "NOT INSTALLED"
        info["cuda"] = "False"
        info["device"] = "cpu"
        info["gpu_name"] = "N/A"
        print("ERROR: PyTorch not found. Install it first.")

    # Optional packages
    if require_packages:
        for pkg in require_packages:
            try:
                mod = __import__(pkg)
                info[pkg] = getattr(mod, "__version__", "installed")
            except ImportError:
                info[pkg] = "NOT INSTALLED"
                print(f"WARNING: {pkg} not found. See demos/README.md for install instructions.")

    # Print summary
    print("=" * 55)
    print("PrimateFace Environment")
    print("=" * 55)
    print(f"  Python:   {info['python']}")
    print(f"  PyTorch:  {info.get('torch', '?')}")
    print(f"  CUDA:     {info.get('cuda', '?')}")
    print(f"  Device:   {info.get('device', '?')}")
    if info.get("gpu_name") and info["gpu_name"] != "N/A":
        print(f"  GPU:      {info['gpu_name']} ({info.get('gpu_memory_gb', '?')} GB)")
    if require_packages:
        for pkg in require_packages:
            status = info.get(pkg, "?")
            marker = "OK" if status != "NOT INSTALLED" else "MISSING"
            print(f"  {pkg}: {status} [{marker}]")
    print("=" * 55)

    return info


def get_device(prefer_gpu: bool = True) -> str:
    """Return the best available device string.

    Args:
        prefer_gpu: If True, returns 'cuda:0' when available.

    Returns:
        Device string ('cuda:0' or 'cpu').
    """
    import torch

    if prefer_gpu and torch.cuda.is_available():
        return "cuda:0"
    if prefer_gpu and not torch.cuda.is_available():
        warnings.warn(
            "GPU requested but CUDA not available. Falling back to CPU.",
            stacklevel=2,
        )
    return "cpu"


def resolve_model_paths(
    model_dir: Path, required: List[str]
) -> Dict[str, Path]:
    """Validate that required model files exist and return their paths.

    Args:
        model_dir: Directory where models were downloaded.
        required: List of expected filenames (e.g., ["mmdet_config.py",
            "mmdet_checkpoint.pth"]).

    Returns:
        Dict mapping filename to its absolute Path.

    Raises:
        FileNotFoundError: If any required file is missing, with
            instructions to run download_models.py.
    """
    model_dir = Path(model_dir)
    paths: Dict[str, Path] = {}
    missing: List[str] = []

    for filename in required:
        fp = model_dir / filename
        if fp.exists():
            paths[filename] = fp
            size_mb = fp.stat().st_size / (1024 * 1024)
            print(f"  Found: {filename} ({size_mb:.1f} MB)")
        else:
            missing.append(filename)

    if missing:
        raise FileNotFoundError(
            f"Missing model files in {model_dir}:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\n\nRun:  python demos/download_models.py "
            + str(model_dir)
        )

    return paths


def download_demo_asset(
    gdrive_id: str, output_path: Path, description: str = ""
) -> Path:
    """Download a file from Google Drive if not already present.

    Args:
        gdrive_id: Google Drive file ID.
        output_path: Local path to save the file.
        description: Human-readable description for progress messages.

    Returns:
        Path to the downloaded (or already existing) file.
    """
    output_path = Path(output_path)
    if output_path.exists():
        print(f"  Already exists: {output_path.name}")
        return output_path

    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is required to download demo assets. "
            "Install with: uv pip install gdown"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    desc_str = f" ({description})" if description else ""
    print(f"  Downloading{desc_str}: {output_path.name} ...")
    url = f"https://drive.google.com/uc?id={gdrive_id}"
    gdown.download(url, str(output_path), quiet=False)
    return output_path
