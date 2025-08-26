# Troubleshooting

Common issues and solutions for PrimateFace.

## Installation Issues

### CUDA/GPU Problems

#### Error: "CUDA out of memory"
**Solution:**
```python
# Reduce batch size
processor = PrimateFaceProcessor(batch_size=1)

# Clear GPU cache
import torch
torch.cuda.empty_cache()

# Use CPU instead
processor = PrimateFaceProcessor(device='cpu')
```

#### Error: "No CUDA GPUs are available"
**Solution:**
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Install correct PyTorch version
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
```

### Dependency Conflicts

#### Error: "Module 'cv2' not found"
**Solution:**
```bash
pip install opencv-python opencv-contrib-python
```

#### Error: "ImportError: cannot import name 'xxx' from 'mmpose'"
**Solution:**
```bash
# Install specific MMPose version
pip install mmpose==0.29.0 mmdet==2.25.0
```

## Model Loading Issues

### Missing Model Files

#### Error: "Model checkpoint not found"
**Solution:**
```bash
# Download pretrained models
python demos/download_models.py

# Or specify custom path
processor = PrimateFaceProcessor(
    det_checkpoint="path/to/your/model.pth"
)
```

### Config Mismatch

#### Error: "Config type mismatch"
**Solution:**
```python
# Ensure config matches checkpoint
from demos.constants import DEFAULT_DET_CONFIG, DEFAULT_POSE_CONFIG

processor = PrimateFaceProcessor(
    det_config=DEFAULT_DET_CONFIG,
    pose_config=DEFAULT_POSE_CONFIG
)
```

## Data Format Issues

### COCO Format Errors

#### Error: "Invalid COCO format"
**Solution:**
```python
# Validate COCO file
from evals.core.utils import validate_coco

issues = validate_coco("annotations.json")
for issue in issues:
    print(issue)

# Fix common issues
from evals.core.utils import fix_coco_format
fixed_data = fix_coco_format("annotations.json")
```

### Image Loading Problems

#### Error: "Cannot read image file"
**Solution:**
```python
# Check image paths
import os
from pathlib import Path

# Make paths absolute
image_dir = Path("images").absolute()

# Verify images exist
for img in image_list:
    if not os.path.exists(img):
        print(f"Missing: {img}")
```

## GUI Issues

### Tkinter Problems

#### Error: "No module named '_tkinter'"
**Solution:**
```bash
# Linux
sudo apt-get install python3-tk

# macOS
brew install python-tk

# Windows - reinstall Python with tcl/tk
```

### Display Issues

#### Error: "Cannot connect to X server"
**Solution:**
```bash
# SSH with X11 forwarding
ssh -X user@server

# Or use virtual display
export DISPLAY=:0

# Or run without GUI
python process_batch.py --no-gui
```

## Performance Issues

### Slow Inference

**Solutions:**
1. **Enable GPU**:
```python
processor = PrimateFaceProcessor(device='cuda:0')
```

2. **Batch processing**:
```python
results = processor.process_batch(images, batch_size=8)
```

3. **Reduce input size**:
```python
processor = PrimateFaceProcessor(input_size=(640, 480))
```

### Memory Leaks

**Solution:**
```python
# Clear cache periodically
import gc
import torch

for batch in data_loader:
    results = process(batch)
    
    # Clear every 100 batches
    if batch_idx % 100 == 0:
        torch.cuda.empty_cache()
        gc.collect()
```

## Framework-Specific Issues

### MMPose/MMDetection

#### Error: "Registry not found"
**Solution:**
```python
# Import required registries
from mmdet.apis import init_detector
from mmpose.apis import init_pose_model
```

### DeepLabCut

#### Error: "Project config not found"
**Solution:**
```bash
# Initialize project properly
import deeplabcut
deeplabcut.create_new_project('project', 'author')
```

### SLEAP

#### Error: "Cannot load SLEAP model"
**Solution:**
```python
# Use correct model path structure
model_path = "models/sleap_model.zip"
sleap.load_model(model_path)
```

## Common Runtime Errors

### KeyError: 'keypoints'

**Solution:**
```python
# Check annotation structure
if 'keypoints' not in annotation:
    annotation['keypoints'] = [0] * (num_keypoints * 3)
```

### ValueError: "too many values to unpack"

**Solution:**
```python
# Check return value count
# Wrong:
x, y = function_that_returns_three_values()

# Correct:
x, y, z = function_that_returns_three_values()
# Or:
result = function_that_returns_three_values()
x, y = result[:2]
```

### IndexError: "list index out of range"

**Solution:**
```python
# Validate indices
if idx < len(keypoints):
    point = keypoints[idx]
else:
    point = default_value
```

## Data Quality Issues

### Poor Detection Results

**Checklist:**
- [ ] Image quality sufficient (>640x480)
- [ ] Face clearly visible
- [ ] Proper lighting
- [ ] Minimal motion blur

**Solutions:**
```python
# Preprocess images
from demos.utils import enhance_image
enhanced = enhance_image(img, contrast=1.2, brightness=1.1)

# Adjust detection threshold
processor = PrimateFaceProcessor(det_threshold=0.3)
```

### Inaccurate Landmarks

**Solutions:**
```python
# Use higher resolution
processor = PrimateFaceProcessor(pose_input_size=(384, 288))

# Apply smoothing for video
from demos.smooth_utils import smooth_trajectory
smoothed = smooth_trajectory(keypoints, window_size=5)
```

## Getting Help

### Before Asking for Help

1. **Check documentation**
   - [User Guide](user-guide/index.md)
   - [API Reference](api/index.md)
   - [Tutorials](tutorials/index.md)

2. **Search existing issues**
   - [GitHub Issues](https://github.com/KordingLab/PrimateFace/issues)

3. **Gather information**
   ```python
   import sys
   import torch
   print(f"Python: {sys.version}")
   print(f"PyTorch: {torch.__version__}")
   print(f"CUDA: {torch.cuda.is_available()}")
   ```

### Reporting Issues

Include:
- Error message (full traceback)
- Code snippet
- Environment details
- Sample data (if possible)

### Community Support

- GitHub Discussions: [Ask questions](https://github.com/KordingLab/PrimateFace/discussions)
- Email: primateface@gmail.com

## See Also

- [Installation Guide](installation.md)
- [FAQ](faq.md)
- [Getting Started](getting-started/index.md)