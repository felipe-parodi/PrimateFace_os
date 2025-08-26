# Quickstart

Get PrimateFace running in 5 minutes! This guide covers the essential steps to start detecting and analyzing primate faces.

## Prerequisites

- Python 3.8+
- Conda (recommended) or pip
- 10GB free disk space
- GPU (optional but recommended)

## Step 1: Installation (2 minutes)

### Option A: Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/KordingLab/PrimateFace.git
cd PrimateFace

# Create environment
conda env create -f environment.yml
conda activate primateface

# Install package
pip install -e .
```

### Option B: Pip Only

```bash
# Clone and install
git clone https://github.com/KordingLab/PrimateFace.git
cd PrimateFace
pip install -e .
```

## Step 2: Download Models (1 minute)

```bash
# Download pretrained models
python demos/download_models.py
```

This downloads:
- Face detection model (Cascade R-CNN)
- Pose estimation model (HRNet-W32)
- Configuration files

## Step 3: Run Your First Detection (1 minute)

### Single Image

```bash
# Process a single image
python demos/primateface_demo.py process \
    --input samples/macaque.jpg \
    --det-config demos/mmdet_config.py \
    --det-checkpoint demos/mmdet_checkpoint.pth \
    --pose-config demos/mmpose_config.py \
    --pose-checkpoint demos/mmpose_checkpoint.pth \
    --save-viz
```

### Python API

```python
from demos.process import PrimateFaceProcessor

# Initialize processor
processor = PrimateFaceProcessor(
    det_config="demos/mmdet_config.py",
    det_checkpoint="demos/mmdet_checkpoint.pth",
    pose_config="demos/mmpose_config.py",
    pose_checkpoint="demos/mmpose_checkpoint.pth"
)

# Process image
result = processor.process("samples/macaque.jpg")

# Access results
for face in result["detections"]:
    bbox = face["bbox"]  # [x1, y1, x2, y2]
    keypoints = face["keypoints"]  # [[x, y, conf], ...]
    print(f"Face detected at {bbox}")
    print(f"Found {len(keypoints)} landmarks")
```

## Step 4: Visualize Results (30 seconds)

The processed image with annotations is saved as `*_result.jpg`:

```python
import matplotlib.pyplot as plt
from PIL import Image

# Load and display result
img = Image.open("samples/macaque_result.jpg")
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis('off')
plt.title("PrimateFace Detection Results")
plt.show()
```

## Step 5: Process Multiple Images (1 minute)

### Batch Processing

```bash
# Process entire directory
python demos/primateface_demo.py process-dir \
    --input-dir samples/ \
    --output-dir results/ \
    --det-config demos/mmdet_config.py \
    --det-checkpoint demos/mmdet_checkpoint.pth
```

### Python Batch API

```python
# Process multiple images
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = processor.process_batch(images)

# Save as COCO format
processor.save_coco(results, "annotations.json")
```

## What's Next?

### Explore Different Workflows

Now that you have PrimateFace running, explore based on your needs:

#### 🎯 **Detection Only**
```python
# Just detect faces, no landmarks
processor = PrimateFaceProcessor(
    det_config="demos/mmdet_config.py",
    det_checkpoint="demos/mmdet_checkpoint.pth"
    # No pose config/checkpoint
)
```

#### 📍 **Landmarks Only**
```python
# If you already have face bounding boxes
from demos.process import PoseEstimator

estimator = PoseEstimator(
    config="demos/mmpose_config.py",
    checkpoint="demos/mmpose_checkpoint.pth"
)
keypoints = estimator.estimate(image, bbox)
```

#### 🎥 **Video Processing**
```python
# Process video files
result = processor.process_video(
    "video.mp4",
    output_path="output.mp4",
    show_progress=True
)
```

### Try Interactive Tools

#### GUI Annotation
```bash
# Launch annotation interface
python gui/pseudolabel_gui_fm.py --img-dir samples/
```

#### Feature Visualization
```bash
# Extract and visualize DINOv2 features
python dinov2/dinov2_cli.py extract --input samples/
```

### Learn More

- **[Tutorials](../tutorials/index.md)** - Hands-on notebooks
- **[User Guide](../user-guide/index.md)** - Detailed workflows
- **[API Reference](../api/index.md)** - Full documentation
- **[Decision Tree](decision-tree.md)** - Choose your workflow

## Common Issues

### CUDA Not Available
```python
# Use CPU instead
processor = PrimateFaceProcessor(
    det_config="demos/mmdet_config.py",
    det_checkpoint="demos/mmdet_checkpoint.pth",
    device="cpu"  # Force CPU
)
```

### Out of Memory
```python
# Reduce batch size
processor.process_batch(images, batch_size=1)
```

### Model Not Found
```bash
# Re-download models
python demos/download_models.py --force
```

See [Troubleshooting](../troubleshooting.md) for more solutions.

## Success!

You've successfully:
- ✅ Installed PrimateFace
- ✅ Downloaded pretrained models
- ✅ Detected faces and landmarks
- ✅ Visualized results
- ✅ Processed multiple images

**Ready for more?** Check out the [Decision Tree](decision-tree.md) to find the perfect workflow for your research!