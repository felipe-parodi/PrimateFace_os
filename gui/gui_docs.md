# GUI Tools - Technical Documentation

This document provides comprehensive technical documentation for the PrimateFace GUI tools, including pseudo-labeling interfaces, annotation refinement tools, and format converters.

## Module Architecture

```
gui/
├── pseudolabel_gui.py       # Main pseudo-labeling GUI
├── pseudolabel_gui_fm.py    # Feature-matching variant
├── refine_boxes.py          # Bbox refinement interface
├── core/                    # Core functionality
│   ├── detectors.py        # Detection models
│   ├── pose.py             # Pose estimation
│   ├── models.py           # Model management
│   └── sam.py              # SAM integration
├── converters/              # Format conversion
│   ├── base.py            # Base converter class
│   ├── image.py           # Image directory → COCO
│   └── video.py           # Video → COCO
└── constants.py            # GUI configuration

Additional scripts:
├── imgdir2coco_face68.py   # Convert images to COCO (68-point)
├── imgdir2coco_facedet.py  # Convert images to COCO (detection)
├── viddir2coco_facedet.py  # Convert videos to COCO
└── f_mmpose_pseudolabel.py # MMPose-specific labeling
```

## Core Components

### Main Pseudo-Labeling GUI

#### Class: `PseudoLabelGUI`

Main interface for interactive pseudo-labeling with detection and pose models.

```python
class PseudoLabelGUI:
    def __init__(
        self,
        img_dir: str,
        output_dir: str,
        det_config: str,
        det_checkpoint: str,
        pose_config: Optional[str] = None,
        pose_checkpoint: Optional[str] = None,
        review_stage: str = 'det',  # 'det', 'pose', or 'none'
        device: str = 'cuda:0',
        bbox_thr: float = 0.5,
        kpt_thr: float = 0.3
    )
```

**Review Stages:**
- `det`: Review and refine detection bounding boxes
- `pose`: Review and adjust pose keypoints
- `none`: Auto-generate without review

**Key Methods:**

##### `run_detection_review()`
Interactive bbox refinement interface.

```python
def run_detection_review(self):
    """Launch detection review GUI."""
    self.root = tk.Tk()
    self.root.title("Detection Review")
    
    # Canvas for image display
    self.canvas = tk.Canvas(self.root, width=800, height=600)
    
    # Controls
    self.create_detection_controls()
    
    # Keyboard bindings
    self.bind_detection_keys()
    
    self.root.mainloop()
```

##### `run_pose_review()`
Interactive keypoint adjustment interface.

```python
def run_pose_review(self):
    """Launch pose review GUI."""
    self.root = tk.Tk()
    self.root.title("Pose Review")
    
    # Canvas with keypoint overlays
    self.canvas = tk.Canvas(self.root, width=800, height=600)
    
    # Keypoint editing controls
    self.create_pose_controls()
    
    # Mouse interaction for keypoint adjustment
    self.canvas.bind("<Button-1>", self.on_click_keypoint)
    self.canvas.bind("<B1-Motion>", self.on_drag_keypoint)
    
    self.root.mainloop()
```

### Detection and Pose Models

#### `core/detectors.py`

Detection model management and inference.

```python
class DetectorManager:
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = 'cuda:0'
    ):
        self.model = init_detector(
            config_path,
            checkpoint_path,
            device=device
        )
    
    def detect(self, image: np.ndarray, threshold: float = 0.5):
        """Run detection on image."""
        results = inference_detector(self.model, image)
        
        # Filter by threshold
        bboxes = self.filter_detections(results, threshold)
        return bboxes
    
    def batch_detect(self, images: List[np.ndarray], threshold: float = 0.5):
        """Batch detection for efficiency."""
        all_results = []
        for img in tqdm(images, desc="Detecting"):
            results = self.detect(img, threshold)
            all_results.append(results)
        return all_results
```

#### `core/pose.py`

Pose estimation pipeline.

```python
class PoseEstimator:
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = 'cuda:0'
    ):
        self.model = init_pose_estimator(
            config_path,
            checkpoint_path,
            device=device
        )
    
    def estimate_pose(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        format: str = 'xyxy'
    ):
        """Estimate pose for given bboxes."""
        dataset_info = self.model.cfg.data['test'].get('dataset_info', None)
        
        results = inference_topdown(
            self.model,
            image,
            bboxes,
            bbox_format=format,
            dataset_info=dataset_info
        )
        
        return self.extract_keypoints(results)
```

### Format Converters

#### `converters/base.py`

Base converter class for all format conversions.

```python
class BaseConverter(ABC):
    """Abstract base class for format converters."""
    
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.annotations = []
        self.images = []
        self.annotation_id = 1
        self.image_id = 1
    
    @abstractmethod
    def convert(self, input_path: str) -> dict:
        """Convert input to COCO format."""
        pass
    
    def save_coco(self, coco_dict: dict):
        """Save COCO format JSON."""
        with open(self.output_path, 'w') as f:
            json.dump(coco_dict, f, indent=2)
    
    def create_coco_dict(self) -> dict:
        """Create standard COCO dictionary."""
        return {
            'images': self.images,
            'annotations': self.annotations,
            'categories': self.get_categories()
        }
```

#### `converters/image.py`

Image directory to COCO conversion.

```python
class ImageDirConverter(BaseConverter):
    """Convert image directory to COCO format."""
    
    def __init__(
        self,
        output_path: str,
        detector: Optional[DetectorManager] = None,
        pose_estimator: Optional[PoseEstimator] = None
    ):
        super().__init__(output_path)
        self.detector = detector
        self.pose_estimator = pose_estimator
    
    def convert(self, img_dir: str, extensions: List[str] = ['.jpg', '.png']):
        """Convert all images in directory."""
        image_paths = self.get_image_paths(img_dir, extensions)
        
        for img_path in tqdm(image_paths, desc="Processing"):
            # Load image
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            
            # Add image entry
            self.add_image(img_path, w, h)
            
            # Detect if detector provided
            if self.detector:
                bboxes = self.detector.detect(img)
                
                # Estimate pose if available
                if self.pose_estimator and len(bboxes) > 0:
                    keypoints = self.pose_estimator.estimate_pose(img, bboxes)
                    self.add_annotations(bboxes, keypoints)
                else:
                    self.add_annotations(bboxes, None)
        
        # Create and save COCO dict
        coco_dict = self.create_coco_dict()
        self.save_coco(coco_dict)
        
        return coco_dict
```

#### `converters/video.py`

Video to COCO conversion with frame extraction.

```python
class VideoConverter(BaseConverter):
    """Convert video to COCO format with frame extraction."""
    
    def __init__(
        self,
        output_path: str,
        frame_dir: str,
        sample_rate: int = 30
    ):
        super().__init__(output_path)
        self.frame_dir = frame_dir
        self.sample_rate = sample_rate
        os.makedirs(frame_dir, exist_ok=True)
    
    def extract_frames(self, video_path: str):
        """Extract frames from video."""
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        extracted = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % self.sample_rate == 0:
                frame_path = os.path.join(
                    self.frame_dir,
                    f"frame_{frame_count:06d}.jpg"
                )
                cv2.imwrite(frame_path, frame)
                extracted.append(frame_path)
            
            frame_count += 1
        
        cap.release()
        return extracted
```

### GUI Controls and Interactions

#### Bounding Box Adjustment

```python
class BBoxAdjuster:
    """Interactive bounding box adjustment."""
    
    def __init__(self, canvas: tk.Canvas):
        self.canvas = canvas
        self.current_bbox = None
        self.drag_data = {"x": 0, "y": 0, "item": None}
    
    def create_bbox_handles(self, bbox: List[float]):
        """Create draggable handles for bbox corners."""
        x1, y1, x2, y2 = bbox
        
        # Corner handles
        self.handles = {
            'tl': self.canvas.create_oval(x1-5, y1-5, x1+5, y1+5, fill="red"),
            'tr': self.canvas.create_oval(x2-5, y1-5, x2+5, y1+5, fill="red"),
            'bl': self.canvas.create_oval(x1-5, y2-5, x1+5, y2+5, fill="red"),
            'br': self.canvas.create_oval(x2-5, y2-5, x2+5, y2+5, fill="red")
        }
        
        # Bind drag events
        for handle in self.handles.values():
            self.canvas.tag_bind(handle, "<ButtonPress-1>", self.on_press)
            self.canvas.tag_bind(handle, "<B1-Motion>", self.on_drag)
            self.canvas.tag_bind(handle, "<ButtonRelease-1>", self.on_release)
    
    def on_drag(self, event):
        """Handle dragging of bbox corners."""
        dx = event.x - self.drag_data["x"]
        dy = event.y - self.drag_data["y"]
        
        self.canvas.move(self.drag_data["item"], dx, dy)
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y
        
        self.update_bbox()
```

#### Keypoint Editor

```python
class KeypointEditor:
    """Interactive keypoint position editor."""
    
    def __init__(self, canvas: tk.Canvas, skeleton: List[Tuple[int, int]]):
        self.canvas = canvas
        self.skeleton = skeleton
        self.keypoints = []
        self.selected_kpt = None
    
    def draw_keypoints(self, keypoints: np.ndarray):
        """Draw keypoints and skeleton."""
        self.clear_keypoints()
        
        # Draw skeleton connections
        for connection in self.skeleton:
            if self.is_valid_connection(keypoints, connection):
                self.draw_bone(keypoints, connection)
        
        # Draw keypoint circles
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0:
                color = self.get_keypoint_color(i, conf)
                kpt_id = self.canvas.create_oval(
                    x-5, y-5, x+5, y+5,
                    fill=color, outline="white", width=2
                )
                self.keypoints.append(kpt_id)
                
                # Make draggable
                self.canvas.tag_bind(kpt_id, "<Button-1>", 
                                   lambda e, idx=i: self.select_keypoint(idx))
    
    def move_keypoint(self, idx: int, new_x: float, new_y: float):
        """Move keypoint to new position."""
        if 0 <= idx < len(self.keypoints):
            kpt_id = self.keypoints[idx]
            old_coords = self.canvas.coords(kpt_id)
            dx = new_x - (old_coords[0] + old_coords[2]) / 2
            dy = new_y - (old_coords[1] + old_coords[3]) / 2
            self.canvas.move(kpt_id, dx, dy)
            
            # Update skeleton connections
            self.update_skeleton()
```

### Keyboard Shortcuts

```python
class KeyboardShortcuts:
    """Keyboard shortcuts for GUI controls."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.setup_shortcuts()
    
    def setup_shortcuts(self):
        """Bind keyboard shortcuts."""
        shortcuts = {
            '<space>': self.accept_current,
            '<Delete>': self.delete_current,
            '<Left>': self.previous_image,
            '<Right>': self.next_image,
            '<a>': self.add_bbox,
            '<d>': self.duplicate_bbox,
            '<r>': self.reset_current,
            '<s>': self.save_progress,
            '<Escape>': self.quit_gui,
            '<1>': lambda: self.set_confidence(0.1),
            '<2>': lambda: self.set_confidence(0.2),
            '<3>': lambda: self.set_confidence(0.3),
            '<4>': lambda: self.set_confidence(0.4),
            '<5>': lambda: self.set_confidence(0.5),
            '<6>': lambda: self.set_confidence(0.6),
            '<7>': lambda: self.set_confidence(0.7),
            '<8>': lambda: self.set_confidence(0.8),
            '<9>': lambda: self.set_confidence(0.9),
        }
        
        for key, handler in shortcuts.items():
            self.root.bind(key, lambda e, h=handler: h())
```

### Advanced Features

#### SAM Integration

```python
from segment_anything import SamPredictor, sam_model_registry

class SAMRefiner:
    """Segment Anything Model for bbox refinement."""
    
    def __init__(self, sam_checkpoint: str, model_type: str = 'vit_h'):
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device='cuda')
        self.predictor = SamPredictor(self.sam)
    
    def refine_bbox(self, image: np.ndarray, bbox: List[float]):
        """Refine bbox using SAM."""
        self.predictor.set_image(image)
        
        # Convert bbox to SAM format
        input_box = np.array(bbox)
        
        masks, scores, logits = self.predictor.predict(
            box=input_box,
            multimask_output=True
        )
        
        # Select best mask
        best_mask = masks[np.argmax(scores)]
        
        # Convert mask to refined bbox
        refined_bbox = self.mask_to_bbox(best_mask)
        
        return refined_bbox
    
    def mask_to_bbox(self, mask: np.ndarray):
        """Convert binary mask to bounding box."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        
        return [xmin, ymin, xmax, ymax]
```

#### Batch Processing

```python
class BatchProcessor:
    """Batch processing for large datasets."""
    
    def __init__(
        self,
        detector: DetectorManager,
        pose_estimator: Optional[PoseEstimator] = None,
        batch_size: int = 32
    ):
        self.detector = detector
        self.pose_estimator = pose_estimator
        self.batch_size = batch_size
    
    def process_directory(
        self,
        img_dir: str,
        output_path: str,
        review: bool = False
    ):
        """Process entire directory with optional review."""
        converter = ImageDirConverter(
            output_path,
            self.detector,
            self.pose_estimator
        )
        
        if review:
            # Launch review GUI
            gui = PseudoLabelGUI(
                img_dir=img_dir,
                output_dir=os.path.dirname(output_path),
                det_config=self.detector.config,
                det_checkpoint=self.detector.checkpoint,
                review_stage='det'
            )
            gui.run()
        else:
            # Auto-process without review
            converter.convert(img_dir)
```

### Configuration

#### `constants.py`

GUI configuration constants.

```python
# Display settings
CANVAS_WIDTH = 1024
CANVAS_HEIGHT = 768
DEFAULT_SCALE = 1.0
MIN_SCALE = 0.1
MAX_SCALE = 5.0

# Detection settings
DEFAULT_BBOX_THR = 0.5
DEFAULT_NMS_THR = 0.3
MAX_DETECTIONS_PER_IMAGE = 100

# Pose settings
DEFAULT_KPT_THR = 0.3
NUM_KEYPOINTS = 68  # or 49 for simplified
SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3),  # Face contour
    # ... complete skeleton
]

# Colors (BGR for OpenCV, RGB for tkinter)
BBOX_COLOR = (0, 255, 0)  # Green
KEYPOINT_COLORS = [
    (255, 0, 0),    # Red for face contour
    (0, 255, 0),    # Green for eyes
    (0, 0, 255),    # Blue for nose
    # ... color scheme
]

# File formats
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp']
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv']

# GUI settings
BUTTON_WIDTH = 15
ENTRY_WIDTH = 30
FONT_SIZE = 10
FONT_FAMILY = "Arial"
```

## Usage Examples

### Basic Pseudo-Labeling

```python
from gui import PseudoLabelGUI

# Initialize GUI with models
gui = PseudoLabelGUI(
    img_dir="dataset/images",
    output_dir="dataset/annotations",
    det_config="configs/cascade_rcnn.py",
    det_checkpoint="models/cascade_rcnn.pth",
    pose_config="configs/hrnet.py",
    pose_checkpoint="models/hrnet.pth",
    review_stage='det'  # Start with detection review
)

# Run GUI
gui.run()
```

### Programmatic Batch Processing

```python
from gui.core import DetectorManager, PoseEstimator
from gui.converters import ImageDirConverter

# Initialize models
detector = DetectorManager(
    "configs/detection.py",
    "models/detection.pth"
)

pose_estimator = PoseEstimator(
    "configs/pose.py",
    "models/pose.pth"
)

# Convert directory to COCO
converter = ImageDirConverter(
    "annotations.json",
    detector,
    pose_estimator
)

coco_dict = converter.convert("images/")
print(f"Processed {len(coco_dict['images'])} images")
```

### Custom Review Pipeline

```python
class CustomReviewPipeline:
    """Custom review pipeline with multiple stages."""
    
    def __init__(self, config):
        self.config = config
        self.stages = []
    
    def add_stage(self, stage_name: str, reviewer):
        """Add review stage."""
        self.stages.append((stage_name, reviewer))
    
    def run(self):
        """Run all review stages."""
        for stage_name, reviewer in self.stages:
            print(f"Running {stage_name}...")
            reviewer.run()
            
            # Check if user wants to continue
            if not self.confirm_continue():
                break
    
    def confirm_continue(self):
        """Ask user to continue to next stage."""
        response = messagebox.askyesno(
            "Continue?",
            "Proceed to next review stage?"
        )
        return response

# Usage
pipeline = CustomReviewPipeline(config)
pipeline.add_stage("Detection", detection_reviewer)
pipeline.add_stage("Pose", pose_reviewer)
pipeline.add_stage("Quality Check", quality_reviewer)
pipeline.run()
```

## Testing

### Unit Tests

```python
import unittest
from gui.core import DetectorManager

class TestDetector(unittest.TestCase):
    def setUp(self):
        self.detector = DetectorManager(
            "test_config.py",
            "test_model.pth",
            device="cpu"
        )
    
    def test_detection(self):
        # Create test image
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Run detection
        results = self.detector.detect(test_img)
        
        # Check output format
        self.assertIsInstance(results, list)
        if len(results) > 0:
            self.assertEqual(len(results[0]), 5)  # x1, y1, x2, y2, score
```

## Troubleshooting

### Common Issues

#### GUI Not Responding
```python
# Problem: GUI freezes during processing
# Solution: Use threading for heavy operations

import threading

def process_in_background(self):
    thread = threading.Thread(target=self.heavy_processing)
    thread.daemon = True
    thread.start()
```

#### Memory Issues with Large Images
```python
# Problem: Out of memory with high-res images
# Solution: Resize for display, process at full resolution

def load_image_for_display(self, img_path, max_size=1024):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h))
    
    return img
```

## Performance Optimization

### GPU Batch Processing

```python
def batch_inference(self, images, batch_size=8):
    """Batch inference for better GPU utilization."""
    results = []
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        batch_results = self.model.batch_inference(batch)
        results.extend(batch_results)
    
    return results
```

### Caching Results

```python
class ResultCache:
    """Cache detection/pose results."""
    
    def __init__(self, cache_dir="cache/"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_path(self, img_path, model_name):
        img_hash = hashlib.md5(img_path.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{img_hash}_{model_name}.pkl")
    
    def load_cached(self, img_path, model_name):
        cache_path = self.get_cache_path(img_path, model_name)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def save_cache(self, img_path, model_name, results):
        cache_path = self.get_cache_path(img_path, model_name)
        with open(cache_path, 'wb') as f:
            pickle.dump(results, f)
```

## See Also

- [Main README](README.md) - Quick start guide
- [API Documentation](../docs/api/gui.md) - API overview
- [Pseudo-labeling Guide](../docs/guides/pseudo-labeling.md) - Step-by-step workflow
- [MMPose Documentation](https://mmpose.readthedocs.io/) - Pose estimation framework