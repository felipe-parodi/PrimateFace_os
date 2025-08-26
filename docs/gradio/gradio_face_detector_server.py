import gradio as gr
import cv2
from ultralytics import YOLO
from pathlib import Path
import time
import tempfile
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm
import shutil # For checking ffmpeg
import torch # Added for MMDetection
from typing import Tuple, Dict, Any, List # Added Tuple for type hinting compatibility

# --- MMDetection Imports ---
try:
    from mmdet.apis import inference_detector, init_detector
    from mmengine.structures import InstanceData # For creating empty results
    from mmdet.structures import DetDataSample
    from mmpose.utils import adapt_mmdet_pipeline # If needed for your MMDetection model
    from mmpose.evaluation.functional import nms # For NMS
    MMDET_AVAILABLE = True
except ImportError as e:
    print(f"Warning: MMDetection libraries not found or import error: {e}. MMDetection model type will be disabled.")
    MMDET_AVAILABLE = False
    # Define dummy classes if MMDetection is not available to prevent NameErrors later
    class DetDataSample: pass 
    class InstanceData: pass

# --- MMPose Imports ---
try:
    from mmpose.apis import inference_topdown, init_model as init_mmpose_model
    from mmpose.structures import PoseDataSample, merge_data_samples as merge_mmpose_data_samples
    from mmpose.visualization import PoseLocalVisualizer
    MMPOSE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: MMPose libraries not found or import error: {e}. Face Pose Estimation task will be disabled.")
    MMPOSE_AVAILABLE = False
    # Define dummy classes if MMPose is not available
    class PoseDataSample: pass
    class PoseLocalVisualizer:
        def __init__(self, *args, **kwargs): pass
        def set_image(self, *args, **kwargs): pass
        def draw_instance_predictions(self, *args, **kwargs): pass
        def get_image(self, *args, **kwargs): return None
        def draw_bboxes(self, *args, **kwargs): pass
        dataset_meta = {} # Dummy
    def init_mmpose_model(*args, **kwargs): 
        print("Error: init_mmpose_model called but MMPose not available.")
        return None
    def inference_topdown(*args, **kwargs): 
        print("Error: inference_topdown called but MMPose not available.")
        return []
    def merge_mmpose_data_samples(*args, **kwargs): 
        print("Error: merge_mmpose_data_samples called but MMPose not available.")
        return None

# --- Gazelle Imports ---
try:
    from gazelle.model import get_gazelle_model
    GAZELLE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Gazelle libraries not found or import error: {e}. Gaze Estimation task will be disabled.")
    GAZELLE_AVAILABLE = False
    def get_gazelle_model(*args, **kwargs): # Dummy function
        print("Error: get_gazelle_model called but Gazelle not available.")
        return None, None


# --- Configuration ---
MODEL_PATH_YOLO = r"A:\\NonEnclosureProjects\\inprep\\PrimateFace\\data\\seb_faceid\\yolo_face_detection_workspace\\training_runs\\weights\\best.pt"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.4
MAX_DETECTIONS = 2 # Default for YOLO, MMDetection can be controlled separately
VIDEO_DURATION_LIMIT_SECONDS = 30

# MMDET_CONFIG_PATH = r"A:\\NonEnclosureProjects\\inprep\\PrimateFace\\results\\pf_cascrcnn_4k_facedet_250427\\primateface_cascade-rcnn_r101_fpn_1x_coco.py"
# MMDET_CHECKPOINT_PATH = r"A:\\NonEnclosureProjects\\inprep\\PrimateFace\\results\\pf_cascrcnn_4k_facedet_250427\\best_coco_bbox_mAP_epoch_12.pth"
MMDET_CONFIG_PATH = r"A:\NonEnclosureProjects\inprep\PrimateFace\results\casc-rcnn_FaceDet_12k_250602\primateface_cascade-rcnn_r101_fpn_1x_coco.py"
MMDET_CHECKPOINT_PATH = r"A:\NonEnclosureProjects\inprep\PrimateFace\results\casc-rcnn_FaceDet_12k_250602\best_coco_bbox_mAP_epoch_12.pth"

MMDET_CONFIDENCE_THRESHOLD = 0.7
MMDET_NMS_THRESHOLD = 0.3   
MMDET_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MMDET_CLASS_NAMES = ['face'] 
MMDET_COLOR_MAP = {'face': (255, 0, 0)} 
DEFAULT_COLOR_MMDET = (128, 128, 128)

# MMPose Configuration
# MMPOSE_CONFIG_PATH = r"A:\NonEnclosureProjects\inprep\PrimateFace\results\hrnet-dark_rex_1k_68kpt_250510\td-hm_hrnetv2-w18_dark-8xb32-60e_coco-wholebody-face-256x256.py"
# MMPOSE_CHECKPOINT_PATH = r"A:\NonEnclosureProjects\inprep\PrimateFace\results\hrnet-dark_rex_1k_68kpt_250510\best_NME_epoch_58.pth"
# MMPOSE_CONFIG_PATH = r"A:\NonEnclosureProjectsinprepPrimateFaceresultshrnet-dark_rex_1k_68kpt_250527\primateface_68_td-hm_hrnetv2-w18_dark-8xb32-60e_coco-wholebody-face-256x256.py"
# MMPOSE_CHECKPOINT_PATH = r"A:\NonEnclosureProjectsinprepPrimateFaceresultshrnet-dark_rex_1k_68kpt_250527\best_NME_epoch_60.pth"
# MMPOSE_CONFIG_PATH = r"A:\NonEnclosureProjects\inprep\PrimateFace\results\hrnet-dark_rex_4.5k_68kpt_250528\primateface_68_td-hm_hrnetv2-w18_dark-8xb32-60e_coco-wholebody-face-256x256.py"
# MMPOSE_CHECKPOINT_PATH = r"A:\NonEnclosureProjects\inprep\PrimateFace\results\hrnet-dark_rex_4.5k_68kpt_250528\best_NME_epoch_60.pth"
MMPOSE_CONFIG_PATH = r"A:\NonEnclosureProjects\inprep\PrimateFace\results\hrnet-dark_rex_5k_68kpt_250601\primateface_68_td-hm_hrnetv2-w18_dark-8xb32-60e_coco-wholebody-face-256x256.py"
MMPOSE_CHECKPOINT_PATH = r"A:\NonEnclosureProjects\inprep\PrimateFace\results\hrnet-dark_rex_5k_68kpt_250601\best_NME_epoch_156.pth"


MMPOSE_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MMPOSE_VIS_KPT_RADIUS = 3
MMPOSE_VIS_LINE_THICKNESS = 1
MMPOSE_VIS_BBOX_COLOR = (0, 255, 0) # Green for detection bboxes

# Gazelle Configuration
# GAZELLE_MODEL_NAME = "gazelle_dinov2_vitl14_inout" 
# GAZELLE_CHECKPOINT_PATH = r"A:\\NonEnclosureProjects\\inprep\\PrimateFace\\ext_models\\gazelle_dinov2_vitl14_inout.pt"
GAZELLE_MODEL_NAME = "gazelle_dinov2_vitl14"
GAZELLE_CHECKPOINT_PATH = r"A:\\NonEnclosureProjects\\inprep\\PrimateFace\\ext_models\\gazelle_dinov2_vitl14.pt"
GAZELLE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GAZELLE_INOUT_THRESHOLD = 0.3 # Default threshold for drawing gaze vector

# Global model variables
yolo_model_instance = None
mmdet_model_instance = None
mmpose_model_instance = None
gazelle_model_instance = None
gazelle_transform_instance = None
# pose_visualizer_instance = None # Initialize per call due to changing image

def check_ffmpeg():
    """Checks if ffmpeg is in PATH."""
    return shutil.which("ffmpeg") is not None

def load_yolo_model():
    global yolo_model_instance
    if yolo_model_instance is None:
        model_file = Path(MODEL_PATH_YOLO)
        if not model_file.exists():
            raise FileNotFoundError(f"YOLO model not found at {MODEL_PATH_YOLO}. Please update MODEL_PATH_YOLO.")
        yolo_model_instance = YOLO(model_file)
        print(f"Successfully loaded YOLO model from {model_file}")

def load_mmdet_model():
    global mmdet_model_instance
    if not MMDET_AVAILABLE:
        raise RuntimeError("MMDetection libraries are not available. Cannot load MMDetection model.")
    if mmdet_model_instance is None:
        config_file = Path(MMDET_CONFIG_PATH)
        checkpoint_file = Path(MMDET_CHECKPOINT_PATH)
        if not config_file.exists(): raise FileNotFoundError(f"MMDetection config not found: {MMDET_CONFIG_PATH}")
        if not checkpoint_file.exists(): raise FileNotFoundError(f"MMDetection checkpoint not found: {MMDET_CHECKPOINT_PATH}")
        
        mmdet_model_instance = init_detector(str(config_file), str(checkpoint_file), device=MMDET_DEVICE)
        # Attempt to adapt pipeline, common for some MMDetection versions/models
        # This might be specific to models trained with older MMPose/MMDetection
        # or if the config expects a different pipeline structure at inference.
        # For newer MMPose (1.x+), direct usage of MMDetection models is often smoother.
        try:
            if hasattr(mmdet_model_instance, 'cfg') and hasattr(mmdet_model_instance.cfg, 'data_preprocessor'):
                 # If using MMPose v1.x style data preprocessor adaptation:
                 # from mmpose.models.utils.tta import MMSelfTesting
                 # if 'test_cfg' in mmdet_model_instance.cfg and 'rcnn' not in mmdet_model_instance.cfg.model.type.lower(): # Example condition
                 #    mmdet_model_instance = MMSelfTesting(mmdet_model_instance) # Wrap model for TTA if applicable
                 pass # Modern MMDetection models usually don't need adapt_mmdet_pipeline for basic inference
            elif hasattr(mmdet_model_instance, 'cfg'): # Fallback for older style or if adapt_mmdet_pipeline is indeed necessary
                 mmdet_model_instance.cfg = adapt_mmdet_pipeline(mmdet_model_instance.cfg)
                 print("MMDetection model config pipeline adapted using adapt_mmdet_pipeline.")
            else:
                 print("MMDetection model instance or its config not structured as expected for pipeline adaptation check.")

        except Exception as adapt_e:
            print(f"Note: Could not adapt MMDetection pipeline or an issue during check (might be normal): {adapt_e}")
        print(f"Loaded MMDetection model from {checkpoint_file} on {MMDET_DEVICE}")

def load_mmpose_model():
    global mmpose_model_instance
    if not MMPOSE_AVAILABLE:
        raise RuntimeError("MMPose libraries are not available. Cannot load MMPose model.")
    if mmpose_model_instance is None:
        config_file = Path(MMPOSE_CONFIG_PATH)
        checkpoint_file = Path(MMPOSE_CHECKPOINT_PATH)
        if not config_file.exists():
            raise FileNotFoundError(f"MMPose config not found: {MMPOSE_CONFIG_PATH}")
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"MMPose checkpoint not found: {MMPOSE_CHECKPOINT_PATH}")
        
        try:
            # Disable heatmap output during model initialization if not needed for inference
            cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=False)))
            mmpose_model_instance = init_mmpose_model(
                str(config_file), 
                str(checkpoint_file), 
                device=MMPOSE_DEVICE,
                cfg_options=cfg_options
            )
            print(f"Loaded MMPose model from {checkpoint_file} on {MMPOSE_DEVICE}")
        except Exception as e:
            print(f"Error initializing MMPose model: {e}")
            mmpose_model_instance = None # Ensure it's None if loading failed
            raise  # Re-raise the exception to be caught by the caller

def load_gazelle_model():
    global gazelle_model_instance, gazelle_transform_instance
    if not GAZELLE_AVAILABLE:
        # This check is more for direct calls; process_media should also check.
        print("Error: Gazelle libraries are not available. Cannot load Gazelle model.")
        return # Or raise error if this function is called when GAZELLE_AVAILABLE is False

    if gazelle_model_instance is None or gazelle_transform_instance is None:
        if not Path(GAZELLE_CHECKPOINT_PATH).exists():
            raise FileNotFoundError(f"Gazelle checkpoint not found: {GAZELLE_CHECKPOINT_PATH}. Please ensure the path is correct.")
        
        model, transform = get_gazelle_model(GAZELLE_MODEL_NAME)
        if model is None or transform is None:
             raise RuntimeError(f"Failed to get Gazelle model structure for '{GAZELLE_MODEL_NAME}'. Check Gazelle installation.")
        
        # Load state dict with map_location to handle model loading across devices
        try:
            model.load_gazelle_state_dict(torch.load(GAZELLE_CHECKPOINT_PATH, map_location=torch.device(GAZELLE_DEVICE), weights_only=True))
        except Exception as e:
            raise RuntimeError(f"Error loading Gazelle state dict from {GAZELLE_CHECKPOINT_PATH}: {e}")

        model.eval()
        model.to(GAZELLE_DEVICE)
        
        gazelle_model_instance = model
        gazelle_transform_instance = transform
        print(f"Loaded Gazelle model '{GAZELLE_MODEL_NAME}' from {GAZELLE_CHECKPOINT_PATH} on {GAZELLE_DEVICE}")
    # else:
    #     print("Gazelle model and transform already loaded.")

def draw_mmdet_boxes_on_image(image_np_bgr: np.ndarray, detections_processed, conf_threshold: float, max_dets: int) -> np.ndarray:
    if not MMDET_AVAILABLE: return image_np_bgr
    img_copy = image_np_bgr.copy()
    height, width, _ = img_copy.shape
    drawn_detections = 0

    # Ensure detections_processed is not None
    if detections_processed is None:
        print("Warning: detections_processed is None in draw_mmdet_boxes_on_image.")
        return img_copy

    if isinstance(detections_processed, DetDataSample):
        pred_instances = detections_processed.pred_instances
        if len(pred_instances) == 0: return img_copy

        bboxes = pred_instances.bboxes.cpu().numpy()
        scores = pred_instances.scores.cpu().numpy()
        labels_idx = pred_instances.labels.cpu().numpy()
        
        sorted_indices = np.argsort(scores)[::-1]

        for i in sorted_indices:
            if drawn_detections >= max_dets: break
            score = scores[i]
            # conf_threshold is already applied before calling this for DetDataSample usually
            # but good to have if called directly with unpruned data
            if score < conf_threshold: continue 

            box = bboxes[i].astype(int)
            label_index = labels_idx[i]
            label = MMDET_CLASS_NAMES[label_index] if label_index < len(MMDET_CLASS_NAMES) else "Unknown"
            color = MMDET_COLOR_MAP.get(label, DEFAULT_COLOR_MMDET)

            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2)
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            label_text = f"{label}: {score:.2f}"
            (tw, th), bl = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            l_y = y1 - 10 if y1 > 20 else y1 + th + 5
            bg_y1, bg_y2 = max(0, l_y - th - bl), min(height -1, l_y + bl // 2)
            bg_x1, bg_x2 = max(0, x1), min(width -1, x1 + tw)
            cv2.rectangle(img_copy, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
            cv2.putText(img_copy, label_text, (x1, l_y - bl // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            drawn_detections += 1

    elif isinstance(detections_processed, list) and all(isinstance(arr, np.ndarray) for arr in detections_processed):
        all_dets_for_sorting = []
        for class_idx, class_dets in enumerate(detections_processed):
            if class_idx >= len(MMDET_CLASS_NAMES): continue
            if isinstance(class_dets, np.ndarray) and class_dets.ndim == 2 and class_dets.shape[1] == 5:
                for box_info in class_dets: # [x1, y1, x2, y2, score]
                    # conf_threshold already applied before this function for this type
                    all_dets_for_sorting.append((box_info[4], class_idx, box_info))
        
        all_dets_for_sorting.sort(key=lambda x: x[0], reverse=True)

        for score, class_idx, box_info in all_dets_for_sorting:
            if drawn_detections >= max_dets: break
            label = MMDET_CLASS_NAMES[class_idx]
            color = MMDET_COLOR_MAP.get(label, DEFAULT_COLOR_MMDET)
            x1, y1, x2, y2 = box_info[:4].astype(int)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2)
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            label_text = f"{label}: {score:.2f}"
            (tw, th), bl = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            l_y = y1 - 10 if y1 > 20 else y1 + th + 5
            bg_y1, bg_y2 = max(0, l_y - th - bl), min(height -1, l_y + bl // 2)
            bg_x1, bg_x2 = max(0, x1), min(width -1, x1 + tw)
            cv2.rectangle(img_copy, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
            cv2.putText(img_copy, label_text, (x1, l_y - bl // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            drawn_detections += 1
    else:
        print(f"Unhandled MMDetection result type ({type(detections_processed)}) in draw_mmdet_boxes.")
    return img_copy

def _get_file_type(filename_lower: str) -> Tuple[bool, bool]:
    image_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']
    video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.gif']
    is_img = any(filename_lower.endswith(ext) for ext in image_exts)
    is_vid = any(filename_lower.endswith(ext) for ext in video_exts)
    return is_img, is_vid

def handle_file_upload_preview(file_obj):
    raw_img_val = None
    raw_vid_val = None
    raw_img_visible = False
    raw_vid_visible = False
    input_file_visible = True # Input uploader is visible by default

    if file_obj is not None:
        file_path = file_obj if isinstance(file_obj, str) else file_obj.name
        is_image, is_video = _get_file_type(Path(file_path).name.lower())
        if is_image:
            try:
                img_bgr = cv2.imread(file_path)
                if img_bgr is None: gr.Warning("Cannot read image for preview.")
                else:
                    raw_img_val = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                    raw_img_visible = True
                    input_file_visible = False # Hide uploader, show preview
            except Exception as e: gr.Warning(f"Image preview error: {e}")
        elif is_video:
            raw_vid_val = file_path
            raw_vid_visible = True
            input_file_visible = False # Hide uploader, show preview
        else: gr.Info("Unsupported file type for preview.")
    
    return (
        gr.update(value=raw_img_val, visible=raw_img_visible),    # display_raw_image_file
        gr.update(value=raw_vid_val, visible=raw_vid_visible),    # display_raw_video_file
        gr.update(visible=input_file_visible),                    # input_file (uploader)
        gr.update(value=None, visible=False),                     # display_processed_image (clear on new preview)
        gr.update(value=None, visible=False)                      # display_processed_video (clear on new preview)
    )

def handle_webcam_capture(snapshot_from_feed):
    if snapshot_from_feed is None:
        return (
            gr.update(), # display_raw_image_webcam (no change)
            gr.update(), # input_webcam (no change)
            gr.update(value=None, visible=False), # display_processed_image (clear)
            gr.update(value=None, visible=False)  # display_processed_video (clear)
        )
    # Snapshot taken, show it, hide live feed
    return (
        gr.update(value=snapshot_from_feed, visible=True), # display_raw_image_webcam
        gr.update(visible=False),                          # input_webcam (hide live feed)
        gr.update(value=None, visible=False),              # display_processed_image (clear)
        gr.update(value=None, visible=False)               # display_processed_video (clear)
    )

def handle_example_select(evt: gr.SelectData):
    selected_item = None
    if isinstance(evt.value, list) and len(evt.value) > 0:
        selected_item = evt.value[0] 
    elif isinstance(evt.value, dict):
        selected_item = evt.value
    
    path_to_return = None
    # Handle cases where evt.value might be a dict from gr.Dataset (e.g. {'image': 'path/to/img.jpg', 'label': ...})
    # or directly a path string if Dataset components is just one item without a header name.
    if isinstance(selected_item, dict):
        # Try common keys for file paths in Gradio examples/datasets
        possible_keys = ['image', 'video', 'file', 'path'] 
        for key in possible_keys:
            if key in selected_item and isinstance(selected_item[key], str):
                path_to_return = selected_item[key]
                break
        if path_to_return is None and 'path' in selected_item and isinstance(selected_item['path'], str) : # Original check
             path_to_return = selected_item['path']

    elif isinstance(selected_item, str): # If evt.value was already a string path
        path_to_return = selected_item
    
    if path_to_return:
        return path_to_return 
    else:
        if evt.value is not None:
            print(f"Warning: Could not extract valid file path from gr.Dataset selection. evt.value: {evt.value}")
        return gr.update() # No change to input_file

# --- Helper functions for Gazelle Visualization (from test_mmdet_gazelle.py) ---
def normalize_bbox(bbox_abs, img_width, img_height):
    x1, y1, x2, y2 = bbox_abs
    # Ensure coordinates are within image boundaries before normalization
    x1_c = np.clip(x1, 0, img_width)
    y1_c = np.clip(y1, 0, img_height)
    x2_c = np.clip(x2, 0, img_width)
    y2_c = np.clip(y2, 0, img_height)
    return (x1_c / img_width, y1_c / img_height, x2_c / img_width, y2_c / img_height)

def visualize_all(pil_image, heatmaps, bboxes, inout_scores, inout_thresh=0.5):
    colors = ['lime', 'tomato', 'cyan', 'fuchsia', 'yellow', 'orange', 'purple', 'deeppink'] # More colors
    overlay_image = pil_image.convert("RGBA")
    draw = ImageDraw.Draw(overlay_image)
    width, height = pil_image.size

    for i in range(len(bboxes)):
        bbox = bboxes[i] # This is a normalized bbox (xmin, ymin, xmax, ymax)
        xmin, ymin, xmax, ymax = bbox
        color = colors[i % len(colors)]
        
        # Draw bounding box
        line_width = max(1, int(min(width, height) * 0.005)) # Ensure line_width is at least 1
        draw.rectangle([xmin * width, ymin * height, xmax * width, ymax * height], outline=color, width=line_width)

        current_inout_score = None
        if inout_scores is not None and i < len(inout_scores) and inout_scores[i] is not None:
            current_inout_score = inout_scores[i]
            text = f"in: {current_inout_score:.2f}"
            try:
                font_size = max(10, int(min(width, height) * 0.025)) # Ensure min font size
                font = ImageFont.load_default(size=font_size)
            except IOError:
                font = ImageFont.load_default() # Fallback

            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            text_x = xmin * width
            text_y = ymax * height + int(height * 0.005) 
            
            # Background for text for better visibility
            # draw.rectangle([text_x, text_y, text_x + text_width + 2, text_y + text_height + 2], fill=(0,0,0,128)) # Semi-transparent black
            draw.text((text_x + 1, text_y + 1), text, fill=color, font=font)


        if current_inout_score is not None and current_inout_score > inout_thresh:
            if i < len(heatmaps) and heatmaps[i] is not None:
                heatmap = heatmaps[i] 
                heatmap_np = heatmap.detach().cpu().numpy()
                
                if heatmap_np.ndim != 2:
                    print(f"Warning: Heatmap for bbox {i} is not 2D, shape is {heatmap_np.shape}. Skipping gaze vector.")
                    continue
                if heatmap_np.size == 0:
                    print(f"Warning: Heatmap for bbox {i} is empty. Skipping gaze vector.")
                    continue
                    
                max_index = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)
                
                gaze_target_x = max_index[1] / heatmap_np.shape[1] * width
                gaze_target_y = max_index[0] / heatmap_np.shape[0] * height
                
                bbox_center_x = ((xmin + xmax) / 2) * width
                bbox_center_y = ((ymin + ymax) / 2) * height # Center of the head bbox top line is often better: ymin * height
                
                # For gaze vector, let's make the dot slightly larger and line thicker
                gaze_dot_radius = max(2, int(0.007 * min(width, height)))
                gaze_line_width = max(1, int(0.005 * min(width, height)))

                draw.ellipse([(gaze_target_x - gaze_dot_radius, gaze_target_y - gaze_dot_radius), 
                              (gaze_target_x + gaze_dot_radius, gaze_target_y + gaze_dot_radius)], 
                             fill=color, outline=color) # Make dot solid
                draw.line([(bbox_center_x, bbox_center_y), (gaze_target_x, gaze_target_y)], fill=color, width=gaze_line_width)
            else:
                print(f"Warning: Missing heatmap for bbox {i}. Cannot draw gaze vector.")
        elif inout_scores is None: # Model does not produce in/out scores at all (Gazelle non-inout models)
            if i < len(heatmaps) and heatmaps[i] is not None: # Draw gaze vector unconditionally
                heatmap = heatmaps[i]
                heatmap_np = heatmap.detach().cpu().numpy()
                if heatmap_np.ndim != 2: continue
                if heatmap_np.size == 0: continue
                max_index = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)
                gaze_target_x = max_index[1] / heatmap_np.shape[1] * width
                gaze_target_y = max_index[0] / heatmap_np.shape[0] * height
                bbox_center_x = ((xmin + xmax) / 2) * width
                bbox_center_y = ((ymin + ymax) / 2) * height
                
                gaze_dot_radius = max(2, int(0.007 * min(width, height)))
                gaze_line_width = max(1, int(0.005 * min(width, height)))
                draw.ellipse([(gaze_target_x - gaze_dot_radius, gaze_target_y - gaze_dot_radius), 
                              (gaze_target_x + gaze_dot_radius, gaze_target_y + gaze_dot_radius)], 
                             fill=color, outline=color)
                draw.line([(bbox_center_x, bbox_center_y), (gaze_target_x, gaze_target_y)], fill=color, width=gaze_line_width)
    return overlay_image
# --- End Helper functions for Gazelle ---

def process_media(uploaded_file_obj, webcam_image_pil, model_type_choice: str, conf_threshold_ui: float, max_detections_ui: int, task_type: str):
    # Initialize updates for processed outputs, raw previews depend on earlier state handled by preview functions
    proc_img_update = gr.update(value=None, visible=False)
    proc_vid_update = gr.update(value=None, visible=False)

    # Initialize updates for raw previews - these will carry over existing state or be set if no preview happened
    # For file uploads
    raw_img_file_update = gr.update() # No change by default
    raw_vid_file_update = gr.update() # No change by default
    # For webcam
    raw_img_webcam_update = gr.update() # No change by default

    active_input_is_image = False
    active_input_is_video = False
    current_bgr_image_input = None 
    input_file_path_for_video = None 

    # Determine active input and prepare it, also set raw preview values if not already set by preview handlers
    if webcam_image_pil is not None:
        print("Processing webcam snapshot.")
        active_input_is_image = True
        current_bgr_image_input = cv2.cvtColor(np.array(webcam_image_pil), cv2.COLOR_RGB2BGR)
        # If webcam_image_pil is provided, display_raw_image_webcam should have been made visible by its handler
        # So, raw_img_webcam_update = gr.update(value=webcam_image_pil, visible=True) - might be redundant
    elif uploaded_file_obj is not None:
        input_file_path = uploaded_file_obj if isinstance(uploaded_file_obj, str) else uploaded_file_obj.name
        print(f"Processing uploaded file: {input_file_path}")
        input_file_path_for_video = input_file_path
        filename_lower = Path(input_file_path_for_video).name.lower()
        is_img_file, is_vid_file = _get_file_type(filename_lower)

        if is_img_file:
            active_input_is_image = True
            current_bgr_image_input = cv2.imread(input_file_path_for_video)
            if current_bgr_image_input is None:
                gr.Error("Failed to read uploaded image file.")
                # Return structure needs to match all UI outputs controlled by process_media
                return raw_img_file_update, raw_vid_file_update, raw_img_webcam_update, proc_img_update, proc_vid_update
            # raw_img_pil_for_preview = Image.fromarray(cv2.cvtColor(current_bgr_image_input, cv2.COLOR_BGR2RGB))
            # raw_img_file_update = gr.update(value=raw_img_pil_for_preview, visible=True)
        elif is_vid_file:
            active_input_is_video = True
            # raw_vid_file_update = gr.update(value=input_file_path_for_video, visible=True)
        else:
            gr.Error("Unsupported file type from uploader.")
            return raw_img_file_update, raw_vid_file_update, raw_img_webcam_update, proc_img_update, proc_vid_update
    else:
        gr.Warning("No input provided for processing.")
        return raw_img_file_update, raw_vid_file_update, raw_img_webcam_update, proc_img_update, proc_vid_update

    global yolo_model_instance, mmdet_model_instance, mmpose_model_instance, gazelle_model_instance, gazelle_transform_instance
    active_model_instance = None
    try:
        if model_type_choice == "YOLO":
            gr.Warning("YOLO model is selected but support might be limited/deprecated. Using MMDetection if available.")
            if MMDET_AVAILABLE:
                load_mmdet_model()
                active_model_instance = mmdet_model_instance
                model_type_choice = "MMDetection" # Force MMDetection
            else:
                raise RuntimeError("YOLO selected but MMDetection (primary) is not available.")

        elif model_type_choice == "MMDetection":
            if not MMDET_AVAILABLE: raise RuntimeError("MMDetection libraries not available.")
            load_mmdet_model()
            active_model_instance = mmdet_model_instance
        else: raise ValueError(f"Invalid model type: {model_type_choice}")

        if task_type == "Face Pose Estimation":
            if not MMPOSE_AVAILABLE: raise RuntimeError("Face Pose Estimation selected, but MMPose libraries are not available.")
            load_mmpose_model() # Loads mmpose_model_instance globally
            if mmpose_model_instance is None: # Check if loading failed
                 raise RuntimeError("MMPose model failed to load.")
        elif task_type == "Gaze Estimation [experimental]":
            if not GAZELLE_AVAILABLE: raise RuntimeError("Gaze Estimation selected, but Gazelle libraries are not available.")
            load_gazelle_model()
            if gazelle_model_instance is None or gazelle_transform_instance is None:
                raise RuntimeError("Gazelle model or transform failed to load.")

    except Exception as e:
        gr.Error(f"Failed to load model(s) (Detector: {model_type_choice}, Task: {task_type}): {str(e)}")
        return raw_img_file_update, raw_vid_file_update, raw_img_webcam_update, proc_img_update, proc_vid_update

    output_video_temp_path = None

    try:
        if active_input_is_image and current_bgr_image_input is not None:
            print(f"Processing image with {model_type_choice}, Conf: {conf_threshold_ui}, MaxDets: {max_detections_ui}")
            processed_image_np_bgr = current_bgr_image_input.copy()

            if model_type_choice == "YOLO":
                yolo_results = active_model_instance.predict(source=current_bgr_image_input, save=False, verbose=False,
                                                           conf=conf_threshold_ui, iou=IOU_THRESHOLD, max_det=max_detections_ui)
                if yolo_results and yolo_results[0].boxes: processed_image_np_bgr = yolo_results[0].plot()
                else: print("No YOLO detections on image.")
            
            elif model_type_choice == "MMDetection":
                # --- MMDetection Inference ---
                mm_results_raw = inference_detector(active_model_instance, current_bgr_image_input)
                processed_mm_results = None # This will be DetDataSample
                current_device = torch.device(MMDET_DEVICE)

                if isinstance(mm_results_raw, list): 
                    processed_mm_results_list = []
                    for class_detections in mm_results_raw:
                        if isinstance(class_detections, np.ndarray) and class_detections.shape[1] == 5 and len(class_detections) > 0:
                            class_detections_conf_filtered = class_detections[class_detections[:, 4] >= conf_threshold_ui]
                            if len(class_detections_conf_filtered) > 0:
                                keep_indices = nms(class_detections_conf_filtered, MMDET_NMS_THRESHOLD)
                                processed_mm_results_list.append(class_detections_conf_filtered[keep_indices])
                            else: processed_mm_results_list.append(np.empty((0, 5), dtype=np.float32))
                        else: processed_mm_results_list.append(np.empty((0, 5), dtype=np.float32))
                    processed_mm_results = processed_mm_results_list
                
                elif isinstance(mm_results_raw, DetDataSample): 
                    pred_instances = mm_results_raw.pred_instances
                    keep_conf = pred_instances.scores >= conf_threshold_ui
                    pred_instances_conf_filtered = pred_instances[keep_conf]
                    if len(pred_instances_conf_filtered) > 0:
                        bboxes_np = pred_instances_conf_filtered.bboxes.cpu().numpy()
                        scores_np = pred_instances_conf_filtered.scores.cpu().numpy()
                        if bboxes_np.shape[1] == 4 and scores_np.ndim == 1:
                             nms_input = np.hstack((bboxes_np, scores_np[:, None]))
                        else: 
                            print(f"Warning: Unexpected shapes for NMS input: bboxes {bboxes_np.shape}, scores {scores_np.shape}")
                            nms_input = np.empty((0,5), dtype=np.float32)

                        if isinstance(nms_input, np.ndarray) and nms_input.ndim == 2 and nms_input.shape[1] == 5 and len(nms_input)>0:
                            keep_nms_indices = nms(nms_input, MMDET_NMS_THRESHOLD)
                            keep_nms_tensor = torch.from_numpy(np.array(keep_nms_indices)).long().to(pred_instances_conf_filtered.bboxes.device)
                            pred_instances_final = pred_instances_conf_filtered[keep_nms_tensor]
                            current_metainfo = mm_results_raw.metainfo if hasattr(mm_results_raw, 'metainfo') else {}
                            processed_mm_results = DetDataSample(pred_instances=pred_instances_final, metainfo=current_metainfo)
                        else: 
                            current_metainfo = mm_results_raw.metainfo if hasattr(mm_results_raw, 'metainfo') else {}
                            empty_instances = InstanceData(bboxes=torch.empty((0,4),device=current_device), scores=torch.empty((0,),device=current_device), labels=torch.empty((0,),dtype=torch.long,device=current_device))
                            processed_mm_results = DetDataSample(pred_instances=empty_instances, metainfo=current_metainfo)
                    else: 
                        current_metainfo = mm_results_raw.metainfo if hasattr(mm_results_raw, 'metainfo') else {}
                        empty_instances = InstanceData(bboxes=torch.empty((0,4),device=current_device),scores=torch.empty((0,),device=current_device),labels=torch.empty((0,),dtype=torch.long,device=current_device))
                        processed_mm_results = DetDataSample(pred_instances=empty_instances, metainfo=current_metainfo)
                else:
                    print(f"Unhandled MMDetection result type ({type(mm_results_raw)}) for image.")

                if processed_mm_results is not None and task_type == "Face Detection":
                    # Draw only MMDetection boxes if task is just detection
                    processed_image_np_bgr = draw_mmdet_boxes_on_image(current_bgr_image_input, processed_mm_results, conf_threshold_ui, max_detections_ui)
                elif processed_mm_results is not None and task_type == "Face Pose Estimation":
                    # --- MMPose Inference and Visualization ---
                    img_for_pose_viz = current_bgr_image_input.copy()
                    detection_bboxes_for_pose = []
                    detection_scores_for_pose = []

                    if hasattr(processed_mm_results, 'pred_instances') and len(processed_mm_results.pred_instances) > 0:
                        # Extract bboxes and scores respecting max_detections_ui and conf_threshold_ui
                        # The processed_mm_results should already be filtered by conf and NMS, and capped by max_dets if draw_mmdet_boxes_on_image logic was fully applied before this.
                        # For safety, we re-check or rely on the structure of processed_mm_results
                        
                        # We need bboxes in Nx4 format for inference_topdown
                        # MMDetection bboxes are [x1, y1, x2, y2]
                        # Let's use the bboxes from the `processed_mm_results` which are already filtered.
                        # Max detections should also be respected here.
                        
                        # Take top 'max_detections_ui' from the already NMS'd and conf_thresholded results.
                        # If 'processed_mm_results' is a DetDataSample, its pred_instances should already be sorted by score (typically) or we can sort.
                        # For simplicity, let's assume processed_mm_results.pred_instances are the final detections to use.
                        
                        # Limit to max_detections_ui if necessary, though draw_mmdet_boxes_on_image logic also has a cap.
                        # The 'processed_mm_results' (DetDataSample) should ideally contain the final bboxes after NMS and score thresholding
                        # and potentially already capped at max_detections_ui or close to it.
                        
                        # For inference_topdown, we need bboxes as a NumPy array.
                        bboxes_tensor = processed_mm_results.pred_instances.bboxes[:max_detections_ui] # Respect max_detections
                        detection_bboxes_for_pose = bboxes_tensor.cpu().numpy()
                        
                        # Draw these detection boxes first
                        for i in range(len(detection_bboxes_for_pose)):
                            x1, y1, x2, y2 = detection_bboxes_for_pose[i].astype(int)
                            cv2.rectangle(img_for_pose_viz, (x1, y1), (x2, y2), MMPOSE_VIS_BBOX_COLOR, 2)
                            # Optionally, add score/label here if needed. For now, just boxes.

                    if len(detection_bboxes_for_pose) > 0:
                        try:
                            pose_results_list = inference_topdown(mmpose_model_instance, img_for_pose_viz, detection_bboxes_for_pose, bbox_format='xyxy')
                            
                            if pose_results_list: # If any poses were estimated
                                final_pose_datasample = merge_mmpose_data_samples(pose_results_list)
                                
                                # Initialize PoseLocalVisualizer
                                pose_visualizer = PoseLocalVisualizer(
                                    radius=MMPOSE_VIS_KPT_RADIUS, 
                                    line_width=MMPOSE_VIS_LINE_THICKNESS,
                                    kpt_color=[(0, 255, 0)] * 68 # Explicitly set BGR int color for keypoints
                                )
                                if hasattr(mmpose_model_instance, 'dataset_meta'):
                                     pose_visualizer.dataset_meta = mmpose_model_instance.dataset_meta
                                else:
                                     print("Warning: mmpose_model_instance.dataset_meta not found. Visualization might be basic.")
                                
                                # CORRECTED USAGE:
                                # The image provided to add_datasample should be the one with detection boxes already drawn.
                                # The draw_gt=False and draw_pred=True are typical for visualizing predictions.
                                # draw_bbox=False within add_datasample if bboxes are already drawn, 
                                # or True if you want the visualizer to draw bboxes from the datasample (if they exist there).
                                # Since we drew detection boxes manually, we might not need the visualizer to do it again.
                                
                                pose_visualizer.add_datasample(
                                    name='image_with_pose', # A name for this datasample
                                    image=img_for_pose_viz, # Image with detection bboxes
                                    data_sample=final_pose_datasample,
                                    draw_gt=False, # Don't draw ground truth
                                    draw_pred=True, # Draw predictions (keypoints, skeleton)
                                    draw_bbox=False, # Set to True if bboxes in final_pose_datasample should be drawn by visualizer
                                    show=False, # Don't display image using cv2.imshow
                                    wait_time=0,
                                    kpt_thr=0.3 # Keypoint score threshold for visualization
                                )
                                processed_image_np_bgr = pose_visualizer.get_image()

                            else: # No poses estimated
                                processed_image_np_bgr = img_for_pose_viz # Show with bboxes only
                        except Exception as pose_e:
                            print(f"Error during pose estimation or visualization: {pose_e}")
                            processed_image_np_bgr = img_for_pose_viz # Show with bboxes if pose fails
                    else: # No detections for pose
                        processed_image_np_bgr = img_for_pose_viz
                
                elif processed_mm_results is not None and task_type == "Gaze Estimation [experimental]":
                    print("Performing Gaze Estimation with Gazelle...")
                    # current_bgr_image_input is the BGR image from MMDetection
                    # visualize_all expects PIL RGB
                    img_pil_rgb_for_gazelle = Image.fromarray(cv2.cvtColor(current_bgr_image_input, cv2.COLOR_BGR2RGB))
                    
                    # Transform image once for Gazelle (expects RGB PIL)
                    gazelle_input_tensor = gazelle_transform_instance(img_pil_rgb_for_gazelle).unsqueeze(dim=0).to(GAZELLE_DEVICE)
                    
                    detected_bboxes_normalized_gazelle = []
                    heatmaps_gazelle = []
                    inout_scores_gazelle = []

                    # Extract bboxes from MMDetection results (processed_mm_results is DetDataSample)
                    # These bboxes are already filtered by conf_threshold_ui and NMS
                    if hasattr(processed_mm_results, 'pred_instances') and len(processed_mm_results.pred_instances) > 0:
                        # Limit to max_detections_ui
                        # pred_instances should already be sorted by score from MMDetection processing steps
                        bboxes_tensor_gazelle = processed_mm_results.pred_instances.bboxes[:max_detections_ui]
                        abs_bboxes_for_gazelle = bboxes_tensor_gazelle.cpu().numpy()
                        
                        img_h, img_w = current_bgr_image_input.shape[:2]
                        for bbox_abs in abs_bboxes_for_gazelle:
                            detected_bboxes_normalized_gazelle.append(normalize_bbox(bbox_abs, img_w, img_h))
                    
                    if detected_bboxes_normalized_gazelle:
                        for norm_bbox_g in detected_bboxes_normalized_gazelle:
                            input_data_g = {
                                "images": gazelle_input_tensor,
                                "bboxes": [[norm_bbox_g]] # Gazelle expects list of lists of (normalized) bbox tuples
                            }
                            with torch.no_grad():
                                output_g = gazelle_model_instance(input_data_g)
                            
                            # output_g["heatmap"] is [batch, num_people, H, W] -> [0][0] for this case
                            heatmaps_gazelle.append(output_g["heatmap"][0][0].cpu()) 
                            
                            current_inout_score_g = None
                            if output_g.get("inout") is not None:
                                # output_g["inout"] is [batch, num_people, 1] -> [0][0]
                                current_inout_score_g = output_g["inout"][0][0].item()
                            inout_scores_gazelle.append(current_inout_score_g)
                        
                        actual_inout_scores_for_viz_g = None
                        if any(s is not None for s in inout_scores_gazelle):
                            actual_inout_scores_for_viz_g = inout_scores_gazelle

                        # Use visualize_all to draw on the PIL image
                        final_pil_image_gazelle = visualize_all(
                            img_pil_rgb_for_gazelle, # Pass the original PIL RGB
                            heatmaps_gazelle,
                            detected_bboxes_normalized_gazelle,
                            actual_inout_scores_for_viz_g,
                            inout_thresh=GAZELLE_INOUT_THRESHOLD 
                        )
                        # Convert back to BGR numpy for Gradio Image component if it expects that,
                        # or let Gradio handle PIL directly if `type="pil"` is used for output.
                        # Assuming processed_image_np_bgr is what's finally converted.
                        processed_image_np_bgr = cv2.cvtColor(np.array(final_pil_image_gazelle.convert('RGB')), cv2.COLOR_RGB2BGR)
                    else: # No MMDetection detections
                        # processed_image_np_bgr is already current_bgr_image_input.copy()
                        print("No MMDetection faces found to pass to Gazelle.")
                
                elif processed_mm_results is None and task_type == "Face Pose Estimation":
                     print("No MMDetection results to pass to Pose Estimation.")
                     processed_image_np_bgr = current_bgr_image_input.copy() # Show original

                elif processed_mm_results is None and task_type == "Gaze Estimation [experimental]":
                     print("No MMDetection results to pass to Gazelle.")
                     processed_image_np_bgr = current_bgr_image_input.copy() # Show original

            processed_image_pil = Image.fromarray(cv2.cvtColor(processed_image_np_bgr, cv2.COLOR_BGR2RGB))
            proc_img_update = gr.update(value=processed_image_pil, visible=True)
            print("Image processing complete.")

        elif active_input_is_video and input_file_path_for_video is not None:
            if not check_ffmpeg(): gr.Warning("FFmpeg not found. Video processing might fail.")
            print(f"Processing video: {input_file_path_for_video} with {model_type_choice}, Conf: {conf_threshold_ui}, MaxDets: {max_detections_ui}")
            
            cap = cv2.VideoCapture(input_file_path_for_video)
            if not cap.isOpened(): raise ValueError("Failed to open video file.")
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps == 0 or frame_count == 0: raise ValueError("Video metadata invalid (fps/frame_count is 0).")
            duration = frame_count / fps
            if duration > VIDEO_DURATION_LIMIT_SECONDS: raise ValueError(f"Video too long ({duration:.1f}s > {VIDEO_DURATION_LIMIT_SECONDS}s limit).")

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_f:
                output_video_temp_path = temp_f.name
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if fw == 0 or fh == 0: raise ValueError("Video frame dimensions invalid (width/height is 0).")
            out_writer = cv2.VideoWriter(output_video_temp_path, fourcc, fps, (fw, fh))
            if not out_writer.isOpened(): raise RuntimeError(f"Failed to init VideoWriter for {output_video_temp_path}.")

            for frame_idx in tqdm(range(frame_count), desc=f"Processing video frames with {model_type_choice}"):
                ret, frame_bgr = cap.read()
                if not ret: break
                processed_frame_bgr = frame_bgr.copy()

                if model_type_choice == "YOLO":
                    yolo_results_frame = active_model_instance.predict(source=frame_bgr, save=False, verbose=False,
                                                               conf=conf_threshold_ui, iou=IOU_THRESHOLD, max_det=max_detections_ui)
                    if yolo_results_frame and yolo_results_frame[0].boxes: processed_frame_bgr = yolo_results_frame[0].plot()
                
                elif model_type_choice == "MMDetection":
                    # --- MMDetection per frame ---
                    mm_results_raw_frame = inference_detector(active_model_instance, frame_bgr)
                    processed_mm_results_frame = None # This will be DetDataSample
                    current_device_frame = torch.device(MMDET_DEVICE) 
                    
                    if isinstance(mm_results_raw_frame, list):
                        processed_mm_results_list_frame = []
                        for class_detections_f in mm_results_raw_frame:
                            if isinstance(class_detections_f, np.ndarray) and class_detections_f.shape[1] == 5 and len(class_detections_f) > 0:
                                class_detections_conf_filtered_f = class_detections_f[class_detections_f[:, 4] >= conf_threshold_ui]
                                if len(class_detections_conf_filtered_f) > 0:
                                    keep_indices_f = nms(class_detections_conf_filtered_f, MMDET_NMS_THRESHOLD)
                                    processed_mm_results_list_frame.append(class_detections_conf_filtered_f[keep_indices_f])
                                else: processed_mm_results_list_frame.append(np.empty((0,5),dtype=np.float32))
                            else: processed_mm_results_list_frame.append(np.empty((0,5),dtype=np.float32))
                        processed_mm_results_frame = processed_mm_results_list_frame

                    elif isinstance(mm_results_raw_frame, DetDataSample): 
                        pred_instances_f = mm_results_raw_frame.pred_instances
                        keep_conf_f = pred_instances_f.scores >= conf_threshold_ui
                        pred_instances_conf_filtered_f = pred_instances_f[keep_conf_f]
                        if len(pred_instances_conf_filtered_f) > 0:
                            bboxes_np_f = pred_instances_conf_filtered_f.bboxes.cpu().numpy()
                            scores_np_f = pred_instances_conf_filtered_f.scores.cpu().numpy()

                            if bboxes_np_f.shape[1] == 4 and scores_np_f.ndim == 1:
                                nms_input_f = np.hstack((bboxes_np_f, scores_np_f[:, None]))
                            else:
                                print(f"Warning (frame {frame_idx}): Unexpected shapes for NMS input: bboxes {bboxes_np_f.shape}, scores {scores_np_f.shape}")
                                nms_input_f = np.empty((0,5), dtype=np.float32)

                            if isinstance(nms_input_f,np.ndarray) and nms_input_f.ndim==2 and nms_input_f.shape[1]==5 and len(nms_input_f)>0:
                                keep_nms_indices_f = nms(nms_input_f, MMDET_NMS_THRESHOLD)
                                keep_nms_tensor_f = torch.from_numpy(np.array(keep_nms_indices_f)).long().to(pred_instances_conf_filtered_f.bboxes.device)
                                pred_instances_final_f = pred_instances_conf_filtered_f[keep_nms_tensor_f]
                                current_metainfo_f = mm_results_raw_frame.metainfo if hasattr(mm_results_raw_frame, 'metainfo') else {}
                                processed_mm_results_frame = DetDataSample(pred_instances=pred_instances_final_f, metainfo=current_metainfo_f)
                            else: 
                                current_metainfo_f = mm_results_raw_frame.metainfo if hasattr(mm_results_raw_frame, 'metainfo') else {}
                                empty_inst_f=InstanceData(bboxes=torch.empty((0,4),device=current_device_frame),scores=torch.empty((0,),device=current_device_frame),labels=torch.empty((0,),dtype=torch.long,device=current_device_frame))
                                processed_mm_results_frame = DetDataSample(pred_instances=empty_inst_f, metainfo=current_metainfo_f)
                        else: 
                            current_metainfo_f = mm_results_raw_frame.metainfo if hasattr(mm_results_raw_frame, 'metainfo') else {}
                            empty_inst_f=InstanceData(bboxes=torch.empty((0,4),device=current_device_frame),scores=torch.empty((0,),device=current_device_frame),labels=torch.empty((0,),dtype=torch.long,device=current_device_frame))
                            processed_mm_results_frame = DetDataSample(pred_instances=empty_inst_f, metainfo=current_metainfo_f)
                    
                    if processed_mm_results_frame is not None and task_type == "Face Detection":
                         processed_frame_bgr = draw_mmdet_boxes_on_image(frame_bgr, processed_mm_results_frame, conf_threshold_ui, max_detections_ui)
                    elif processed_mm_results_frame is not None and task_type == "Face Pose Estimation":
                        # --- MMPose per frame ---
                        img_for_pose_viz_frame = frame_bgr.copy()
                        detection_bboxes_for_pose_frame = []

                        if hasattr(processed_mm_results_frame, 'pred_instances') and len(processed_mm_results_frame.pred_instances) > 0:
                            bboxes_tensor_f = processed_mm_results_frame.pred_instances.bboxes[:max_detections_ui]
                            detection_bboxes_for_pose_frame = bboxes_tensor_f.cpu().numpy()
                            
                            for i in range(len(detection_bboxes_for_pose_frame)):
                                x1, y1, x2, y2 = detection_bboxes_for_pose_frame[i].astype(int)
                                cv2.rectangle(img_for_pose_viz_frame, (x1, y1), (x2, y2), MMPOSE_VIS_BBOX_COLOR, 2)

                        if len(detection_bboxes_for_pose_frame) > 0:
                            try:
                                pose_results_list_f = inference_topdown(mmpose_model_instance, img_for_pose_viz_frame, detection_bboxes_for_pose_frame, bbox_format='xyxy')
                                if pose_results_list_f:
                                    final_pose_datasample_f = merge_mmpose_data_samples(pose_results_list_f)
                                    
                                    pose_visualizer_f = PoseLocalVisualizer(
                                        radius=MMPOSE_VIS_KPT_RADIUS, 
                                        line_width=MMPOSE_VIS_LINE_THICKNESS,
                                        kpt_color=[(0, 255, 0)] * 68 # Explicitly set BGR int color for keypoints
                                    )
                                    if hasattr(mmpose_model_instance, 'dataset_meta'):
                                         pose_visualizer_f.dataset_meta = mmpose_model_instance.dataset_meta
                                    
                                    # CORRECTED USAGE:
                                    pose_visualizer_f.add_datasample(
                                        name='frame_with_pose',
                                        image=img_for_pose_viz_frame, # Frame with detection bboxes
                                        data_sample=final_pose_datasample_f,
                                        draw_gt=False,
                                        draw_pred=True,
                                        draw_bbox=False, # Or True, depending on if bboxes from pose_datasample are desired
                                        show=False,
                                        wait_time=0,
                                        kpt_thr=0.3
                                    )
                                    processed_frame_bgr = pose_visualizer_f.get_image()
                                else:
                                    processed_frame_bgr = img_for_pose_viz_frame
                            except Exception as pose_e_f:
                                print(f"Error during pose estimation/visualization for frame: {pose_e_f}")
                                processed_frame_bgr = img_for_pose_viz_frame
                        else: # No detections for pose on this frame
                            processed_frame_bgr = img_for_pose_viz_frame
                    
                    elif processed_mm_results_frame is not None and task_type == "Gaze Estimation [Experimental]":
                        # --- Gazelle per frame ---
                        frame_pil_rgb_for_gazelle = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                        gazelle_input_tensor_f = gazelle_transform_instance(frame_pil_rgb_for_gazelle).unsqueeze(dim=0).to(GAZELLE_DEVICE)

                        detected_bboxes_normalized_gazelle_f = []
                        heatmaps_gazelle_f = []
                        inout_scores_gazelle_f = []

                        if hasattr(processed_mm_results_frame, 'pred_instances') and len(processed_mm_results_frame.pred_instances) > 0:
                            bboxes_tensor_gazelle_f = processed_mm_results_frame.pred_instances.bboxes[:max_detections_ui]
                            abs_bboxes_for_gazelle_f = bboxes_tensor_gazelle_f.cpu().numpy()
                            
                            frame_h, frame_w = frame_bgr.shape[:2] # Use current frame's dimensions
                            for bbox_abs_f in abs_bboxes_for_gazelle_f:
                                detected_bboxes_normalized_gazelle_f.append(normalize_bbox(bbox_abs_f, frame_w, frame_h))

                        if detected_bboxes_normalized_gazelle_f:
                            for norm_bbox_gf in detected_bboxes_normalized_gazelle_f:
                                input_data_gf = {"images": gazelle_input_tensor_f, "bboxes": [[norm_bbox_gf]]}
                                with torch.no_grad():
                                    output_gf = gazelle_model_instance(input_data_gf)
                                heatmaps_gazelle_f.append(output_gf["heatmap"][0][0].cpu())
                                inout_score_gf = None
                                if output_gf.get("inout") is not None:
                                    inout_score_gf = output_gf["inout"][0][0].item()
                                inout_scores_gazelle_f.append(inout_score_gf)
                            
                            actual_inout_scores_for_viz_gf = None
                            if any(s is not None for s in inout_scores_gazelle_f):
                                actual_inout_scores_for_viz_gf = inout_scores_gazelle_f
                            
                            # Use visualize_all for the frame
                            final_frame_pil_gazelle_viz = visualize_all(
                                frame_pil_rgb_for_gazelle, # PIL RGB of current frame
                                heatmaps_gazelle_f,
                                detected_bboxes_normalized_gazelle_f,
                                actual_inout_scores_for_viz_gf,
                                inout_thresh=GAZELLE_INOUT_THRESHOLD
                            )
                            processed_frame_bgr = cv2.cvtColor(np.array(final_frame_pil_gazelle_viz.convert('RGB')), cv2.COLOR_RGB2BGR)
                        else: # No MMDetection detections on this frame
                            # processed_frame_bgr remains frame_bgr.copy() if no Gazelle processing done
                            pass 
                    
                    elif processed_mm_results_frame is None and task_type == "Gaze Estimation [experimental]":
                        # processed_frame_bgr is already frame_bgr.copy() from earlier
                        pass # No Gazelle if no detections

                    elif processed_mm_results_frame is None : # Covers both tasks if no detection
                         # processed_frame_bgr remains frame_bgr.copy()
                         pass # No change needed, already a copy

                out_writer.write(processed_frame_bgr)
            
            cap.release(); out_writer.release()
            if not Path(output_video_temp_path).exists() or os.path.getsize(output_video_temp_path) == 0:
                raise RuntimeError(f"Output video not created or empty: {output_video_temp_path}")
            
            proc_vid_update = gr.update(value=output_video_temp_path, visible=True)
            print(f"Video processing complete. Output: {output_video_temp_path}")
        else:
            if not active_input_is_image and not active_input_is_video:
                 gr.Info("No valid input (image/video) identified for processing.")

    except Exception as e:
        import traceback; tb_str = traceback.format_exc()
        error_msg = f"Error ({model_type_choice}, Conf: {conf_threshold_ui}, MaxDets: {max_detections_ui}): {str(e)}"
        print(f"{error_msg}\n{tb_str}")
        gr.Error(error_msg)
        if output_video_temp_path and Path(output_video_temp_path).exists():
            try: os.remove(output_video_temp_path)
            except OSError as oe: print(f"Could not cleanup temp file {output_video_temp_path}: {oe}")
        proc_img_update = gr.update(value=None, visible=False)
        proc_vid_update = gr.update(value=None, visible=False)

    # Return order must match outputs list for submit_button.click
    # display_raw_image_file, display_raw_video_file, display_raw_image_webcam, 
    # display_processed_image, display_processed_video
    return raw_img_file_update, raw_vid_file_update, raw_img_webcam_update, proc_img_update, proc_vid_update

def clear_all_media_and_outputs():
    return (
        gr.update(value=None, visible=True),  # input_file (show)
        gr.update(value=None, visible=True),  # input_webcam (show)
        gr.update(value=None, visible=False), # display_raw_image_file (hide)
        gr.update(value=None, visible=False), # display_raw_video_file (hide)
        gr.update(value=None, visible=False), # display_raw_image_webcam (hide)
        gr.update(value=None, visible=False), # display_processed_image (hide)
        gr.update(value=None, visible=False)  # display_processed_video (hide)
    )

# --- Gradio UI Definition ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("<center><h1>PrimateFace Detection, Pose Estimation, and Gaze Estimation Demo</h1></center>")
    gr.Markdown("Upload an image/video or use your webcam. For webcam, press 'Enter' to take a snapshot.")
    gr.Markdown("Click 'Detect Faces' for results.")
  
    with gr.Row():
        with gr.Column(scale=1): # Left column for Inputs
            with gr.Tabs() as input_tabs:
                with gr.TabItem("Upload File"):
                    input_file = gr.File(label="Upload Image or Video Here", file_types=["image", ".mp4", ".avi", ".mov", ".mkv", ".webm", ".gif"])
                    display_raw_image_file = gr.Image(label="Raw Image Preview", type="pil", interactive=False, visible=False, height=None, width=None)
                    display_raw_video_file = gr.Video(label="Raw Video Preview", interactive=False, visible=False, height=None, width=None)
                with gr.TabItem("Webcam"):
                    gr.Markdown("""
**Using the Webcam:**

1.  **Activate Feed:** If prompted, allow browser access to your camera. You should see a live feed.
2.  **Select Camera (Optional):** Use any camera selection options (often a small camera icon on/near the feed) if you have multiple cameras.
3.  **Capture Frame:**
    *   **Desktop:** Click directly *on the live video feed* or press **Enter** when the feed is active.
    *   **Mobile:** Tap *on the live video feed*. You might then need to tap a "capture" or "use photo" button provided by your phone's camera interface.
4.  **Confirm & Detect:** Once the captured snapshot appears below the (now hidden) live feed, click "Detect Faces".

*(If the snapshot doesn't appear, try interacting with the live feed again. To retry, click "Clear All Inputs & Outputs".)*
""")
                    input_webcam = gr.Image(sources=["webcam"], type="pil", label="Live Webcam: Click feed or press Enter to capture")
                    display_raw_image_webcam = gr.Image(label="Captured Snapshot Preview", type="pil", interactive=False, visible=False, height=None, width=None)
            
            clear_all_button = gr.Button("Clear All Inputs & Outputs")

        with gr.Column(scale=1): # Right column for Processed Outputs ONLY
            gr.Markdown("### Processed Output")
            display_processed_image = gr.Image(label="Processed Image", type="pil", interactive=False, visible=False, height=None, width=None)
            display_processed_video = gr.Video(label="Processed Video", interactive=False, visible=False, height=None, width=None)

    # Example images and submit button (below inputs and outputs, above controls)
    example_image_paths = [
        r"A:\NonEnclosureProjects\inprep\PrimateFace\data\bing_images\allocebus\allocebus_000003.jpeg",
        r"A:\NonEnclosureProjects\inprep\PrimateFace\data\bing_images\tarsius\tarsius_000120.jpeg",
        r"A:\NonEnclosureProjects\inprep\PrimateFace\data\bing_images\nasalis\nasalis_proboscis-monkey.png",
        r"A:\NonEnclosureProjects\inprep\PrimateFace\data\bing_images\macaca\macaca_000032.jpeg",
        r"A:\NonEnclosureProjects\inprep\PrimateFace\data\bing_images\mandrillus\mandrillus_000011.jpeg",
        r"A:\NonEnclosureProjects\inprep\PrimateFace\data\bing_images\pongo\pongo_000006.jpeg"
        
    ]
    example_data = [[path] for path in example_image_paths] # Ensure it's a list of lists

    example_dataset = gr.Dataset(
        components=["image"], # Corrected: use "image" type for image file paths
        samples=example_data,
        label="Example Images (Click to use with File Uploader)",
        samples_per_page=6, 
    )
    
    submit_button = gr.Button("Detect Faces", variant="primary", scale=2) 

    with gr.Column(): # Controls area at the bottom
        gr.Markdown("### Detection Controls")
        model_choice_radio = gr.Radio(
            choices=["MMDetection"], # Only MMDetection
            value="MMDetection",    # Default to MMDetection
            label="Inferencer", 
            visible=False           # Hide this choice from the user
        )
        task_type_dropdown = gr.Dropdown(
            choices=["Face Detection", "Face Pose Estimation", "Gaze Estimation [experimental]"],
            value="Face Detection",
            label="Select Task",
            info="Choose detection, detection + pose, or detection + gaze."
        )
        conf_slider_ctrl = gr.Slider(minimum=0.05, maximum=0.95, value=0.25, step=0.05, label="Confidence Threshold (for Detection)")
        max_det_slider_ctrl = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Max Detections (for Detection & Pose)")
          
    # --- Event Wiring ---

    # Define component lists for outputs of event handlers
    # For file uploads preview:
    file_preview_outputs = [
        display_raw_image_file, 
        display_raw_video_file, 
        input_file, # To control its visibility
        display_processed_image, # Clear processed from previous run
        display_processed_video  # Clear processed from previous run
    ]
    # For webcam snapshot preview (now handle_webcam_capture):
    webcam_capture_outputs = [
        display_raw_image_webcam, 
        input_webcam, 
        display_processed_image, 
        display_processed_video  
    ]
    # For main processing function (submit_button):
    # Note: The raw preview components (file and webcam) are not directly set by process_media.
    # Their state is determined by the preview handlers and should persist.
    # process_media only updates the processed outputs.
    # However, to simplify, we can make process_media return gr.update() for them to not change.
    process_media_outputs = [
        display_raw_image_file, 
        display_raw_video_file, 
        display_raw_image_webcam, # This now holds the confirmed webcam snapshot for processing
        display_processed_image,
        display_processed_video
    ]
    # For clear button:
    clear_button_outputs = [
        input_file, input_webcam, 
        display_raw_image_file, display_raw_video_file, display_raw_image_webcam, 
        display_processed_image, display_processed_video
    ]

    input_file.change( 
        fn=handle_file_upload_preview,
        inputs=[input_file],
        outputs=file_preview_outputs
    )
    
    input_webcam.change(
        fn=handle_webcam_capture,
        inputs=[input_webcam],
        outputs=webcam_capture_outputs
    )

    example_dataset.select(
        fn=handle_example_select,
        inputs=None, 
        outputs=[input_file] 
    )

    clear_all_button.click(
        fn=clear_all_media_and_outputs,
        inputs=[],
        outputs=clear_button_outputs
    )
 
    submit_button.click(
        fn=process_media,
        inputs=[input_file, display_raw_image_webcam, model_choice_radio, conf_slider_ctrl, max_det_slider_ctrl, task_type_dropdown], # Added task_type_dropdown
        outputs=process_media_outputs
    )

if __name__ == "__main__":
    print("Launching Gradio demo...")
    print(f"FFmpeg: {'Found' if check_ffmpeg() else 'Not found'}")
    print(f"MMDetection available: {MMDET_AVAILABLE}")
    print(f"MMPose available: {MMPOSE_AVAILABLE}")
    print(f"Gazelle available: {GAZELLE_AVAILABLE}")

    models_loaded_successfully = True

    # Critical check and pre-load for MMDetection
    if MMDET_AVAILABLE:
        if not Path(MMDET_CONFIG_PATH).exists(): 
            print(f"Warning: MMDetection config not found: {MMDET_CONFIG_PATH}")
        if not Path(MMDET_CHECKPOINT_PATH).exists(): 
            print(f"Warning: MMDetection checkpoint not found: {MMDET_CHECKPOINT_PATH}")
        try:
            print("Pre-loading MMDetection model...")
            load_mmdet_model() # This will load into global mmdet_model_instance
            if mmdet_model_instance is None:
                raise RuntimeError("MMDetection model is None after loading attempt.")
            print("MMDetection model pre-loaded successfully.")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to pre-load MMDetection model: {e}")
            models_loaded_successfully = False
    else:
        print("CRITICAL ERROR: MMDetection libraries are not available.")
        print("The application cannot function without MMDetection.")
        models_loaded_successfully = False

    # Check and pre-load for MMPose (only if MMDetection loaded successfully)
    if models_loaded_successfully and MMPOSE_AVAILABLE:
        if not Path(MMPOSE_CONFIG_PATH).exists():
            print(f"Warning: MMPose config not found: {MMPOSE_CONFIG_PATH}")
        if not Path(MMPOSE_CHECKPOINT_PATH).exists():
            print(f"Warning: MMPose checkpoint not found: {MMPOSE_CHECKPOINT_PATH}")
        try:
            print("Pre-loading MMPose model...")
            load_mmpose_model() # This will load into global mmpose_model_instance
            if mmpose_model_instance is None:
                raise RuntimeError("MMPose model failed to load.")
            print("MMPose model pre-loaded successfully.")
        except Exception as e:
            print(f"ERROR: Failed to pre-load MMPose model: {e}. Face Pose Estimation task might fail.")
            # We can still proceed if detection is the only task desired and MMPose fails
            # Or set models_loaded_successfully = False if MMPose is critical for any operation
    elif not MMPOSE_AVAILABLE:
        print("Info: MMPose libraries not available. 'Face Pose Estimation' task will not function.")

    # Check and pre-load for Gazelle (only if MMDetection loaded successfully)
    if models_loaded_successfully and GAZELLE_AVAILABLE:
        # Check for Gazelle files (optional, load_gazelle_model will error if not found)
        if not Path(GAZELLE_CHECKPOINT_PATH).exists():
            print(f"Warning: Gazelle checkpoint not found at {GAZELLE_CHECKPOINT_PATH}. 'Gaze Estimation' will fail if selected.")
        try:
            print("Pre-loading Gazelle model...")
            load_gazelle_model() 
            if gazelle_model_instance is None or gazelle_transform_instance is None:
                # This condition might be redundant if load_gazelle_model raises on failure
                raise RuntimeError("Gazelle model or transform is None after loading attempt.")
            print("Gazelle model pre-loaded successfully.")
        except Exception as e:
            print(f"ERROR: Failed to pre-load Gazelle model: {e}. 'Gaze Estimation [experimental]' task might fail if selected.")
            # Potentially set models_loaded_successfully = False if Gazelle is critical for *any* default operation,
            # but for now, allow Gradio to start and let the task fail if selected.
    elif not GAZELLE_AVAILABLE:
        print("Info: Gazelle libraries not available. 'Gaze Estimation [experimental]' task will not function.")

    if models_loaded_successfully:
        print("Starting Gradio server...")
        demo.launch(share=True, show_error=True)
    else:
        print("Gradio server will not start due to critical model loading errors.") 