# GUI Module Migration Checklist

## Files to Delete After Testing

Once you've verified the new consolidated system works with your models, you can safely delete these redundant files:

### 1. Redundant Pseudo-labeling Scripts
- [ ] `pseudolabel_gui.py` (2545 lines) - Replaced by `pseudolabel.py`
- [ ] `pseudolabel_gui_fm.py` (853 lines) - Merged into core modules
- [ ] `pseudolabel_gui_fm copy.py` (668 lines) - Best parts extracted to core modules

### 2. Redundant Conversion Scripts  
- [ ] `imgdir2coco_face68.py` (339 lines) - **WARNING: Has hardcoded paths on lines 48-58**
- [ ] `imgdir2coco_facedet.py` (142 lines) - Functionality in `converters/image.py`
- [ ] `viddir2coco_facedet.py` (196 lines) - Functionality in `converters/video.py`
- [ ] `f_mmpose_pseudolabel.py` (835 lines) - Best parts merged into converters

### 3. Files to Keep
- ✅ `refine_boxes.py` - Interactive refinement GUI (works with COCO output)

## Total Reduction
- **Before**: ~8,344 lines across 8 files
- **After**: ~3,500 lines in modular structure
- **Reduction**: 58% fewer lines, 100% better organization

## Testing Before Deletion

Test each replacement to ensure functionality:

```bash
# Test image processing (replaces imgdir2coco_facedet.py)
python pseudolabel.py --input ./test_images --type images \
    --det-config your_config.py --det-checkpoint your_model.pth \
    --output ./test_output

# Test video processing (replaces viddir2coco_facedet.py)
python pseudolabel.py --input ./test_videos --type videos \
    --det-config your_config.py --det-checkpoint your_model.pth \
    --output ./test_output

# Test with pose estimation (replaces pseudolabel_gui*.py)
python pseudolabel.py --input ./test_images --type images \
    --det-config det.py --det-checkpoint det.pth \
    --pose-config pose.py --pose-checkpoint pose.pth \
    --output ./test_output

# Test refinement GUI (uses refine_boxes.py)
python refine_boxes.py test_output/annotations.json
```

## Backup Recommendation

Before deleting, consider creating a backup:

```bash
mkdir gui_backup
mv pseudolabel_gui*.py gui_backup/
mv imgdir2coco*.py gui_backup/
mv viddir2coco*.py gui_backup/
mv f_mmpose_pseudolabel.py gui_backup/
```