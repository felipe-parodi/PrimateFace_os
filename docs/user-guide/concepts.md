# Concepts

Key concepts and theory behind PrimateFace.

## Facial Landmarks

### What are Facial Landmarks?

Facial landmarks are specific points on the face that correspond to anatomical features. They enable:
- Precise face alignment
- Expression analysis
- Individual identification
- Behavioral studies

### 68-Point vs 48-Point Systems

#### 68-Point System (Human-Oriented)
Originally designed for human faces with detailed coverage:
- **Jaw line**: 17 points
- **Eyebrows**: 10 points (5 each)
- **Eyes**: 12 points (6 each)
- **Nose**: 9 points
- **Mouth**: 20 points

#### 48-Point System (Primate-Optimized)
Adapted for non-human primates:
- **Reduced jaw points**: Better fits primate facial structure
- **Simplified eyebrows**: Accounts for fur coverage
- **Core features preserved**: Eyes, nose, mouth remain detailed
- **Species-agnostic**: Works across primate genera

### Why Convert Between Systems?

1. **Dataset Compatibility**: Leverage human face datasets
2. **Cross-Species Transfer**: Apply human-trained models to primates
3. **Tool Interoperability**: Use various annotation tools

## DINOv2 Features

### Self-Supervised Vision Transformers

DINOv2 is a vision transformer trained without labels that learns:
- **Universal visual features**: Not task-specific
- **Semantic understanding**: Groups similar content
- **Fine-grained details**: Captures subtle differences

### Why DINOv2 for Primates?

1. **No Species Bias**: Not trained specifically on humans
2. **Robust Features**: Works across lighting, pose, species
3. **Zero-Shot Transfer**: No primate-specific training needed

### Feature Applications

- **Smart Selection**: Choose diverse training samples
- **Quality Assessment**: Identify good/bad images
- **Clustering**: Group by species or individuals
- **Anomaly Detection**: Find unusual cases

## Cascade R-CNN Architecture

### Multi-Stage Detection

Cascade R-CNN improves detection through:
1. **Stage 1**: Rough face localization
2. **Stage 2**: Refined bounding boxes
3. **Stage 3**: Final precise detection

### Why Cascade for Primates?

- **Handles Occlusion**: Partial faces in natural settings
- **Multi-Scale**: Detects faces at various distances
- **High Precision**: Reduces false positives

## HRNet Architecture

### High-Resolution Networks

HRNet maintains high-resolution representations throughout:
- **Parallel branches**: Multiple resolution streams
- **Information exchange**: Cross-resolution fusion
- **Detail preservation**: Fine landmark localization

### Advantages for Landmarks

- **Precise localization**: Sub-pixel accuracy
- **Robust to blur**: Handles motion and focus issues
- **Efficient**: Good accuracy/speed trade-off

## Semi-Supervised Learning

### Pseudo-Labeling Strategy

Combine model predictions with human verification:
1. **Initial Prediction**: Model generates candidates
2. **Human Review**: Quick verification/correction
3. **Iterative Improvement**: Retrain with new data

### Benefits

- **Faster Annotation**: 5-10x speedup
- **Consistency**: Reduces human annotator variance
- **Scalability**: Handle large datasets

## Cross-Framework Training

### Why Multiple Frameworks?

Different frameworks excel at different tasks:
- **MMPose**: Best accuracy, production-ready
- **DeepLabCut**: Behavioral analysis, tracking
- **SLEAP**: Multi-animal, social interactions
- **YOLO**: Real-time, edge deployment

### Unified Data Format

COCO format as universal exchange:
```json
{
  "images": [...],
  "annotations": [{
    "keypoints": [x1,y1,v1, x2,y2,v2, ...],
    "bbox": [x, y, width, height]
  }],
  "categories": [...]
}
```

## Evaluation Metrics

### NME (Normalized Mean Error)

Distance error normalized by face size:
```
NME = (1/N) * Σ ||predicted - ground_truth|| / face_size
```
- **Good**: < 5%
- **Acceptable**: 5-10%
- **Poor**: > 10%

### PCK (Percentage of Correct Keypoints)

Percentage within threshold distance:
```
PCK@α = (# keypoints within α * face_size) / (# total keypoints)
```

### mAP (Mean Average Precision)

Detection quality across IoU thresholds:
- **mAP@50**: 50% overlap required
- **mAP@75**: 75% overlap required
- **mAP@50-95**: Average across range

## Species-Specific Considerations

### Anatomical Variations

Different primates require adjustments:
- **Great Apes**: Pronounced brow ridges
- **Prosimians**: Large eyes, small faces
- **New World Monkeys**: Varied nose structures

### Behavioral Context

Consider natural behaviors:
- **Arboreal species**: Often partially occluded
- **Social species**: Multiple faces in frame
- **Nocturnal species**: Low-light conditions

## See Also

- [Getting Started](../getting-started/index.md)
- [Core Workflows](./core-workflows/index.md)
- [API Reference](../api/index.md)