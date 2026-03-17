# PrimateFace v2 Plan

**Date**: 2026-03-16 (last updated: 2026-03-17)
**Authors**: Felipe Parodi, Claude
**Status**: Draft — brainstorming & research phase. Core API + analysis module shipped.

---

## 1. What IS PrimateFace?

### The Identity Tension

PrimateFace currently uses multiple framings: "Resource" (README title), "ecosystem" (README body), "data, models, and tutorials" (quick start). v2 needs clarity.

PrimateFace wants to be:
1. A **simple library** — `pip install primateface`, detect faces in 3 lines (InsightFace/DeepFace)
2. A **training framework** — active learning loop, fine-tuning, custom species (DLC/SLEAP)
3. An **interop hub** — import/export to DLC, SLEAP, MMPose formats; use their training backends
4. A **dataset + benchmark** — 230K images, evaluation protocols

These aren't contradictory — they're **layers**. DeepLabCut does exactly this:
```
Layer 1: deeplabcut.video_inference_superanimal("video.mp4")  ← library mode
Layer 2: deeplabcut.train_network(config)                      ← framework mode
Layer 3: GUI + active learning + model zoo + DLC2Kinematics    ← ecosystem mode
```

### Recommended Positioning: "Toolkit" that scales to "Ecosystem"

> **PrimateFace meets you where you are:**
> - Never coded before? → HF Space / Colab notebooks
> - Want quick results? → `pip install primateface`, 3 lines of code
> - Need custom models for your species? → Active learning loop + fine-tuning
> - Building a research pipeline? → DLC/SLEAP interop, DINOv2 embeddings, evaluation framework

**Dual framing**:
- **Paper** (Nature Methods Resource): "...a comprehensive resource comprising a large-scale annotated dataset, pretrained models, and an open-source analysis toolbox"
- **GitHub README**: "PrimateFace is an open-source toolbox for primate face detection, pose estimation, and facial behavior analysis. Works across 60+ primate genera out of the box."

### The Unique Gap

**There is no pip-installable, general-purpose primate/animal face analysis library.**

| Category | Existing tools | PrimateFace's advantage |
|----------|---------------|------------------------|
| Human face analysis | InsightFace, DeepFace, face_recognition | Doesn't work on non-human primates |
| Animal pose (body) | DeepLabCut, SLEAP | Body pose, not face-specific; no face detection |
| Animal re-ID | MegaDescriptor, wildlife-tools | No face landmarks, no face detection |
| Primate face (research) | PrimNet, LemurFaceID, ChimpFace | Not pip-installable, species-specific, not maintained |

PrimateFace is the **InsightFace for primates** — that's the pitch.

---

## 2. Competitive Reference

### The "Hello World" Test

| Library | Lines | Code |
|---------|-------|------|
| face_recognition | 2 | `img = load_image_file("x.jpg"); locs = face_locations(img)` |
| DeepFace | 1 | `faces = DeepFace.extract_faces("x.jpg")` |
| InsightFace | 3 | `app = FaceAnalysis(); app.prepare(ctx_id=0); faces = app.get(img)` |
| **PrimateFace v0.1** | **10+** | Create processor w/ config paths, checkpoint paths, load image... |
| **PrimateFace v2 target** | **3** | `pf = PrimateFace(); faces = pf.detect("monkey.jpg")` |

### Key Patterns to Steal

| From | Pattern | How to apply |
|------|---------|-------------|
| **InsightFace** | `FaceAnalysis` app object, rich `Face` objects, model packs, ONNX | `primateface.PrimateFace()` with auto-download ONNX models |
| **DeepFace** | String-based backend swapping, accepts file paths | `PrimateFace(detector="yolo", pose="hrnet")` |
| **face_recognition** | Zero-config defaults, intuitive function names | `primateface.face_locations("monkey.jpg")` |
| **Py-Feat** | `Detector` returning `Fex` DataFrame with AUs, emotions, pose | `pf.analyze()` returning rich result with all attributes |
| **DeepLabCut** | SuperAnimal Model Zoo, active learning loop, GUI | PrimateFace Model Zoo + AL loop + pseudo-labeling GUI |
| **SLEAP** | GUI-first AL loop, fast iteration, confidence-based sampling | Gradio-based review + training trigger |
| **Ultralytics** | CLI: `yolo predict model=X source=Y` | `primateface detect --input X --output Y` |

---

## 3. The Dream API

### Tier 1: The Library (`pip install primateface`)

```python
import primateface

# Zero-config — auto-downloads default models on first use
pf = primateface.PrimateFace()

# Detect faces
faces = pf.detect("monkey.jpg")
# → [Face(bbox=[x1,y1,x2,y2], score=0.97)]

# Full analysis — detection + landmarks + all derived features
faces = pf.analyze("monkey.jpg")
# → [Face with bbox, keypoints, head_pose, quality, symmetry, mouth_aperture, ...]

# Process video
results = pf.process_video("video.mp4", output="results/")

# Visualize
pf.draw(faces, "monkey.jpg", output="result.jpg")

# Backend swapping (power users)
pf = primateface.PrimateFace(detector="yolo", pose_model="hrnet-w18", device="cuda:0")
```

### The Rich `Face` Object

```python
face = faces[0]

# Core (available from v2 launch)
face.bbox              # [x1, y1, x2, y2]
face.score             # detection confidence
face.keypoints         # ndarray(68, 3) — [x, y, visibility]
face.crop              # cropped face image

# Landmark-derived (no new models — pure geometry, ships at launch)
face.head_pose         # [yaw, pitch, roll] from solvePnP
face.quality           # face quality score (blur, occlusion, size)
face.symmetry          # facial symmetry score (fluctuating asymmetry)
face.mouth_aperture    # upper/lower lip distance — vocalization proxy
face.eye_aperture      # eye openness — blink detection
face.brow_position     # relative brow height — AU1/2/4 proxy

# Model-based (require additional models, phased rollout)
face.genus             # predicted genus/species
face.sex               # male/female/unknown
face.age_class         # infant/juvenile/subadult/adult
face.identity          # embedding vector for re-ID
face.gaze              # [gaze_x, gaze_y] direction

# Research frontier (future)
face.aus               # dict of AU predictions (species-specific FACS)
face.expression        # expression classification
face.pain_score        # grimace scale score
```

### Functional API (face_recognition pattern)

```python
import primateface

# Module-level functions — no object creation needed
locations = primateface.face_locations("monkey.jpg")
landmarks = primateface.face_landmarks("monkey.jpg")
embeddings = primateface.face_embeddings("monkey.jpg")
```

### CLI

```bash
primateface detect --input image.jpg --output results/
primateface analyze --input ./images/ --output results/ --tasks detect,pose,genus
primateface models list
primateface models download hrnet-w18-dark
```

### Tier 2: Low/No-Code (Primatologists)

1. **HuggingFace Space** — upload image/video, get results, zero install
2. **Google Colab notebooks** — guided tutorials with "Run All"
3. **Gradio local app** — `primateface launch` starts a web UI
4. **Pseudo-labeling GUI** — for creating custom annotations

### Tier 3: Power Users / Ecosystem

- Active learning loop + fine-tuning
- DINOv2 embeddings + UMAP visualization
- Landmark converter (68 ↔ 49 kpts)
- Multi-framework evaluation (MMPose vs DLC vs SLEAP)
- DLC/SLEAP/BORIS/NWB interop
- Model export (ONNX, TorchScript)

---

## 4. Package Restructure

### Proposed v2 structure

```
primateface/                       # rename from primateface_oss
  src/primateface/
    __init__.py                    # PrimateFace class + top-level functions
    core/
      face.py                      # Face dataclass
      detector.py                  # Unified detector interface
      pose.py                      # Unified pose interface
      models.py                    # Model registry + auto-download from HF
      config.py                    # Default configs
    backends/
      onnx.py                      # ONNX Runtime (default, lightweight)
      mmdet.py                     # MMDetection backend
      mmpose.py                    # MMPose backend
      ultralytics.py               # YOLO backend
    analysis/                      # <-- THIS IS THE SHORT-TERM PRIORITY
      kinematics.py                # Mouth aperture, lip-smack, brow movement
      head_pose.py                 # Yaw/pitch/roll from landmarks
      symmetry.py                  # Fluctuating asymmetry from landmark pairs
      quality.py                   # Blur, occlusion, lighting, size scores
      genus.py                     # Genus/species classification
      sex.py                       # Sex classification
      age.py                       # Age class prediction
      embeddings.py                # DINOv2 / MegaDescriptor / ArcFace
      gaze.py                      # Gaze estimation
    training/
      active_learning.py           # AL loop orchestrator
      finetune.py                  # Fine-tuning API
      data.py                      # Dataset creation from annotations
      augmentation.py              # Primate-specific augmentations
    interop/
      deeplabcut.py                # DLC project ↔ PrimateFace
      sleap.py                     # SLEAP labels ↔ PrimateFace
      coco.py                      # COCO JSON I/O
      csv.py                       # CSV export
      boris.py                     # BORIS behavioral coding export
      nwb.py                       # Neurodata Without Borders export
    tools/
      converter.py                 # Landmark converter (68 ↔ 49)
      smoother.py                  # Temporal smoothing
      visualizer.py                # Visualization utilities
    evaluation/
      metrics.py                   # NME, PCK, OKS
      benchmark.py                 # Framework comparison
    gui/
      pseudolabel.py               # Pseudo-labeling
      refine.py                    # Annotation refinement
      gradio_app.py                # Gradio web interface
    cli.py                         # CLI entry point
    constants.py                   # Centralized constants
  notebooks/
  docs/
  tests/
  pyproject.toml
  README.md
```

---

## 5. Short-Term Analysis Features (No New Models Needed)

These derive directly from existing 68-point landmarks. **Ship at launch.**

### 5a. Landmark-Based Kinematics

```python
from primateface.analysis.kinematics import (
    mouth_aperture,       # distance between upper/lower lip landmarks (51-57 or 62-66)
    eye_aperture,         # distance between upper/lower eyelid landmarks
    brow_height,          # brow landmarks relative to eye landmarks
    lip_smack_detector,   # detect rapid mouth open/close cycling in video
    jaw_displacement,     # chin landmark movement over time
)

# Usage on video results
timeseries = primateface.analysis.kinematics.extract_all(video_results)
# → DataFrame with columns: frame, mouth_aperture, left_eye_aperture, right_eye_aperture,
#   left_brow_height, right_brow_height, jaw_displacement
```

Already demonstrated in PrimateFace paper: unsupervised lip-smack detection via landmark kinematics clustering.

### 5b. Head Pose (Yaw/Pitch/Roll)

Standard approach: `cv2.solvePnP` with a 3D reference face model and 2D landmark correspondences.

```python
face.head_pose  # → [yaw, pitch, roll] in degrees
```

Challenge: need a species-appropriate 3D reference. Options:
1. Use a generic primate face model (good enough for most species)
2. Use species-specific references for major clades (catarrhines vs. platyrrhines vs. strepsirrhines)
3. Derive approximate pose from landmark PCA (less accurate but model-free)

### 5c. Facial Symmetry

**What it is**: Comparing left vs. right landmark distances to the facial midline. Fluctuating asymmetry (FA) is proposed as a biomarker of developmental stress — more symmetric faces = healthier development. Evidence is mixed but influential (Little et al. 2012: FA negatively correlated with health in rhesus macaques).

**How it works** with dlib 68 landmarks:
1. Fit midline through nose/chin landmarks (28, 31, 34, 9)
2. For each left-right pair (e.g., left eye corner ↔ right eye corner):
   - `asymmetry_i = |d_left_to_midline - d_right_to_midline|`
3. Normalize by face size (interocular distance)
4. Average across all pairs → single FA score

Left-right pairs in 68-pt scheme: jaw (1-8 ↔ 17-10), eyebrows (18-22 ↔ 27-23), eyes (37-42 ↔ 46-43), mouth corners, etc.

```python
face.symmetry  # → float (0 = perfect symmetry, higher = more asymmetric)
```

With 230K+ annotated faces, PrimateFace could enable the **largest comparative FA study across primates ever**.

### 5d. Face Quality Assessment

Simple heuristics, no model needed:
- **Blur score**: Laplacian variance of face crop
- **Face size**: bbox area relative to image (too small = unreliable landmarks)
- **Visibility ratio**: fraction of keypoints with visibility > threshold
- **Aspect ratio**: degenerate bboxes
- **Brightness**: mean pixel intensity of face crop

```python
face.quality  # → float (0-1, higher = better quality)
```

### 5e. AU Proxy from Landmarks (Geometric FACS)

Not full AU detection, but landmark-derived approximations:

| Proxy | Landmarks used | Approximates |
|-------|---------------|--------------|
| Mouth aperture | Upper lip (51) ↔ Lower lip (57) | AU25 (Lips Part), AU26 (Jaw Drop) |
| Lip stretch width | Mouth corners (49 ↔ 55) | AU20 (Lip Stretcher) |
| Brow raise | Brow (19-24) relative to eye (37-46) | AU1/2 (Brow Raiser) |
| Eye aperture | Upper eyelid ↔ Lower eyelid | AU5 (Upper Lid Raiser), AU43/45 (Blink) |
| Nose wrinkle zone | Nose landmarks (31-36) displacement | AU9 (Nose Wrinkler) — very rough |
| Lip pucker | Mouth width / mouth height ratio | AU18/22 (Pucker/Funneler) |

These are NOT real FACS coding — they're geometric proxies. But they're immediately useful for:
- Vocalization analysis (mouth aperture dynamics)
- Blink detection (eye aperture)
- Lip-smack detection (rapid mouth cycling)
- Gross expression classification (open mouth threat vs. neutral)

---

## 6. Medium-Term Features (Need Models or Data)

### 6a. Sex & Age Estimation: Architecture

**Key insight from human face literature**: Classification into bins + expected value (DEX approach) beats regression for age. And frozen pretrained features + lightweight head is the simplest transfer approach.

**Recommended approach (3 phases):**

```
Phase 1: DINOv2 linear probe (hours, simplest)
  - Extract DINOv2-ViT-B features from MFD (29K mandrill face crops, frozen backbone)
  - Sex: linear classifier (binary)
  - Age: linear head with ~46 bins (0.5yr resolution, 0-23yr) + DEX expected value
  - Cross-validate by INDIVIDUAL ID (critical to avoid data leakage)
  - DINOv2 is species-agnostic — already in PrimateFace pipeline

Phase 2: Fine-tune if DINOv2 isn't good enough
  - Option A: nateraw/vit-age-classifier (HuggingFace, ViT pretrained on human FairFace age bins)
  - Option B: MiVOLO v2 (SOTA human age, MAE 3.65, Apache 2.0, HuggingFace: iitolstykh/mivolo_v2)
  - Option C: InsightFace attribute model (MobileNet-0.25, 0.3M params, tiny + fast)

Phase 3: Cross-species generalization
  - Train on mandrill → test on chimp (zero-shot transfer)
  - Train on mandrill + chimp → test on PrimateFace genera
  - Tells us if features are species-generalizable
```

**DEX technique** (from DeepFace, dominant approach):
- Softmax over N age bins → expected value = Σ(p_i × bin_center_i)
- For mandrills (0-23yr): ~46 bins at 0.5yr resolution
- For chimps (0-45yr): ~90 bins at 0.5yr resolution
- Consistently beats direct regression due to richer gradient signal

**Reusable human models/tools:**

| Model | What | Params | License | HF Downloads |
|-------|------|--------|---------|-------------|
| `nateraw/vit-age-classifier` | ViT for 9 age bins (FairFace) | 85.8M | -- | 303K+ |
| `iitolstykh/mivolo_v2` | VOLO-D1 age+gender | 28.8M | Apache 2.0 | 5M+ |
| InsightFace attribute | MobileNet-0.25 age+gender ONNX | 0.3M | MIT | -- |
| FairFace ResNet-34 | Age (9 bins) + gender + race | ~21M | -- | -- |

**Data quality filters for MFD training:**
- Quality ≥ 2: 25,223 images (85% of dataset)
- Frontal view (FaceView=1): 26,846 images (91%)
- Combined (quality ≥ 2 AND frontal): ~22K images
- Exclude sex="unknown" (110 images)

### 6b. Sex Classification — Data

**Feasibility**: HIGH for sexually dimorphic species.

| Dataset | Species | Images | Sex labels | Status |
|---------|---------|--------|-----------|--------|
| **Mandrillus Face DB** | Mandrill | 29,495 | 16K F / 13K M / 110 unknown | **Downloaded** |
| **CTai** | Chimp (wild) | 5,078 | 2.3K M / 2.3K F / 372 unknown | **Downloaded** |
| **Duke Lemur Center** | 27 lemur spp. | Demographics only | YES | Collaboration needed |
| **Cayo Santiago** | Rhesus macaque | Colony records | YES | Collaboration (Platt lab) |

### 6c. Age Estimation — Data

| Dataset | Species | Age type | Range | Status |
|---------|---------|----------|-------|--------|
| **Mandrillus Face DB** | Mandrill | Continuous (DOB + photo date) | 0-23yr | **Downloaded** |
| **CTai** | Chimp (wild) | Continuous + 5 age groups | 0-45yr | **Downloaded** |
| **Duke Lemur Center** | 27 lemur spp. | DOB available | Varies | Collaboration needed |

**CTai age groups**: Infant / Juvenile / SubAdult / Adult / Elderly

**Unsupervised approach (no labels):**
- PCA on Procrustes-aligned landmarks → infant/juvenile vs. adult separation
- Infants: rounder faces, proportionally larger eyes, different face-to-head ratios
- Validate against labeled data later

### 6c. Genus Classification

Already prototyped with VLMs (SmolVLM, InternVL2-2B) in `demos/classify_genus.py`. Polish and integrate into the main `Face` object.

### 6d. Individual Re-Identification

**Current**: ArcFace embeddings → SVM → 86% accuracy across 62 macaque individuals.

**MegaDescriptor** — first foundation model for wildlife re-ID (BVRA, Czech Technical University, WACV 2024):
- Swin Transformer or DINOv2 backbone, trained on 52+ wildlife re-ID datasets
- Outputs embedding vector; compare with cosine similarity + k-NN
- Available via `timm`: `timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True)`
- Species-agnostic but whole-body, not face-specific
- License: CC-BY-NC-4.0 (non-commercial only — check compatibility)

**Plan**: Benchmark MegaDescriptor against ArcFace and DINOv2 on PrimateFace face crops in the **App2: Macaque Face Recognition** Colab notebook. Compare:
1. ArcFace (current, human-pretrained)
2. MegaDescriptor-L-384 (wildlife-pretrained, Swin Transformer)
3. MegaDescriptor-DINOv2-518 (wildlife-pretrained, DINOv2 backbone)
4. DINOv2-base (general-purpose, already computed)
5. Fine-tuned ArcFace on PrimateFace crops (if time permits)

### 6e. Gaze Estimation

Already prototyped with Gazelle (DINOv2-ViT-L14). Challenge: sclera visibility varies by species (e.g., cooperative eye hypothesis — humans have unique visible sclera; many primates don't).

---

## 7. Research Frontier Features (Longer-Term)

### 7a. Automated FACS / Action Unit Detection

**The opportunity**: All primate FACS coding is currently manual. Only one prototype exists (Morozov et al. 2021: automated MaqFACS, 72-84% accuracy, only 6 of ~19 AUs, PCA+KNN, not deep learning).

**Bridget Waller is a PrimateFace co-author** — direct access to FACS expertise and coding manuals.

Existing primate FACS systems (all freely available at [animalfacs.com](https://animalfacs.github.io/AnimalFACS/)):

| System | Species | AUs | Year | Key author |
|--------|---------|-----|------|------------|
| ChimpFACS | *Pan troglodytes* | ~15 | 2007 | Vick, Waller |
| BonoboFACS | *Pan paniscus* | 22 AUs + 3 ADs + 3 EADs | 2025 | Correia-Caeiro |
| MaqFACS | *Macaca* spp. | 19 AUs + 15 ADs | 2010/2015 | Parr, Waller |
| OrangFACS | *Pongo* spp. | 17 AUs + 7 ADs | 2012 | Caeiro |
| GibbonFACS | Hylobatidae | 18 movements | 2012 | Waller |
| GorillaFACS | *Gorilla* spp. | 28 AUs + 14 ADs | 2024 | Correia-Caeiro |
| CalliFACS | *Callithrix jacchus* | 15 AUs + 15 ADs + 3 EADs | 2022 | Correia-Caeiro |

**Path forward**:
1. Ship landmark-based AU proxies first (Section 5e) — immediately useful
2. Collect FACS-coded training data (via Waller collaboration)
3. Train AU classifiers (Py-Feat-style XGBoost on HOG features, or CNN)
4. Start with macaques (most data, most demand)

### 7b. Expression Classification

Key primate expressions: silent bared-teeth display, play face, open-mouth threat, lip smack, fear grimace, pout. No universal ontology across species. Emerging work: Fang et al. 2025 achieved 94.5% on golden snub-nosed monkey expressions using deep learning (EfficientNet).

### 7c. Pain/Grimace Scale Automation

Cynomolgus Macaque Grimace Scale is validated (Nature Scientific Reports 2023). Currently manual. Automation would be high-impact for animal welfare.

### 7d. 2D Statistical Shape Model (Practical "PrimateFLAME")

**FLAME** is a 3D parametric face model (Max Planck, 2017) that describes any human face as ~400 numbers: 300 for identity shape, 100 for expression, ~15 for pose. Built from 3,800 3D face scans. Lets you disentangle "who is this" from "what are they doing with their face."

A full 3D version for primates is unrealistic now (would need hundreds of 3D primate face scans, 4D expression capture, cross-species mesh topology). **But a 2D version is very feasible**:

- PCA on Procrustes-aligned 68-landmark configurations
- With 230K+ annotated faces across species, this would learn axes of variation:
  - Species shape differences (macaque face vs. lemur face vs. chimp face)
  - Within-species identity (individual face shapes)
  - Expression (mouth open vs. closed, brow position)
- Same conceptual benefits as FLAME: shape/expression disentanglement, dense correspondence, data augmentation
- **This could be a standalone publication** — "A 2D Statistical Face Model for Primates"

### 7e. Multi-Animal Face Tracking

Identity persistence across video frames. Significant scope but essential for social network analysis.

### 7f. Real-Time / Streaming Mode

DLC-Live equivalent for PrimateFace. ONNX makes this feasible.

---

## 8. Framework Interoperability

### Import/Export Matrix

| Format | Import | Export | Priority | Use case |
|--------|--------|--------|----------|----------|
| **COCO JSON** | Yes (native) | Yes (native) | P0 | Standard format |
| **CSV** | Yes | Yes | P0 | Universal |
| **DLC project** | Yes | Yes | P1 | Largest animal behavior user base |
| **Lightning Pose** | Yes | Yes | P1 | Same DLC CSV format — `to_dlc_csv`/`from_dlc` work directly |
| **SLEAP labels** | Yes | Yes | P1 | Second largest, growing |
| **BORIS** | No | Yes | P2 | Behavioral ethogram coding |
| **NWB** | No | Yes | P2 | Neuroscience data standard |

### API

```python
primateface.io.to_coco(results, "output.json")
primateface.io.to_deeplabcut(results, "dlc_project/")
primateface.io.to_sleap(results, "output.slp")
primateface.io.to_csv(results, "output.csv")
primateface.io.to_boris(results, "events.csv", event_type="face_visible")

results = primateface.io.from_deeplabcut("body_project/")
combined = primateface.io.merge_body_face(body_pose, face_pose)
```

---

## 9. Active Learning & Training Loop

### The User Story

```
Step 1: Zero-shot inference → PrimateFace detects + estimates landmarks
Step 2: Review & correct in GUI (50-100 frames, ~30 min)
Step 3: Fine-tune (one command, ~15 min on GPU or Colab)
Step 4: Re-run → much better results
Step 5: Export to CSV/DLC/SLEAP/BORIS
```

### API

```python
# Fine-tune
trainer = primateface.Trainer(base_model="hrnet-w18-dark")
trainer.finetune(train_json="corrected.json", img_dir="frames/", epochs=50)

# Full active learning loop
primateface.active_learning_loop(
    video="tamarin_video.mp4", output_dir="project/",
    max_rounds=3, review_mode="gui"
)
```

### Training Backend

MMPose primary (what PrimateFace models were trained with), behind a clean API so users never touch configs. Lightweight PyTorch path (freeze backbone, train head) for Colab.

---

## 10. Model Distribution

### HuggingFace Hub + local cache

Models at `fparodi/primateface-models` on HF, cached at `~/.primateface/models/`. ONNX format for zero-framework-dependency inference.

**Model packs**: `default` (Cascade R-CNN + HRNet-W18), `fast` (YOLO + lightweight pose), `accurate` (larger models).

### Installation

```bash
pip install primateface                  # core + onnxruntime
pip install primateface[gpu]             # + onnxruntime-gpu
pip install primateface[all]             # + training, GUI, DINOv2
```

---

## 11. Colab Notebook Strategy

### Principles
1. Every notebook works with "Run All"
2. First cell: `!pip install primateface`
3. Keep it short (10-15 cells for tutorials)
4. Visual outputs in every notebook

### Priority

| # | Notebook | Status | Priority |
|---|----------|--------|----------|
| **NEW** | **Quick Start: Detection + Landmarks** | -- | **P0** |
| 1 | Lemur Face Visibility Time-Stamping | Exists | P0 (polish) |
| 2 | Macaque Face Recognition (**+ MegaDescriptor benchmark**) | Exists | P0 (polish + extend) |
| **NEW** | **Facial Kinematics & Symmetry Analysis** | -- | **P1** |
| **NEW** | **Fine-Tune on Your Species** | -- | P1 |
| 3 | Howler Vocal-Motor Coupling | Planned | P1 |
| 4 | Human Infant Social Gaze Tracking | Exists | P1 (polish) |
| **NEW** | **Age & Sex from Landmarks** | -- | **P1** |
| 5 | Data-Driven Discovery of Facial Actions | Planned | P2 |
| 6 | Cross-Subject Neural Decoding | Planned | P2 |
| **NEW** | **Kin Recognition from Embeddings** | -- | P3 |

**App2 extension**: Benchmark ArcFace vs MegaDescriptor vs DINOv2 on PrimateFace face crops for individual re-ID. Felipe is building this now.

**Age & Sex notebook**: 7-part structure — landmarks→sex, landmarks→age, DINOv2 comparison, human model baselines, cross-species transfer. PrimateFace innovation = landmark-based demographics, not another DINOv2 fine-tune. Data: MFD (mandrill, 29K) + CTai (chimp, 5K).

**Kin Recognition notebook** (future): Low-dimensional face embeddings → show related individuals cluster → kinship prediction from embedding distance. Ref: Charpentier et al. 2020 "Same father, same face" in mandrills (Science Advances).

---

## 12. Development Phases (Revised)

### Phase 0: Foundation (weeks 1-2)
- [ ] Convert best detection + pose models to ONNX
- [ ] Create `fparodi/primateface-models` HuggingFace repo
- [ ] Set up `src/primateface/` package structure
- [ ] Implement `Face` dataclass with all fields
- [ ] Model download + caching (`~/.primateface/models/`)

### Phase 1: Core API + Landmark-Derived Analysis (weeks 3-6)
- [ ] `PrimateFace` class with `.detect()`, `.analyze()`
- [ ] ONNX backend for detection + pose
- [ ] Input handling (path, numpy, PIL)
- [ ] Visualization (`pf.draw()`)
- [ ] **`analysis.kinematics`**: mouth aperture, eye aperture, brow height, lip-smack detection
- [ ] **`analysis.head_pose`**: yaw/pitch/roll from solvePnP
- [ ] **`analysis.symmetry`**: FA from landmark pairs
- [ ] **`analysis.quality`**: blur, occlusion, size, visibility ratio
- [ ] **AU proxies**: landmark-geometric FACS approximations
- [ ] Module-level convenience functions (`primateface.face_locations()`)

### Phase 2: CLI + Distribution + Notebooks (weeks 7-8)
- [ ] CLI entry points (`primateface detect`, `primateface analyze`)
- [ ] Publish to PyPI (`pip install primateface`)
- [ ] Quick Start Colab notebook
- [ ] Facial Kinematics & Symmetry Colab notebook
- [ ] Deploy HuggingFace Space with Gradio demo
- [ ] Polish App1 (Lemur) and App2 (Macaque Re-ID + MegaDescriptor benchmark)

### Phase 3: Sex, Age, Genus, Re-ID (weeks 9-14)
- [ ] **Sex classification**: train on CTai/CZoo chimps, extend to macaques via collaborations
- [ ] **Age class**: unsupervised morphometric clustering first, then supervised if data available
- [ ] **Genus classification**: polish existing VLM-based module
- [ ] **Re-ID**: benchmark MegaDescriptor vs ArcFace vs DINOv2, pick best, integrate
- [ ] Sex & Age Classification Colab notebook
- [ ] Data acquisition: reach out to Cayo Santiago, NPRCs for labeled images

### Phase 4: Interop + I/O (weeks 15-16)
- [ ] `primateface.io` module
- [ ] COCO JSON, CSV (P0)
- [ ] DLC project import/export (P1)
- [ ] SLEAP labels import/export (P1)
- [ ] Body+face merge utility

### Phase 5: Training + Active Learning (weeks 17-20)
- [ ] `primateface.training.Trainer` class
- [ ] Confidence-based frame sampling
- [ ] Active learning loop orchestrator
- [ ] Lightweight fine-tuning (freeze backbone, Colab-friendly)
- [ ] Full MMPose-backed training
- [ ] Fine-Tune on Your Species Colab notebook

### Phase 6: Ecosystem (weeks 21-24)
- [ ] Port DINOv2 module into `primateface.analysis.embeddings`
- [ ] Port landmark converter into `primateface.tools.converter`
- [ ] Port evaluation framework into `primateface.evaluation`
- [ ] Backend swapping (ONNX default, MMDet/MMPose/Ultralytics optional)
- [ ] BORIS export, NWB export
- [ ] Polish remaining Colab notebooks

### Phase 7: Action Units, Parsing, Blendshapes (weeks 25-28)

#### 7a. Action Unit Proxies
- [ ] Map landmark kinematics → FACS-like AU signals:
  - AU1/AU2 (brow raise) ← `brow_height`
  - AU26/AU27 (jaw drop) ← `mouth_aperture`
  - AU43 (eye closure) ← `eye_aperture`
  - AU10 (upper lip raiser) ← upper lip landmark displacement
  - AU20 (lip stretcher) ← `mouth_width`
- [ ] `face.aus` property returning `Dict[str, float]` of AU proxy intensities
- [ ] Long-term: train supervised AU detector on MaqFACS/ChimpFACS labeled data (Waller collaboration)
- [ ] Spatiotemporal AU detection for video (transformer or temporal graph net)

#### 7b. Face Parsing / Segmentation
- [ ] Use SAM2/3 with open-vocab prompts ("eye", "nose", "mouth", "skin") for region segmentation
- [ ] `face.parse()` → dict of binary masks per region
- [ ] Use cases: fur color analysis, sexual dimorphism (color vs geometry), occlusion detection
- [ ] No primate-specific face parsing model exists — SAM generalizes well

#### 7c. Blendshape Mapping
- [ ] Map kinematics → ARKit-style standardized names:
  - `jawOpen` ← `mouth_aperture`
  - `eyeBlink{Left,Right}` ← `eye_aperture`
  - `browInnerUp` ← `brow_height`
  - `mouthSmile{Left,Right}` ← mouth corner displacement
- [ ] `face.blendshapes` → `Dict[str, float]` with 0.0–1.0 range
- [ ] Document which of the 52 ARKit blendshapes we can approximate and which we can't

### Phase 8: Research Frontier (weeks 29+)
- [ ] Expression classification (threat, fear grimace, lip smack, yawn, neutral)
- [ ] Pain/grimace scale automation
- [ ] 2D statistical shape model ("PrimateFLAME-2D")
- [ ] Multi-animal face tracking
- [ ] Real-time / streaming mode

---

## 12b. Video Processing & Inference Performance

### Current Bottlenecks
- **Serial frame processing**: `PrimateFace.analyze()` runs detection + pose per frame sequentially
- **Video decoding**: `cv2.VideoCapture` is single-threaded, decoding often slower than GPU inference
- **No batching**: Each frame is a separate forward pass through the detector and pose model
- **Memory**: Full video keypoints in memory for timeseries analysis

### Short-term: Batched Inference
```python
# Target API
pf = PrimateFace()
results = pf.analyze_video("video.mp4", batch_size=8)
# → yields Face objects per frame, batches GPU inference
```

**Implementation approach:**
1. **Threaded video decoder**: Separate thread reads frames into a queue while GPU processes the previous batch. Use `cv2.VideoCapture` in a producer thread or `decord` (GPU-accelerated video decoding).
2. **Batched detection**: MMDetection already supports batch inference via `inference_detector` with multiple images. Collect N frames → single forward pass.
3. **Batched pose**: MMPose `inference_topdown` processes all bboxes from one frame, but across frames we can batch. Alternatively, batch all crops from N frames into one pass.
4. **Streaming output**: Yield results per frame (or per batch) rather than accumulating everything in memory.

### Medium-term: Efficient Video Backends
- **decord** (`pip install decord`): GPU-accelerated video decoding, 2-3x faster than OpenCV for sequential reads. Apache 2.0 license.
- **PyAV** (`pip install av`): FFmpeg Python bindings, better codec support, can seek efficiently.
- **TorchVision video reader**: `torchvision.io.read_video()` with hardware acceleration.

### Parallelization Strategies
1. **Multi-video**: `pf.analyze_videos(["v1.mp4", "v2.mp4"], n_workers=4)` — process multiple videos in parallel across GPUs
2. **Frame-level**: Split long video into chunks, process on different GPUs, merge results
3. **Pipeline parallelism**: Decode on CPU → detect on GPU0 → pose on GPU1 (if multi-GPU)

### Performance Targets
| Video length | Current (est.) | Target with batching | Target with ONNX |
|---|---|---|---|
| 30s @ 30fps (900 frames) | ~3 min | ~30s | ~15s |
| 5 min @ 30fps (9000 frames) | ~30 min | ~5 min | ~2 min |
| 1 hr @ 30fps (108K frames) | ~6 hr | ~1 hr | ~25 min |

### Relevant Existing Code
- `demos/process.py`: `PrimateFaceProcessor.process_video()` — already processes videos frame-by-frame with optional smoothing
- `gui/core/models.py`: `ModelManager.distribute_gpus()` — multi-GPU device assignment
- `scripts/run_parallel_inference.py` — existing parallel inference script (check for reusable patterns)

### Open Questions
- Decord vs PyAV vs OpenCV: which to default to?
- Should `analyze_video()` return a generator (streaming) or a list (all-at-once)?
- How to handle multi-face tracking across frames? (Currently each frame is independent — no identity persistence)
- ONNX export: mmdeploy supports Cascade R-CNN + HRNet but needs testing

---

## 13. Data Acquisition Strategy

The biggest bottleneck for sex/age classification is **labeled data**. Strategy: maximize taxonomic diversity — mandrill + chimp + lemur covers Old World monkey + great ape + strepsirrhine.

### Tier 1: Download Now (publicly available, confirmed sex+age)

| Dataset | Species | Images | Individuals | Sex | Age | License | Access |
|---------|---------|--------|-------------|-----|-----|---------|--------|
| **Mandrillus Face DB** | Mandrill | 29,495 | 397 (191F, 203M) | YES | YES (exact DOB, 0-23yr) | CC BY | [Zenodo 10.5281/zenodo.7467318](https://doi.org/10.5281/zenodo.7467318) |
| **C-Zoo** | Chimp (captive, Leipzig) | ~2,109 | 24 | YES | YES (float + age group) | Non-commercial | [github.com/cvjena/chimpanzee_faces](https://github.com/cvjena/chimpanzee_faces) |
| **C-Tai** | Chimp (wild, Cote d'Ivoire) | ~5,078 | 78 | Partial | Partial | Non-commercial | Same GitHub |
| **PetFace chimp subset** | Chimp | -- | 446 | YES | YES | -- | [dahlian00.github.io/PetFacePage](https://dahlian00.github.io/PetFacePage/) |

**Action items:**
- [x] ~~Download CTai/CZoo chimp datasets~~ → via `wildlife-datasets` or GitHub
- [ ] **Download Mandrillus Face DB from Zenodo** — 29K images, CC BY, best single dataset
- [ ] Download PetFace chimp subset
- [ ] Inspect all datasets: verify sex/age columns, image quality, face crop consistency

### Tier 2: Collaboration-based (confirmed records exist, need access)

| Source | Species | Sex | Age | Contact |
|--------|---------|-----|-----|---------|
| **Duke Lemur Center** | 27 lemur spp. (3,700+ individuals) | YES | YES (DOB) | [lemur.duke.edu](https://lemur.duke.edu/duke-lemur-center-database/) — registration required |
| **Cayo Santiago** | Rhesus macaque (~1,400 current) | YES | YES (DOB since 1938) | Platt lab (PrimateFace co-PI) |
| **NPRCs** | Multiple macaque spp. | YES | YES | California, Oregon, Emory |
| **Schofield/Oxford** | Chimp (Bossou, 23 ind.) | YES (96% acc) | -- | Contact authors |

**Duke Lemur Center is the priority** — would give us strepsirrhine coverage (lemurs are completely unrepresented in labeled face data). They have demographic records for every individual since the center opened. Need to pair face images with their database.

### Tier 3: Medium-term (pseudo-labeling & unsupervised)
- [ ] Train sex classifier on Tier 1 data (mandrill + chimp) → pseudo-label PrimateFace's 230K images
- [ ] Train age-class classifier on mandrill (continuous age) + chimp (age groups)
- [ ] Unsupervised morphometric clustering as age proxy: PCA on Procrustes-aligned landmarks separates infant/juvenile from adult shapes
- [ ] For monomorphic species (many lemurs): report sex as "unknown" with confidence score

### Taxonomic Coverage of Current Strategy

| Clade | Species | Source | Status |
|-------|---------|--------|--------|
| **Old World monkey** | Mandrill | Mandrillus Face DB | Download now |
| **Great ape** | Chimpanzee | C-Zoo + C-Tai + PetFace | Download now |
| **Strepsirrhine** | 27 lemur spp. | Duke Lemur Center | Collaboration needed |
| **Old World monkey** | Rhesus macaque | Cayo Santiago + NPRCs | Collaboration needed |

This gives 3 major primate clades covered in Tier 1 (mandrill + chimp), with lemurs in Tier 2.

---

## 14. Open Questions

1. **PyPI name**: `primateface`? `primate-face`? Check availability.
2. **ONNX conversion**: Can Cascade R-CNN + HRNet cleanly convert? Test with mmdeploy.
3. **MegaDescriptor license**: CC-BY-NC-4.0 — compatible with MIT? May need to keep as optional dependency.
4. **Python version**: Drop 3.8? Most modern ML tools require 3.9+.
5. **Training backend**: MMPose-only or also pure PyTorch for fine-tuning?
6. **Active learning in Colab**: Can the GUI review loop work in a notebook? (Gradio inline?)
7. **DLC format version**: v2 vs v3 project formats — which to target?
8. **SLEAP I/O**: Use `sleap-io` (lightweight, pip-installable) directly?
9. **Multi-animal tracking**: v2 scope or v3?
10. **Repo rename**: `primateface_oss` → `primateface`? When?
11. **Head pose 3D reference**: Single generic primate model or species-specific?
12. **Symmetry validation**: Landmark precision vs. FA signal size — is our landmark accuracy sufficient?
13. **Sex classification for monomorphic species**: Report "unknown" or attempt with lower confidence?
14. **Community platform**: Discord? GitHub Discussions? Where do primatologists hang out?

---

## 15. Reference: Competitors

| Library | First line | Killer feature | Stars |
|---------|-----------|---------------|-------|
| **face_recognition** | "The world's simplest facial recognition api" | 1-line detection, zero config | 56K |
| **InsightFace** | "State-of-the-art 2D and 3D Face Analysis Project" | `FaceAnalysis()` app, ONNX model packs | 28K |
| **DeepFace** | "A Lightweight Face Recognition and Facial Attribute Analysis Library" | Backend swapping, REST API | 22K |
| **Py-Feat** | "Python Facial Expression Analysis Toolbox" | AU detection, Fex DataFrame | 600 |
| **DeepLabCut** | "A toolbox for state-of-the-art markerless pose estimation" | SuperAnimal model zoo, active learning | 5.5K |
| **SLEAP** | "A deep learning framework for multi-animal pose tracking" | GUI-first AL loop, 800+ FPS | 1.4K |
| **MegaDescriptor** | "Foundation model for wildlife re-identification" | Multi-species re-ID, timm/HF | 160 |

Key insight: face_recognition (56K stars) is also the simplest. **Simplicity wins adoption.**

---

## 16. Success Metrics

| Metric | Target | How to measure |
|--------|--------|----------------|
| PyPI installs | 1,000/month within 6 months | PyPI stats |
| GitHub stars | 500 within 1 year | GitHub |
| HF Space usage | 100 unique users/week | HF analytics |
| Colab notebook runs | 50/week | Colab metrics |
| Citations | 20+ papers using PrimateFace within 1 year | Google Scholar |
| Community | Active GitHub Discussions | GitHub |
| Contributors | 5+ external PRs | GitHub |

---

## Appendix A: Nature Methods "Resource" Criteria

From [Nature Methods editorial](https://www.nature.com/articles/s41592-023-01926-8):

> "Resources describe a collection of tools or a large dataset of broad utility, interest and significance to a field of research."

PrimateFace hits both computational platform (subtype 2) and large dataset (subtype 3).

## Appendix B: Primate FACS Reference

All systems freely available at [animalfacs.com](https://animalfacs.github.io/AnimalFACS/). Key collaborator: **Bridget Waller** (PrimateFace co-author, NTU, central figure in all primate FACS development).

## Appendix C: Wildlife Re-ID Landscape

| Project | Type | Species | pip-installable? | Active? |
|---------|------|---------|-----------------|---------|
| MegaDescriptor | Foundation model | 52+ datasets | `timm` | Yes |
| PrimNet | Research code | 14 primate spp | No | No |
| LemurFaceID | Research code | Lemurs | No | No |
| MacaqueFaces | Dataset | Rhesus | Via wildlife-datasets | Low |
| **PrimateFace** | **Toolkit + dataset** | **60+ genera, 68 face kpts** | **v2 goal** | **Yes** |

---

## 17. Progress Log (2026-03-17)

### Completed (merged to main on felipe-parodi/PrimateFace_os)

**Core API (Phase 0-1):**
- [x] `primateface/` package consolidated — `PrimateFace` class, `Face` dataclass, `io` module
- [x] 3-line API: `pf = PrimateFace(); faces = pf.analyze("img.jpg")`
- [x] `analysis/` module: kinematics, symmetry, head_pose, quality — all shipped
- [x] HF model hosting at `fparodi/primateface-models` with auto-download via `download_models_hf()`
- [x] `model_registry.py` — centralized model metadata
- [x] `Face` dataclass with bbox, keypoints, head_pose, quality, symmetry, mouth_aperture
- [x] ViTPose backend support (alongside HRNet)
- [x] Face embeddings + verification module (`_embedding.py`)
- [x] CLI entry points (`cli.py`)

**Interop (Phase 4):**
- [x] `primateface.io` — DLC, SLEAP, NWB, Lightning Pose import/export
- [x] COCO JSON native

**Notebooks (Phase 2):**
- [x] `quickstart.ipynb` — 3-line detection + pose on single image
- [x] `lemur_video_timestamping.ipynb` — video face detection + BORIS export
- [x] `macaque_face_recognition.ipynb` — 3-way embedding benchmark (ArcFace 84.1%, DINOv2 83.0%, MegaDescriptor 74.7%)
- [x] `howler_vocal_motor_coupling.ipynb` — uses `analysis.extract_timeseries()` for mouth aperture
- [x] `macaque_gaze_following.ipynb` — runs on human infant video (Adobe Stock twin babies), validates cross-species generalization
- [x] `landmark_demographics.ipynb` — age/sex from 68-pt landmarks on mandrill + chimp
- [x] `facial_action_discovery.ipynb` — unsupervised facial action discovery (App5)
- [x] All notebooks auto-download models from HF on first run
- [x] Publication-quality figures (Nature style, PNG+SVG+PDF)
- [x] Renamed from `AppN_*` to descriptive `species_task.ipynb`

**Documentation:**
- [x] Colab references removed (local-only execution)
- [x] notebooks/README.md, demos/README.md, main README.md all updated
- [x] Cross-references in docs/ updated to new notebook names

### Key Results

**3-way embedding benchmark** (same PrimateFace-aligned 112×112 crops, SVM RBF C=10, 70/30 split):
| Model | Domain | Dim | Accuracy | Balanced Acc |
|-------|--------|-----|----------|-------------|
| ArcFace (buffalo_l) | Human face rec | 512 | 84.1% | 78.0% |
| DINOv2 (ViT-S/14) | General vision | 384 | 83.0% | 79.4% |
| MegaDescriptor-L-384 | Wildlife re-ID | 1536 | 74.7% | 65.1% |

**Takeaway**: PrimateFace detection + alignment is the hard part. The embedding model is plug-and-play. ArcFace and DINOv2 are interchangeable; MegaDescriptor (despite being wildlife-specific) lags, possibly because it was trained on whole-body re-ID, not face crops.

### Not Yet Done

- [ ] ONNX model conversion (required for `pip install primateface` without mmdet/mmpose)
- [ ] PyPI publish
- [ ] Repo rename `primateface_oss` → `primateface`
- [ ] HuggingFace Space
- [ ] Sex/age models (beyond landmark-based demo)
- [ ] Genus classification integration into `Face` object
- [ ] Batched video inference
- [ ] Module-level convenience functions (`primateface.face_locations()`)
- [ ] Lemur notebook needs re-run (papermill hangs on video processing — likely I/O bottleneck)

---

## 18. Multi-Animal Face Tracking (Expanded)

### The Problem

Current PrimateFace processes each frame independently — no identity persistence. For social behavior studies (grooming, gaze following, dominance interactions), you need to know *which* individual is *which* across frames.

### Current State

- App1 (lemur) uses **ByteTracker** for simple IoU-based tracking
- App4 (gaze) uses center-distance assignment (2 individuals only)
- No general-purpose multi-face tracking module exists in the API

### Proposed Architecture

```python
# API
pf = PrimateFace()
tracks = pf.track_video("video.mp4")
# → Dict[int, List[Face]]  — track_id → list of Face per frame

# Or streaming
for frame_faces in pf.track_video("video.mp4", stream=True):
    for face in frame_faces:
        print(f"Track {face.track_id}, frame {face.frame_idx}")
```

### Approach (3 tiers of sophistication)

**Tier 1: IoU + embedding tracker (short-term)**
- ByteTrack/BoT-SORT for bbox tracking (already in App1)
- When track breaks (occlusion, exit/re-enter), use face embeddings (ArcFace) to re-associate
- Simple, works for <10 individuals in controlled settings
- Requires: bbox tracker + embedding model (both already available)

**Tier 2: Appearance-based re-ID (medium-term)**
- Extract face embeddings per detection
- Hungarian algorithm for frame-to-frame assignment (embedding distance + IoU)
- Gallery of known identities built incrementally
- Handles longer occlusions and entry/exit events
- Ref: FairMOT, BoTrack patterns adapted for faces

**Tier 3: Graph-based social tracking (research)**
- Build a graph: nodes = face detections, edges = same-identity associations
- GNN or transformer for identity propagation across frames
- Jointly model social interactions (who looks at whom, proximity, approach/withdraw)
- Ref: Social Force Models, attention-based MOT

### Key Challenges

1. **Primate faces look similar** — within-species variation is subtle compared to human faces
2. **Occlusion is frequent** — primates groom, huddle, and turn away constantly
3. **Group size** — field studies can have 20+ visible individuals; captive studies usually <10
4. **No ground truth** — need to create tracking benchmarks (annotated multi-primate videos)

### Downstream Applications

- **Social network analysis**: who associates with whom, dominance hierarchies
- **Behavioral bout detection**: grooming duration, play episodes
- **Attention dynamics**: gaze following over time (who watches whom first)
- **Mother-infant interactions**: nursing, carrying, proximity
- **Aggression/conflict**: chase sequences, displacement events

---

## 19. Cross-Species Generalization

### Evidence So Far

PrimateFace was trained on 230K+ images spanning 60+ primate genera. The gaze-following notebook now demonstrates it works on **human infants** (Adobe Stock twin babies video) without any retraining — detection, landmarks, and Gazelle gaze estimation all transfer.

### What This Means

The 68-point landmark model generalizes beyond its training distribution. This suggests:
1. Primate facial geometry is conserved enough that a pan-primate model works on humans
2. The detection model (Cascade R-CNN) is learning "face-like" features, not species-specific ones
3. Potential to extend to other mammals (carnivores, ungulates) with appropriate landmarks

### Validation Experiments to Run

- [ ] Systematic evaluation on human face benchmarks (300W, WFLW) using PrimateFace models
- [ ] Evaluate on non-primate faces (dogs, cats — PetFace dataset)
- [ ] Compare landmark accuracy: PrimateFace on humans vs. specialized human models
- [ ] Test on great ape species not in training set (bonobos, gibbons if underrepresented)

### Cross-Species Transfer for Demographics

The landmark_demographics notebook shows:
- Landmark geometry alone predicts mandrill age class at ~66.5% (4-class, chance=25%)
- DINOv2 reaches 81% for sex on mandrills
- Key question: do models trained on mandrills transfer to chimps? (Cross-species experiment in notebook)

---

## 20. Additional Feature Ideas

### 20a. Thermal / IR Face Detection

Thermal imaging is increasingly used in field primatology (body temperature, fever detection, stress). PrimateFace currently only handles RGB. Adding thermal/IR support would:
- Enable nocturnal primate studies (many strepsirrhines are nocturnal)
- Support health monitoring (fever, inflammation)
- Complement RGB for all-weather/low-light conditions

**Approach**: Fine-tune detection model on thermal primate face images. Challenge is data — very few labeled thermal primate face datasets exist.

### 20b. Comparative Morphometrics at Scale

With 230K+ annotated faces across 60+ genera, PrimateFace enables population-level morphometric analyses that were previously impossible:
- **Allometry**: How do face proportions scale with body size across species?
- **Sexual dimorphism quantification**: Which facial features are most dimorphic, and does this vary across clades?
- **Phylogenetic signal in face shape**: Do closely related species have more similar face shapes? (Procrustes distances vs. phylogenetic distances)
- **Ontogenetic trajectories**: How does face shape change with age (mandrill data has exact DOBs)?

This could be a standalone comparative biology paper — "Primate Facial Diversity at Scale" or similar.

### 20c. Integration with Audio/Vocal Analysis

The howler vocal-motor coupling notebook shows the potential. Expand to:
- **Automatic call segmentation** + face kinematics alignment
- **Multi-modal behavior annotation**: vocal + facial + postural
- **Speaker diarization for primates**: who is vocalizing, based on mouth aperture timing
- Integration with bioacoustics tools (e.g., OpenSoundscape, BirdNET patterns)

### 20d. Edge Deployment / Embedded Systems

For field researchers with camera traps or live monitoring:
- ONNX Runtime on Jetson Nano / Raspberry Pi
- Model distillation: train smaller student models from PrimateFace teacher
- INT8 quantization for mobile/embedded
- Target: real-time face detection on edge device, landmarks computed server-side or batch

### 20e. Longitudinal Individual Monitoring

Combine tracking + re-ID + demographics for long-term studies:
- Track individuals across days/weeks/months
- Build "face galleries" that grow over time
- Detect new individuals entering a group
- Monitor health changes via face quality, symmetry trends, body condition

### 20f. Social Attention Networks

Combine gaze estimation + multi-face tracking to build:
- **Attention matrices**: who looks at whom, how often, for how long
- **Social network graphs** weighted by visual attention
- **Dominance hierarchy inference** from gaze patterns (subordinates look at dominants more)
- **Joint attention detection**: two individuals looking at same target
- Ref: Shepherd et al. 2006, Emery 2000 on primate gaze and social cognition

---

## 21. Open Questions (Additions)

15. **ViTPose vs HRNet**: Which should be the default? ViTPose may be more accurate but heavier.
16. **Embedding model for re-ID**: ArcFace won the benchmark but is human-pretrained. Fine-tune on primate faces?
17. **Notebook execution**: Papermill hangs on video-heavy notebooks (lemur). Alternative: `nbconvert --execute` or `jupyter nbconvert`?
18. **Cross-species landmark consistency**: Are the same 68 points anatomically homologous across all primate genera? (Probably not perfectly — jaw shape varies hugely between lemurs and great apes)
19. **Cloud processing**: Offer a paid API or free tier for researchers without GPUs? (HF Inference API?)
20. **Data contribution pipeline**: How can other labs contribute annotated images? (HF datasets with community contributions?)
