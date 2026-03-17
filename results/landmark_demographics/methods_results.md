# Landmark-Based Demographic Estimation in Primates

**Status**: Pre-results (methods finalized, awaiting landmark extraction completion)
**Notebook**: `demos/notebooks/Landmark_Demographics.ipynb`
**Target**: Supplementary figure in PrimateFace paper

---

## Motivation

PrimateFace extracts 68-point facial landmarks across 60+ primate genera. We ask: **do these landmarks encode sufficient geometric information to predict age and sex?** If so, demographic estimation becomes a zero-cost byproduct of the existing pipeline — no additional model, no GPU at inference, fully interpretable features.

This is complementary to image-based approaches (e.g., Renoult et al. 2025 fine-tuned DINOv2 on mandrill face images, achieving MAE = 0.58 years). Our contribution is landmark-based: interpretable, lightweight, and cross-species compatible.

---

## Methods

### Datasets

| Dataset | Species | Images | Individuals | Sex labels | Age labels | Source |
|---------|---------|--------|-------------|-----------|-----------|--------|
| Mandrillus Face Database (MFD) | *Mandrillus sphinx* | 29,495 | 397 | M/F/Unknown | Continuous (DOB + photo date) | Zenodo, CC BY |
| CTai Chimpanzee Faces | *Pan troglodytes* | 5,078 | 78 | M/F/Unknown | Continuous + age groups | Freytag et al. 2016 |

**Quality filtering (MFD)**: FaceQual >= 2, frontal view (FaceView=1), known sex, mean keypoint score > 0.3. Yields ~20-22K images.

**Quality filtering (CTai)**: Known sex (Male/Female), mean keypoint score > 0.3. Yields ~4,000 images.

### Landmark Extraction

PrimateFace HRNet-V2-W18-DARK pose model (trained on 4.5K primate face images with 68-point annotations) was applied to all pre-cropped face images. Since images are already face crops (224x224 for MFD, variable for CTai), we passed `bbox=[0, 0, w, h]` covering the full image to `mmpose.apis.inference_topdown()`.

Landmarks were cached as `.npz` files for reuse.

### Feature Extraction

From each face's 68 landmarks, we computed ~20 interpretable geometric features using PrimateFace's `analysis` module, all normalized by interocular distance (IOD) to remove scale effects:

**Mouth features:**
- `mouth_aperture` — inner lip vertical distance (landmarks 62-66) / IOD
- `mouth_width` — mouth corner distance (landmarks 48-54) / IOD
- `mouth_aspect_ratio` — aperture / width

**Eye features:**
- `right_eye_aperture`, `left_eye_aperture` — mean upper-lower eyelid distance / IOD

**Brow features:**
- `right_brow_height`, `left_brow_height` — brow center to eye center distance / IOD

**Face geometry:**
- `face_height` — chin (8) to brow midpoint (27) / IOD
- `face_width` — maximum jaw width (landmarks 1-15 or 2-14) / IOD
- `face_aspect_ratio` — height / width
- `jaw_width` — lower jaw width (landmarks 4-12) / IOD
- `nose_length` — nose bridge top (27) to nose tip (30) / IOD
- `eye_to_mouth` — eye center midpoint to mouth center / IOD

**Symmetry:**
- `symmetry` — fluctuating asymmetry score (midline method)
- `symmetry_jaw`, `symmetry_eyes`, `symmetry_mouth` — per-region FA

**Head pose:**
- `yaw`, `pitch`, `roll` — from solvePnP with generic 3D reference

### Classification

**Sex classification**: Logistic regression with balanced class weights. Features standardized (zero mean, unit variance).

**Age classification**:
- Age class bins: Infant (0-1yr), Juvenile (1-4yr), Subadult (4-7yr), Adult (7+yr)
- Multinomial logistic regression with balanced class weights
- Also: Ridge regression for continuous age (MAE metric)

**Train/test split**: GroupShuffleSplit (80/20) by **individual ID**, ensuring no individual appears in both train and test. This prevents data leakage from multiple images of the same individual.

### Cross-Species Transfer

- Train on MFD mandrills → test on CTai chimpanzees
- Train on CTai chimpanzees → test on MFD mandrills
- Train on combined → test on held-out from both (split by individual)

---

## Expected Results

### Sex Classification

**Within-species (expected)**:
- MFD mandrill: **75-90% balanced accuracy**. Mandrills are highly sexually dimorphic (males have larger, more colorful faces, broader jaws). Landmark geometry should capture jaw width and face proportions well.
- CTai chimp: **65-80% balanced accuracy**. Chimps are less dimorphic than mandrills. May see lower accuracy.

**Feature importance (expected)**:
- `jaw_width` and `face_width` should be top features for mandrills (strong male jaw)
- `face_aspect_ratio` may differentiate (males typically have more elongated faces)
- `symmetry` features may contribute (literature suggests sex-linked FA differences)

**Cross-species (expected)**:
- Mandrill → chimp: **55-65%** (above chance, some shared dimorphism patterns)
- Chimp → mandrill: **55-70%** (mandrill dimorphism is so strong that even imperfect features help)
- Combined: **65-80%** (more training data, shared + species-specific features)

### Age Classification

**Within-species (expected)**:
- Age class (4-class): **55-75% balanced accuracy**. Infant vs. adult should be easy (different face proportions). Juvenile vs. subadult will be hardest (gradual change).
- Continuous age MAE: **3-5 years** from landmarks alone. Much worse than Renoult's 0.58yr from DINOv2 (expected — landmarks miss texture/color cues).

**Feature importance (expected)**:
- `face_aspect_ratio` should be top feature (infant faces are rounder, adult faces more elongated)
- `eye_to_mouth` / `face_height` should capture developmental growth
- `interocular_distance` (absolute, unnormalized) correlates with head size → age

**PCA visualization (expected)**:
- PC1 should separate infants from adults
- PC2 may capture sex-related variation
- Subadults and juveniles likely overlap in the middle

### Key Comparisons

| Task | Method | Expected Performance | Notes |
|------|--------|---------------------|-------|
| Sex (mandrill) | Landmark LR | 75-90% bal. acc. | Strong dimorphism |
| Sex (chimp) | Landmark LR | 65-80% bal. acc. | Weaker dimorphism |
| Sex (mandrill) | DINOv2 probe | 85-95% bal. acc. | Image adds color/texture |
| Age class (mandrill) | Landmark LR | 55-75% bal. acc. | Good for infant/adult |
| Continuous age (mandrill) | Landmark Ridge | 3-5yr MAE | Much worse than DINOv2 |
| Continuous age (mandrill) | DINOv2 (Renoult) | 0.58yr MAE | Image-based upper bound |
| Cross-species sex | Mandrill→Chimp | 55-65% bal. acc. | Above chance? |

---

## Desired Visualizations (Supplementary Figure)

### Layout: 2 rows x 3 columns = 6 panels

**Row 1: Sex Classification**

- **Panel A**: Feature importance bar chart (top 10 logistic regression coefficients for mandrill sex). Red bars = positive (male-associated), blue = negative (female-associated). Shows WHICH geometric features drive sex prediction.

- **Panel B**: ROC curves for sex classification. Two curves: mandrill (red) and chimp (blue), each with AUC annotation. Dashed diagonal = chance. Shows HOW WELL landmarks predict sex in each species.

- **Panel C**: Violin plots of top discriminative feature by sex. E.g., jaw_width distribution for females vs males in mandrills. Shows the RAW SIGNAL that the classifier is using.

**Row 2: Age Prediction**

- **Panel D**: Confusion matrix for age class prediction (mandrill). 4x4 grid: Infant/Juvenile/Subadult/Adult. Shows WHERE the model succeeds and fails (expect infant/adult diagonal strong, juvenile/subadult confused).

- **Panel E**: Scatter plot of predicted vs. actual continuous age (mandrill). Each dot = one face. Diagonal line = perfect prediction. Color by age class. Annotate MAE. Shows the CONTINUOUS age regression quality.

- **Panel F**: PCA of landmark features colored by age class (mandrill). PC1 vs PC2, four colors for Infant/Juvenile/Subadult/Adult. Shows whether landmark geometry NATURALLY separates age groups without any training.

### Additional figures (in notebook, not supplementary)

- Cross-species transfer bar chart: 3 bars (mandrill→chimp, chimp→mandrill, combined) vs chance line
- Feature correlation heatmap across all ~20 features
- Age distribution histograms for both datasets
- Landmark overlay examples (spot-check that PrimateFace landmarks look correct on mandrill/chimp crops)

### Figure conventions (per CLAUDE.md)
- Save as both PNG (300 dpi) and SVG
- Nature-style: large bold labels (12-14pt), no gridlines, no chartjunk
- Color-blind friendly palette
- All text editable in SVG for Illustrator polishing

---

## References

1. Tieo, C. et al. (2023). "Mandrillus Face Database." Data in Brief. Zenodo doi:10.5281/zenodo.7467318
2. Freytag, A. et al. (2016). "Chimpanzee Faces in the Wild." GCPR.
3. Renoult, J. et al. (2025). "Age prediction from mandrill face images." Methods in Ecology and Evolution.
4. Parodi, F. et al. (2025). "PrimateFace: A Machine Learning Resource for Automated Face Analysis." bioRxiv.
5. Little, A.C. et al. (2012). "Facial asymmetry and health in rhesus macaques." Behavioral Ecology and Sociobiology.
