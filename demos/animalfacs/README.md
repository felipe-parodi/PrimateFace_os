# PrimateFace x AnimalFACS Demo

Cross-species facial Action Unit prediction using PrimateFace landmarks.

## Purpose

This demo shows that PrimateFace extracts anatomically meaningful facial landmarks across multiple primate species, and that these geometric representations are predictive of expert-coded Facial Action Units (AUs). The LOSO (Leave-One-Species-Out) evaluation demonstrates cross-species transfer — PrimateFace captures conserved facial anatomy rather than species-specific artifacts.

## Data

Videos and manuals are sourced from the [AnimalFACS](https://animalfacs.github.io/AnimalFACS/) project. **Bridget Waller**, creator and maintainer of AnimalFACS, is a co-author on the PrimateFace paper and has given explicit written permission to use these datasets for this demo.

### Species covered

| Species | FACS System | Training Videos | Test Materials |
|---------|-------------|:---:|:---:|
| Chimpanzee (*Pan troglodytes*) | ChimpFACS | Y | Y |
| Rhesus macaque (*Macaca mulatta*) | MaqFACS | Y | Y |
| Gibbon (*Hylobatidae* spp.) | GibbonFACS | Y | Y |
| Orangutan (*Pongo* spp.) | OrangFACS | Y | Y |
| Common marmoset (*Callithrix jacchus*) | CalliFACS | - | Y |
| Gorilla (*Gorilla* sp.) | GorillaFACS | - | - |

## Quick start

```bash
# Activate environment
conda activate omlab310

# Dry run — show what would be downloaded
python -m demos.animalfacs.run_pipeline --dry-run

# Download + run full pipeline
python -m demos.animalfacs.run_pipeline

# Quick test with limited data
python -m demos.animalfacs.run_pipeline --species chimp,macaque --max-clips 20

# Skip download (use existing data)
python -m demos.animalfacs.run_pipeline --skip-download

# RF baseline only (no neural models)
python -m demos.animalfacs.run_pipeline --skip-neural
```

## Pipeline phases

1. **Scrape** — Fetch AnimalFACS species pages, build manifest
2. **Download** — Download videos and manuals from Google Drive via gdown
3. **Parse** — Extract AU labels from folder/filename structure
4. **Build** — Construct structured parquet dataset with video-level splits
5. **Preprocess** — Extract frames, run PrimateFace detection + 68-point landmarks
6. **Features** — Compute geometric features (kinematics + pairwise distances) and landmark sequences
7. **Evaluate** — Train models and evaluate:
   - Within-species cross-validation
   - Leave-One-Species-Out (LOSO) transfer
   - Pooled-species
8. **Visualize** — Generate publication figures + demo videos with landmark overlay

## Models

- **G1 (RF baseline):** Random Forest on per-clip geometric features from PrimateFace kinematics
- **G2 (TCN):** Temporal Convolutional Network on (T, 68, 2) landmark sequences
- **G3 (ST-GCN):** Spatial-Temporal Graph Convolutional Network on the 68-point face graph

## Leakage prevention

- Splits are by **video**, not by frame — all frames from one video stay in the same fold
- LOSO splits are species-disjoint by definition
- AU labels come only from folder structure (ground truth), never inferred at test time

## Adding your own videos

Place videos in `data/animalfacs/raw/{species_id}/training_videos/AU{N}/` where `{N}` is the AU number. The parser will pick up AU labels from folder names automatically.

## Citation

If you use this demo, please cite both PrimateFace and the relevant AnimalFACS systems:

- **ChimpFACS:** Vick et al. (2007) *Evolution of Communication*
- **MaqFACS:** Parr et al. (2010) *Am J Phys Anthropol*
- **GibbonFACS:** Waller et al. (2012) *Int J Primatol*
- **OrangFACS:** Caeiro et al. (2013) *Int J Primatol*
- **CalliFACS:** Caeiro et al. (2022) *PLoS ONE*
- **GorillaFACS:** Caeiro et al. (2024) *PLoS ONE*
