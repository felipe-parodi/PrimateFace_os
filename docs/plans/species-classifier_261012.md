# Primate species/genus face classifier — plan (2026-10-12)

Branch: `feat/species-classifier` (worktree `primateface_oss/PF-feat-species`, off `origin/main`).
Scratch/data root (local SSD): `C:/pfspecies/` (venv, crops, features, checkpoints).
Durable outputs: `A:/.../PrimateFace/results/species_classifier_261012/`.

## Goal
Fast face-crop classifiers that predict primate family → genus → species, distilled from
modern foundation-model teachers, benchmarked from ~3M params to teacher scale, and exposed in
the PrimateFace API (`Face.taxon`). MVP first, then scale.

## Agreed design (see alignment Q&A)
- Input: face crops from the PrimateFace detector (square, 1.2× bbox context).
- Label space: Mammal Diversity Database v2.5 (514 spp / 84 genera / 19 families), names mapped
  through MDD synonyms. Species head only for species with ≥ `MIN_SPECIES_IMGS` (default 20) train
  crops; all images contribute to genus/family heads.
- Ground truth: source metadata (iNat research grade, GBIF species-rank, TreeOfLife) = hard labels.
  Bing folder genus = genus-only noisy label. Unlabeled YouTube faces = pseudo-label pool.
- Teachers: BioCLIP 2 (MIT), iNat21-finetuned ConvNeXt-L / ViT-L (timm), DINOv3/SigLIP2/PE as
  benchmark rows. No closed APIs in this phase.
- Release: students on Apache/MIT backbones. Per-image license recorded. DINOv3 = benchmark only.
- Open set: calibrated abstention, roll up species → genus → family.

## Phases

### Phase 0 — Environment & taxonomy (parallel)
- uv venv `C:/pfspecies/.venv` (torch cu124, timm, open_clip, transformers).
- `taxonomy.py`: download MDD v2.5, build primate table, synonym resolver
  (`resolve(name) -> (family, genus, species|None)`), unit tests.
- Success: ≥95% of distinct source scientific names in face manifests resolve to an MDD genus;
  unresolved names listed in a report.

### Phase 1 — Labeled face-crop manifest
- Join acquisition detections + candidates + downloads (GBIF/iNat API, 93k faces); iNat S3
  candidates (taxonomy re-derived from `taxa.csv.gz` ancestry) where faces exist; Bing genus folders.
- Fields: crop_id, image_path, bbox, det score, source, scientific_name, family/genus/species
  (MDD), label_level, license, group_id (iNat observation / GBIF occurrence / observer), split.
- Crops: 256px square JPEG on local SSD.
- Splits: grouped by observer/observation (no leakage). Test = 10% of groups, val = 5%.
  Bing used only for train (genus) since it lacks provenance groups; separate Bing-genus test.
- Success: ≥80k labeled crops, ≥60 genera, ≥200 species with ≥20 crops; split leakage check = 0
  shared group ids; review grid published for human spot-check.

### Phase 2 — Teacher features & probes (MVP baseline)
- Extract frozen embeddings (BioCLIP 2, iNat21 ConvNeXt-L, DINOv3-S, SigLIP2) for all crops.
- BioCLIP 2 zero-shot (MDD names) + linear/hierarchical probes on each embedding.
- Success: first benchmark table with genus/species top-1, macro-F1 on test.

### Phase 3 — First trained student (MVP; hard deadline 5h after start = 2026-10-12 20:45)
- Fine-tune a small timm student (ConvNeXt-Atto or MobileNetV4-S) and a mid model with
  hierarchical heads on hard labels.
- Success: checkpoint saved; test metrics recorded; ≥1 model beats BioCLIP 2 zero-shot at genus.

### Phase 4 — Distillation & pseudo-labels
- Best teacher = full fine-tune of BioCLIP 2 (or best probe). Soft labels (T≈2) on all crops,
  incl. YouTube/unlabeled pool, with confidence + teacher-consensus filtering.
- Students trained with CE(hard) + KL(teacher) + taxonomy-aware label smoothing.
- Success: distilled students ≥ non-distilled at equal size on genus & species macro-F1.

### Phase 5 — Benchmark ladder
- Models: MobileNetV4-S, ConvNeXt-Atto, FastViT/RepViT, ConvNeXt-Nano/Tiny, EfficientViT,
  ViT-S, ViT-B (BioCLIP-init), teacher(s), zero-shot rows.
- Metrics: family/genus/species top-1, top-5, macro-F1 (head/tail), ECE, coverage-accuracy
  (abstention roll-up), external tests (Rare Species primate rows, CTai/C-Zoo, MacaqueFaces,
  Bing-genus test), GPU throughput (bs=64, fp16) and CPU ms/crop (bs=1, ONNX).
- Success: table + plots (PNG+SVG) + Pareto plot (accuracy vs latency).

### Phase 6 — API integration & demos
- `primateface/taxonomy/` module: `TaxonClassifier`, model registry entries, `Face.taxon`
  lazy property, CLI `primateface classify`, tests.
- Demo images: detector + pose + taxon overlay on held-out photos & demo videos frames.
- Success: `pf.analyze(img)[0].taxon` works; pytest green; ruff clean.

### Phase 7 — Scale (after MVP)
- Face detection on remaining iNat S3 images (153k downloaded), TreeOfLife primate rows,
  re-train ladder; pseudo-label YouTube 173k for genus.

## Test strategy
- Unit tests for taxonomy resolver, split grouping (no leakage), hierarchy roll-up, API property.
- Metrics computed only on held-out groups; teachers never trained on test groups.

## Risks
- Label noise (Bing queries, GBIF misidentifications) → genus-only for Bing; teacher-consensus filter.
- Captive macaque dominance → class-balanced sampling, macro metrics.
- Network drive IO → all crops/features on local SSD.

## Outcome (2026-10-14)

- Teacher: BioCLIP 2 full fine-tune, test genus 98.0 / species 91.9 (Rare Species 90.6 / 77.1).
- API default: TinyViT-5M (plain CE), test genus 97.5 / species 90.9, 8.8 ms CPU (ONNX, 4 threads).
  Alternatives registered: `taxon_repvit` (distilled, 97.6 / 91.3, 7.4 ms), `taxon_convnext_atto`
  (96.5 / 88.8, 3.8 ms), `taxon_bioclip2` (teacher, needs open_clip).
- Distillation from BioCLIP 2 changed species top-1 by -0.7 to +0.7 points across ten students.
- Dropped by user decision: PE-L and DINOv3-L fine-tunes, and with them the two-teacher YouTube
  consensus test. DINOv3-L remains as a linear-probe reference row (97.1 / 89.9).
- Report: `results/species_classifier_261012/primate_taxon_benchmark.html`; figures, CSV, models,
  demos and review grids alongside it.
