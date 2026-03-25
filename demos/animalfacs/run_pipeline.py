#!/usr/bin/env python
"""PrimateFace x AnimalFACS Demo Pipeline.

Main entry point that orchestrates all phases:
  A. Scrape AnimalFACS website → manifest
  B. Download videos from Google Drive
  C. Parse AU labels from folder structure
  D. Build structured dataset
  E. Extract frames + run PrimateFace landmarks
  F. Compute geometric + temporal features
  G. Train models (RF, TCN, ST-GCN)
  H. Evaluate (within-species, LOSO, pooled)
  I. Generate figures and demo videos

Usage:
    python -m demos.animalfacs.run_pipeline
    python -m demos.animalfacs.run_pipeline --dry-run
    python -m demos.animalfacs.run_pipeline --species chimp,macaque
    python -m demos.animalfacs.run_pipeline --skip-download
    python -m demos.animalfacs.run_pipeline --max-clips 20
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from .au_parser import parse_all_species
from .dataset_builder import build_dataset
from .downloader import build_video_inventory, download_all
from .evaluate import run_all_evaluations
from .features import extract_all_features, extract_appearance_embeddings
from .preprocess import process_dataset
from .qa import qa_dataset
from .scraper import build_manifest
from .utils import check_environment, load_config, set_seed, setup_logging
from .visualize import generate_all_figures, render_demo_video

logger = logging.getLogger("animalfacs")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="PrimateFace x AnimalFACS Demo Pipeline"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (default: demos/animalfacs/config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build manifest and print plan without downloading",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading, use existing data",
    )
    parser.add_argument(
        "--species",
        type=str,
        default=None,
        help="Comma-separated species to process (e.g. chimp,macaque)",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Limit to N clips per species (for quick testing)",
    )
    parser.add_argument(
        "--skip-neural",
        action="store_true",
        help="Skip G2/G3 neural models, only run RF baseline",
    )
    parser.add_argument(
        "--phases",
        type=str,
        default="all",
        help="Comma-separated phases to run: scrape,download,parse,build,preprocess,features,evaluate,visualize (default: all)",
    )
    return parser.parse_args()


def run_pipeline(
    cfg: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
    skip_download: bool = False,
    species_filter: Optional[List[str]] = None,
    max_clips: Optional[int] = None,
    skip_neural: bool = False,
    phases: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run the full demo pipeline.

    Args:
        cfg: Loaded config dict.
        dry_run: Only print plan without executing.
        skip_download: Skip download phase.
        species_filter: Only process these species.
        max_clips: Limit clips per species.
        skip_neural: Skip TCN/ST-GCN training.
        phases: Which phases to run.

    Returns:
        Dict with all pipeline outputs.
    """
    if cfg is None:
        cfg = load_config()

    if phases is None:
        phases = ["scrape", "download", "parse", "build", "preprocess",
                  "qa", "features", "appearance", "evaluate", "visualize"]

    set_seed(cfg.get("seed", 42))
    results: Dict[str, Any] = {}

    logger.info("=" * 60)
    logger.info("PrimateFace x AnimalFACS Demo Pipeline")
    logger.info("=" * 60)

    env = check_environment()
    results["environment"] = env

    # === Phase A: Scrape ===
    if "scrape" in phases:
        logger.info("\n=== Phase A: Scraping AnimalFACS ===")
        manifest = build_manifest(cfg)
        results["manifest"] = manifest

        if dry_run:
            logger.info("\n[DRY RUN] Manifest built. Species available:")
            for entry in manifest:
                logger.info(
                    "  %s (%s): training_videos=%s, test_materials=%s",
                    entry["species_id"],
                    entry["common_name"],
                    "YES" if entry["has_training_videos"] else "NO",
                    "YES" if entry["has_test_materials"] else "NO",
                )
            return results

    # === Phase B: Download ===
    if "download" in phases and not skip_download:
        logger.info("\n=== Phase B: Downloading ===")
        download_statuses = download_all(
            cfg, species_filter=species_filter, dry_run=dry_run
        )
        results["download_statuses"] = download_statuses

        logger.info("\n=== Building video inventory ===")
        build_video_inventory(cfg)

    # === Phase C: Parse AU labels ===
    if "parse" in phases:
        logger.info("\n=== Phase C: Parsing AU Labels ===")
        au_records = parse_all_species(cfg)
        results["au_records"] = au_records

        if species_filter:
            au_records = [r for r in au_records if r["species"] in species_filter]
    else:
        au_records = []

    # === Phase D: Build dataset ===
    if "build" in phases and au_records:
        logger.info("\n=== Phase D: Building Dataset ===")
        dataset_df = build_dataset(au_records, cfg)
        results["dataset_size"] = len(dataset_df)

        if max_clips and len(dataset_df) > 0:
            # Sample per species
            sampled = []
            for species in dataset_df["species"].unique():
                sp_df = dataset_df[dataset_df["species"] == species]
                n = min(max_clips, len(sp_df))
                sampled.append(sp_df.sample(n=n, random_state=42))
            import pandas as pd
            dataset_df = pd.concat(sampled, ignore_index=True)
            logger.info("Clipped to %d clips (max %d/species)", len(dataset_df), max_clips)
    elif "build" in phases:
        logger.warning("No AU records to build dataset from")
        return results
    else:
        # Try loading existing dataset
        import pandas as pd
        ds_path = Path(cfg["paths"]["au_dataset"])
        if ds_path.exists():
            dataset_df = pd.read_parquet(ds_path)
        else:
            logger.error("No dataset found at %s", ds_path)
            return results

    if dataset_df.empty:
        logger.error("Empty dataset. Cannot continue.")
        return results

    # === Phase E: Preprocess ===
    if "preprocess" in phases:
        logger.info("\n=== Phase E: Preprocessing + PrimateFace Landmarks ===")
        dataset_df = process_dataset(dataset_df, cfg)

    # === Phase E.5: QA ===
    if "qa" in phases:
        logger.info("\n=== Phase E.5: Video QA ===")
        dataset_df = qa_dataset(dataset_df, cfg)
        filter_mode = cfg.get("qa", {}).get("filter_mode", "pass_and_warn")
        if filter_mode == "pass_and_warn":
            before = len(dataset_df)
            dataset_df = dataset_df[dataset_df["qa_status"] != "fail"]
            logger.info("QA filter: %d → %d clips", before, len(dataset_df))

    # === Phase F: Feature extraction ===
    if "features" in phases:
        logger.info("\n=== Phase F: Feature Extraction ===")
        geo_features, seq_features, clip_ids, dataset_df = extract_all_features(
            dataset_df, cfg
        )
    else:
        # Try loading cached features
        feat_path = Path(cfg["paths"]["features_matrix"])
        if feat_path.exists():
            cached = np.load(feat_path, allow_pickle=True)
            geo_features = cached["geo_features"]
            seq_features = cached["seq_features"]
            clip_ids = list(cached["clip_ids"])
        else:
            logger.error("No features found at %s", feat_path)
            return results

    if geo_features.shape[0] == 0:
        logger.error("No features extracted. Cannot evaluate.")
        return results

    # Determine AU set from data
    all_aus: set = set()
    for labels in dataset_df["normalized_labels"]:
        if isinstance(labels, str):
            aus = [int(x) for x in labels.split(",") if x.strip()]
        else:
            aus = labels
        all_aus.update(aus)
    au_set = sorted(all_aus)
    logger.info("AU set: %s (%d total)", au_set, len(au_set))

    # === Phase F.5: Appearance embeddings (FMAE-IAT ViT) ===
    appearance_embs = None
    appearance_ids = None
    if "appearance" in phases:
        logger.info("\n=== Phase F.5: FMAE-IAT Appearance Embeddings ===")
        appearance_embs, appearance_ids = extract_appearance_embeddings(
            dataset_df, cfg
        )
        if appearance_embs.shape[0] > 0:
            # Save embeddings
            data_root = Path(cfg["paths"]["data_root"])
            np.savez_compressed(
                data_root / "appearance_embeddings.npz",
                embeddings=appearance_embs,
                clip_ids=np.array(appearance_ids),
            )
    else:
        # Try loading cached
        data_root = Path(cfg["paths"]["data_root"])
        emb_path = data_root / "appearance_embeddings.npz"
        if emb_path.exists():
            cached = np.load(emb_path, allow_pickle=True)
            appearance_embs = cached["embeddings"]
            appearance_ids = list(cached["clip_ids"])
            logger.info(
                "Loaded cached appearance embeddings: %s", appearance_embs.shape
            )

    # === Phase G+H: Evaluate ===
    if "evaluate" in phases:
        logger.info("\n=== Phases G+H: Training & Evaluation ===")

        if skip_neural:
            orig_device = cfg["primateface"]["device"]
            cfg["primateface"]["device"] = "cpu"

        eval_results = run_all_evaluations(
            geo_features, seq_features, dataset_df, clip_ids, au_set, cfg
        )
        results["evaluation"] = eval_results

        # Appearance-based evaluation (linear probe)
        if appearance_embs is not None and appearance_embs.shape[0] > 0:
            logger.info("\n=== FMAE-IAT Linear Probe Evaluation ===")
            from .evaluate import within_species_cv, loso_evaluation, pooled_evaluation

            # Align appearance embeddings with clip_ids
            app_id_set = set(appearance_ids)
            shared_ids = [c for c in clip_ids if c in app_id_set]
            if len(shared_ids) >= 10:
                app_idx = {c: i for i, c in enumerate(appearance_ids)}
                shared_app = np.stack([appearance_embs[app_idx[c]] for c in shared_ids])

                logger.info("=== FMAE Within-species CV ===")
                app_within = within_species_cv(
                    shared_app, dataset_df, shared_ids, au_set, cfg
                )
                eval_results["fmae_within_species"] = app_within

                logger.info("=== FMAE LOSO ===")
                app_loso = loso_evaluation(
                    shared_app, dataset_df, shared_ids, au_set, cfg
                )
                eval_results["fmae_loso"] = app_loso

                logger.info("=== FMAE Pooled ===")
                app_pooled = pooled_evaluation(
                    shared_app, dataset_df, shared_ids, au_set, cfg
                )
                eval_results["fmae_pooled"] = app_pooled

        if skip_neural:
            cfg["primateface"]["device"] = orig_device

    # === Phase I: Visualize ===
    if "visualize" in phases:
        logger.info("\n=== Phase I: Visualization ===")
        eval_results = results.get("evaluation", {})
        generate_all_figures(eval_results, dataset_df, cfg)

        # Render demo videos for a few clips per species
        vid_dir = Path(cfg["paths"]["results_root"]) / "videos"
        data_root = Path(cfg["paths"]["data_root"])

        for species in dataset_df["species"].unique():
            sp_df = dataset_df[dataset_df["species"] == species]
            # Pick up to 3 clips with AU labels
            sample = sp_df.head(3)
            for _, row in sample.iterrows():
                vid_path = Path(row["source_video_path"])
                npz_path_str = row.get("npz_path", "")
                if not npz_path_str:
                    feat_dir = data_root / "features" / species
                    npz_path = feat_dir / f"{row['clip_id']}.npz"
                else:
                    npz_path = Path(npz_path_str)

                labels = row["normalized_labels"]
                if isinstance(labels, str):
                    gt_aus = [int(x) for x in labels.split(",") if x.strip()]
                else:
                    gt_aus = labels

                render_demo_video(
                    video_path=vid_path,
                    npz_path=npz_path,
                    species=species,
                    gt_aus=gt_aus,
                    pred_aus=None,  # TODO: add model predictions
                    output_path=vid_dir / f"{species}_{row['clip_id']}.mp4",
                )

    # === Summary ===
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline complete!")
    logger.info("=" * 60)

    # Save results summary
    results_root = Path(cfg["paths"]["results_root"])
    results_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "n_species": len(dataset_df["species"].unique()),
        "n_clips": len(dataset_df),
        "n_features_geo": geo_features.shape[1] if geo_features.ndim == 2 else 0,
        "au_set": au_set,
        "species": sorted(dataset_df["species"].unique().tolist()),
    }

    if "evaluation" in results:
        ev = results["evaluation"]
        if "within_species" in ev:
            for sp, metrics in ev["within_species"].items():
                summary[f"within_{sp}_f1"] = metrics.get("f1_macro", 0)
        if "loso" in ev:
            for sp, metrics in ev["loso"].items():
                summary[f"loso_{sp}_f1"] = metrics.get("f1_macro", 0)

    with open(results_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Summary saved to %s", results_root / "summary.json")

    return results


def main() -> None:
    """CLI entry point."""
    setup_logging()
    args = parse_args()
    cfg = load_config(args.config)

    species_filter = None
    if args.species:
        species_filter = [s.strip() for s in args.species.split(",")]

    phases = None
    if args.phases != "all":
        phases = [p.strip() for p in args.phases.split(",")]

    run_pipeline(
        cfg=cfg,
        dry_run=args.dry_run,
        skip_download=args.skip_download,
        species_filter=species_filter,
        max_clips=args.max_clips,
        skip_neural=args.skip_neural,
        phases=phases,
    )


if __name__ == "__main__":
    main()
