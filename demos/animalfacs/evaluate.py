"""Phase H: Evaluation — within-species, LOSO, pooled.

Computes per-AU and aggregate metrics, generates interpretive summaries.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    hamming_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from .au_homology import get_shared_aus
from .models.baseline_rf import GeometricAUClassifier, prepare_labels
from .utils import load_config, set_seed

logger = logging.getLogger("animalfacs.evaluate")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    au_list: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Compute multi-label classification metrics.

    Args:
        y_true: (N, C) binary ground truth.
        y_pred: (N, C) binary predictions.
        y_proba: (N, C) probability scores (optional).
        au_list: List of AU integers for per-AU reporting.

    Returns:
        Dict of metric names to values.
    """
    results: Dict[str, Any] = {}

    # Only evaluate columns with both positive and negative examples
    valid_cols = [
        c for c in range(y_true.shape[1])
        if y_true[:, c].sum() > 0 and y_true[:, c].sum() < y_true.shape[0]
    ]

    if not valid_cols:
        logger.warning("No AU columns have both positive and negative examples")
        return {"f1_macro": 0.0, "f1_micro": 0.0, "n_valid_aus": 0}

    y_t = y_true[:, valid_cols]
    y_p = y_pred[:, valid_cols]

    results["f1_macro"] = float(f1_score(y_t, y_p, average="macro", zero_division=0))
    results["f1_micro"] = float(f1_score(y_t, y_p, average="micro", zero_division=0))
    results["hamming_loss"] = float(hamming_loss(y_t, y_p))
    results["n_valid_aus"] = len(valid_cols)
    results["n_samples"] = y_true.shape[0]

    # Per-AU F1 and balanced accuracy
    per_au_f1 = {}
    per_au_balanced_acc = {}
    for i, c in enumerate(valid_cols):
        au = au_list[c] if au_list else c
        per_au_f1[au] = float(f1_score(y_t[:, i], y_p[:, i], zero_division=0))
        per_au_balanced_acc[au] = float(
            balanced_accuracy_score(y_t[:, i], y_p[:, i])
        )
    results["per_au_f1"] = per_au_f1
    results["per_au_balanced_acc"] = per_au_balanced_acc
    results["balanced_acc_macro"] = float(
        np.mean(list(per_au_balanced_acc.values()))
    ) if per_au_balanced_acc else 0.0

    # AUROC and mAP if probabilities available
    if y_proba is not None:
        y_prob = y_proba[:, valid_cols]
        try:
            results["auroc_macro"] = float(
                roc_auc_score(y_t, y_prob, average="macro")
            )
        except ValueError:
            results["auroc_macro"] = float("nan")
        try:
            results["map"] = float(
                average_precision_score(y_t, y_prob, average="macro")
            )
        except ValueError:
            results["map"] = float("nan")

    return results


def within_species_cv(
    geo_features: np.ndarray,
    dataset_df: pd.DataFrame,
    clip_ids: List[str],
    au_set: List[int],
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Per-species stratified k-fold cross-validation.

    Splits by clip (video-level) to prevent temporal leakage.

    Args:
        geo_features: (N, D) geometric feature matrix.
        dataset_df: Dataset DataFrame.
        clip_ids: Ordered clip IDs.
        au_set: List of AU integers.
        cfg: Config dict.

    Returns:
        Dict mapping species to metrics dict.
    """
    if cfg is None:
        cfg = load_config()

    n_folds = cfg["evaluation"]["cv_folds"]
    min_clips = cfg["evaluation"]["min_clips_per_species"]
    seed = cfg.get("seed", 42)
    set_seed(seed)

    y, _ = prepare_labels(dataset_df, clip_ids, au_set)

    # Map clip_ids to species
    cid_to_species = dict(zip(dataset_df["clip_id"], dataset_df["species"]))
    species_arr = np.array([cid_to_species.get(cid, "unknown") for cid in clip_ids])

    results = {}
    for species in sorted(set(species_arr)):
        mask = species_arr == species
        n_sp = mask.sum()
        if n_sp < min_clips:
            logger.info("  Skip %s: only %d clips < %d minimum", species, n_sp, min_clips)
            continue

        x_sp = geo_features[mask]
        y_sp = y[mask]

        # Use dominant AU for stratification
        dominant_au = y_sp.sum(axis=1).astype(int)
        actual_folds = min(n_folds, len(np.unique(dominant_au)))
        if actual_folds < 2:
            actual_folds = min(n_folds, n_sp)
            if actual_folds < 2:
                logger.info("  Skip %s: too few samples for CV", species)
                continue
            # Fall back to non-stratified
            kf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=seed)
            # Use dummy stratification
            dummy_strat = np.zeros(n_sp, dtype=int)
            dummy_strat[:n_sp // 2] = 1
            splits = list(kf.split(x_sp, dummy_strat))
        else:
            kf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=seed)
            splits = list(kf.split(x_sp, dominant_au))

        fold_metrics = []
        for train_idx, test_idx in splits:
            model = GeometricAUClassifier(
                model_type="rf",
                model_params=cfg["models"]["g1_rf"],
            )
            model.fit(x_sp[train_idx], y_sp[train_idx], au_set)
            preds = model.predict(x_sp[test_idx])
            proba = model.predict_proba(x_sp[test_idx])
            metrics = compute_metrics(y_sp[test_idx], preds, proba, au_set)
            fold_metrics.append(metrics)

        # Average across folds
        avg = {"f1_macro": np.mean([m["f1_macro"] for m in fold_metrics])}
        avg["f1_micro"] = np.mean([m["f1_micro"] for m in fold_metrics])
        avg["n_clips"] = int(n_sp)
        avg["n_folds"] = actual_folds

        # Average per-AU F1
        all_au_f1: Dict[int, List[float]] = {}
        for m in fold_metrics:
            for au, f1 in m.get("per_au_f1", {}).items():
                all_au_f1.setdefault(au, []).append(f1)
        avg["per_au_f1"] = {au: np.mean(vals) for au, vals in all_au_f1.items()}

        results[species] = avg
        logger.info(
            "  %s: macro-F1=%.3f, micro-F1=%.3f (%d clips, %d folds)",
            species, avg["f1_macro"], avg["f1_micro"], n_sp, actual_folds,
        )

    return results


def loso_evaluation(
    geo_features: np.ndarray,
    dataset_df: pd.DataFrame,
    clip_ids: List[str],
    au_set: List[int],
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Leave-One-Species-Out cross-species evaluation.

    The crown jewel: train on N-1 species, test on held-out.
    Only evaluates on AUs shared between train and test species.

    Args:
        geo_features: (N, D) geometric feature matrix.
        dataset_df: Dataset DataFrame.
        clip_ids: Ordered clip IDs.
        au_set: List of AU integers.
        cfg: Config dict.

    Returns:
        Dict mapping held-out species to metrics dict.
    """
    if cfg is None:
        cfg = load_config()

    min_clips = cfg["evaluation"]["min_clips_per_species"]
    set_seed(cfg.get("seed", 42))

    y, _ = prepare_labels(dataset_df, clip_ids, au_set)

    cid_to_species = dict(zip(dataset_df["clip_id"], dataset_df["species"]))
    species_arr = np.array([cid_to_species.get(cid, "unknown") for cid in clip_ids])
    unique_species = sorted(set(species_arr))

    if len(unique_species) < 2:
        logger.warning("LOSO requires >= 2 species, got %d", len(unique_species))
        return {}

    results = {}
    for held_out in unique_species:
        test_mask = species_arr == held_out
        train_mask = ~test_mask

        if test_mask.sum() < min_clips:
            logger.info("  Skip LOSO %s: too few test clips (%d)", held_out, test_mask.sum())
            continue

        # Find shared AUs between training species and test species
        train_species = [s for s in unique_species if s != held_out]
        shared = get_shared_aus([held_out] + train_species)
        shared_indices = [i for i, au in enumerate(au_set) if au in shared]

        if not shared_indices:
            logger.info("  Skip LOSO %s: no shared AUs", held_out)
            continue

        y_shared = y[:, shared_indices]
        shared_au_list = [au_set[i] for i in shared_indices]

        model = GeometricAUClassifier(
            model_type="rf",
            model_params=cfg["models"]["g1_rf"],
        )
        model.fit(
            geo_features[train_mask],
            y_shared[train_mask],
            shared_au_list,
        )
        preds = model.predict(geo_features[test_mask])
        proba = model.predict_proba(geo_features[test_mask])
        metrics = compute_metrics(
            y_shared[test_mask], preds, proba, shared_au_list
        )
        metrics["train_species"] = train_species
        metrics["shared_aus"] = sorted(shared)

        results[held_out] = metrics
        logger.info(
            "  LOSO %s: macro-F1=%.3f (%d shared AUs, trained on %s)",
            held_out,
            metrics.get("f1_macro", 0),
            len(shared),
            train_species,
        )

    return results


def pooled_evaluation(
    geo_features: np.ndarray,
    dataset_df: pd.DataFrame,
    clip_ids: List[str],
    au_set: List[int],
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Pooled-species evaluation with species as a covariate.

    Args:
        geo_features: (N, D) geometric feature matrix.
        dataset_df: Dataset DataFrame.
        clip_ids: Ordered clip IDs.
        au_set: List of AU integers.
        cfg: Config dict.

    Returns:
        Metrics dict for pooled model.
    """
    if cfg is None:
        cfg = load_config()

    set_seed(cfg.get("seed", 42))
    y, _ = prepare_labels(dataset_df, clip_ids, au_set)

    # Add species one-hot as features
    cid_to_species = dict(zip(dataset_df["clip_id"], dataset_df["species"]))
    species_arr = np.array([cid_to_species.get(cid, "unknown") for cid in clip_ids])
    unique_sp = sorted(set(species_arr))
    sp_onehot = np.zeros((len(clip_ids), len(unique_sp)), dtype=np.float32)
    for i, sp in enumerate(species_arr):
        sp_onehot[i, unique_sp.index(sp)] = 1.0
    x_aug = np.concatenate([geo_features, sp_onehot], axis=1)

    # Use split from dataset_df
    cid_to_split = dict(zip(dataset_df["clip_id"], dataset_df["split"]))
    split_arr = np.array([cid_to_split.get(cid, "train") for cid in clip_ids])

    train_mask = split_arr == "train"
    test_mask = split_arr == "test"

    if test_mask.sum() == 0:
        # Fall back to val as test
        test_mask = split_arr == "val"
    if test_mask.sum() == 0:
        logger.warning("No test data for pooled evaluation")
        return {}

    model = GeometricAUClassifier(
        model_type="rf",
        model_params=cfg["models"]["g1_rf"],
    )
    model.fit(x_aug[train_mask], y[train_mask], au_set)
    preds = model.predict(x_aug[test_mask])
    proba = model.predict_proba(x_aug[test_mask])
    metrics = compute_metrics(y[test_mask], preds, proba, au_set)
    metrics["n_species"] = len(unique_sp)

    logger.info(
        "  Pooled: macro-F1=%.3f, micro-F1=%.3f (%d species)",
        metrics.get("f1_macro", 0),
        metrics.get("f1_micro", 0),
        len(unique_sp),
    )
    return metrics


def kinematic_au_correlations(
    geo_features: np.ndarray,
    dataset_df: pd.DataFrame,
    clip_ids: List[str],
    au_set: List[int],
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute per-AU correlations with kinematic features.

    The key scientific result: which PrimateFace kinematics features
    correlate with which AUs, and do these correlations hold cross-species?

    Uses only the first 14 features (kinematics from PrimateFace) for
    interpretability. The remaining features are pairwise distances.

    Args:
        geo_features: (N, D) geometric feature matrix.
        dataset_df: Dataset DataFrame.
        clip_ids: Ordered clip IDs.
        au_set: List of AU integers.
        cfg: Config dict.

    Returns:
        Dict with per-species and cross-species correlation matrices.
    """
    from scipy.stats import pointbiserialr

    if cfg is None:
        cfg = load_config()

    y, _ = prepare_labels(dataset_df, clip_ids, au_set)
    cid_to_species = dict(zip(dataset_df["clip_id"], dataset_df["species"]))
    species_arr = np.array([cid_to_species.get(cid, "unknown") for cid in clip_ids])

    # Use first 14 features (kinematics) for interpretability
    n_kin = 14
    kin_names = [
        "mouth_aperture", "mouth_width", "mouth_aspect_ratio",
        "right_eye_aperture", "left_eye_aperture",
        "right_brow_height", "left_brow_height",
        "face_height", "face_width", "face_aspect_ratio",
        "jaw_width", "nose_length", "eye_to_mouth", "interocular_distance",
    ]
    # Use mean stats only (first n_kin of the geo features)
    x_kin = geo_features[:, :n_kin]

    results: Dict[str, Any] = {"feature_names": kin_names[:n_kin]}

    # Per-species correlations
    for species in sorted(set(species_arr)):
        mask = species_arr == species
        x_sp = x_kin[mask]
        y_sp = y[mask]
        n_sp = mask.sum()

        corr_matrix = np.zeros((n_kin, len(au_set)))
        pval_matrix = np.ones((n_kin, len(au_set)))

        for j, au in enumerate(au_set):
            if y_sp[:, j].sum() < 2 or y_sp[:, j].sum() >= n_sp - 1:
                continue  # Need variance in both label and feature
            for i in range(min(n_kin, x_sp.shape[1])):
                if np.std(x_sp[:, i]) < 1e-8:
                    continue
                try:
                    r, p = pointbiserialr(y_sp[:, j], x_sp[:, i])
                    corr_matrix[i, j] = r
                    pval_matrix[i, j] = p
                except Exception:
                    pass

        results[species] = {
            "correlations": corr_matrix,
            "pvalues": pval_matrix,
            "n_clips": int(n_sp),
        }

        # Log top correlations
        for j, au in enumerate(au_set):
            top_feat = np.argmax(np.abs(corr_matrix[:, j]))
            r_val = corr_matrix[top_feat, j]
            if abs(r_val) > 0.2:
                logger.info(
                    "  %s AU%d: top feature=%s (r=%.3f)",
                    species, au, kin_names[top_feat] if top_feat < len(kin_names) else f"f{top_feat}", r_val,
                )

    # Cross-species consistency: correlation of correlations
    species_list = sorted(set(species_arr))
    if len(species_list) >= 2:
        consistency = {}
        for i, sp1 in enumerate(species_list):
            for sp2 in species_list[i + 1:]:
                c1 = results[sp1]["correlations"].ravel()
                c2 = results[sp2]["correlations"].ravel()
                valid = (c1 != 0) | (c2 != 0)
                if valid.sum() >= 10:
                    r_consistency = np.corrcoef(c1[valid], c2[valid])[0, 1]
                    consistency[f"{sp1}_vs_{sp2}"] = float(r_consistency)
                    logger.info(
                        "  Cross-species consistency %s vs %s: r=%.3f",
                        sp1, sp2, r_consistency,
                    )
        results["cross_species_consistency"] = consistency

    return results


def run_all_evaluations(
    geo_features: np.ndarray,
    seq_features: np.ndarray,
    dataset_df: pd.DataFrame,
    clip_ids: List[str],
    au_set: List[int],
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run all evaluation regimes.

    Args:
        geo_features: (N, D) geometric features.
        seq_features: (N, T, 68, 2) landmark sequences.
        dataset_df: Dataset DataFrame.
        clip_ids: Ordered clip IDs.
        au_set: List of AU integers.
        cfg: Config dict.

    Returns:
        Dict with all evaluation results.
    """
    if cfg is None:
        cfg = load_config()

    all_results: Dict[str, Any] = {}

    logger.info("=== Kinematic-AU Correlations ===")
    all_results["correlations"] = kinematic_au_correlations(
        geo_features, dataset_df, clip_ids, au_set, cfg
    )

    logger.info("=== Within-species CV (G1: RF) ===")
    all_results["within_species"] = within_species_cv(
        geo_features, dataset_df, clip_ids, au_set, cfg
    )

    logger.info("=== LOSO Evaluation (G1: RF) ===")
    all_results["loso"] = loso_evaluation(
        geo_features, dataset_df, clip_ids, au_set, cfg
    )

    logger.info("=== Pooled Evaluation (G1: RF) ===")
    all_results["pooled"] = pooled_evaluation(
        geo_features, dataset_df, clip_ids, au_set, cfg
    )

    # Prototypical classifier on geometric features
    logger.info("=== Prototypical Classifier (Geo Features) ===")
    try:
        from primateface.analysis.proto import PrototypicalClassifier

        y, _ = prepare_labels(dataset_df, clip_ids, au_set)
        cid_to_split_proto = dict(zip(dataset_df["clip_id"], dataset_df["split"]))
        split_arr_proto = np.array([cid_to_split_proto.get(c, "train") for c in clip_ids])
        train_m = split_arr_proto == "train"
        test_m = (split_arr_proto == "test") | (split_arr_proto == "val")

        if train_m.sum() >= 5 and test_m.sum() >= 3:
            proto = PrototypicalClassifier(distance="cosine")
            proto.fit(geo_features[train_m], y[train_m], class_labels=au_set)
            proto_preds, proto_scores = proto.predict(geo_features[test_m])
            proto_metrics = compute_metrics(y[test_m], proto_preds, None, au_set)
            all_results["proto"] = proto_metrics
            logger.info(
                "  Proto: macro-F1=%.3f, balanced_acc=%.3f",
                proto_metrics.get("f1_macro", 0),
                proto_metrics.get("balanced_acc_macro", 0),
            )
    except Exception as e:
        logger.warning("Prototypical classifier failed: %s", e)

    # Neural models: train/evaluate if enough data
    cid_to_split = dict(zip(dataset_df["clip_id"], dataset_df["split"]))
    split_arr = np.array([cid_to_split.get(cid, "train") for cid in clip_ids])

    y, _ = prepare_labels(dataset_df, clip_ids, au_set)
    train_mask = split_arr == "train"
    val_mask = split_arr == "val"
    test_mask = split_arr == "test"

    if val_mask.sum() == 0:
        val_mask = test_mask

    # Save checkpoints directory
    ckpt_dir = Path(cfg["paths"]["results_root"]) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if train_mask.sum() >= 10 and val_mask.sum() >= 3:
        device = cfg["primateface"]["device"]
        import torch

        # G2: TCN
        logger.info("=== Training G2: TCN ===")
        try:
            from .models.temporal_cnn import predict_tcn, train_tcn

            tcn_model, tcn_history = train_tcn(
                seq_features[train_mask],
                y[train_mask],
                seq_features[val_mask],
                y[val_mask],
                cfg["models"]["g2_tcn"],
                device=device,
            )
            torch.save(tcn_model.state_dict(), ckpt_dir / "g2_tcn.pt")
            tcn_preds, tcn_proba = predict_tcn(
                tcn_model, seq_features[test_mask], device=device
            )
            tcn_metrics = compute_metrics(
                y[test_mask], tcn_preds, tcn_proba, au_set
            )
            all_results["tcn"] = tcn_metrics
            all_results["tcn_history"] = tcn_history
            logger.info("  TCN test: macro-F1=%.3f", tcn_metrics.get("f1_macro", 0))
        except Exception as e:
            logger.warning("TCN training failed: %s", e)

        # G3: ST-GCN
        logger.info("=== Training G3: ST-GCN ===")
        try:
            from .models.stgcn import predict_stgcn, train_stgcn

            stgcn_model, stgcn_history = train_stgcn(
                seq_features[train_mask],
                y[train_mask],
                seq_features[val_mask],
                y[val_mask],
                cfg["models"]["g3_stgcn"],
                device=device,
            )
            torch.save(stgcn_model.state_dict(), ckpt_dir / "g3_stgcn.pt")
            stgcn_preds, stgcn_proba = predict_stgcn(
                stgcn_model, seq_features[test_mask], device=device
            )
            stgcn_metrics = compute_metrics(
                y[test_mask], stgcn_preds, stgcn_proba, au_set
            )
            all_results["stgcn"] = stgcn_metrics
            all_results["stgcn_history"] = stgcn_history
            logger.info("  ST-GCN test: macro-F1=%.3f", stgcn_metrics.get("f1_macro", 0))
        except Exception as e:
            logger.warning("ST-GCN training failed: %s", e)

        # G4: Two-Stream Face GCN (adaptive, joint+bone)
        logger.info("=== Training G4: Two-Stream Face GCN ===")
        try:
            from primateface.analysis.face_gcn import TwoStreamFaceGCN
            from primateface.analysis.losses import AsymmetricLoss

            g4_cfg = cfg["models"].get("g3_stgcn", {})  # reuse stgcn config
            g4_model = TwoStreamFaceGCN(
                num_classes=y.shape[1],
                channels=g4_cfg.get("channels", [32, 64]),
                temporal_kernel=g4_cfg.get("temporal_kernel", 9),
                dropout=g4_cfg.get("dropout", 0.3),
                adaptive=True,
            ).to(device)

            g4_optimizer = torch.optim.AdamW(
                g4_model.parameters(),
                lr=g4_cfg.get("lr", 1e-3),
                weight_decay=g4_cfg.get("weight_decay", 1e-4),
            )
            g4_epochs = g4_cfg.get("epochs", 100)
            g4_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                g4_optimizer, g4_epochs
            )
            g4_criterion = AsymmetricLoss()
            g4_patience = g4_cfg.get("patience", 15)

            from .models.stgcn import GraphSequenceDataset

            g4_train_ds = GraphSequenceDataset(
                seq_features[train_mask], y[train_mask], augment=True
            )
            g4_val_ds = GraphSequenceDataset(
                seq_features[val_mask], y[val_mask], augment=False
            )
            from torch.utils.data import DataLoader

            g4_train_dl = DataLoader(g4_train_ds, batch_size=16, shuffle=True)
            g4_val_dl = DataLoader(g4_val_ds, batch_size=16)

            best_val = float("inf")
            best_state = None
            wait = 0
            g4_history = {"train_loss": [], "val_loss": []}

            for epoch in range(g4_epochs):
                g4_model.train()
                t_loss = 0.0
                for xb, yb in g4_train_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = g4_model(xb)
                    loss = g4_criterion(logits, yb)
                    g4_optimizer.zero_grad()
                    loss.backward()
                    g4_optimizer.step()
                    t_loss += loss.item() * xb.size(0)
                t_loss /= len(g4_train_ds)

                g4_model.eval()
                v_loss = 0.0
                with torch.no_grad():
                    for xb, yb in g4_val_dl:
                        xb, yb = xb.to(device), yb.to(device)
                        v_loss += g4_criterion(g4_model(xb), yb).item() * xb.size(0)
                v_loss /= max(len(g4_val_ds), 1)

                g4_scheduler.step()
                g4_history["train_loss"].append(t_loss)
                g4_history["val_loss"].append(v_loss)

                if v_loss < best_val:
                    best_val = v_loss
                    best_state = {
                        k: v.cpu().clone()
                        for k, v in g4_model.state_dict().items()
                    }
                    wait = 0
                else:
                    wait += 1

                if (epoch + 1) % 10 == 0:
                    logger.info(
                        "  2s-GCN epoch %d/%d: train=%.4f val=%.4f",
                        epoch + 1, g4_epochs, t_loss, v_loss,
                    )
                if wait >= g4_patience:
                    logger.info("  Early stopping at epoch %d", epoch + 1)
                    break

            if best_state is not None:
                g4_model.load_state_dict(best_state)
            g4_model.eval()
            torch.save(g4_model.state_dict(), ckpt_dir / "g4_2s_gcn.pt")

            # Predict
            g4_probs_list = []
            g4_test_ds = GraphSequenceDataset(
                seq_features[test_mask], np.zeros((test_mask.sum(), 1))
            )
            g4_test_dl = DataLoader(g4_test_ds, batch_size=32)
            with torch.no_grad():
                for xb, _ in g4_test_dl:
                    xb = xb.to(device)
                    probs = torch.sigmoid(g4_model(xb)).cpu().numpy()
                    g4_probs_list.append(probs)
            g4_probs = np.concatenate(g4_probs_list)
            g4_preds = (g4_probs >= 0.5).astype(int)

            g4_metrics = compute_metrics(y[test_mask], g4_preds, g4_probs, au_set)
            all_results["two_stream_gcn"] = g4_metrics
            all_results["two_stream_gcn_history"] = g4_history
            logger.info(
                "  2s-GCN test: macro-F1=%.3f", g4_metrics.get("f1_macro", 0)
            )
        except Exception as e:
            logger.warning("Two-Stream GCN training failed: %s", e)

        # G5: Pre-trained STGCN (NTU60 body → face transfer)
        logger.info("=== Training G5: Pre-trained STGCN (NTU60 transfer) ===")
        try:
            from primateface.analysis.pretrained_stgcn import PretrainedFaceSTGCN
            from primateface.analysis.losses import AsymmetricLoss as ASL5

            g5_model = PretrainedFaceSTGCN(
                num_classes=y.shape[1], freeze_blocks=7
            ).to(device)

            g5_opt = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, g5_model.parameters()),
                lr=5e-4, weight_decay=1e-4,
            )
            g5_sched = torch.optim.lr_scheduler.CosineAnnealingLR(g5_opt, 100)
            g5_loss_fn = ASL5()

            g5_train_ds = GraphSequenceDataset(
                seq_features[train_mask], y[train_mask], augment=True
            )
            g5_val_ds = GraphSequenceDataset(
                seq_features[val_mask], y[val_mask], augment=False
            )
            g5_train_dl = DataLoader(g5_train_ds, batch_size=16, shuffle=True)
            g5_val_dl = DataLoader(g5_val_ds, batch_size=16)

            best_v5 = float("inf")
            best_s5 = None
            wait5 = 0
            for ep in range(100):
                g5_model.train()
                tl = 0.0
                for xb, yb in g5_train_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    loss = g5_loss_fn(g5_model(xb), yb)
                    g5_opt.zero_grad()
                    loss.backward()
                    g5_opt.step()
                    tl += loss.item() * xb.size(0)
                tl /= len(g5_train_ds)

                g5_model.eval()
                vl = 0.0
                with torch.no_grad():
                    for xb, yb in g5_val_dl:
                        xb, yb = xb.to(device), yb.to(device)
                        vl += g5_loss_fn(g5_model(xb), yb).item() * xb.size(0)
                vl /= max(len(g5_val_ds), 1)
                g5_sched.step()

                if vl < best_v5:
                    best_v5 = vl
                    best_s5 = {
                        k: v.cpu().clone() for k, v in g5_model.state_dict().items()
                    }
                    wait5 = 0
                else:
                    wait5 += 1
                if (ep + 1) % 10 == 0:
                    logger.info(
                        "  Pretrained-STGCN epoch %d: train=%.4f val=%.4f",
                        ep + 1, tl, vl,
                    )
                if wait5 >= 15:
                    logger.info("  Early stopping at epoch %d", ep + 1)
                    break

            if best_s5:
                g5_model.load_state_dict(best_s5)
            g5_model.eval()
            torch.save(g5_model.state_dict(), ckpt_dir / "g5_pretrained_stgcn.pt")

            g5_probs_list = []
            g5_test_ds = GraphSequenceDataset(
                seq_features[test_mask], np.zeros((test_mask.sum(), 1))
            )
            g5_test_dl = DataLoader(g5_test_ds, batch_size=32)
            with torch.no_grad():
                for xb, _ in g5_test_dl:
                    xb = xb.to(device)
                    g5_probs_list.append(
                        torch.sigmoid(g5_model(xb)).cpu().numpy()
                    )
            g5_probs = np.concatenate(g5_probs_list)
            g5_preds = (g5_probs >= 0.5).astype(int)
            g5_metrics = compute_metrics(
                y[test_mask], g5_preds, g5_probs, au_set
            )
            all_results["pretrained_stgcn"] = g5_metrics
            logger.info(
                "  Pretrained-STGCN test: macro-F1=%.3f",
                g5_metrics.get("f1_macro", 0),
            )
        except Exception as e:
            logger.warning("Pre-trained STGCN failed: %s", e)
    else:
        logger.info(
            "Skipping neural models: not enough data (train=%d, val=%d)",
            train_mask.sum(),
            val_mask.sum(),
        )

    logger.info("Checkpoints saved to %s", ckpt_dir)
    return all_results
