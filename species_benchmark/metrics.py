"""Classification metrics for hierarchical taxon prediction."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax.

    Args:
        logits: Array of logits.
        axis: Axis to normalize over.

    Returns:
        Probabilities with the same shape as ``logits``.
    """
    z = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Top-1 expected calibration error.

    Args:
        probs: Probabilities of shape (n, k).
        labels: Integer labels of shape (n,).
        n_bins: Number of equal-width confidence bins.

    Returns:
        ECE in [0, 1].
    """
    conf = probs.max(1)
    correct = probs.argmax(1) == labels
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def level_metrics(logits: np.ndarray, labels: np.ndarray, class_counts: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Metrics for one taxonomic level, ignoring rows with label < 0.

    Args:
        logits: Logits of shape (n, k).
        labels: Integer labels of shape (n,), -1 for unlabeled at this level.
        class_counts: Optional training counts per class, used to split macro-F1 into
            head (top third by count) and tail (bottom third) classes.

    Returns:
        Dict with n, top1, top5, macro_f1, balanced_acc, ece (and head/tail F1).
    """
    m = labels >= 0
    if m.sum() == 0:
        return {"n": 0}
    lg, y = logits[m], labels[m]
    probs = softmax(lg)
    pred = probs.argmax(1)
    k = min(5, lg.shape[1])
    top5 = np.argsort(-lg, axis=1)[:, :k]
    present = np.unique(y)
    out = {
        "n": int(m.sum()),
        "n_classes_present": int(len(present)),
        "top1": float((pred == y).mean()),
        "top5": float((top5 == y[:, None]).any(1).mean()),
        "macro_f1": float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "ece": expected_calibration_error(probs, y),
    }
    if class_counts is not None and len(present) >= 6:
        order = present[np.argsort(class_counts[present])]
        third = max(1, len(order) // 3)
        out["macro_f1_tail"] = float(f1_score(y, pred, labels=order[:third], average="macro", zero_division=0))
        out["macro_f1_head"] = float(f1_score(y, pred, labels=order[-third:], average="macro", zero_division=0))
    return out


def coverage_curve(
    genus_logits: np.ndarray,
    species_logits: np.ndarray,
    genus_labels: np.ndarray,
    species_labels: np.ndarray,
    species_to_genus: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> List[Dict[str, float]]:
    """Abstention roll-up: answer species when confident, else genus.

    A species answer is correct if it matches the species label; a genus answer is
    correct if it matches the genus label. Evaluated on rows with a species label.

    Args:
        genus_logits: (n, G) genus logits.
        species_logits: (n, S) species logits.
        genus_labels: (n,) genus labels.
        species_labels: (n,) species labels (-1 = none).
        species_to_genus: (S,) genus index for each species class.
        thresholds: Species-confidence thresholds to sweep.

    Returns:
        List of dicts with threshold, species_coverage, accuracy (hierarchical).
    """
    thresholds = thresholds or [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
    m = species_labels >= 0
    ps = softmax(species_logits[m])
    pg = softmax(genus_logits[m])
    sp_pred, sp_conf = ps.argmax(1), ps.max(1)
    g_pred = pg.argmax(1)
    rows = []
    for t in thresholds:
        use_sp = sp_conf >= t
        correct = np.where(use_sp, sp_pred == species_labels[m], g_pred == genus_labels[m])
        rows.append(
            {
                "threshold": t,
                "species_coverage": float(use_sp.mean()),
                "hier_accuracy": float(correct.mean()),
                "species_acc_when_answered": float((sp_pred == species_labels[m])[use_sp].mean())
                if use_sp.any()
                else float("nan"),
                "genus_consistency": float((species_to_genus[sp_pred] == g_pred).mean()),
            }
        )
    return rows
