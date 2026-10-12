"""Tests for split grouping, label space, and metrics."""

import numpy as np
import pandas as pd

from species_benchmark.labels import assign_splits, build_groups, build_label_space
from species_benchmark.metrics import coverage_curve, expected_calibration_error, level_metrics


def _toy_manifest(n_obs: int = 60) -> pd.DataFrame:
    rows = []
    for i in range(n_obs):
        sp = ["Macaca mulatta", "Macaca fuscata", "Pan troglodytes"][i % 3]
        for k in range(2):
            rows.append(
                {
                    "crop_id": f"c{i}_{k}",
                    "observer_id": f"u{i // 2}",  # two observations per observer
                    "group_id": f"obs{i}",
                    "image_sha256": f"h{i}_{k}",
                    "species": sp,
                    "genus": sp.split()[0],
                    "family": "Hominidae" if sp.startswith("Pan") else "Cercopithecidae",
                }
            )
    return pd.DataFrame(rows)


def test_groups_merge_shared_observer_and_hash():
    df = _toy_manifest()
    df.loc[5, "image_sha256"] = df.loc[100, "image_sha256"]  # duplicated photo across observers
    g = build_groups(df)
    assert g.iloc[0] == g.iloc[2]  # same observer u0
    assert g.iloc[5] == g.iloc[100]
    assert g.nunique() < len(df)


def test_splits_have_no_group_leakage_and_hold_out_each_class():
    df = _toy_manifest()
    df["group"] = build_groups(df)
    df["split"] = assign_splits(df, "species", test_frac=0.2, val_frac=0.1).values
    assert (df.groupby("group")["split"].nunique() == 1).all()
    assert set(df.loc[df["split"] == "test", "species"]) == set(df["species"])


def test_label_space_min_species():
    df = _toy_manifest()
    df.loc[df["species"] == "Pan troglodytes", "species"] = None
    space = build_label_space(df, min_species=5)
    assert "Pan troglodytes" not in space["species"]
    assert "Pan" in space["genus"]
    assert space["species_to_genus"]["Macaca mulatta"] == "Macaca"


def test_level_metrics_perfect_and_ignore():
    logits = np.eye(4) * 10
    labels = np.array([0, 1, 2, -1])
    m = level_metrics(logits, labels)
    assert m["n"] == 3 and m["top1"] == 1.0 and m["macro_f1"] == 1.0


def test_ece_zero_for_calibrated_onehot():
    probs = np.eye(3)
    assert expected_calibration_error(probs, np.arange(3)) == 0.0


def test_coverage_curve_rolls_up_to_genus():
    # species head wrong but unconfident, genus right -> high threshold gives correct answers
    sp_logits = np.array([[0.1, 0.0], [0.0, 0.1]])
    g_logits = np.array([[5.0, 0.0], [5.0, 0.0]])
    rows = coverage_curve(g_logits, sp_logits, np.array([0, 0]), np.array([1, 0]), np.array([0, 0]), [0.0, 0.9])
    assert rows[0]["species_coverage"] == 1.0 and rows[0]["hier_accuracy"] == 0.0
    assert rows[1]["species_coverage"] == 0.0 and rows[1]["hier_accuracy"] == 1.0
