"""G1: Random Forest / XGBoost baseline for AU prediction.

Geometric features → multi-label binary AU prediction.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("animalfacs.models.baseline_rf")


class GeometricAUClassifier:
    """Multi-label AU classifier from geometric landmark features.

    Wraps sklearn MultiOutputClassifier with feature scaling.

    Args:
        model_type: "rf" for RandomForest or "xgb" for GradientBoosting.
        model_params: Kwargs passed to the base estimator.
    """

    def __init__(
        self,
        model_type: str = "rf",
        model_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_type = model_type
        self.model_params = model_params or {}
        self.scaler = StandardScaler()
        self.model: Optional[MultiOutputClassifier] = None
        self.au_columns: List[int] = []

    def _make_estimator(self) -> Any:
        """Create the base estimator."""
        seed = self.model_params.get("seed", 42)
        if self.model_type == "rf":
            return RandomForestClassifier(
                n_estimators=self.model_params.get("n_estimators", 200),
                max_depth=self.model_params.get("max_depth", 10),
                min_samples_leaf=self.model_params.get("min_samples_leaf", 5),
                class_weight=self.model_params.get("class_weight", "balanced"),
                random_state=seed,
                n_jobs=1,  # deterministic
            )
        else:
            return GradientBoostingClassifier(
                n_estimators=self.model_params.get("n_estimators", 200),
                max_depth=self.model_params.get("max_depth", 6),
                learning_rate=self.model_params.get("learning_rate", 0.1),
                random_state=seed,
            )

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        au_columns: List[int],
    ) -> "GeometricAUClassifier":
        """Fit the multi-label classifier.

        Args:
            x: (N, D) feature matrix.
            y: (N, C) binary label matrix.
            au_columns: List of AU integers corresponding to y columns.

        Returns:
            self.
        """
        self.au_columns = au_columns
        x_scaled = self.scaler.fit_transform(x)

        # Handle case where some AUs have only one class in training
        valid_cols = []
        for c in range(y.shape[1]):
            if len(np.unique(y[:, c])) > 1:
                valid_cols.append(c)

        if not valid_cols:
            logger.warning("No AU columns with both positive and negative examples.")
            self.model = None
            return self

        y_valid = y[:, valid_cols]
        self._valid_cols = valid_cols
        self._valid_aus = [au_columns[c] for c in valid_cols]

        base = self._make_estimator()
        self.model = MultiOutputClassifier(base, n_jobs=-1)
        self.model.fit(x_scaled, y_valid)

        logger.info(
            "Fit %s on %d samples, %d features, %d AUs",
            self.model_type.upper(),
            x.shape[0],
            x.shape[1],
            len(valid_cols),
        )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict AU labels.

        Args:
            x: (N, D) feature matrix.

        Returns:
            (N, C) binary prediction matrix (full AU set).
        """
        if self.model is None:
            return np.zeros((x.shape[0], len(self.au_columns)), dtype=int)

        x_scaled = self.scaler.transform(x)
        y_pred_valid = self.model.predict(x_scaled)

        # Map back to full AU columns
        y_pred = np.zeros((x.shape[0], len(self.au_columns)), dtype=int)
        for i, c in enumerate(self._valid_cols):
            y_pred[:, c] = y_pred_valid[:, i]
        return y_pred

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict AU probabilities.

        Args:
            x: (N, D) feature matrix.

        Returns:
            (N, C) probability matrix.
        """
        if self.model is None:
            return np.zeros((x.shape[0], len(self.au_columns)))

        x_scaled = self.scaler.transform(x)
        probas = np.zeros((x.shape[0], len(self.au_columns)))

        for i, est in enumerate(self.model.estimators_):
            col = self._valid_cols[i]
            p = est.predict_proba(x_scaled)
            # predict_proba returns (N, 2) — take probability of class 1
            if p.shape[1] == 2:
                probas[:, col] = p[:, 1]
            else:
                probas[:, col] = p[:, 0]
        return probas

    def feature_importances(self) -> Dict[int, np.ndarray]:
        """Get per-AU feature importances.

        Returns:
            Dict mapping AU number to importance array of shape (D,).
        """
        if self.model is None:
            return {}

        importances = {}
        for i, est in enumerate(self.model.estimators_):
            au = self._valid_aus[i]
            importances[au] = est.feature_importances_
        return importances


def prepare_labels(
    dataset_df: "pd.DataFrame",
    clip_ids: List[str],
    au_set: List[int],
) -> Tuple[np.ndarray, List[int]]:
    """Build binary label matrix from dataset DataFrame.

    Args:
        dataset_df: DataFrame with 'clip_id' and 'normalized_labels'.
        clip_ids: Ordered clip IDs matching feature matrix rows.
        au_set: List of AU integers to predict.

    Returns:
        Tuple of (N, C) binary label matrix and the AU list.
    """
    # Build lookup
    label_lookup: Dict[str, List[int]] = {}
    for _, row in dataset_df.iterrows():
        cid = row["clip_id"]
        labels = row["normalized_labels"]
        if isinstance(labels, str):
            labels = [int(x) for x in labels.split(",") if x.strip()]
        label_lookup[cid] = labels

    y = np.zeros((len(clip_ids), len(au_set)), dtype=int)
    for i, cid in enumerate(clip_ids):
        clip_aus = set(label_lookup.get(cid, []))
        for j, au in enumerate(au_set):
            if au in clip_aus:
                y[i, j] = 1

    return y, au_set
