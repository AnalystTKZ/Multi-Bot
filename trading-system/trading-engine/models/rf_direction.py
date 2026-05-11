"""
rf_direction.py — Random Forest directional classifier.

Trains on a snapshot of the last bar of each GRU sequence window (tabular
features) and predicts p_bull_rf ∈ [0, 1].  Its output is blended with the
GRU direction signal:

    p_bull_blend = (1 - RF_BLEND_WEIGHT) * p_bull_gru + RF_BLEND_WEIGHT * p_bull_rf

RF_BLEND_WEIGHT defaults to 0.30 (env-configurable).

Features: a fixed 30-feature subset from SEQUENCE_FEATURES (last bar of each
window — no sequence needed). This list is version-locked; adding features
requires bumping RF_FEATURE_VERSION and re-training.

Why Random Forest alongside GRU?
  - Tree ensembles consistently outperform neural nets on tabular data at
    moderate feature counts (Grinsztajn et al. 2022; e-forex.net 2025).
  - Interpretable: feature_importances_ surfaces what's actually predictive.
  - Trains in minutes vs. hours; fast to retrain after each backtest round.
  - Provides ensemble diversity: RF captures monotone / threshold relationships
    that the GRU sequence model may under-weight.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

RF_FEATURE_VERSION = 1

# ── Feature contract ──────────────────────────────────────────────────────────
# Subset of SEQUENCE_FEATURES (last-bar snapshot only — no sequential context).
# Order fixed; do not re-sort.
RF_FEATURES: List[str] = [
    "rsi_14",
    "rsi_slope_5",
    "macd_line_atr",
    "macd_hist_atr",
    "macd_hist_slope_3",
    "macd_cross_age",
    "adx_15m",
    "adx_cat",
    "atr_pctile",
    "vol_expansion",
    "ema21_dist",
    "ema50_dist",
    "bb_position",
    "ema_pullback_zone",
    "ema21_slope_15m",
    "ema_stack_15m",
    "hh_hl_structure",
    "lh_ll_structure",
    "candle_body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "sweep_wick_depth_atr",
    "body_recovery_ratio",
    "vwap_dist_atr",
    "vwap_band_position",
    "external_trend_direction",
    "external_structure_score",
    "internal_structure_state",
    "swing_sequence_score",
    "atr_normalized",
]

N_RF_FEATURES = len(RF_FEATURES)

_MODEL_ROOT = Path(__file__).resolve().parent.parent
WEIGHT_DIR  = _MODEL_ROOT / "weights" / "rf_direction"
WEIGHT_FILE = WEIGHT_DIR / "model.pkl"
META_FILE   = WEIGHT_DIR / "meta.json"

RF_BLEND_WEIGHT_DEFAULT = 0.30


class RFDirectionClassifier:
    """
    Random Forest directional classifier — tabular ensemble member.

    Public API mirrors other model classes:
      - is_trained  → bool
      - train(X, y) → dict of metrics
      - predict_proba(row_dict) → float (p_bull)
      - predict_proba_batch(X: np.ndarray) → np.ndarray (p_bull per row)
      - save() / load()
    """

    def __init__(self):
        self._model = None
        self._feature_importances: Optional[np.ndarray] = None
        self._meta: dict = {}
        self._load_if_exists()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    @property
    def feature_importances(self) -> Optional[np.ndarray]:
        return self._feature_importances

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_estimators: int = 400,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 20,
        class_weight: str = "balanced",
        random_state: int = 42,
        val_X: Optional[np.ndarray] = None,
        val_y: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Train Random Forest on tabular snapshot features.

        X: shape (N, N_RF_FEATURES) — last-bar features per window
        y: shape (N,) — 1 = bull (price rose to TP), 0 = bear/neutral

        Returns metrics dict.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, roc_auc_score

        assert X.shape[1] == N_RF_FEATURES, (
            f"RFDirectionClassifier: expected {N_RF_FEATURES} features, got {X.shape[1]}"
        )

        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)
        logger.info(
            "RFDirection train: N=%d pos=%d neg=%d ratio=%.3f",
            len(y), n_pos, n_neg, n_pos / max(1, len(y)),
        )

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=random_state,
        )
        rf.fit(X, y)

        self._model = rf
        self._feature_importances = rf.feature_importances_

        train_preds = rf.predict_proba(X)[:, 1]
        metrics: dict = {
            "n_train": len(y),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "train_acc": float(accuracy_score(y, (train_preds >= 0.5).astype(int))),
            "train_auc": float(roc_auc_score(y, train_preds)) if n_pos > 0 and n_neg > 0 else float("nan"),
            "feature_version": RF_FEATURE_VERSION,
        }

        # Top-5 feature importances for log
        top5_idx = np.argsort(self._feature_importances)[::-1][:5]
        top5 = [(RF_FEATURES[i], float(self._feature_importances[i])) for i in top5_idx]
        metrics["top5_features"] = top5
        logger.info("RFDirection top-5 importances: %s", top5)

        if val_X is not None and val_y is not None and len(val_y) > 0:
            val_preds = rf.predict_proba(val_X)[:, 1]
            n_val_pos = int(val_y.sum())
            n_val_neg = int(len(val_y) - n_val_pos)
            metrics["val_acc"] = float(accuracy_score(val_y, (val_preds >= 0.5).astype(int)))
            if n_val_pos > 0 and n_val_neg > 0:
                metrics["val_auc"] = float(roc_auc_score(val_y, val_preds))
            logger.info(
                "RFDirection val: N=%d acc=%.4f auc=%.4f",
                len(val_y), metrics["val_acc"], metrics.get("val_auc", float("nan")),
            )

        self.save()
        return metrics

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, row: dict) -> float:
        """Single-row inference. Returns p_bull ∈ [0, 1]."""
        if self._model is None:
            return 0.5
        x = np.array(
            [float(row.get(f, 0.0) or 0.0) for f in RF_FEATURES],
            dtype=np.float32,
        ).reshape(1, -1)
        return float(self._model.predict_proba(x)[0, 1])

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Batch inference.

        X: shape (N, N_RF_FEATURES)
        Returns: shape (N,) float32 p_bull probabilities
        """
        if self._model is None:
            return np.full(len(X), 0.5, dtype=np.float32)
        return self._model.predict_proba(X)[:, 1].astype(np.float32)

    def feature_importance_report(self) -> List[tuple]:
        """Returns sorted (feature_name, importance) list."""
        if self._feature_importances is None:
            return []
        idx = np.argsort(self._feature_importances)[::-1]
        return [(RF_FEATURES[i], float(self._feature_importances[i])) for i in idx]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
        with open(WEIGHT_FILE, "wb") as f:
            pickle.dump(self._model, f, protocol=5)
        meta = {
            "feature_version": RF_FEATURE_VERSION,
            "n_features": N_RF_FEATURES,
            "features": RF_FEATURES,
            "importances": (
                self._feature_importances.tolist()
                if self._feature_importances is not None
                else []
            ),
        }
        with open(META_FILE, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("RFDirectionClassifier saved to %s", WEIGHT_FILE)

    def load(self) -> bool:
        if not WEIGHT_FILE.exists():
            return False
        try:
            with open(WEIGHT_FILE, "rb") as f:
                self._model = pickle.load(f)
            if META_FILE.exists():
                with open(META_FILE) as f:
                    self._meta = json.load(f)
                if self._meta.get("feature_version") != RF_FEATURE_VERSION:
                    logger.warning(
                        "RFDirection: weight feature_version=%s != current=%s — weights may be stale",
                        self._meta.get("feature_version"), RF_FEATURE_VERSION,
                    )
                imp = self._meta.get("importances")
                if imp:
                    self._feature_importances = np.array(imp, dtype=np.float32)
            logger.info("RFDirectionClassifier loaded from %s", WEIGHT_FILE)
            return True
        except Exception as exc:
            logger.warning("RFDirectionClassifier.load failed: %s", exc)
            self._model = None
            return False

    def _load_if_exists(self) -> None:
        self.load()


def blend_p_bull(p_bull_gru: float, p_bull_rf: float, weight: float = RF_BLEND_WEIGHT_DEFAULT) -> float:
    """
    Weighted blend of GRU and RF bull probabilities.
    weight = fraction assigned to RF (default 0.30).
    """
    return float((1.0 - weight) * p_bull_gru + weight * p_bull_rf)


def build_rf_feature_matrix(df_features: "pd.DataFrame") -> np.ndarray:
    """
    Build RF input matrix from a DataFrame of per-bar features.
    Expects columns matching RF_FEATURES; missing columns filled with 0.
    Returns shape (N, N_RF_FEATURES) float32.
    """
    import pandas as pd
    out = np.zeros((len(df_features), N_RF_FEATURES), dtype=np.float32)
    for i, col in enumerate(RF_FEATURES):
        if col in df_features.columns:
            vals = df_features[col].to_numpy(dtype=np.float32)
            np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
            out[:, i] = vals
    return out
