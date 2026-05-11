"""
kmeans_regime.py — K-Means unsupervised market regime discovery.

Clusters 4H feature vectors into K data-driven market regimes without
pre-defined labels.  Cluster IDs are surfaced as an additional feature
(`kmeans_regime_id`, normalised to [0,1]) fed to the GRU and regime
classifier, helping both models condition on discovered regime structure
beyond the hand-crafted hh_hl / lh_ll labels.

Why K-Means alongside supervised regime?
  - Supervised regime classes (TRENDING/RANGING/CONSOLIDATING/VOLATILE)
    are defined by rule-based labels that may not match true market clusters.
  - K-Means discovers *data-driven* groupings: it can find, e.g., a
    "low-volatility trend" regime distinct from "high-volatility trend".
  - Cluster assignment is a fast O(K × D) lookup at inference time.
  - Feature importances from the RF model will reveal if kmeans_regime_id
    actually adds signal; if not, it can be zeroed out via env flag.

Architecture:
  - sklearn KMeans (k-means++ init, n_jobs=-1)
  - K defaults to 8 (env: KMEANS_N_CLUSTERS); range 4–16 typical
  - Input: REGIME_4H_FEATURES (34 features) — same as HTF regime classifier
  - StandardScaler applied before clustering (persisted with model)
  - cluster_id output: integer 0 to K-1, normalised to [0,1] for use as feature

Usage as feature:
  - At backtest/live inference, after building the 4H feature matrix,
    call kmeans.predict_normalised(X_4h) → float array [0,1]
  - This becomes column `kmeans_regime_id` in the feature frame
  - Added to SEQUENCE_FEATURES and REGIME_4H_FEATURES via feature_engine.py

Weights:
  - weights/kmeans_regime/model.pkl    — KMeans object
  - weights/kmeans_regime/scaler.pkl   — StandardScaler
  - weights/kmeans_regime/meta.json    — k, inertia, cluster_sizes, feature_version
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

KMEANS_FEATURE_VERSION = 1

# ── Feature contract ──────────────────────────────────────────────────────────
# Subset of REGIME_4H_FEATURES — excludes symbol_group_code and macro_vix_level
# (categorical/macro context shouldn't dominate cluster geometry).
KMEANS_FEATURES: List[str] = [
    "adx_14_base",
    "ema_stack_score",
    "mtf_1d_adx",
    "mtf_1d_ema_stack",
    "mtf_1d_atr_ratio",
    "rsi_14",
    "rsi_slope_5",
    "macd_line_atr",
    "macd_hist_atr",
    "macd_hist_slope_3",
    "efficiency_ratio",
    "plus_di",
    "minus_di",
    "di_spread",
    "adx_slope_10",
    "ema_50_slope",
    "ema_200_slope",
    "ema_50_dist_atr",
    "ema_200_dist_atr",
    "atr_percentile_500",
    "rolling_vol_percentile",
    "bb_width_percentile",
    "rolling_range_percentile",
    "range_expansion_zscore",
    "wick_ratio",
    "hh_hl_structure",
    "lh_ll_structure",
    "external_trend_direction",
    "external_structure_score",
    "internal_structure_state",
    "swing_sequence_score",
    "directional_bars_20",
    "trend_age_40",
    "candle_body_atr",
]

N_KMEANS_FEATURES = len(KMEANS_FEATURES)

_MODEL_ROOT = Path(__file__).resolve().parent.parent
WEIGHT_DIR   = _MODEL_ROOT / "weights" / "kmeans_regime"
MODEL_FILE   = WEIGHT_DIR / "model.pkl"
SCALER_FILE  = WEIGHT_DIR / "scaler.pkl"
META_FILE    = WEIGHT_DIR / "meta.json"

KMEANS_N_CLUSTERS_DEFAULT = 8


class KMeansRegimeModel:
    """
    Unsupervised K-Means market regime clustering.

    Public API:
      - is_trained      → bool
      - n_clusters      → int
      - train(X)        → metrics dict
      - predict(X)      → np.ndarray int cluster IDs [0, K-1]
      - predict_normalised(X) → np.ndarray float [0, 1]
      - cluster_sizes   → dict {cluster_id: count} from training
      - save() / load()
    """

    def __init__(self):
        self._model = None
        self._scaler = None
        self._meta: dict = {}
        self._load_if_exists()

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    @property
    def n_clusters(self) -> int:
        if self._model is None:
            return 0
        return int(self._model.n_clusters)

    @property
    def cluster_sizes(self) -> dict:
        return self._meta.get("cluster_sizes", {})

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        n_clusters: Optional[int] = None,
        n_init: int = 20,
        max_iter: int = 500,
        random_state: int = 42,
    ) -> dict:
        """
        Fit K-Means on 4H feature matrix.

        X: shape (N, N_KMEANS_FEATURES) — one row per 4H bar
        Returns metrics dict with inertia, silhouette score, cluster_sizes.
        """
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        assert X.shape[1] == N_KMEANS_FEATURES, (
            f"KMeansRegime: expected {N_KMEANS_FEATURES} features, got {X.shape[1]}"
        )

        k = n_clusters or int(os.environ.get("KMEANS_N_CLUSTERS", KMEANS_N_CLUSTERS_DEFAULT))
        k = max(2, min(16, k))

        logger.info("KMeansRegime train: N=%d k=%d", len(X), k)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        np.nan_to_num(X_scaled, nan=0.0, posinf=3.0, neginf=-3.0, copy=False)
        X_scaled = np.clip(X_scaled, -5.0, 5.0)

        km = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
            n_jobs=-1 if hasattr(KMeans(), "n_jobs") else None,
        )
        km.fit(X_scaled)

        self._model  = km
        self._scaler = scaler

        labels = km.labels_
        cluster_sizes = {
            int(c): int(np.sum(labels == c)) for c in range(k)
        }

        metrics: dict = {
            "n_clusters": k,
            "n_train": len(X),
            "inertia": float(km.inertia_),
            "cluster_sizes": cluster_sizes,
            "feature_version": KMEANS_FEATURE_VERSION,
        }

        # Silhouette score (sampled for speed on large datasets)
        try:
            from sklearn.metrics import silhouette_score
            sample = min(10000, len(X_scaled))
            idx = np.random.default_rng(42).choice(len(X_scaled), size=sample, replace=False)
            sil = float(silhouette_score(X_scaled[idx], labels[idx], sample_size=None))
            metrics["silhouette"] = round(sil, 4)
            logger.info("KMeansRegime silhouette=%.4f inertia=%.1f", sil, km.inertia_)
        except Exception as exc:
            logger.warning("KMeansRegime silhouette failed: %s", exc)

        logger.info("KMeansRegime cluster sizes: %s", cluster_sizes)
        self._meta = metrics
        self.save()
        return metrics

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Assign cluster IDs to rows of X.
        X: shape (N, N_KMEANS_FEATURES)
        Returns: shape (N,) int32 cluster IDs in [0, K-1]
        """
        if self._model is None or self._scaler is None:
            return np.zeros(len(X), dtype=np.int32)
        X_scaled = self._scaler.transform(X)
        np.nan_to_num(X_scaled, nan=0.0, posinf=3.0, neginf=-3.0, copy=False)
        X_scaled = np.clip(X_scaled, -5.0, 5.0)
        return self._model.predict(X_scaled).astype(np.int32)

    def predict_normalised(self, X: np.ndarray) -> np.ndarray:
        """
        Assign cluster IDs normalised to [0, 1].
        Returns float32 array suitable for use as a continuous feature.
        """
        k = self.n_clusters
        if k == 0:
            return np.full(len(X), 0.0, dtype=np.float32)
        ids = self.predict(X)
        return (ids.astype(np.float32) / max(1, k - 1))

    def predict_single(self, row: dict) -> tuple:
        """
        Single-row cluster assignment.
        Returns (cluster_id: int, normalised: float)
        """
        x = np.array(
            [float(row.get(f, 0.0) or 0.0) for f in KMEANS_FEATURES],
            dtype=np.float32,
        ).reshape(1, -1)
        ids = self.predict(x)
        k   = self.n_clusters
        return int(ids[0]), float(ids[0]) / max(1, k - 1)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(self._model, f, protocol=5)
        with open(SCALER_FILE, "wb") as f:
            pickle.dump(self._scaler, f, protocol=5)
        with open(META_FILE, "w") as f:
            json.dump(self._meta, f, indent=2)
        logger.info("KMeansRegimeModel saved to %s (k=%d)", MODEL_FILE, self.n_clusters)

    def load(self) -> bool:
        if not MODEL_FILE.exists():
            return False
        try:
            with open(MODEL_FILE, "rb") as f:
                self._model = pickle.load(f)
            if SCALER_FILE.exists():
                with open(SCALER_FILE, "rb") as f:
                    self._scaler = pickle.load(f)
            if META_FILE.exists():
                with open(META_FILE) as f:
                    self._meta = json.load(f)
                if self._meta.get("feature_version") != KMEANS_FEATURE_VERSION:
                    logger.warning(
                        "KMeansRegime: weight feature_version=%s != current=%s — stale weights",
                        self._meta.get("feature_version"), KMEANS_FEATURE_VERSION,
                    )
            logger.info(
                "KMeansRegimeModel loaded from %s (k=%d)",
                MODEL_FILE, self.n_clusters,
            )
            return True
        except Exception as exc:
            logger.warning("KMeansRegimeModel.load failed: %s", exc)
            self._model = None
            self._scaler = None
            return False

    def _load_if_exists(self) -> None:
        self.load()


def build_kmeans_feature_matrix(df: "pd.DataFrame") -> np.ndarray:
    """
    Build KMeans input matrix from a DataFrame with KMEANS_FEATURES columns.
    Missing columns filled with 0. Returns (N, N_KMEANS_FEATURES) float32.
    """
    out = np.zeros((len(df), N_KMEANS_FEATURES), dtype=np.float32)
    for i, col in enumerate(KMEANS_FEATURES):
        if col in df.columns:
            vals = df[col].to_numpy(dtype=np.float32)
            np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
            out[:, i] = vals
    return out


