"""
regime_classifier.py — GPU-native PyTorch MLP hierarchical regime classifier.

Hierarchical market structure framework:
  HTF classifier (4H) — "What is overall direction?" (mode="htf_bias")
    3 classes: 0=BIAS_UP, 1=BIAS_DOWN, 2=BIAS_NEUTRAL
  LTF score model (1H) — "How is price behaving NOW?" (mode="ltf_behaviour")
    5 independent scores: trend/range/chop/volatility/consolidation

Architecture: N_FEATURES → 128 → 64 → N_CLASSES  (BN + Dropout + residual skip)
DataParallel across both T4 GPUs during training and batch inference.
3-bar hysteresis: regime must persist 3 bars before switching.
"""

from __future__ import annotations

import copy
import logging
import os
import pickle
import threading
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from models.base_model import BaseModel
from services.feature_engine import REGIME_4H_FEATURES, REGIME_1H_FEATURES
from services.regime_scores import LTF_SCORE_COLUMNS

logger = logging.getLogger(__name__)

# ── New hierarchical class definitions ───────────────────────────────────────
HTF_CLASSES = ["BIAS_UP", "BIAS_DOWN", "BIAS_NEUTRAL"]          # 3-class HTF bias
LTF_CLASSES = ["TRENDING", "RANGING", "CONSOLIDATING", "VOLATILE"]  # legacy derived labels
LTF_SCORE_OUTPUTS = list(LTF_SCORE_COLUMNS)

# CLASSES kept as LTF_CLASSES for backward compat (most code paths use LTF or check by name)
CLASSES = LTF_CLASSES
N_FEATURES  = len(REGIME_1H_FEATURES)   # default LTF score-head input width
N_CLASSES   = len(CLASSES)           # legacy default label count
_ALLOWED_FEATURE_NAMES = frozenset(REGIME_4H_FEATURES) | frozenset(REGIME_1H_FEATURES)

# Mapping: timeframe → feature names
_TF_FEATURE_MAP: dict = {
    "4H": REGIME_4H_FEATURES,
    "1H": REGIME_1H_FEATURES,
    None: REGIME_1H_FEATURES,
}
_MODEL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHT_PATH = os.path.join(_MODEL_ROOT, "weights", "regime_classifier.pkl")

# Per-mode weight paths for the hierarchical cascade.
# regime_htf.pkl: 3-class HTF bias (BIAS_UP/DOWN/NEUTRAL) — trained on 4H data.
# regime_ltf.pkl: 5-output LTF behaviour scores — trained on 1H data.
# Legacy paths kept for backward compat during transition.
WEIGHT_PATH_HTF = os.path.join(_MODEL_ROOT, "weights", "regime_htf.pkl")
WEIGHT_PATH_LTF = os.path.join(_MODEL_ROOT, "weights", "regime_ltf.pkl")
# Legacy aliases (old pkl files cold-start on n_classes mismatch detection — no manual deletion needed)
WEIGHT_PATH_4H = WEIGHT_PATH_HTF
WEIGHT_PATH_1H = WEIGHT_PATH_LTF


# ── Device selection ──────────────────────────────────────────────────────────

def _get_device():
    import torch
    # Remove any empty CUDA_VISIBLE_DEVICES mask
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        del os.environ["CUDA_VISIBLE_DEVICES"]
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        logger.info("RegimeClassifier: %d GPU(s) available — training on CUDA", n)
        for i in range(n):
            logger.info("  GPU %d: %s (%.1f GB)", i,
                        torch.cuda.get_device_name(i),
                        torch.cuda.get_device_properties(i).total_memory / 1e9)
        torch.backends.cudnn.benchmark        = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        return torch.device("cuda")
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") and not os.environ.get("INFERENCE_ONLY"):
        raise RuntimeError(
            "RegimeClassifier: CUDA not available on Kaggle — "
            "enable GPU accelerator in notebook settings."
        )
    logger.warning("RegimeClassifier: CUDA unavailable — using CPU")
    import torch as _t
    return _t.device("cpu")


DEVICE = _get_device()


# ── Model architecture ────────────────────────────────────────────────────────

def _build_mlp(n_features: int = N_FEATURES, n_classes: int = N_CLASSES):
    """
    53 → 256 → 128 → 64 → 4

    Each hidden block: Linear → BatchNorm → GELU → Dropout(0.3)
    Residual projection from input to first hidden output for gradient flow.
    """
    import torch.nn as nn

    class _RegimeMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_bn = nn.BatchNorm1d(n_features)

            self.fc1 = nn.Linear(n_features, 128)
            self.bn1 = nn.BatchNorm1d(128)

            self.fc2 = nn.Linear(128, 64)
            self.bn2 = nn.BatchNorm1d(64)

            self.head = nn.Linear(64, n_classes)

            self.drop = nn.Dropout(0.5)
            self.act  = nn.GELU()

            # Residual skip: input → 128
            self.skip = nn.Linear(n_features, 128, bias=False)

        def forward(self, x):
            x = self.input_bn(x)
            skip = self.skip(x)
            h = self.act(self.bn1(self.fc1(x))) + skip   # residual
            h = self.drop(h)
            h = self.act(self.bn2(self.fc2(h)))
            h = self.drop(h)
            return self.head(h)   # raw logits

    return _RegimeMLP()


class ModelNotTrainedError(RuntimeError):
    pass


# ── Classifier ────────────────────────────────────────────────────────────────

class RegimeClassifier(BaseModel):
    """
    GPU-native PyTorch MLP hierarchical regime classifier.

    timeframe: "4H" (HTF bias) | "1H" (LTF behaviour) | None (legacy default).
    mode: "htf_bias" → 3-class (BIAS_UP/DOWN/NEUTRAL) trained on 4H data.
          "ltf_behaviour" → 5 independent behaviour scores trained on 1H data.
    Each mode trains and saves to its own weight file so both can coexist.
    DataParallel is used across all available GPUs for both training and batch predict.
    """

    weight_path = WEIGHT_PATH

    _TF_TO_PATH = {
        "4H": WEIGHT_PATH_HTF,
        "1H": WEIGHT_PATH_LTF,
    }

    def __init__(self, timeframe: Optional[str] = None, mode: Optional[str] = None):
        super().__init__()
        self._model = None
        self._hysteresis_buffer: List[int] = []
        self._inference_lock = threading.RLock()
        self._timeframe = timeframe.upper() if timeframe else None

        # Determine mode: explicit > inferred from timeframe > default ltf_behaviour
        if mode is not None:
            self._mode = mode
        elif self._timeframe == "4H":
            self._mode = "htf_bias"
        elif self._timeframe == "1H":
            self._mode = "ltf_behaviour"
        else:
            self._mode = "ltf_behaviour"  # backward compat default

        # Pick class list and output size based on mode
        if self._mode == "htf_bias":
            self._class_list = HTF_CLASSES
            self._n_output_classes = len(HTF_CLASSES)  # 3
            self._output_type = "classification"
            self._current_regime_id: int = 2   # default BIAS_NEUTRAL
        else:
            self._class_list = LTF_CLASSES
            self._n_output_classes = len(LTF_SCORE_OUTPUTS)  # 5 independent scores
            self._output_type = "behaviour_scores"
            self._current_regime_id: int = 1   # default RANGING

        # Route weight file: per-TF/mode if specified, else legacy path
        self.weight_path = self._TF_TO_PATH.get(self._timeframe, WEIGHT_PATH)
        # Feature names and count for this TF's classifier
        self._feature_names = list(_TF_FEATURE_MAP.get(self._timeframe, _TF_FEATURE_MAP[None]))
        self._n_features = len(self._feature_names)
        self._htf_directional_threshold = float(os.getenv("REGIME_HTF_DIRECTIONAL_PROBA_THRESHOLD", "0.60"))
        self._htf_directional_margin = float(os.getenv("REGIME_HTF_DIRECTIONAL_MARGIN", "0.10"))
        logger.debug("RegimeClassifier[%s mode=%s]: %d features, %d classes, weight=%s",
                     self._timeframe or "default", self._mode, self._n_features,
                     self._n_output_classes, self.weight_path)
        os.makedirs(os.path.join(_MODEL_ROOT, "weights"), exist_ok=True)
        if self.is_trained:
            self.load(self.weight_path)
            self._last_mtime = os.path.getmtime(self.weight_path)

    # ── Predict ───────────────────────────────────────────────────────────────

    @staticmethod
    def _htf_bias_decision(
        proba: np.ndarray,
        threshold: float,
        margin: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Neutral-default HTF decision policy for imbalanced directional labels."""
        p = np.asarray(proba, dtype=np.float32)
        if p.ndim == 1:
            p2 = p.reshape(1, -1)
        else:
            p2 = p
        if p2.shape[1] != len(HTF_CLASSES):
            raise ValueError(f"HTF decision expected {len(HTF_CLASSES)} probabilities, got {p2.shape[1]}")
        p_up = p2[:, 0]
        p_down = p2[:, 1]
        p_neutral = p2[:, 2]
        labels = np.full(len(p2), 2, dtype=np.int32)
        up_ok = (p_up >= threshold) & (p_up >= p_neutral + margin) & (p_up >= p_down + margin)
        down_ok = (p_down >= threshold) & (p_down >= p_neutral + margin) & (p_down >= p_up + margin)
        labels[up_ok] = 0
        labels[down_ok] = 1
        confidence = np.where(labels == 0, p_up, np.where(labels == 1, p_down, p_neutral)).astype(np.float32)
        return labels, confidence

    @staticmethod
    def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: Sequence[str]) -> dict:
        n_cls = len(classes)
        confusion = np.zeros((n_cls, n_cls), dtype=np.int64)
        for true_id, pred_id in zip(y_true.astype(np.int64), y_pred.astype(np.int64)):
            if 0 <= true_id < n_cls and 0 <= pred_id < n_cls:
                confusion[true_id, pred_id] += 1
        recall = {}
        precision = {}
        f1 = {}
        for c, name in enumerate(classes):
            tp = float(confusion[c, c])
            fp = float(confusion[:, c].sum() - confusion[c, c])
            fn = float(confusion[c, :].sum() - confusion[c, c])
            rec = tp / (tp + fn + 1e-9)
            prec = tp / (tp + fp + 1e-9)
            f1_v = 2.0 * prec * rec / (prec + rec + 1e-9)
            recall[name] = rec
            precision[name] = prec
            f1[name] = f1_v
        accuracy = float((y_pred == y_true).mean()) if len(y_true) else 0.0
        balanced = float(np.mean(list(recall.values()))) if recall else 0.0
        return {
            "accuracy": accuracy,
            "balanced_accuracy": balanced,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "confusion": confusion,
        }

    @classmethod
    def _select_htf_bias_policy(cls, proba: np.ndarray, y_true: np.ndarray) -> tuple[float, float, dict]:
        min_precision = float(os.getenv("REGIME_MIN_DIRECTIONAL_PRECISION", "0.30"))
        min_recall = float(os.getenv("REGIME_MIN_DIRECTIONAL_RECALL", "0.20"))
        thresholds = np.linspace(0.40, 0.85, 10)
        margins = np.linspace(0.00, 0.25, 6)
        best: tuple[float, float, dict] | None = None
        best_score = -1e9
        for threshold in thresholds:
            for margin in margins:
                pred, _ = cls._htf_bias_decision(proba, float(threshold), float(margin))
                metrics = cls._classification_metrics(y_true, pred, HTF_CLASSES)
                up_p = metrics["precision"]["BIAS_UP"]
                down_p = metrics["precision"]["BIAS_DOWN"]
                up_r = metrics["recall"]["BIAS_UP"]
                down_r = metrics["recall"]["BIAS_DOWN"]
                up_f1 = metrics["f1"]["BIAS_UP"]
                down_f1 = metrics["f1"]["BIAS_DOWN"]
                neutral_r = metrics["recall"]["BIAS_NEUTRAL"]
                meets_floor = (
                    up_p >= min_precision
                    and down_p >= min_precision
                    and up_r >= min_recall
                    and down_r >= min_recall
                )
                score = (
                    2.0 * min(up_p, down_p)
                    + 1.5 * min(up_f1, down_f1)
                    + 0.75 * min(up_r, down_r)
                    + 0.50 * neutral_r
                    + metrics["accuracy"]
                )
                if meets_floor:
                    score += 10.0
                if score > best_score:
                    best_score = score
                    best = (float(threshold), float(margin), metrics)
        if best is None:
            pred, _ = cls._htf_bias_decision(proba, 0.60, 0.10)
            best = (0.60, 0.10, cls._classification_metrics(y_true, pred, HTF_CLASSES))
        return best

    def predict(self, df: Optional[pd.DataFrame], symbol: Optional[str] = None,
                df_htf: Optional[dict] = None,
                df_h4: Optional[pd.DataFrame] = None) -> Dict:
        if not self.is_trained or self._model is None:
            raise ModelNotTrainedError(
                "RegimeClassifier has no trained weights. "
                "Run: python scripts/retrain_incremental.py --model regime"
            )
        if df is None:
            raise ValueError("RegimeClassifier.predict: df cannot be None")

        self.reload_if_updated()
        if self._model is None:
            raise ModelNotTrainedError(
                "RegimeClassifier weights are missing or incompatible with the current regime architecture."
            )

        htf = dict(df_htf) if df_htf else {}
        if df_h4 is not None and "4H" not in htf:
            htf["4H"] = df_h4

        try:
            import torch
            X_feat = self._build_feature_matrix(
                df,
                htf,
                symbol,
                feature_names=self._feature_names,
            )
            if len(X_feat) == 0:
                raise ValueError("RegimeClassifier.predict built an empty feature matrix")
            feat = X_feat[-1]
            x = torch.tensor(feat.reshape(1, -1), dtype=torch.float32).to(DEVICE)

            # Unwrap DataParallel — DP cannot split a batch of 1 across 2 GPUs
            _infer_m = self._model.module if isinstance(self._model, torch.nn.DataParallel) else self._model
            _infer_m.eval()
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                    logits = _infer_m(x)
                logits_f = logits.float()
                if self._output_type == "behaviour_scores":
                    score_t = torch.sigmoid(logits_f)[0]
                    raw_scores = score_t.cpu().numpy().astype(np.float32)
                    score_payload = {
                        name: float(raw_scores[i])
                        for i, name in enumerate(LTF_SCORE_OUTPUTS)
                    }
                    # Backward-compatible aliases used by older gates/reports.
                    score_payload["volatility_score"] = score_payload["volatility_percentile"]
                    from services.regime_scores import build_regime_score_frame

                    primitive_df = build_regime_score_frame(df, symbol=symbol)
                    if primitive_df.empty:
                        raise ValueError("RegimeClassifier.predict primitive score frame is empty")
                    primitive_last = primitive_df.iloc[-1]
                    score_payload["efficiency_ratio_20"] = float(primitive_last["efficiency_ratio_20"])
                    score_payload["atr_percentile_500"] = float(primitive_last["atr_percentile_500"])
                    from services.regime_scores import classify_trade_regime, legacy_ltf_label_from_scores

                    legacy_label = legacy_ltf_label_from_scores(score_payload)
                    raw_id = LTF_CLASSES.index(legacy_label) if legacy_label in LTF_CLASSES else 1
                    proba = [float(score_payload[name]) for name in LTF_SCORE_OUTPUTS]
                    confidence = float(max(proba))
                    extra = {
                        "regime_scores": score_payload,
                        "trade_regime": classify_trade_regime(score_payload),
                    }
                else:
                    proba_t = torch.softmax(logits_f, dim=1)[0]
                    proba = proba_t.cpu().numpy().tolist()
                    ids, conf = self._htf_bias_decision(
                        np.asarray(proba, dtype=np.float32),
                        self._htf_directional_threshold,
                        self._htf_directional_margin,
                    )
                    raw_id = int(ids[0])
                    confidence = float(conf[0])
                    extra = {}

            # 3-bar hysteresis
            self._hysteresis_buffer.append(raw_id)
            if len(self._hysteresis_buffer) > 3:
                self._hysteresis_buffer.pop(0)
            if len(self._hysteresis_buffer) == 3 and len(set(self._hysteresis_buffer)) == 1:
                self._current_regime_id = self._hysteresis_buffer[0]

            return {
                "regime":             self._class_list[self._current_regime_id],
                "regime_id":          self._current_regime_id,
                "proba":              proba,
                "regime_confidence":  confidence,
                **extra,
            }
        except Exception as exc:
            logger.error("RegimeClassifier.predict failed: %s", exc)
            raise

    def predict_batch(self, X: np.ndarray, batch_size: int = 4096) -> tuple:
        """
        CPU inference on a pre-built feature matrix X (N, F_tf).
        Returns (labels: np.ndarray int32, confidences: np.ndarray float32).

        Small MLP (input→128→64→classes): CPU is faster than GPU for this because
        the H2D transfer and kernel launch overhead dominates at these batch sizes.
        GPU is reserved for the GRU which has large recurrent state and benefits
        from CUDA parallelism.
        """
        labels, conf, _scores = self.predict_batch_scores(X, batch_size=batch_size)
        return labels, conf

    def predict_batch_scores(self, X: np.ndarray, batch_size: int = 4096) -> tuple:
        """
        Batch inference with optional LTF behaviour score matrix.

        Returns (labels, confidences, scores). For HTF classification scores is
        None. For LTF behaviour scores is an (N, 5) matrix ordered by
        LTF_SCORE_OUTPUTS and labels are only the backward-compatible derived
        4-class behaviour ids.
        """
        if not self.is_trained or self._model is None:
            raise ModelNotTrainedError(
                "RegimeClassifier has no trained weights. "
                "Run: python scripts/retrain_incremental.py --model regime"
            )
        import torch
        from services.regime_scores import legacy_ltf_label_from_scores

        self.reload_if_updated()
        if self._model is None:
            raise ModelNotTrainedError(
                "RegimeClassifier weights are missing or incompatible with the current regime architecture."
            )
        with self._inference_lock:
            if X.shape[1] != len(self._feature_names):
                raise ValueError(
                    f"RegimeClassifier[{self._timeframe} mode={self._mode}] expected "
                    f"{len(self._feature_names)} features {self._feature_names}; got {X.shape[1]}"
                )
            # Clone to CPU for inference — never mutate the live shared model.
            _raw = self._model.module if isinstance(self._model, torch.nn.DataParallel) else self._model
            _cpu_m = copy.deepcopy(_raw).to("cpu").eval()
            all_labels = []
            all_conf   = []
            all_scores = []
            with torch.no_grad():
                for s in range(0, len(X), batch_size):
                    xb = torch.from_numpy(X[s: s + batch_size])  # stays on CPU
                    logits = _cpu_m(xb).float()
                    if self._output_type == "behaviour_scores":
                        scores = torch.sigmoid(logits).numpy().astype(np.float32)
                        labels = []
                        for row in scores:
                            payload = {
                                name: float(row[i])
                                for i, name in enumerate(LTF_SCORE_OUTPUTS)
                            }
                            payload["volatility_score"] = payload.get("volatility_percentile", 0.0)
                            label = legacy_ltf_label_from_scores(payload)
                            labels.append(LTF_CLASSES.index(label) if label in LTF_CLASSES else 1)
                        all_labels.append(np.asarray(labels, dtype=np.int32))
                        all_conf.append(scores.max(axis=1).astype(np.float32))
                        all_scores.append(scores)
                    else:
                        proba = torch.softmax(logits, dim=1).numpy()
                        labels, conf = self._htf_bias_decision(
                            proba,
                            self._htf_directional_threshold,
                            self._htf_directional_margin,
                        )
                        all_labels.append(labels.astype(np.int32))
                        all_conf.append(conf.astype(np.float32))
            score_mat = np.concatenate(all_scores) if all_scores else None
            return np.concatenate(all_labels), np.concatenate(all_conf), score_mat

    # ── Labels ────────────────────────────────────────────────────────────────

    def create_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Institutional-grade regime labeling via unsupervised GMM clustering.

        Uses 4 components to match LTF_CLASSES exactly:
          VOLATILE (3):       highest (vol - eff) — chaotic expansion
          TRENDING (0):       highest efficiency + highest abs(drift) — direction-agnostic
          CONSOLIDATING (2):  lowest atr_pctile + lowest autocorr — pre-breakout compression
          RANGING (1):        remainder — moderate vol, near-zero drift

        Mirrors fit_global_gmm (ltf_behaviour) exactly so per-symbol and global
        paths produce consistent semantics. Raises if sklearn is not available
        or if the data is insufficient — callers must supply clean data.
        """
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler

        n_bar = self._TF_NBAR.get(self._timeframe, self._DEFAULT_NBAR)
        feat_df, _ = self._extract_gmm_features(df, n_bar=n_bar)

        if len(feat_df) < 50:
            raise ValueError(
                f"create_labels: insufficient data ({len(feat_df)} rows after feature extraction, "
                f"need ≥50). Provide a longer history or use create_rule_labels()."
            )

        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(feat_df.values)

        n_components = 3 if self._mode == "htf_bias" else 4
        gmm = GaussianMixture(n_components=n_components, covariance_type="full",
                              random_state=42, max_iter=200)
        cluster_ids = gmm.fit_predict(X_scaled)

        centroids = scaler.inverse_transform(gmm.means_)
        # centroids cols: [eff, vol, drift, comp, vol_slope, atr_pctile, autocorr, hurst_proxy]
        remaining = list(range(n_components))
        cluster_labels: dict[int, int] = {}

        if self._mode == "htf_bias":
            bu_c = max(remaining, key=lambda c: centroids[c, 2])
            cluster_labels[bu_c] = 0
            remaining.remove(bu_c)
            bd_c = min(remaining, key=lambda c: centroids[c, 2])
            cluster_labels[bd_c] = 1
            remaining.remove(bd_c)
            cluster_labels[remaining[0]] = 2
        else:
            # VOLATILE (3): highest (vol - eff)
            vol_c = max(remaining, key=lambda c: centroids[c, 1] - centroids[c, 0])
            cluster_labels[vol_c] = 3
            remaining.remove(vol_c)
            # TRENDING (0): highest efficiency + highest abs(drift), direction-agnostic
            tr_c = max(remaining, key=lambda c: centroids[c, 0] + abs(centroids[c, 2]))
            cluster_labels[tr_c] = 0
            remaining.remove(tr_c)
            # CONSOLIDATING (2): lowest atr_pctile + lowest autocorr
            consol = min(remaining, key=lambda c: centroids[c, 5] + max(centroids[c, 6], 0))
            cluster_labels[consol] = 2
            remaining.remove(consol)
            # RANGING (1): the remainder
            cluster_labels[remaining[0]] = 1

        default_id = 2 if self._mode == "htf_bias" else 1
        labels = pd.Series(default_id, index=df.index, dtype=int)
        labels.loc[feat_df.index] = [cluster_labels[int(c)] for c in cluster_ids]
        return labels.astype(int)

    # Lookback window per timeframe — must span roughly one full regime cycle.
    # 4H: 50 bars ≈ 2.5 weeks (trend impulse); 1H: 24 bars ≈ 1 trading day;
    # 15M/default: 14 bars ≈ 3.5 hours. Using 14 everywhere collapses all
    # 4H distributions to nearly identical centroids → poor GMM separation.
    _TF_NBAR: dict = {"4H": 50, "1H": 24, "15M": 14, "5M": 10}
    _TF_LABEL_HORIZON: dict = {"4H": 12, "1H": 12, "15M": 16, "5M": 24}
    _DEFAULT_NBAR = 14

    @staticmethod
    def _infer_nbar_from_index(index: pd.Index) -> int:
        """Infer the regime lookback from bar spacing so features match labels."""
        try:
            if len(index) >= 3:
                deltas = pd.Series(index).diff().dropna()
                minutes = float(deltas.median().total_seconds() / 60.0)
                if minutes <= 7:
                    return RegimeClassifier._TF_NBAR["5M"]
                if minutes <= 30:
                    return RegimeClassifier._TF_NBAR["15M"]
                if minutes <= 90:
                    return RegimeClassifier._TF_NBAR["1H"]
                if minutes <= 300:
                    return RegimeClassifier._TF_NBAR["4H"]
        except Exception:
            pass
        return RegimeClassifier._DEFAULT_NBAR

    @staticmethod
    def _extract_gmm_features(df: pd.DataFrame, n_bar: int = 14) -> tuple[pd.DataFrame, pd.Index]:
        """Extract 8 GMM labeling features from a single df. Returns (feat_df, valid_index).

        Features:
          eff        — efficiency ratio: how directional price movement is [0→1]
          vol        — relative volatility: ATR / close
          drift      — signed n-bar momentum / close (positive=up, negative=down)
          comp       — price range / ATR: how wide the range is vs current noise
          vol_slope  — Δ(ATR/close) over n_bar: volatility expanding or contracting
          atr_pctile — ATR percentile rank in own 3×n_bar history [0→1]
          autocorr   — lag-1 autocorrelation of log-returns over n_bar window.
                       High autocorr (>0) = momentum/trending. Near-zero = ranging.
                       Mean-reverting = negative. This is a true time-series discriminator.
          hurst_proxy — simplified Hurst exponent proxy: ratio of n_bar range to
                       sqrt(n_bar) × realized vol. H>1 ≈ trending, H<1 ≈ mean-reverting.
                       Uses log-log slope approximation: range_n / range_1 vs sqrt(n).
        """
        from indicators.market_structure import compute_atr
        close = df["close"]
        atr = compute_atr(df, n_bar)
        abs_moves = np.abs(close.diff()).rolling(n_bar, min_periods=n_bar).sum()
        net_move  = np.abs(close - close.shift(n_bar))
        eff_ratio = (net_move / (abs_moves + 1e-9)).clip(0, 1)
        rel_vol   = atr / (close + 1e-9)
        drift     = (close - close.shift(n_bar)) / (n_bar * close + 1e-9)
        hi        = df["high"].rolling(n_bar, min_periods=n_bar).max()
        lo        = df["low"].rolling(n_bar, min_periods=n_bar).min()
        compression = (hi - lo) / (atr + 1e-9)
        vol_slope = rel_vol.diff(n_bar)
        _hist_window = n_bar * 3
        atr_pctile = atr.rolling(_hist_window, min_periods=n_bar).apply(
            lambda x: float(np.searchsorted(np.sort(x[:-1]), x[-1])) / max(len(x) - 1, 1)
            if len(x) > 1 else 0.5, raw=True
        ).clip(0.0, 1.0)

        # Lag-1 autocorrelation of log-returns over a rolling n_bar window.
        # Trending markets have positive autocorr (momentum); ranging markets ≈ 0;
        # mean-reverting have negative autocorr. This is the core time-series discriminator
        # that pure cross-sectional features (ADX, ATR) miss.
        log_ret = np.log(close / close.shift(1))
        autocorr = log_ret.rolling(n_bar, min_periods=max(4, n_bar // 2)).apply(
            lambda x: float(pd.Series(x).autocorr(lag=1)) if len(x) > 3 else 0.0,
            raw=False
        ).fillna(0.0).clip(-1.0, 1.0)

        # Hurst exponent proxy: R/S statistic over n_bar window, normalized.
        # R/S ∝ n^H; we approximate H by comparing range at n_bar vs range at n_bar//2.
        # H > 0.5 → trending (persistent), H < 0.5 → mean-reverting (anti-persistent).
        # Proxy: (range_n / range_half) / sqrt(2) — equals 1.0 at H=0.5, >1 at H>0.5.
        range_n    = (hi - lo).clip(1e-9)
        hi_half    = df["high"].rolling(max(2, n_bar // 2), min_periods=2).max()
        lo_half    = df["low"].rolling(max(2, n_bar // 2), min_periods=2).min()
        range_half = (hi_half - lo_half).clip(1e-9)
        hurst_proxy = (range_n / range_half / (2 ** 0.5)).clip(0.2, 3.0)

        feat_df = pd.DataFrame({
            "eff": eff_ratio, "vol": rel_vol, "drift": drift,
            "comp": compression, "vol_slope": vol_slope, "atr_pctile": atr_pctile,
            "autocorr": autocorr, "hurst_proxy": hurst_proxy,
        }).dropna()
        return feat_df, feat_df.index

    @staticmethod
    def fit_global_gmm(dfs: list[pd.DataFrame], timeframe: str = None,
                       mode: str = "ltf_behaviour") -> tuple:
        """
        Fit one GMM on combined features from all dfs and return (gmm, scaler, cluster_labels).
        Call once before labeling — guarantees consistent regime semantics across all symbols/TFs.
        timeframe: used to pick the correct lookback window ("4H", "1H", "15M", etc.).
        mode: "htf_bias" → 3 clusters (BIAS_UP/DOWN/NEUTRAL by drift direction).
              "ltf_behaviour" → 4 clusters (TRENDING/RANGING/CONSOLIDATING/VOLATILE by behaviour).
        """
        try:
            from sklearn.mixture import GaussianMixture
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            return None, None, None

        n_bar = RegimeClassifier._TF_NBAR.get(timeframe, RegimeClassifier._DEFAULT_NBAR)
        n_components = 3 if mode == "htf_bias" else 4
        logger.info("GMM fit: timeframe=%s mode=%s → n_bar=%d n_components=%d",
                    timeframe or "default", mode, n_bar, n_components)

        all_feats = []
        for df in dfs:
            feat_df, _ = RegimeClassifier._extract_gmm_features(df, n_bar=n_bar)
            if len(feat_df) >= 50:
                # Subsample to avoid XAUUSD (5× longer) dominating the GMM
                step = max(1, len(feat_df) // 10_000)
                all_feats.append(feat_df.iloc[::step].values)

        if not all_feats:
            return None, None, None

        X_all = np.concatenate(all_feats, axis=0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)

        gmm = GaussianMixture(n_components=n_components, covariance_type="full",
                              random_state=42, max_iter=300)
        gmm.fit(X_scaled)

        centroids = scaler.inverse_transform(gmm.means_)
        # centroids cols: [eff, vol, drift, comp, vol_slope, atr_pctile, autocorr, hurst_proxy]
        remaining = list(range(n_components))
        cluster_labels: dict[int, int] = {}

        if mode == "htf_bias":
            # HTF (3-class): assign by drift direction — most-distinguishable first
            # BIAS_UP (0): highest signed drift
            bu_c = max(remaining, key=lambda c: centroids[c, 2])
            cluster_labels[bu_c] = 0
            remaining.remove(bu_c)
            # BIAS_DOWN (1): lowest signed drift
            bd_c = min(remaining, key=lambda c: centroids[c, 2])
            cluster_labels[bd_c] = 1
            remaining.remove(bd_c)
            # BIAS_NEUTRAL (2): the remainder
            cluster_labels[remaining[0]] = 2
        else:
            # LTF (4-class): assign by behaviour profile
            # VOLATILE (3): highest (vol - eff) — chaotic expansion
            vol_c = max(remaining, key=lambda c: centroids[c, 1] - centroids[c, 0])
            cluster_labels[vol_c] = 3
            remaining.remove(vol_c)
            # TRENDING (0): highest efficiency AND highest abs(drift)
            tr_c = max(remaining, key=lambda c: centroids[c, 0] + abs(centroids[c, 2]))
            cluster_labels[tr_c] = 0
            remaining.remove(tr_c)
            # CONSOLIDATING (2): lowest atr_pctile + lowest autocorr (pre-breakout compression)
            consol = min(remaining, key=lambda c: centroids[c, 5] + max(centroids[c, 6], 0))
            cluster_labels[consol] = 2
            remaining.remove(consol)
            # RANGING (1): the last cluster — moderate vol, near-zero drift
            cluster_labels[remaining[0]] = 1

        class_list = HTF_CLASSES if mode == "htf_bias" else LTF_CLASSES
        dist = {class_list[v]: 0 for v in range(n_components)}
        for v in cluster_labels.values():
            dist[class_list[v]] += 1
        logger.info("GMM fitted on %d samples (mode=%s) — cluster→regime: %s dist: %s",
                    len(X_all), mode, cluster_labels, dist)
        return gmm, scaler, cluster_labels

    def create_labels_with_gmm(self, df: pd.DataFrame, gmm, scaler, cluster_labels: dict,
                               n_bar: int = None) -> pd.Series:
        """Label a single df using a pre-fitted global GMM (consistent across all symbols).
        n_bar: lookback used when the GMM was fitted — must match to get consistent features.
               Defaults to the classifier's own timeframe n_bar if not provided.
        """
        if n_bar is None:
            n_bar = self._TF_NBAR.get(self._timeframe, self._DEFAULT_NBAR)
        feat_df, _ = self._extract_gmm_features(df, n_bar=n_bar)
        labels = pd.Series(2, index=df.index, dtype=int)
        if len(feat_df) < 10 or gmm is None:
            return labels
        X_scaled = scaler.transform(feat_df.values)
        ids = gmm.predict(X_scaled)
        labels.loc[feat_df.index] = [cluster_labels[int(c)] for c in ids]
        return labels.astype(int)

    @staticmethod
    def _future_rolling(series: pd.Series, horizon: int, op: str) -> pd.Series:
        future = series.shift(-1).iloc[::-1]
        rolled = getattr(future.rolling(horizon, min_periods=horizon), op)()
        return rolled.iloc[::-1].reindex(series.index)

    @staticmethod
    def create_structural_labels(
        df: pd.DataFrame,
        timeframe: str = "4H",
        mode: str = "ltf_behaviour",
        return_confidence: bool = False,
        symbol: Optional[str] = None,
    ):
        """
        Outcome-aware regime labels for supervised regime-classifier training.

        These labels describe the realised forward path over the next regime
        horizon, while the model features remain strictly backward-looking.
        Do not feed these labels directly into GRU/sequence features.
        """
        from indicators.market_structure import compute_atr
        from services.regime_scores import build_regime_score_frame

        _tf = (timeframe or "4H").upper()
        horizon = int(RegimeClassifier._TF_LABEL_HORIZON.get(_tf, 12))
        n_bar = int(RegimeClassifier._TF_NBAR.get(_tf, RegimeClassifier._DEFAULT_NBAR))

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        atr = compute_atr(df, max(14, min(n_bar, 50))).astype(float).replace(0.0, np.nan)

        f_high = RegimeClassifier._future_rolling(high, horizon, "max")
        f_low = RegimeClassifier._future_rolling(low, horizon, "min")
        f_close = close.shift(-horizon)
        f_abs_path = (
            close.diff().abs().shift(-1).iloc[::-1]
            .rolling(horizon, min_periods=horizon).sum()
            .iloc[::-1].reindex(close.index)
        )

        up_exc = ((f_high - close) / (atr + 1e-9)).clip(lower=0.0)
        down_exc = ((close - f_low) / (atr + 1e-9)).clip(lower=0.0)
        terminal = (f_close - close) / (atr + 1e-9)
        f_range = ((f_high - f_low) / (atr + 1e-9)).clip(lower=0.0)
        efficiency = ((f_close - close).abs() / (f_abs_path + 1e-9)).clip(0.0, 1.0)

        valid = (
            up_exc.notna()
            & down_exc.notna()
            & terminal.notna()
            & f_range.notna()
            & efficiency.notna()
            & atr.notna()
        )
        if valid.any():
            dominant = pd.concat([up_exc, down_exc], axis=1).max(axis=1)
            trend_thr = max(0.80, float(dominant[valid].quantile(0.55)))
            range_hi = max(1.25, float(f_range[valid].quantile(0.75)))
            vol_thr = max(1.75, float(f_range[valid].quantile(0.80)))
            consol_thr = max(0.45, float(f_range[valid].quantile(0.25)))
        else:
            trend_thr, range_hi, vol_thr, consol_thr = 1.0, 1.5, 2.0, 0.6

        score_df = build_regime_score_frame(df, symbol=symbol, window=n_bar)
        plus_di = score_df["plus_di"]
        minus_di = score_df["minus_di"]
        adx_score = score_df.get("adx_14", pd.Series(0.0, index=df.index))
        ema50_slope = score_df["ema_50_slope"]
        ema50_dist = score_df["ema_50_dist_atr"]
        ema200_dist = score_df["ema_200_dist_atr"]
        er_now = score_df["efficiency_ratio_20"]
        trend_score = score_df["trend_score"]
        range_score = score_df["range_score"]
        chop_score = score_df["chop_score"]
        vol_score = score_df.get("volatility_percentile", score_df["volatility_score"])
        consol_score = score_df["consolidation_score"]
        atr_pctile = score_df["atr_percentile_500"]
        bbw_pctile = score_df["bb_width_percentile"]
        rolling_range_pctile = score_df["rolling_range_percentile"]
        range_exp_z = score_df["range_expansion_zscore"]

        if mode == "htf_bias":
            labels = pd.Series(2, index=df.index, dtype=int)
            conf = pd.Series(0.0, index=df.index, dtype=np.float32)

            spread = (up_exc - down_exc)
            dominance = (spread.abs() / (f_range + 1e-9)).clip(0.0, 1.0)
            dominant_exc = pd.concat([up_exc, down_exc], axis=1).max(axis=1)
            barrier_atr = float(os.getenv("REGIME_HTF_BARRIER_ATR", "0.80"))
            terminal_min = float(os.getenv("REGIME_HTF_TERMINAL_ATR", "0.25"))
            close_arr = close.to_numpy(dtype=np.float64, copy=False)
            high_arr = high.to_numpy(dtype=np.float64, copy=False)
            low_arr = low.to_numpy(dtype=np.float64, copy=False)
            atr_arr = atr.to_numpy(dtype=np.float64, copy=False)
            up_level = close_arr + barrier_atr * atr_arr
            down_level = close_arr - barrier_atr * atr_arr
            no_hit = horizon + 1
            up_hit = np.full(len(df), no_hit, dtype=np.int16)
            down_hit = np.full(len(df), no_hit, dtype=np.int16)
            for step_i in range(1, horizon + 1):
                future_high = np.empty_like(high_arr)
                future_low = np.empty_like(low_arr)
                future_high[:-step_i] = high_arr[step_i:]
                future_high[-step_i:] = np.nan
                future_low[:-step_i] = low_arr[step_i:]
                future_low[-step_i:] = np.nan
                hit_up_now = np.isfinite(future_high) & (future_high >= up_level) & (up_hit == no_hit)
                hit_down_now = np.isfinite(future_low) & (future_low <= down_level) & (down_hit == no_hit)
                up_hit[hit_up_now] = step_i
                down_hit[hit_down_now] = step_i
            terminal_arr = terminal.to_numpy(dtype=np.float64, copy=False)
            up_first = pd.Series(
                (up_hit <= horizon)
                & (
                    (down_hit > up_hit)
                    | ((down_hit == up_hit) & (terminal_arr > terminal_min))
                ),
                index=df.index,
            )
            down_first = pd.Series(
                (down_hit <= horizon)
                & (
                    (up_hit > down_hit)
                    | ((up_hit == down_hit) & (terminal_arr < -terminal_min))
                ),
                index=df.index,
            )
            up_path = up_first | (
                (up_exc >= barrier_atr * 1.25)
                & (terminal > terminal_min)
                & (dominance >= 0.35)
            )
            down_path = down_first | (
                (down_exc >= barrier_atr * 1.25)
                & (terminal < -terminal_min)
                & (dominance >= 0.35)
            )
            hh_hl_structure = score_df["hh_hl_structure"]
            lh_ll_structure = score_df["lh_ll_structure"]
            up_structure = (
                (plus_di > minus_di * 1.08)
                & ((adx_score > 18.0) | (trend_score > 0.55))
                & (ema50_slope > 0.015)
                & (
                    ((ema50_dist > 0.10) & (ema200_dist > 0.0))
                    | (score_df["bias_up_score"] > 0.60)
                )
                & (er_now > 0.24)
                & (hh_hl_structure >= lh_ll_structure * 0.80)
            )
            down_structure = (
                (minus_di > plus_di * 1.08)
                & ((adx_score > 18.0) | (trend_score > 0.55))
                & (ema50_slope < -0.015)
                & (
                    ((ema50_dist < -0.10) & (ema200_dist < 0.0))
                    | (score_df["bias_down_score"] > 0.60)
                )
                & (er_now > 0.24)
                & (lh_ll_structure >= hh_hl_structure * 0.80)
            )
            up_mask = (
                valid
                & up_structure
                & up_path
                & (up_exc >= max(barrier_atr, trend_thr * 0.60))
                & (up_exc >= down_exc * 1.15)
                & (terminal > terminal_min)
            )
            down_mask = (
                valid
                & down_structure
                & down_path
                & (down_exc >= max(barrier_atr, trend_thr * 0.60))
                & (down_exc >= up_exc * 1.15)
                & (terminal < -terminal_min)
            )
            labels[up_mask] = 0
            labels[down_mask] = 1

            directional_conf = (
                0.35 * (dominance / 0.60).clip(0.0, 1.0)
                + 0.25 * (dominant_exc / (trend_thr + 1e-9)).clip(0.0, 1.0)
                + 0.20 * efficiency
                + 0.20 * er_now
            ).clip(0.0, 1.0)
            conf[up_mask | down_mask] = (0.45 + 0.55 * directional_conf[up_mask | down_mask]).astype(np.float32)

            no_clear_path = ~(up_path | down_path)
            neutral_structure = (
                (
                    no_clear_path
                    | (dominance < 0.45)
                    | (dominant_exc < barrier_atr)
                    | (terminal.abs() < terminal_min)
                )
                & (
                    (adx_score < 20.0)
                    | (er_now < 0.32)
                    | (trend_score < 0.55)
                    | (ema50_slope.abs() < 0.05)
                )
            )
            neutral_mask = valid & neutral_structure & ~(up_mask | down_mask)
            neutral_conf = (
                0.30 * (1.0 - dominance).clip(0.0, 1.0)
                + 0.25 * (1.0 - (terminal.abs() / 0.65).clip(0.0, 1.0))
                + 0.20 * (1.0 - (er_now / 0.35).clip(0.0, 1.0))
                + 0.15 * (1.0 - (adx_score / 20.0).clip(0.0, 1.0))
                + 0.10 * (1.0 - (trend_score / 0.45).clip(0.0, 1.0))
            ).clip(0.0, 1.0)
            conf[neutral_mask] = (0.40 + 0.60 * neutral_conf[neutral_mask]).astype(np.float32)

            dist = {HTF_CLASSES[c]: int((labels == c).sum()) for c in range(len(HTF_CLASSES))}
            ambiguous = int((conf < 0.4).sum())
            logger.info(
                "Structural labels HTF_BIAS [%s]: %s  ambiguous=%d (total=%d) horizon=%d",
                timeframe or "?", dist, ambiguous, len(labels), horizon,
            )
            if return_confidence:
                return labels.astype(int), conf.astype(np.float32)
            return labels.astype(int)

        labels = pd.Series(1, index=df.index, dtype=int)
        conf = pd.Series(0.0, index=df.index, dtype=np.float32)

        dominance = ((up_exc - down_exc).abs() / (f_range + 1e-9)).clip(0.0, 1.0)
        two_sided = pd.concat([up_exc, down_exc], axis=1).min(axis=1)
        dominant = pd.concat([up_exc, down_exc], axis=1).max(axis=1)

        trend_mask = (
            valid
            & (trend_score >= 0.60)
            & (dominant >= max(0.60, trend_thr * 0.70))
            & (dominance >= 0.25)
            & (terminal.abs() >= 0.20)
            & (efficiency >= 0.28)
            & (chop_score < 0.70)
        )
        volatile_mask = (
            valid
            & ~trend_mask
            & (
                (vol_score >= 0.75)
                | (atr_pctile >= 0.85)
                | (range_exp_z >= 2.0)
                | (f_range >= vol_thr)
            )
        )
        consol_mask = (
            valid
            & ~trend_mask
            & ~volatile_mask
            & (
                (consol_score >= 0.60)
                | (
                    (atr_pctile <= 0.35)
                    & (bbw_pctile <= 0.35)
                    & (rolling_range_pctile <= 0.35)
                )
                | ((f_range <= consol_thr) & (dominant <= max(trend_thr, 1.25)))
            )
        )
        ranging_mask = (
            valid
            & ~trend_mask
            & ~volatile_mask
            & ~consol_mask
            & (range_score >= 0.40)
            & (trend_score < 0.58)
            & (chop_score < 0.90)
        )
        chop_mask = (
            valid
            & ~(trend_mask | volatile_mask | consol_mask | ranging_mask)
            & (chop_score >= 0.80)
            & (range_score < 0.40)
        )

        labels[trend_mask] = 0
        labels[ranging_mask] = 1
        labels[consol_mask] = 2
        labels[volatile_mask] = 3
        labels[chop_mask] = 1

        trend_conf = (
            0.35 * trend_score
            + 0.25 * (dominance / 0.70).clip(0.0, 1.0)
            + 0.20 * (dominant / (trend_thr + 1e-9)).clip(0.0, 1.0)
            + 0.20 * efficiency
        ).clip(0.0, 1.0)
        vol_conf = (
            0.45 * vol_score
            + 0.30 * (f_range / (vol_thr + 1e-9)).clip(0.0, 1.0)
            + 0.25 * (1.0 - efficiency).clip(0.0, 1.0)
        ).clip(0.0, 1.0)
        consol_conf = (
            0.60 * consol_score
            + 0.40 * (1.0 - (f_range / (consol_thr + 1e-9)).clip(0.0, 1.0))
        ).clip(0.0, 1.0)
        range_conf = (
            0.45 * range_score
            + 0.25 * (1.0 - dominance).clip(0.0, 1.0)
            + 0.15 * (two_sided / (0.75 + 1e-9)).clip(0.0, 1.0)
            + 0.15 * (1.0 - efficiency).clip(0.0, 1.0)
        ).clip(0.0, 1.0)

        conf[trend_mask] = (0.45 + 0.55 * trend_conf[trend_mask]).astype(np.float32)
        conf[volatile_mask] = (0.45 + 0.55 * vol_conf[volatile_mask]).astype(np.float32)
        conf[consol_mask] = (0.45 + 0.55 * consol_conf[consol_mask]).astype(np.float32)
        conf[ranging_mask] = (0.45 + 0.55 * range_conf[ranging_mask]).astype(np.float32)
        conf[chop_mask] = 0.0

        dist = {LTF_CLASSES[c]: int((labels == c).sum()) for c in range(len(LTF_CLASSES))}
        ambiguous = int((conf < 0.4).sum())
        logger.info(
            "Structural labels LTF_BEHAVIOUR [%s]: %s  ambiguous=%d (total=%d) horizon=%d",
            timeframe or "?", dist, ambiguous, len(labels), horizon,
        )
        if return_confidence:
            return labels.astype(int), conf.astype(np.float32)
        return labels.astype(int)

    @staticmethod
    def create_behaviour_score_targets(
        df: pd.DataFrame,
        timeframe: str = "1H",
        symbol: Optional[str] = None,
        return_confidence: bool = False,
    ):
        """
        Causal multi-output LTF behaviour targets.

        This replaces the old forced 4-class LTF target. Each bar receives five
        independent scores, so a market can be trending and volatile, or ranging
        and compressing, without losing information to a single softmax class.
        """
        from services.regime_scores import build_regime_score_frame

        _tf = (timeframe or "1H").upper()
        n_bar = int(RegimeClassifier._TF_NBAR.get(_tf, RegimeClassifier._DEFAULT_NBAR))
        score_df = build_regime_score_frame(df, symbol=symbol, window=n_bar)
        targets = (
            score_df[LTF_SCORE_OUTPUTS]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(0.0, 1.0)
            .astype(np.float32)
        )
        conf = pd.Series(1.0, index=df.index, dtype=np.float32)
        if return_confidence:
            return targets, conf
        return targets

    @staticmethod
    def create_rule_labels(
        df: pd.DataFrame,
        timeframe: str = "4H",
        mode: str = "ltf_behaviour",
        return_confidence: bool = False,
    ):
        """
        Mode-aware rule-based regime labels with per-bar confidence scores.

        mode="htf_bias" (4H): 3-class direction labels
          - BIAS_UP (0):      strong ADX, full bullish EMA stack, directional drift, trend efficiency
          - BIAS_DOWN (1):    strong ADX, full bearish EMA stack, directional drift, trend efficiency
          - BIAS_NEUTRAL (2): explicit low-ADX, weak-stack, low-drift, low-efficiency middle-volatility state
          Persistence: 20 bars (4H) for BIAS_UP/DOWN; 5 bars for BIAS_NEUTRAL.

        mode="ltf_behaviour" (1H): independent behaviour scores, not one forced
          flat class. The target columns are trend_score, range_score, chop_score,
          volatility_percentile, and consolidation_score. A derived trade_regime
          is produced downstream from those scores.

        Bars with confidence < 0.4 are "ambiguous" — by default retraining drops
        them before fitting the MLP so the classifier learns clean regime boundaries.

        Returns:
          labels pd.Series[int]                        (always)
          confidence pd.Series[float32]  (only if return_confidence=True)
        """
        from indicators.market_structure import compute_atr, compute_adx, compute_ema_stack_score

        _tf = (timeframe or "4H").upper()
        n_bar = RegimeClassifier._TF_NBAR.get(_tf, RegimeClassifier._DEFAULT_NBAR)
        close = df["close"]
        conf  = pd.Series(0.5, index=df.index, dtype=np.float32)

        adx       = df["adx_14"]    if "adx_14"    in df.columns else compute_adx(df, 14)
        ema_stack = df["ema_stack"] if "ema_stack" in df.columns else compute_ema_stack_score(df)
        drift = (close - close.shift(n_bar)) / (n_bar * close.shift(n_bar) + 1e-9)
        drift_abs = drift.abs()
        drift_p35 = float(drift_abs.quantile(0.35)) + 1e-9
        drift_p40 = float(drift_abs.quantile(0.40)) + 1e-9
        drift_p60 = float(drift_abs.quantile(0.60)) + 1e-9
        drift_p80 = float(drift_abs.quantile(0.80)) + 1e-9

        atr = compute_atr(df, n_bar)
        _hist = n_bar * 3
        atr_pctile = atr.rolling(_hist, min_periods=n_bar).apply(
            lambda x: float(np.searchsorted(np.sort(x[:-1]), x[-1])) / max(len(x) - 1, 1)
            if len(x) > 1 else 0.5, raw=True
        ).clip(0.0, 1.0).fillna(0.5)

        atr_slope = atr.rolling(n_bar, min_periods=max(2, n_bar // 2)).apply(
            lambda x: (x[-1] - x[0]) / (x[0] + 1e-9) if len(x) > 1 else 0.0, raw=True
        ).fillna(0.0)

        ret = close.pct_change()
        direction = (close - close.shift(n_bar)).abs()
        path = close.diff().abs().rolling(n_bar, min_periods=max(2, n_bar // 2)).sum()
        efficiency = (direction / (path + 1e-9)).fillna(0.0).clip(0.0, 1.0)
        autocorr_lag1 = ret.rolling(_hist, min_periods=n_bar).corr(ret.shift(1)).fillna(0.0).clip(-1.0, 1.0)
        hurst_proxy = (0.5 + 0.5 * autocorr_lag1).clip(0.0, 1.0)

        # ── HTF BIAS mode (3-class: direction-focused) ────────────────────────
        if mode == "htf_bias":
            labels = pd.Series(2, index=df.index, dtype=int)  # default BIAS_NEUTRAL
            conf[:] = 0.0

            trend_quality = (efficiency >= 0.40) & ((autocorr_lag1 >= 0.03) | (hurst_proxy >= 0.55))

            # BIAS_UP/DOWN require full EMA alignment and efficient movement.
            # Weak or conflicted directional bars remain neutral-labeled but
            # zero-confidence so training can drop them as ambiguous.
            bu_mask = (adx >= 28) & (ema_stack == 2) & (drift > drift_p60) & trend_quality
            labels[bu_mask] = 0
            adx_conf_bu   = ((adx - 28) / 22.0).clip(0.0, 1.0)
            stack_conf_bu = np.where(ema_stack == 2, 1.0, 0.0)
            drift_conf_bu = (drift.abs() / drift_p80).clip(0.0, 1.0)
            eff_conf_bu = efficiency.clip(0.0, 1.0)
            bu_conf = (adx_conf_bu * stack_conf_bu * drift_conf_bu * eff_conf_bu).astype(np.float32)
            conf[bu_mask] = (0.5 + 0.5 * pd.Series(bu_conf, index=df.index)[bu_mask]).astype(np.float32)

            bd_mask = (adx >= 28) & (ema_stack == -2) & (drift < -drift_p60) & trend_quality
            labels[bd_mask] = 1
            adx_conf_bd   = ((adx - 28) / 22.0).clip(0.0, 1.0)
            stack_conf_bd = np.where(ema_stack == -2, 1.0, 0.0)
            drift_conf_bd = (drift.abs() / drift_p80).clip(0.0, 1.0)
            eff_conf_bd = efficiency.clip(0.0, 1.0)
            bd_conf = (adx_conf_bd * stack_conf_bd * drift_conf_bd * eff_conf_bd).astype(np.float32)
            conf[bd_mask] = (0.5 + 0.5 * pd.Series(bd_conf, index=df.index)[bd_mask]).astype(np.float32)

            # BIAS_NEUTRAL is explicit market classification, not "not trend".
            neutral_clean = (
                (adx <= 20)
                & (ema_stack == 0)
                & (drift_abs <= drift_p35)
                & (efficiency <= 0.35)
                & (atr_pctile >= 0.25)
                & (atr_pctile <= 0.75)
            )
            labels[neutral_clean] = 2
            adx_neutral_conf = (1.0 - (adx / 20.0).clip(0.0, 1.0))
            drift_neutral_conf = (1.0 - (drift_abs / (drift_p35 + 1e-9)).clip(0.0, 1.0))
            eff_neutral_conf = (1.0 - (efficiency / 0.35).clip(0.0, 1.0))
            neutral_conf = (
                0.40 * adx_neutral_conf
                + 0.30 * drift_neutral_conf
                + 0.30 * eff_neutral_conf
            ).clip(0.0, 1.0)
            conf[neutral_clean] = (0.5 + 0.5 * neutral_conf[neutral_clean]).astype(np.float32)

            # ── Persistence filter (HTF) ──────────────────────────────────────
            # BIAS_UP/DOWN: a 4H structural bias should hold for at least 8 bars
            # (32 hours). Prior value of 96 on 1H = 4 days was too strict.
            _persist_by_class = {
                0: {"5M": 96, "15M": 32, "1H": 8, "4H": 8, "1D": 3},  # BIAS_UP
                1: {"5M": 96, "15M": 32, "1H": 8, "4H": 8, "1D": 3},  # BIAS_DOWN
                2: {"5M": 48, "15M": 16, "1H": 4, "4H": 3, "1D": 2},  # BIAS_NEUTRAL
            }
            _runs = (labels != labels.shift()).cumsum()
            _run_len = _runs.map(_runs.value_counts())
            _short_run_mask = pd.Series(False, index=labels.index)
            for cls_id, tf_thresholds in _persist_by_class.items():
                cls_mask = labels == cls_id
                min_p = tf_thresholds.get(_tf, tf_thresholds.get("4H", 5))
                _short_run_mask |= (cls_mask & (_run_len < min_p))
            conf[_short_run_mask] = 0.0

            dist = {HTF_CLASSES[c]: int((labels == c).sum()) for c in range(len(HTF_CLASSES))}
            ambiguous = int((conf < 0.4).sum())
            logger.info("Rule labels HTF_BIAS [%s]: %s  ambiguous=%d (total=%d)  short_runs_zeroed=%d",
                        timeframe or "?", dist, ambiguous, len(labels), int(_short_run_mask.sum()))

            labels = labels.astype(int)
            if return_confidence:
                return labels, conf.astype(np.float32)
            return labels

        # ── LTF BEHAVIOUR mode (4-class: behaviour-focused, direction-agnostic) ─
        else:  # ltf_behaviour (default)
            labels = pd.Series(1, index=df.index, dtype=int)  # default RANGING
            conf[:] = 0.0
            vol_thresh    = max(float(atr_pctile.quantile(0.80)), 0.85)
            consol_thresh = min(float(atr_pctile.quantile(0.25)), 0.20)
            trend_quality = (efficiency >= 0.45) & ((autocorr_lag1 >= 0.03) | (hurst_proxy >= 0.55))

            # VOLATILE (3): ATR expanding — chaotic, unpredictable
            volatile_mask = (atr_pctile >= vol_thresh) & (atr_slope > 0)
            labels[volatile_mask] = 3
            vol_conf = ((atr_pctile - vol_thresh) / (1.0 - vol_thresh + 1e-9)).clip(0.0, 1.0)
            conf[volatile_mask] = (0.5 + 0.5 * vol_conf[volatile_mask]).astype(np.float32)

            # TRENDING (0): directional momentum — direction-agnostic (we know direction from HTF)
            trend_mask = (
                (adx >= 28)
                & (ema_stack.abs() == 2)
                & (drift_abs > drift_p60)
                & trend_quality
                & (atr_pctile < 0.85)
                & ~volatile_mask
            )
            labels[trend_mask] = 0
            adx_conf_t   = ((adx - 28) / 22.0).clip(0.0, 1.0)
            stack_conf_t = np.where(ema_stack.abs() == 2, 1.0, 0.0)
            drift_conf_t = (drift_abs / drift_p80).clip(0.0, 1.0)
            eff_conf_t = efficiency.clip(0.0, 1.0)
            t_conf = (adx_conf_t * stack_conf_t * drift_conf_t * eff_conf_t).astype(np.float32)
            conf[trend_mask] = (0.5 + 0.5 * pd.Series(t_conf, index=df.index)[trend_mask]).astype(np.float32)

            # CONSOLIDATING (2): ATR at multi-period low AND falling — pre-breakout compression
            consol_mask = (
                (atr_pctile <= consol_thresh)
                & (atr_slope < 0)
                & (adx <= 22)
                & (efficiency <= 0.30)
                & ~volatile_mask
                & ~trend_mask
            )
            labels[consol_mask] = 2
            consol_atr_conf   = (1.0 - (atr_pctile / (consol_thresh + 1e-9)).clip(0.0, 1.0))
            consol_slope_conf = (-atr_slope).clip(0.0, 0.5) / 0.5
            consol_conf = (0.5 * consol_atr_conf + 0.5 * consol_slope_conf).clip(0.1, 1.0)
            conf[consol_mask] = (0.5 + 0.5 * consol_conf[consol_mask]).astype(np.float32)

            # RANGING (1): explicit sideways oscillation in the middle volatility band.
            ranging_mask = (
                ~trend_mask & ~volatile_mask & ~consol_mask
                & (adx <= 20)
                & (ema_stack == 0)
                & (drift_abs <= drift_p40)
                & (efficiency <= 0.35)
                & (autocorr_lag1 >= -0.15)
                & (autocorr_lag1 <= 0.10)
                & (atr_pctile >= 0.25)
                & (atr_pctile <= 0.70)
            )
            labels[ranging_mask] = 1
            adx_range_conf = (1.0 - (adx / 20.0).clip(0.0, 1.0))
            drift_range_conf = (1.0 - (drift_abs / (drift_p40 + 1e-9)).clip(0.0, 1.0))
            eff_range_conf = (1.0 - (efficiency / 0.35).clip(0.0, 1.0))
            ranging_conf = (
                0.35 * adx_range_conf
                + 0.35 * drift_range_conf
                + 0.30 * eff_range_conf
            ).clip(0.0, 1.0)
            conf[ranging_mask] = (0.5 + 0.5 * ranging_conf[ranging_mask]).astype(np.float32)

            # Ambiguous bars: don't fit any explicit definition cleanly — assign most
            # likely class by ADX/atr_pctile but zero their confidence so the MLP
            # learns uncertainty rather than memorising a noisy hard label.
            ambig_mask = ~trend_mask & ~volatile_mask & ~consol_mask & ~ranging_mask
            labels[ambig_mask] = 1  # default to RANGING for ambiguous mid-band bars
            conf[ambig_mask] = 0.0

            # ── Persistence filter (LTF) ──────────────────────────────────────
            # Thresholds represent the minimum run length (in bars) for a regime
            # label to be considered stable enough to train on.
            # Prior values (TRENDING=48h, VOLATILE=24h on 1H) zeroed ~83% of bars.
            # Realistic thresholds: a 6-bar trend = 6H of directional momentum is
            # sufficient; VOLATILE only needs 3-4 bars to be real.
            _persist_by_class = {
                0: {"5M": 48, "15M": 16, "1H": 6, "4H": 3, "1D": 2},    # TRENDING
                1: {"5M": 24, "15M":  8, "1H": 4, "4H": 2, "1D": 1},    # RANGING
                2: {"5M": 24, "15M":  8, "1H": 4, "4H": 2, "1D": 1},    # CONSOLIDATING
                3: {"5M": 24, "15M":  8, "1H": 4, "4H": 2, "1D": 1},    # VOLATILE
            }
            _runs = (labels != labels.shift()).cumsum()
            _run_len = _runs.map(_runs.value_counts())
            _short_run_mask = pd.Series(False, index=labels.index)
            for cls_id, tf_thresholds in _persist_by_class.items():
                cls_mask = labels == cls_id
                min_p = tf_thresholds.get(_tf, tf_thresholds.get("4H", 5))
                _short_run_mask |= (cls_mask & (_run_len < min_p))
            conf[_short_run_mask] = 0.0

            dist = {LTF_CLASSES[c]: int((labels == c).sum()) for c in range(len(LTF_CLASSES))}
            ambiguous = int((conf < 0.4).sum())
            logger.info("Rule labels LTF_BEHAVIOUR [%s]: %s  ambiguous=%d (total=%d)  short_runs_zeroed=%d",
                        timeframe or "?", dist, ambiguous, len(labels), int(_short_run_mask.sum()))

            labels = labels.astype(int)
            if return_confidence:
                return labels, conf.astype(np.float32)
            return labels

    # ── Train ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _ensure_structure_columns(
        df: pd.DataFrame,
        *,
        require_bos: bool = True,
        require_sweep: bool = True,
    ) -> pd.DataFrame:
        """Populate causal BOS and sweep columns used by regime features."""
        required = {"open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Regime feature input missing OHLC columns: {sorted(missing)}")

        bos_columns = {
            "swing_high", "swing_low", "bos_bull", "bos_bear",
            "last_swing_high", "last_swing_low",
        }
        sweep_columns = {
            "sweep_bull", "sweep_bear", "sweep_low_level",
            "sweep_high_level", "sweep_bull_wick", "sweep_bear_wick",
        }
        needs_bos = require_bos and not bos_columns.issubset(df.columns)
        needs_sweep = require_sweep and not sweep_columns.issubset(df.columns)
        if not needs_bos and not needs_sweep:
            return df

        from indicators.market_structure import detect_break_of_structure, detect_liquidity_sweeps

        out = df.copy()
        if needs_bos:
            bos = detect_break_of_structure(out)
            missing_bos = bos_columns - set(bos.columns)
            if missing_bos:
                raise RuntimeError(f"BOS detector did not return required columns: {sorted(missing_bos)}")
            for col in bos_columns:
                out[col] = bos[col]
        if needs_sweep:
            sweeps = detect_liquidity_sweeps(out)
            missing_sweep = sweep_columns - set(sweeps.columns)
            if missing_sweep:
                raise RuntimeError(f"Sweep detector did not return required columns: {sorted(missing_sweep)}")
            for col in sweep_columns:
                out[col] = sweeps[col]
        return out

    @staticmethod
    def _build_feature_matrix(df: pd.DataFrame, htf_full: dict,
                               symbol: Optional[str],
                               feature_names: Optional[Sequence[str]] = None) -> np.ndarray:
        """
        Vectorised feature extraction for an explicit feature contract.

        Only the named columns are computed and the return shape is
        (N, len(feature_names)).
        """
        from indicators.market_structure import (
            compute_adx, compute_atr, compute_ema_stack_score,
            compute_bollinger_bands,
        )
        from services.feature_engine import INDEX_NAMES, _vec_atr_pctile, _vec_autocorr

        if feature_names is None:
            raise ValueError("_build_feature_matrix requires explicit feature_names")
        requested = list(feature_names)
        unknown = [name for name in requested if name not in _ALLOWED_FEATURE_NAMES]
        if unknown:
            raise ValueError(f"_build_feature_matrix requested unknown regime features: {unknown}")
        if len(set(requested)) != len(requested):
            raise ValueError("_build_feature_matrix received duplicate feature names")

        requested_set = set(requested)
        strict_requested = feature_names is not None
        need_bos = "swing_hh_hl_count" in requested_set
        need_sweep = "liquidity_sweep_24h" in requested_set
        df = RegimeClassifier._ensure_structure_columns(
            df,
            require_bos=need_bos,
            require_sweep=need_sweep,
        )
        n = len(df)
        n_feat = len(requested)
        X = np.zeros((n, n_feat), dtype=np.float32)
        pos = {name: i for i, name in enumerate(requested)}
        assigned: set[str] = set()
        regime_n_bar = RegimeClassifier._infer_nbar_from_index(df.index)

        close_s = df["close"].astype(float)
        close = close_s.to_numpy(dtype=np.float64)
        atr_series = None
        adx_series = None
        stack_series = None
        bb_width_series = None

        def _has(name: str) -> bool:
            return name in pos

        def _has_any(names: Sequence[str]) -> bool:
            return any(name in pos for name in names)

        def _set(name: str, values) -> None:
            if name not in pos:
                return
            arr = np.asarray(values, dtype=np.float32)
            if arr.ndim == 0:
                arr = np.full(n, float(arr), dtype=np.float32)
            if len(arr) != n:
                raise RuntimeError(
                    f"_build_feature_matrix produced wrong length for {name}: {len(arr)} != {n}"
                )
            X[:, pos[name]] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            assigned.add(name)

        def _atr() -> pd.Series:
            nonlocal atr_series
            if atr_series is None:
                atr_series = compute_atr(df, 14).astype(float)
            return atr_series

        def _adx() -> pd.Series:
            nonlocal adx_series
            if adx_series is None:
                adx_series = (
                    df["adx_14"].astype(float)
                    if "adx_14" in df.columns
                    else compute_adx(df, 14).astype(float)
                )
            return adx_series

        def _stack() -> pd.Series:
            nonlocal stack_series
            if stack_series is None:
                stack_series = (
                    df["ema_stack"].astype(float)
                    if "ema_stack" in df.columns
                    else compute_ema_stack_score(df).astype(float)
                )
            return stack_series

        def _bb_width() -> pd.Series:
            nonlocal bb_width_series
            if bb_width_series is None:
                if "bb_width" in df.columns:
                    bb_width_series = df["bb_width"].astype(float)
                else:
                    bb_u, bb_m, bb_l = compute_bollinger_bands(df["close"])
                    bb_width_series = ((bb_u - bb_l) / (bb_m + 1e-9)).astype(float)
            return bb_width_series

        # ── Base structural features ────────────────────────────────────────
        if _has("adx_14_base"):
            _set("adx_14_base", np.clip(_adx().to_numpy(dtype=np.float64), 0, 100))
        if _has("ema_stack_score"):
            _set("ema_stack_score", np.clip(_stack().to_numpy(dtype=np.float64), -2, 2))
        if _has("atr_ratio"):
            _set("atr_ratio", np.clip((_atr() / (close_s + 1e-9) * 1000).to_numpy(dtype=np.float64), 0, 10))
        if _has("bb_width_pct"):
            _set("bb_width_pct", np.clip(_bb_width().to_numpy(dtype=np.float64), 0, 0.1))
        if _has("realized_vol_20"):
            ret = pd.Series(close, index=df.index).pct_change()
            rv = ret.rolling(20).std() * 100
            _set("realized_vol_20", np.clip(rv.to_numpy(dtype=np.float64), 0, 5))
        if _has("session_code"):
            if not hasattr(df.index, "hour"):
                raise ValueError("_build_feature_matrix requires a DatetimeIndex for session_code")
            h = df.index.hour
            sc = np.where((h >= 2) & (h < 7), 1,
                 np.where((h >= 7) & (h < 12), 2,
                 np.where(h == 12, 4,
                 np.where((h >= 13) & (h < 18), 3, 0))))
            _set("session_code", sc.astype(np.float32))

        if _has("swing_hh_hl_count"):
            _bos_count = (
                df["bos_bull"].fillna(False).astype(np.int8)
                + df["bos_bear"].fillna(False).astype(np.int8)
            ).rolling(24, min_periods=1).sum()
            _set("swing_hh_hl_count", np.clip(_bos_count.to_numpy(dtype=np.float32), 0, 20))
        if _has("liquidity_sweep_24h"):
            _sw_count = (
                df["sweep_bull"].fillna(False).astype(np.int8)
                + df["sweep_bear"].fillna(False).astype(np.int8)
            ).rolling(24, min_periods=1).sum()
            _set("liquidity_sweep_24h", np.clip(_sw_count.to_numpy(dtype=np.float32), 0, 20))

        # ── MTF features ─────────────────────────────────────────────────────
        _tf_specs = {
            "5M": ("5m", ("5M", "5m")),
            "15M": ("15m", ("15M", "15m")),
            "1H": ("1h", ("1H", "H1")),
            "4H": ("4h", ("4H", "H4")),
            "1D": ("1d", ("1D", "D1")),
        }
        htf_full = htf_full or {}
        for canonical_tf, (slug, aliases) in _tf_specs.items():
            names = {
                "adx": f"mtf_{slug}_adx",
                "ema_stack": f"mtf_{slug}_ema_stack",
                "atr_ratio": f"mtf_{slug}_atr_ratio",
                "bb_width": f"mtf_{slug}_bb_width",
            }
            needed = {metric: fname for metric, fname in names.items() if fname in requested_set}
            if not needed:
                continue
            tf_df = None
            for alias in aliases:
                if alias in htf_full and htf_full[alias] is not None:
                    tf_df = htf_full[alias]
                    break
            if tf_df is None or len(tf_df) < 14:
                raise RuntimeError(
                    f"_build_feature_matrix requires {canonical_tf} context for {sorted(needed.values())}"
                )

            def _align_mtf(series: pd.Series, feature_name: str) -> np.ndarray:
                aligned = pd.Series(series.to_numpy(dtype=np.float32), index=tf_df.index).reindex(
                    df.index, method="ffill"
                )
                first_valid = aligned.first_valid_index()
                if first_valid is None:
                    raise RuntimeError(
                        f"_build_feature_matrix could not causally align any {feature_name} values"
                    )
                if aligned.loc[first_valid:].isna().any():
                    raise RuntimeError(
                        f"_build_feature_matrix found internal alignment gaps for {feature_name}"
                    )
                aligned = aligned.fillna(0.0)
                return aligned.to_numpy(dtype=np.float32)

            try:
                if "adx" in needed:
                    tf_adx = (
                        tf_df["adx_14"].astype(float)
                        if "adx_14" in tf_df.columns
                        else compute_adx(tf_df, 14).astype(float)
                    )
                    _set(needed["adx"], np.clip(_align_mtf(tf_adx, needed["adx"]), 0, 100))
                if "ema_stack" in needed:
                    tf_stk = (
                        tf_df["ema_stack"].astype(float)
                        if "ema_stack" in tf_df.columns
                        else compute_ema_stack_score(tf_df).astype(float)
                    )
                    _set(needed["ema_stack"], np.clip(_align_mtf(tf_stk, needed["ema_stack"]), -2, 2))
                if "atr_ratio" in needed:
                    tf_atr = compute_atr(tf_df, 14).astype(float)
                    tf_c = tf_df["close"].astype(float)
                    _set(
                        needed["atr_ratio"],
                        np.clip(_align_mtf(tf_atr / (tf_c + 1e-9) * 1000, needed["atr_ratio"]), 0, 10),
                    )
                if "bb_width" in needed:
                    if "bb_width" in tf_df.columns:
                        tf_bbw = tf_df["bb_width"].astype(float)
                    else:
                        tf_bu, tf_bm, tf_bl = compute_bollinger_bands(tf_df["close"])
                        tf_bbw = ((tf_bu - tf_bl) / (tf_bm + 1e-9)).astype(float)
                    _set(needed["bb_width"], np.clip(_align_mtf(tf_bbw, needed["bb_width"]), 0, 0.1))
            except Exception as exc:
                raise RuntimeError(
                    f"_build_feature_matrix: MTF feature extraction failed for tf={canonical_tf}: {exc}"
                ) from exc

        # ── S/R zone features (indices 28–33) ─────────────────────────────────
        # Kept zero to preserve the trained regime feature distribution. The
        # underlying detector is causal now; enabling these columns requires a
        # deliberate retrain and manifest update.
        # Full legacy matrix callers still receive zeroes for these columns.

        # ── Regime dynamics ──────────────────────────────────────────────────
        # vol_slope: Δ(ATR/close) over the regime lookback — positive = expanding.
        if _has("vol_slope"):
            try:
                rel_vol = _atr() / (df["close"] + 1e-9)
                vol_slope = rel_vol.diff(regime_n_bar)
                _set("vol_slope", np.clip(np.nan_to_num(vol_slope.values * 1000, nan=0.0), -5, 5))
            except Exception as exc:
                raise RuntimeError(f"_build_feature_matrix: vol_slope failed: {exc}") from exc

        # regime_duration: bars since last close-direction flip (fully vectorised, O(N))
        # At each flip the counter resets; between flips it counts up.
        if _has("regime_duration"):
            try:
                direction = np.sign(np.diff(df["close"].values, prepend=df["close"].values[0]))
                flip_mask = np.concatenate(([True], direction[1:] != direction[:-1]))
                flip_indices = np.where(flip_mask)[0].astype(np.int64)
                bar_indices = np.arange(n, dtype=np.int64)
                group_starts = flip_indices[np.searchsorted(flip_indices, bar_indices, side="right") - 1]
                duration = (bar_indices - group_starts).astype(np.float32)
                _set("regime_duration", np.clip(duration, 0, 50) / 50.0)
            except Exception as exc:
                raise RuntimeError(f"_build_feature_matrix: regime_duration failed: {exc}") from exc

        # ── ATR percentile (index 36) ─────────────────────────────────────────
        # Mirrors the timeframe-specific regime label window.
        if _has("atr_pctile"):
            try:
                _atr_hist_window = regime_n_bar * 3
                _set(
                    "atr_pctile",
                    _vec_atr_pctile(
                        _atr().to_numpy(dtype=np.float64),
                        window=_atr_hist_window, min_periods=min(regime_n_bar, 14),
                    ),
                )
            except Exception as exc:
                raise RuntimeError(f"_build_feature_matrix: atr_pctile failed: {exc}") from exc

        # ── Time-series discriminators ───────────────────────────────────────
        ts_features = {"efficiency_ratio", "autocorr_lag1", "hurst_proxy"}
        if requested_set & ts_features:
            try:
                _n_bar = regime_n_bar
                _close = df["close"]
                if _has_any(("efficiency_ratio", "autocorr_lag1")):
                    _log_ret = np.log(_close / _close.shift(1))
                if _has("efficiency_ratio"):
                    _abs_moves = np.abs(_close.diff()).rolling(_n_bar, min_periods=_n_bar).sum()
                    _net_move = np.abs(_close - _close.shift(_n_bar))
                    _eff_ratio = (_net_move / (_abs_moves + 1e-9)).clip(0, 1)
                    _set("efficiency_ratio", np.nan_to_num(_eff_ratio.values.astype(np.float32), nan=0.5))
                if _has("autocorr_lag1"):
                    _autocorr_arr = _vec_autocorr(
                        np.nan_to_num(np.asarray(_log_ret, dtype=np.float64), nan=0.0),
                        window=_n_bar,
                    )
                    _set("autocorr_lag1", _autocorr_arr)
                if _has("hurst_proxy"):
                    _hi_n = df["high"].rolling(_n_bar, min_periods=_n_bar).max()
                    _lo_n = df["low"].rolling(_n_bar, min_periods=_n_bar).min()
                    _range_n = (_hi_n - _lo_n).clip(1e-9)
                    _hi_h = df["high"].rolling(max(2, _n_bar // 2), min_periods=2).max()
                    _lo_h = df["low"].rolling(max(2, _n_bar // 2), min_periods=2).min()
                    _range_h = (_hi_h - _lo_h).clip(1e-9)
                    _hurst_raw = (_range_n / _range_h / (2 ** 0.5)).clip(0.2, 3.0)
                    _hurst_norm = ((_hurst_raw - 0.2) / 2.8).clip(0.0, 1.0)
                    _set("hurst_proxy", np.nan_to_num(_hurst_norm.values.astype(np.float32), nan=0.5))
            except Exception as exc:
                raise RuntimeError(f"_build_feature_matrix: ts_discriminators failed: {exc}") from exc

        # ── Causal regime primitives (direction, volatility percentiles, candle
        # structure, symbol group). These are per-symbol normalised and match the
        # score-based labeller used by create_structural_labels().
        try:
            from services.regime_scores import REGIME_PRIMITIVE_COLUMNS, build_regime_score_frame

            primitive_requested = requested_set.intersection(REGIME_PRIMITIVE_COLUMNS)
            if primitive_requested:
                score_df = build_regime_score_frame(df, symbol=symbol, window=regime_n_bar)
                missing_primitives = [name for name in primitive_requested if name not in score_df.columns]
                if missing_primitives:
                    raise RuntimeError(f"regime score frame missing primitives: {missing_primitives}")
                for name in primitive_requested:
                    _set(name, score_df[name].to_numpy(dtype=np.float32))
        except Exception as exc:
            raise RuntimeError(f"_build_feature_matrix: regime score primitives failed: {exc}") from exc

        # ── Macro features ───────────────────────────────────────────────────
        macro_names = [f"idx_{name}_ret" for name in INDEX_NAMES] + [
            "macro_vix_level",
            "macro_yield_spread",
        ]
        if requested_set.intersection(macro_names):
            try:
                from services.feature_engine import FeatureEngine

                fe = FeatureEngine()
                macro_df = fe._build_macro_frame(df.index, symbol)
                for index_name in INDEX_NAMES:
                    feature_name = f"idx_{index_name}_ret"
                    if _has(feature_name):
                        if feature_name not in macro_df.columns:
                            raise RuntimeError(f"macro frame missing {feature_name}")
                        _set(feature_name, np.clip(macro_df[feature_name].to_numpy(dtype=np.float64) * 100, -5, 5))
                if _has("macro_vix_level"):
                    _set("macro_vix_level", np.clip(macro_df["macro_vix_level"].to_numpy(dtype=np.float64), 0, 2))
                if _has("macro_yield_spread"):
                    _set(
                        "macro_yield_spread",
                        np.clip(macro_df["macro_yield_spread"].to_numpy(dtype=np.float64), -0.2, 0.4),
                    )
            except Exception as exc:
                raise RuntimeError(f"_build_feature_matrix: macro features failed: {exc}") from exc

        if strict_requested:
            missing = [name for name in requested if name not in assigned]
            if missing:
                raise RuntimeError(f"_build_feature_matrix did not build requested features: {missing}")

        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def train_on_arrays(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        sample_weight_val: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Train directly on pre-built feature matrix X.

        HTF bias uses a 3-class label array y (N,). LTF behaviour uses a
        5-column score target y (N, 5), ordered by LTF_SCORE_OUTPUTS.
        sample_weight: float32 array (N,) in [0, 1] — confidence per bar from rule labeling.
          High-confidence bars (strong ADX + full stack + clear drift) get weight 1.0.
          Ambiguous bars (borderline ADX, partial stack, weak drift) are dropped by
          default when REGIME_DROP_AMBIGUOUS=1, with a safe fallback if filtering
          would remove a class.
        """
        X = np.asarray(X)
        if self._output_type == "behaviour_scores":
            return self._fit_behaviour_scores(
                X,
                np.asarray(y, dtype=np.float32),
                sample_weight=sample_weight,
                X_val=X_val,
                y_val=y_val,
                sample_weight_val=sample_weight_val,
            )

        y = np.asarray(y, dtype=np.int64)
        X_val_arr = None if X_val is None else np.asarray(X_val)
        y_val_arr = None if y_val is None else np.asarray(y_val, dtype=np.int64)
        if sample_weight is not None and len(sample_weight) == len(y):
            drop_ambiguous = os.getenv("REGIME_DROP_AMBIGUOUS", "1").lower() in (
                "1", "true", "yes",
            )
            min_conf = float(os.getenv("REGIME_MIN_LABEL_CONFIDENCE", "0.4"))
            if drop_ambiguous:
                sw = sample_weight.astype(np.float32, copy=False)
                clean_mask = np.isfinite(sw) & (sw >= min_conf)
                n_clean = int(clean_mask.sum())
                n_total = int(len(clean_mask))
                min_clean_ratio = float(os.getenv("REGIME_MIN_CLEAN_RATIO", "0.05"))
                _n_cls = self._n_output_classes
                clean_counts = np.bincount(y[clean_mask].astype(np.int64), minlength=_n_cls)
                has_all_classes = bool((clean_counts[:_n_cls] > 0).all())
                if n_total > 0 and (n_clean / n_total) < min_clean_ratio:
                    return {
                        "error": (
                            f"Regime labels too sparse after confidence filter: "
                            f"kept={n_clean}/{n_total} < {min_clean_ratio:.1%}. "
                            "Check structural label rules before trusting accuracy."
                        )
                    }
                if n_clean >= 100 and has_all_classes:
                    dropped = n_total - n_clean
                    logger.info(
                        "RegimeClassifier[mode=%s]: dropped ambiguous labels below %.2f "
                        "(kept=%d dropped=%d classes=%s)",
                        self._mode, min_conf, n_clean, dropped,
                        {
                            self._class_list[i]: int(clean_counts[i])
                            for i in range(_n_cls)
                        },
                    )
                    X = X[clean_mask]
                    y = y[clean_mask]
                    sample_weight = sw[clean_mask]
                else:
                    logger.warning(
                        "RegimeClassifier[mode=%s]: keeping ambiguous labels because "
                        "clean filter would be unsafe (kept=%d/%d class_counts=%s)",
                        self._mode, n_clean, n_total,
                        {
                            self._class_list[i]: int(clean_counts[i])
                            for i in range(_n_cls)
                        },
                    )
        if X.shape[1] != len(self._feature_names):
            raise ValueError(
                f"RegimeClassifier[{self._timeframe} mode={self._mode}] expected "
                f"{len(self._feature_names)} features {self._feature_names}; got {X.shape[1]}"
            )
        if X_val_arr is not None and X_val_arr.shape[1] != len(self._feature_names):
            raise ValueError(
                f"RegimeClassifier[{self._timeframe} mode={self._mode}] validation expected "
                f"{len(self._feature_names)} features; got {X_val_arr.shape[1]}"
            )
        return self._fit(
            X,
            y,
            sample_weight=sample_weight,
            X_val=X_val_arr,
            y_val=y_val_arr,
            sample_weight_val=sample_weight_val,
        )

    def _fit_behaviour_scores(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        sample_weight_val: Optional[np.ndarray] = None,
        _cold_start: bool = False,
    ) -> dict:
        """Train the LTF behaviour head as five independent score outputs."""
        try:
            import torch

            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)
            if y.ndim != 2 or y.shape[1] != len(LTF_SCORE_OUTPUTS):
                return {
                    "error": (
                        f"LTF score targets must have shape (N, {len(LTF_SCORE_OUTPUTS)}); "
                        f"got {tuple(y.shape)}"
                    )
                }
            if len(X) < 100:
                return {"error": f"Insufficient data ({len(X)} rows)"}

            if X.shape[1] != len(self._feature_names):
                raise ValueError(
                    f"RegimeClassifier[{self._timeframe} mode={self._mode}] expected "
                    f"{len(self._feature_names)} features {self._feature_names}; got {X.shape[1]}"
                )
            if X_val is not None:
                X_val = np.asarray(X_val, dtype=np.float32)
                if X_val.shape[1] != len(self._feature_names):
                    raise ValueError(
                        f"RegimeClassifier[{self._timeframe} mode={self._mode}] validation expected "
                        f"{len(self._feature_names)} features; got {X_val.shape[1]}"
                    )
            y = np.clip(np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
            finite_mask = np.isfinite(X).all(axis=1) & np.isfinite(y).all(axis=1)
            if sample_weight is not None and len(sample_weight) == len(y):
                finite_mask &= np.isfinite(sample_weight)
            if not finite_mask.all():
                X = X[finite_mask]
                y = y[finite_mask]
                if sample_weight is not None and len(sample_weight) == len(finite_mask):
                    sample_weight = np.asarray(sample_weight, dtype=np.float32)[finite_mask]

            if X_val is not None and y_val is not None:
                y_val = np.asarray(y_val, dtype=np.float32)
                if y_val.ndim != 2 or y_val.shape[1] != len(LTF_SCORE_OUTPUTS):
                    return {
                        "error": (
                            f"LTF validation targets must have shape (N, {len(LTF_SCORE_OUTPUTS)}); "
                            f"got {tuple(y_val.shape)}"
                        )
                    }
                y_val = np.clip(np.nan_to_num(y_val, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
                val_mask = np.isfinite(X_val).all(axis=1) & np.isfinite(y_val).all(axis=1)
                X_va = X_val[val_mask]
                y_va = y_val[val_mask]
                X_tr, y_tr = X, y
                sw_tr = (
                    np.asarray(sample_weight, dtype=np.float32)
                    if sample_weight is not None and len(sample_weight) == len(y)
                    else np.ones(len(X_tr), dtype=np.float32)
                )
            else:
                split = int(len(X) * 0.8)
                X_tr, X_va = X[:split], X[split:]
                y_tr, y_va = y[:split], y[split:]
                sw_tr = (
                    np.asarray(sample_weight[:split], dtype=np.float32)
                    if sample_weight is not None and len(sample_weight) == len(X)
                    else np.ones(len(X_tr), dtype=np.float32)
                )

            if len(X_tr) < 50 or len(X_va) < 10:
                return {"error": "Not enough LTF score data after split"}

            n_feat = X_tr.shape[1]
            _loaded_n_cls = getattr(self, "_n_classes", len(LTF_SCORE_OUTPUTS))
            _feature_mismatch = self._model is not None and self._n_features != n_feat
            _output_mismatch = self._model is not None and _loaded_n_cls != len(LTF_SCORE_OUTPUTS)
            _warm_start = (
                self._model is not None
                and not _feature_mismatch
                and not _output_mismatch
                and not _cold_start
            )
            if not _warm_start:
                self._model = _build_mlp(n_feat, len(LTF_SCORE_OUTPUTS)).to(DEVICE)
                self._n_features = n_feat
                self._n_classes = len(LTF_SCORE_OUTPUTS)
                logger.info("RegimeClassifier[mode=%s]: cold start score head", self._mode)
            else:
                logger.info("RegimeClassifier[mode=%s]: warm start score head", self._mode)
            if DEVICE.type == "cuda" and torch.cuda.device_count() > 1:
                if not isinstance(self._model, torch.nn.DataParallel):
                    self._model = torch.nn.DataParallel(self._model)
                logger.info("RegimeClassifier score head: DataParallel across %d GPUs",
                            torch.cuda.device_count())

            batch_size = 4096
            X_tr_gpu = torch.from_numpy(X_tr).to(DEVICE)
            y_tr_gpu = torch.from_numpy(y_tr).to(DEVICE)
            sw_tr_gpu = torch.from_numpy(np.clip(sw_tr, 0.05, 1.0)).to(DEVICE)
            X_va_gpu = torch.from_numpy(X_va.astype(np.float32, copy=False)).to(DEVICE)
            y_va_gpu = torch.from_numpy(y_va.astype(np.float32, copy=False)).to(DEVICE)
            n_tr = len(X_tr_gpu)
            n_va = len(X_va_gpu)
            steps_per_epoch = max(1, (n_tr + batch_size - 1) // batch_size)
            tr_idx = np.arange(n_tr, dtype=np.int64)

            optimiser = torch.optim.AdamW(
                self._model.parameters(),
                lr=8e-4 if not _warm_start else 2e-4,
                weight_decay=1e-2,
            )
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimiser,
                max_lr=8e-4 if not _warm_start else 2e-4,
                epochs=50,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.2,
            )
            use_amp = DEVICE.type == "cuda"
            amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            best_loss = float("inf")
            best_state = None
            patience, no_improve = 8, 0

            def _score_loss(logits: "torch.Tensor", target: "torch.Tensor",
                            weight: "torch.Tensor") -> "torch.Tensor":
                pred = torch.sigmoid(logits.float())
                per_row = torch.mean((pred - target.float()) ** 2, dim=1)
                weight = torch.clamp(weight.float(), min=0.05)
                return (per_row * weight).sum() / (weight.sum() + 1e-9)

            def _val_stats() -> tuple[float, np.ndarray]:
                all_pred = []
                loss_acc = 0.0
                with torch.no_grad():
                    for v_s in range(0, n_va, batch_size * 2):
                        xb = X_va_gpu[v_s: v_s + batch_size * 2]
                        yb = y_va_gpu[v_s: v_s + batch_size * 2]
                        with torch.amp.autocast("cuda", enabled=use_amp):
                            logits_v = self._model(xb)
                        pred = torch.sigmoid(logits_v.float())
                        loss_acc += torch.mean((pred - yb) ** 2).item() * len(xb)
                        all_pred.append(pred.cpu().numpy())
                return loss_acc / max(1, n_va), np.concatenate(all_pred, axis=0)

            for epoch in range(50):
                self._model.train()
                np.random.shuffle(tr_idx)
                tr_idx_t = torch.from_numpy(tr_idx).to(DEVICE)
                optimiser.zero_grad()
                tr_loss = 0.0
                for step in range(steps_per_epoch):
                    b_s = step * batch_size
                    b_e = min(b_s + batch_size, n_tr)
                    idx_b = tr_idx_t[b_s:b_e]
                    xb = X_tr_gpu[idx_b]
                    yb = y_tr_gpu[idx_b]
                    wb = sw_tr_gpu[idx_b]
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        logits_tr = self._model(xb)
                    loss = _score_loss(logits_tr, yb, wb)
                    amp_scaler.scale(loss).backward()
                    amp_scaler.unscale_(optimiser)
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                    amp_scaler.step(optimiser)
                    amp_scaler.update()
                    optimiser.zero_grad()
                    scheduler.step()
                    tr_loss += loss.item() * (b_e - b_s)
                tr_loss /= max(1, n_tr)

                self._model.eval()
                va_loss, va_pred = _val_stats()
                mae = np.mean(np.abs(va_pred - y_va), axis=0)
                if epoch == 0 or (epoch + 1) % 5 == 0:
                    logger.info(
                        "Regime score epoch %2d/50 — tr=%.4f va=%.4f mae=%s",
                        epoch + 1,
                        tr_loss,
                        va_loss,
                        {name: round(float(mae[i]), 4) for i, name in enumerate(LTF_SCORE_OUTPUTS)},
                    )
                else:
                    logger.info("Regime score epoch %2d/50 — tr=%.4f va=%.4f",
                                epoch + 1, tr_loss, va_loss)

                if va_loss < best_loss:
                    best_loss = va_loss
                    no_improve = 0
                    m_bs = self._model.module if isinstance(
                        self._model, torch.nn.DataParallel) else self._model
                    best_state = {k: v.cpu().clone() for k, v in m_bs.state_dict().items()}
                else:
                    no_improve += 1
                    if no_improve >= patience and epoch + 1 >= 10:
                        logger.info("Regime score early stop at epoch %d", epoch + 1)
                        break

            if best_state is not None:
                m = self._model.module if isinstance(
                    self._model, torch.nn.DataParallel) else self._model
                m.load_state_dict(best_state)

            self._model.eval()
            _, val_pred = _val_stats()
            mae = np.mean(np.abs(val_pred - y_va), axis=0)
            mse = np.mean((val_pred - y_va) ** 2, axis=0)
            target_std = np.std(y_va, axis=0)
            pred_std = np.std(val_pred, axis=0)
            corr = {}
            for i, name in enumerate(LTF_SCORE_OUTPUTS):
                if target_std[i] < 1e-6 or pred_std[i] < 1e-6:
                    corr[name] = 0.0
                else:
                    corr[name] = float(np.corrcoef(y_va[:, i], val_pred[:, i])[0, 1])
            score_mae = {name: round(float(mae[i]), 4) for i, name in enumerate(LTF_SCORE_OUTPUTS)}
            score_mse = {name: round(float(mse[i]), 5) for i, name in enumerate(LTF_SCORE_OUTPUTS)}
            score_corr = {name: round(float(corr[name]), 4) for name in LTF_SCORE_OUTPUTS}
            pred_std_map = {name: round(float(pred_std[i]), 4) for i, name in enumerate(LTF_SCORE_OUTPUTS)}
            target_std_map = {name: round(float(target_std[i]), 4) for i, name in enumerate(LTF_SCORE_OUTPUTS)}
            logger.info(
                "RegimeClassifier[mode=%s] score validation mae=%s mse=%s corr=%s pred_std=%s target_std=%s",
                self._mode,
                score_mae,
                score_mse,
                score_corr,
                pred_std_map,
                target_std_map,
            )

            del X_tr_gpu, y_tr_gpu, sw_tr_gpu, X_va_gpu, y_va_gpu
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

            max_mae = float(os.getenv("REGIME_SCORE_MAX_MAE", "0.30"))
            weak_scores = [name for name, value in score_mae.items() if float(value) > max_mae]
            min_pred_std = float(os.getenv("REGIME_SCORE_MIN_PRED_STD", "0.015"))
            collapsed_scores = [
                name
                for i, name in enumerate(LTF_SCORE_OUTPUTS)
                if target_std[i] > 0.03 and pred_std[i] < min_pred_std
            ]
            if weak_scores or collapsed_scores:
                return {
                    "error": (
                        f"Regime score validation below acceptance floor: "
                        f"mae={score_mae} max_mae={max_mae:.3f} "
                        f"weak_scores={weak_scores} collapsed_scores={collapsed_scores}. "
                        "Refusing to save misleading LTF score weights."
                    )
                }

            self.save(self.weight_path)
            mean_mae = float(np.mean(mae))
            return {
                "accuracy": round(max(0.0, 1.0 - mean_mae), 4),
                "n_train": len(X_tr),
                "n_val": len(X_va),
                "val_loss": round(float(best_loss), 6),
                "score_mae": score_mae,
                "score_mse": score_mse,
                "score_corr": score_corr,
                "score_outputs": list(LTF_SCORE_OUTPUTS),
                "timeframe": self._timeframe or "default",
            }
        except Exception as exc:
            logger.error("RegimeClassifier._fit_behaviour_scores failed: %s", exc)
            raise

    def _fit(self, X: np.ndarray, y: np.ndarray,
             sample_weight: Optional[np.ndarray] = None,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None,
             sample_weight_val: Optional[np.ndarray] = None,
             _cold_start: bool = False) -> dict:
        """Core GPU training loop. X: (N, F) float32, y: (N,) int64.

        sample_weight: optional (N,) float32 — per-bar confidence from labeling.
          Implemented as weighted CrossEntropyLoss (reduction='none' × weight).
          When REGIME_DROP_AMBIGUOUS=1, low-confidence bars are removed before this
          method is called. If they are kept, confidence weights scale each bar's
          contribution without softening the realised structural target.
        """
        try:
            import torch
            import torch.nn as nn

            # Mode-specific class definitions
            _n_cls   = self._n_output_classes
            _classes = self._class_list

            if len(X) < 100:
                return {"error": f"Insufficient data ({len(X)} rows)"}

            class_counts = {
                _classes[int(c)]: int(cnt)
                for c, cnt in zip(*np.unique(y, return_counts=True))
                if int(c) < len(_classes)
            }
            logger.info("RegimeClassifier[mode=%s]: %d samples, classes=%s, device=%s",
                        self._mode, len(y), class_counts, DEVICE)

            # Sanity check: all expected classes must be present with at least 1% of samples.
            if len(class_counts) < _n_cls:
                missing = [c for c in _classes if c not in class_counts]
                logger.error(
                    "RegimeClassifier[mode=%s]: MISSING CLASSES %s — label generation is broken. "
                    "All %d classes must be present. Check create_rule_labels().",
                    self._mode, missing, _n_cls
                )
                return {"error": f"Missing classes: {missing}"}
            min_class_pct = min(class_counts.get(c, 0) for c in _classes) / max(len(y), 1)
            if min_class_pct < 0.01:
                rare = {c: v for c, v in class_counts.items() if v / len(y) < 0.01}
                logger.warning(
                    "RegimeClassifier[mode=%s]: classes with <1%% of samples: %s — "
                    "model may collapse to majority class", self._mode, rare
                )
            max_class_share = max(class_counts.values()) / max(len(y), 1)
            max_allowed_share = float(os.getenv("REGIME_MAX_CLASS_SHARE", "0.90"))
            if max_class_share > max_allowed_share:
                return {
                    "error": (
                        f"Regime label distribution is degenerate: max_class_share="
                        f"{max_class_share:.1%} > {max_allowed_share:.1%}. "
                        "Refusing to train a classifier that can pass by predicting one regime."
                    )
                }

            # ── Majority-class undersampling ──────────────────────────────────
            # Preserve most of the HTF neutral base rate. Directional bias is rare;
            # forcing an almost-balanced training set makes the model flood neutral
            # validation bars as UP/DOWN, producing high recall but poor precision.
            _us_counts = np.bincount(y.astype(np.int64), minlength=_n_cls)
            _present = _us_counts[_us_counts > 0]
            _minority_n = int(_present.min()) if len(_present) > 0 else 1
            _cap_ratio = float(os.getenv(
                "REGIME_HTF_MAJORITY_CAP_RATIO" if self._mode == "htf_bias" else "REGIME_MAJORITY_CAP_RATIO",
                "12.0" if self._mode == "htf_bias" else "2.0",
            ))
            _majority_cap = max(_minority_n, int(_minority_n * _cap_ratio))
            _keep_mask = np.ones(len(y), dtype=bool)
            for _cls_id in range(_n_cls):
                _cls_idx = np.where(y == _cls_id)[0]
                if len(_cls_idx) > _majority_cap:
                    # Evenly-spaced subset preserves temporal distribution
                    _keep = np.round(
                        np.linspace(0, len(_cls_idx) - 1, _majority_cap)
                    ).astype(int)
                    _drop = np.ones(len(_cls_idx), dtype=bool)
                    _drop[_keep] = False
                    _keep_mask[_cls_idx[_drop]] = False
                    logger.info(
                        "RegimeClassifier[mode=%s]: undersample class %s: %d → %d",
                        self._mode, _classes[_cls_id], len(_cls_idx), _majority_cap,
                    )
            _keep_idx = np.sort(np.where(_keep_mask)[0])
            if len(_keep_idx) < len(y):
                X = X[_keep_idx]
                y = y[_keep_idx]
                if sample_weight is not None:
                    sample_weight = sample_weight[_keep_idx]
                _new_counts = {_classes[c]: int((y == c).sum()) for c in range(_n_cls)}
                logger.info(
                    "RegimeClassifier[mode=%s]: after undersampling: %d samples classes=%s",
                    self._mode, len(y), _new_counts,
                )

            # ── Temporal split ────────────────────────────────────────────────
            # Prefer explicit rolling-window validation arrays from the retrainer.
            # Legacy callers still fall back to the final 20% of X.
            if X_val is not None and y_val is not None:
                X_tr, y_tr = X, y
                X_va = np.asarray(X_val, dtype=np.float32)
                y_va = np.asarray(y_val, dtype=np.int64)
                if sample_weight is not None and len(sample_weight) == len(X_tr):
                    sw_tr = sample_weight.astype(np.float32)
                else:
                    sw_tr = np.ones(len(X_tr), dtype=np.float32)
            else:
                split      = int(len(X) * 0.8)
                X_tr, X_va = X[:split],  X[split:]
                y_tr, y_va = y[:split],  y[split:]

                # Sample weights: confidence per bar (rule strength).
                # Uniform 1.0 if not provided (e.g. legacy callers).
                if sample_weight is not None and len(sample_weight) == len(X):
                    sw_tr = sample_weight[:split].astype(np.float32)
                else:
                    sw_tr = np.ones(len(X_tr), dtype=np.float32)

            _ambig_pct = float((sw_tr < 0.4).mean()) * 100 if len(sw_tr) else 0.0
            logger.info("RegimeClassifier: sample weights — mean=%.3f  ambiguous(<0.4)=%.1f%%",
                        float(sw_tr.mean()) if len(sw_tr) else 0.0, _ambig_pct)

            if len(X_tr) < 50 or len(X_va) < 10:
                return {"error": "Not enough data after split"}

            # ── Build/warm-start model ────────────────────────────────────────
            n_feat = X.shape[1]
            _feature_mismatch = (self._model is not None and self._n_features != n_feat)
            # Force cold start if loaded model has wrong number of output classes.
            # This fires when old pkl is loaded but mode changed (e.g. 5-class → 3-class HTF).
            _loaded_n_cls = getattr(self, "_n_classes", _n_cls)
            _class_mismatch = (self._model is not None and _loaded_n_cls != _n_cls)
            if _feature_mismatch:
                logger.warning("RegimeClassifier[mode=%s]: feature count changed %d→%d, resetting",
                               self._mode, self._n_features, n_feat)
            if _class_mismatch:
                logger.warning("RegimeClassifier[mode=%s]: class count changed %d→%d, resetting",
                               self._mode, _loaded_n_cls, _n_cls)
            _warm_start = (
                self._model is not None
                and not _feature_mismatch
                and not _class_mismatch
                and not _cold_start
            )
            if not _warm_start:
                # Cold start: fresh random init
                self._model      = _build_mlp(n_feat, _n_cls).to(DEVICE)
                self._n_features = n_feat
                self._n_classes  = _n_cls
                logger.info("RegimeClassifier[mode=%s]: cold start (no existing weights)", self._mode)
            else:
                # Warm start: continue from loaded weights — preserves learned structure
                logger.info("RegimeClassifier[mode=%s]: warm start from existing weights", self._mode)
            if DEVICE.type == "cuda" and torch.cuda.device_count() > 1:
                if not isinstance(self._model, torch.nn.DataParallel):
                    self._model = torch.nn.DataParallel(self._model)
                logger.info("RegimeClassifier: DataParallel across %d GPUs",
                            torch.cuda.device_count())

            # ── Class weights (handle imbalance) ─────────────────────────────
            counts  = np.bincount(y_tr, minlength=_n_cls).astype(np.float32)
            counts  = np.where(counts == 0, 1.0, counts)
            # Squared inverse-frequency: boosts minority classes more aggressively
            # than linear inverse-frequency. Linear weights weren't sufficient to
            # prevent the LTF classifier from collapsing (TRENDING/RANGING recall
            # 0.20/0.24) and the HTF classifier from predicting mostly BIAS_DOWN.
            inv_freq = counts.sum() / (_n_cls * counts)
            _weight_power = float(os.getenv(
                "REGIME_HTF_CLASS_WEIGHT_POWER" if self._mode == "htf_bias" else "REGIME_CLASS_WEIGHT_POWER",
                "0.25" if self._mode == "htf_bias" else "1.0",
            ))
            class_w  = inv_freq ** _weight_power
            class_w  = class_w / class_w.mean()   # normalise so mean weight = 1.0
            class_w  = torch.tensor(class_w, dtype=torch.float32).to(DEVICE)

            # ── GPU-resident tensors ──────────────────────────────────────────
            batch_size = 4096
            X_tr_gpu  = torch.from_numpy(X_tr).to(DEVICE)
            y_tr_gpu  = torch.from_numpy(y_tr).to(DEVICE)
            sw_tr_gpu = torch.from_numpy(sw_tr).to(DEVICE)   # per-bar confidence weights
            X_va_gpu  = torch.from_numpy(X_va).to(DEVICE)
            y_va_gpu  = torch.from_numpy(y_va).to(DEVICE)
            n_tr = len(X_tr_gpu)
            n_va = len(X_va_gpu)
            steps_per_epoch = max(1, (n_tr + batch_size - 1) // batch_size)
            tr_idx = np.arange(n_tr, dtype=np.int64)

            # ── Loss functions ────────────────────────────────────────────────
            # Focal loss (gamma=2): down-weights easy (well-classified) examples
            # and focuses gradient on hard boundary examples. Standard CE + class
            # weights alone produced TRENDING/RANGING recall of 0.20/0.24 and
            # BIAS_DOWN recall of 0.239 — the model was collapsing to majority class.
            _base_ce = nn.CrossEntropyLoss(weight=class_w, reduction="none")
            _focal_gamma = 2.0

            def _hybrid_loss(logits: "torch.Tensor", labels: "torch.Tensor",
                             bar_weights: "torch.Tensor") -> "torch.Tensor":
                """
                Class-weighted focal loss with per-bar confidence weighting.

                Focal loss = -(1-pt)^gamma * log(pt). This reduces the gradient
                contribution from bars that are already classified confidently and
                focuses learning on ambiguous boundary bars (NEUTRAL vs DOWN, etc.).
                """
                logits_f = logits.float()
                ce = _base_ce(logits_f, labels)  # already class-weighted
                # pt = probability assigned to the correct class
                with torch.no_grad():
                    proba = torch.softmax(logits_f, dim=1)
                    pt = proba.gather(1, labels.unsqueeze(1)).squeeze(1).clamp(1e-7, 1.0)
                focal_w = (1.0 - pt) ** _focal_gamma
                effective_w = torch.clamp(bar_weights.float(), min=0.25) * focal_w
                return (ce * effective_w).sum() / (effective_w.sum() + 1e-9)

            # ── Optimiser + scheduler ─────────────────────────────────────────
            # Lowered from 3e-4 to 1e-4: warm-start collapse was observed even at
            # 3e-4/5=6e-5 — BIAS_NEUTRAL recall fell 0.405→0.078 over 50 epochs
            # while val_loss kept decreasing. Lower LR makes the warm start
            # move more slowly away from the loaded weights.
            _base_lr  = 1e-4
            _train_lr = _base_lr / 5.0 if _warm_start else _base_lr
            optimiser = torch.optim.AdamW(self._model.parameters(),
                                          lr=_train_lr, weight_decay=1e-1)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimiser,
                max_lr=_train_lr,
                epochs=50,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.2,
            )
            use_amp = DEVICE.type == "cuda"
            amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

            # ── Training loop ─────────────────────────────────────────────────
            # Early stopping based on BALANCED ACCURACY (mean per-class recall),
            # not val_loss. Val_loss decreased monotonically 0.824→0.813 while
            # BIAS_NEUTRAL recall crashed 0.405→0.078 — the model learned to
            # produce confident wrong predictions (lower CE) at the cost of
            # minority-class recall. Balanced accuracy exposes this collapse.
            best_balanced_acc  = -1.0
            epoch_1_balanced_acc: float | None = None
            epoch_1_state: dict | None = None
            patience, no_improve = 10, 0
            min_epochs_before_stop = 5
            best_state = None

            def _compute_val_stats() -> tuple[float, float, np.ndarray, np.ndarray]:
                va_loss_acc = 0.0
                all_preds_v, all_true_v = [], []
                with torch.no_grad():
                    for v_s in range(0, n_va, batch_size * 2):
                        xb = X_va_gpu[v_s: v_s + batch_size * 2]
                        yb = y_va_gpu[v_s: v_s + batch_size * 2]
                        with torch.amp.autocast("cuda", enabled=use_amp):
                            logits_v = self._model(xb).float()
                        va_loss_acc += _base_ce(logits_v, yb).mean().item() * len(xb)
                        if self._mode == "htf_bias":
                            proba_v = torch.softmax(logits_v, dim=1).cpu().numpy()
                            pred_v, _ = self._htf_bias_decision(
                                proba_v,
                                self._htf_directional_threshold,
                                self._htf_directional_margin,
                            )
                            all_preds_v.append(pred_v)
                        else:
                            all_preds_v.append(logits_v.argmax(1).cpu().numpy())
                        all_true_v.append(yb.cpu().numpy())
                va_loss_out = va_loss_acc / max(1, n_va)
                va_preds = np.concatenate(all_preds_v)
                va_trues = np.concatenate(all_true_v)
                val_acc_out = float((va_preds == va_trues).mean())
                return va_loss_out, val_acc_out, va_preds, va_trues

            for epoch in range(50):
                self._model.train()
                tr_loss = 0.0
                np.random.shuffle(tr_idx)
                tr_idx_t = torch.from_numpy(tr_idx).to(DEVICE)
                optimiser.zero_grad()
                for step in range(steps_per_epoch):
                    b_s = step * batch_size
                    b_e = min(b_s + batch_size, n_tr)
                    idx_b  = tr_idx_t[b_s:b_e]
                    xb     = X_tr_gpu[idx_b]
                    yb     = y_tr_gpu[idx_b]
                    wb     = sw_tr_gpu[idx_b]
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        logits_tr = self._model(xb)
                    loss = _hybrid_loss(logits_tr, yb, wb)
                    amp_scaler.scale(loss).backward()
                    amp_scaler.unscale_(optimiser)
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                    amp_scaler.step(optimiser)
                    amp_scaler.update()
                    optimiser.zero_grad()
                    scheduler.step()
                    tr_loss += loss.item() * (b_e - b_s)
                tr_loss /= max(1, n_tr)

                self._model.eval()
                va_loss, val_acc, va_p, va_t = _compute_val_stats()
                per_class_recall = [
                    float((va_p[va_t == c] == c).mean()) if (va_t == c).sum() > 0 else 0.0
                    for c in range(_n_cls)
                ]
                balanced_acc = float(np.mean(per_class_recall))

                if (epoch + 1) % 5 == 0 or epoch == 0:
                    per_class = {_classes[c]: round(per_class_recall[c], 3) for c in range(_n_cls)}
                    logger.info("Regime epoch %2d/50 — tr=%.4f va=%.4f acc=%.3f bal=%.3f per_class=%s",
                                epoch + 1, tr_loss, va_loss, val_acc, balanced_acc, per_class)
                else:
                    logger.info("Regime epoch %2d/50 — tr=%.4f va=%.4f acc=%.3f bal=%.3f",
                                epoch + 1, tr_loss, va_loss, val_acc, balanced_acc)

                # Save epoch-1 checkpoint for warm-start degradation protection
                if epoch == 0:
                    epoch_1_balanced_acc = balanced_acc
                    m_e1 = self._model.module if isinstance(
                        self._model, torch.nn.DataParallel) else self._model
                    epoch_1_state = {k: v.cpu().clone() for k, v in m_e1.state_dict().items()}

                    # If warm-start weights are barely above random (loaded pkl is
                    # biased), continuing from them makes things worse — the observed
                    # pattern is balanced_acc degrading from ~0.34 → ~0.24 over 50
                    # epochs while val_loss decreases (confident wrong predictions).
                    # Reset to cold start and retry with the same (already-undersampled)
                    # data so we don't waste the majority-class filtering work.
                    _random_baseline = 1.0 / _n_cls
                    if _warm_start and balanced_acc < _random_baseline + 0.05:
                        logger.warning(
                            "Regime[mode=%s]: warm-start epoch-1 balanced_acc=%.3f "
                            "barely above random (baseline=%.3f) — forcing cold restart",
                            self._mode, balanced_acc, _random_baseline,
                        )
                        del X_tr_gpu, y_tr_gpu, sw_tr_gpu, X_va_gpu, y_va_gpu, tr_idx_t
                        if DEVICE.type == "cuda":
                            torch.cuda.empty_cache()
                        # Reset model to random init so next call sees cold start
                        self._model = None
                        return self._fit(X, y, sample_weight=sample_weight, _cold_start=True)

                # Early abort: if balanced accuracy degraded >4pp from epoch 1
                # after 3 warm-up epochs, training is making things worse.
                if epoch >= 3 and epoch_1_balanced_acc is not None:
                    if balanced_acc < epoch_1_balanced_acc - 0.04:
                        logger.info(
                            "Regime: balanced_acc degraded %.3f→%.3f at epoch %d — "
                            "reverting to epoch-1 checkpoint to prevent collapse",
                            epoch_1_balanced_acc, balanced_acc, epoch + 1,
                        )
                        best_state = epoch_1_state
                        break

                # Best state = highest balanced accuracy
                if balanced_acc > best_balanced_acc:
                    best_balanced_acc = balanced_acc
                    no_improve = 0
                    m_bs = self._model.module if isinstance(
                        self._model, torch.nn.DataParallel) else self._model
                    best_state = {k: v.cpu().clone() for k, v in m_bs.state_dict().items()}
                else:
                    no_improve += 1
                    if no_improve >= patience and epoch + 1 >= min_epochs_before_stop:
                        logger.info("Regime early stop at epoch %d (no_improve=%d)",
                                    epoch + 1, no_improve)
                        break

            # Restore best weights (highest balanced accuracy seen during training)
            if best_state is not None:
                m = self._model.module if isinstance(
                    self._model, torch.nn.DataParallel) else self._model
                m.load_state_dict(best_state)

            # Final accuracy on val set (reuse GPU tensors already resident)
            self._model.eval()
            all_preds = []
            all_proba = []
            with torch.no_grad():
                val_bs = batch_size * 2
                for v_s in range(0, n_va, val_bs):
                    xb = X_va_gpu[v_s: v_s + val_bs]
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        logits_eval = self._model(xb).float()
                    if self._mode == "htf_bias":
                        proba_eval = torch.softmax(logits_eval, dim=1).cpu().numpy()
                        all_proba.append(proba_eval)
                    else:
                        preds = logits_eval.argmax(1).cpu().numpy()
                        all_preds.extend(preds)
            if self._mode == "htf_bias":
                proba_arr = np.concatenate(all_proba, axis=0)
                threshold, margin, policy_metrics = self._select_htf_bias_policy(proba_arr, y_va)
                self._htf_directional_threshold = threshold
                self._htf_directional_margin = margin
                all_preds_arr, _ = self._htf_bias_decision(proba_arr, threshold, margin)
                logger.info(
                    "RegimeClassifier[mode=%s] selected HTF decision policy threshold=%.3f margin=%.3f "
                    "policy_accuracy=%.3f policy_balanced=%.3f",
                    self._mode,
                    threshold,
                    margin,
                    policy_metrics["accuracy"],
                    policy_metrics["balanced_accuracy"],
                )
            else:
                all_preds_arr = np.array(all_preds)
            accuracy = float(np.mean(all_preds_arr == y_va))
            pred_counts = np.bincount(all_preds_arr.astype(np.int64), minlength=_n_cls)
            pred_share = pred_counts / max(len(all_preds_arr), 1)
            max_pred_share = float(pred_share.max()) if len(pred_share) else 0.0
            max_pred_allowed = float(os.getenv("REGIME_MAX_PRED_SHARE", "0.85"))
            min_pred_share = float(os.getenv("REGIME_MIN_PRED_CLASS_SHARE", "0.01"))
            collapsed_classes = [
                _classes[c]
                for c in range(_n_cls)
                if (y_va == c).sum() > 0 and pred_share[c] < min_pred_share
            ]
            per_class_accuracy = {
                _classes[c]: (
                    round(float((all_preds_arr[y_va == c] == c).mean()), 3)
                    if (y_va == c).sum() > 0 else 0.0
                )
                for c in range(_n_cls)
            }
            confusion = np.zeros((_n_cls, _n_cls), dtype=np.int64)
            for true_id, pred_id in zip(y_va.astype(np.int64), all_preds_arr.astype(np.int64)):
                if 0 <= true_id < _n_cls and 0 <= pred_id < _n_cls:
                    confusion[true_id, pred_id] += 1
            per_class_precision = {}
            per_class_f1 = {}
            for c in range(_n_cls):
                tp = float(confusion[c, c])
                fp = float(confusion[:, c].sum() - confusion[c, c])
                fn = float(confusion[c, :].sum() - confusion[c, c])
                precision = tp / (tp + fp + 1e-9)
                recall = tp / (tp + fn + 1e-9)
                f1 = 2.0 * precision * recall / (precision + recall + 1e-9)
                per_class_precision[_classes[c]] = round(float(precision), 3)
                per_class_f1[_classes[c]] = round(float(f1), 3)
            logger.info(
                "RegimeClassifier[mode=%s] validation precision=%s recall=%s f1=%s confusion=%s",
                self._mode,
                per_class_precision,
                per_class_accuracy,
                per_class_f1,
                confusion.tolist(),
            )
            del X_tr_gpu, y_tr_gpu, sw_tr_gpu, X_va_gpu, y_va_gpu, tr_idx_t
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            if max_pred_share > max_pred_allowed or collapsed_classes:
                return {
                    "error": (
                        f"Regime prediction distribution collapsed: "
                        f"pred_share={dict(zip(_classes, np.round(pred_share, 4).tolist()))}, "
                        f"max_pred_share={max_pred_share:.1%}, "
                        f"collapsed_classes={collapsed_classes}. "
                        "Refusing to save misleading regime weights."
                    )
                }
            default_min_overall = (1.0 / max(_n_cls, 1)) + 0.03
            min_overall_accuracy = float(
                os.getenv("REGIME_MIN_OVERALL_ACCURACY", f"{default_min_overall:.6f}")
            )
            min_class_accuracy = float(os.getenv("REGIME_MIN_CLASS_ACCURACY", "0.10"))
            weak_classes = [
                name
                for name, acc in per_class_accuracy.items()
                if (y_va == _classes.index(name)).sum() > 0 and float(acc) < min_class_accuracy
            ]
            if self._mode == "htf_bias":
                min_directional_precision = float(os.getenv("REGIME_MIN_DIRECTIONAL_PRECISION", "0.30"))
                min_directional_f1 = float(os.getenv("REGIME_MIN_DIRECTIONAL_F1", "0.30"))
                directional_names = ("BIAS_UP", "BIAS_DOWN")
                weak_precision = [
                    name for name in directional_names
                    if float(per_class_precision.get(name, 0.0)) < min_directional_precision
                ]
                weak_f1 = [
                    name for name in directional_names
                    if float(per_class_f1.get(name, 0.0)) < min_directional_f1
                ]
                if weak_precision or weak_f1:
                    return {
                        "error": (
                            f"Regime HTF directional validation below acceptance floor: "
                            f"precision={per_class_precision} min_directional_precision="
                            f"{min_directional_precision:.3f} f1={per_class_f1} "
                            f"min_directional_f1={min_directional_f1:.3f} "
                            f"weak_precision={weak_precision} weak_f1={weak_f1}. "
                            "Refusing to save directional-bias weights that flood neutral bars."
                        )
                    }
            if accuracy < min_overall_accuracy or weak_classes:
                return {
                    "error": (
                        f"Regime validation below acceptance floor: accuracy={accuracy:.3f} "
                        f"min_overall={min_overall_accuracy:.3f} "
                        f"per_class={per_class_accuracy} "
                        f"min_class={min_class_accuracy:.3f} "
                        f"weak_classes={weak_classes}. "
                        "Refusing to save misleading regime weights."
                    )
                }
            default_warn_accuracy = (1.0 / max(_n_cls, 1)) + 0.15
            warn_accuracy = float(
                os.getenv("REGIME_WARN_ACCURACY", f"{default_warn_accuracy:.6f}")
            )
            if accuracy < warn_accuracy:
                logger.warning(
                    "RegimeClassifier accuracy %.3f < warning floor %.3f "
                    "(harder structural labels; check blind backtest economics)",
                    accuracy,
                    warn_accuracy,
                )

            self.save(self.weight_path)
            logger.info("RegimeClassifier[%s] saved to %s",
                        self._timeframe or "default", self.weight_path)
            return {
                "accuracy":  accuracy,
                "n_train":   len(X_tr),
                "n_val":     len(X_va),
                "val_loss":  round(va_loss, 6),
                "per_class_accuracy": per_class_accuracy,
                "per_class_precision": per_class_precision,
                "per_class_f1": per_class_f1,
                "confusion_matrix": confusion.tolist(),
                "timeframe": self._timeframe or "default",
            }

        except Exception as exc:
            logger.error("RegimeClassifier._fit failed: %s", exc)
            raise

    def train(self, df: pd.DataFrame, continue_training: bool = False,
              symbol: Optional[str] = None,
              df_htf: Optional[dict] = None,
              df_h4: Optional[pd.DataFrame] = None) -> dict:
        """Train on a single symbol/timeframe DataFrame — builds arrays then calls _fit."""
        try:
            htf_full: dict = dict(df_htf) if df_htf else {}
            if df_h4 is not None and "4H" not in htf_full:
                htf_full["4H"] = df_h4

            n = len(df)
            X_all = self._build_feature_matrix(
                df,
                htf_full,
                symbol,
                feature_names=self._feature_names,
            )

            MAX_ROWS = 100_000
            step = max(1, (n - 50) // MAX_ROWS)
            idx  = np.arange(50, n, step)
            X    = X_all[idx]
            if self._output_type == "behaviour_scores":
                labels_frame = self.create_behaviour_score_targets(
                    df, timeframe=self._timeframe or "1H", symbol=symbol
                )
                y = labels_frame.iloc[idx].to_numpy(dtype=np.float32)
            else:
                labels_series = self.create_labels(df)
                y = labels_series.iloc[idx].values.astype(np.int64)
            logger.info("RegimeClassifier: vectorised extraction — %d rows (step=%d)", len(X), step)
            return self.train_on_arrays(X, y)
        except Exception as exc:
            logger.error("RegimeClassifier.train failed: %s", exc)
            raise

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        if self._model is None:
            return
        try:
            import torch
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            m = self._model.module if isinstance(
                self._model, torch.nn.DataParallel) else self._model
            payload = {
                "state_dict": {k: v.cpu() for k, v in m.state_dict().items()},
                "n_features": self._n_features,
                "n_classes":  self._n_output_classes,
                "mode":       self._mode,
                "output_type": self._output_type,
                "feature_names": list(self._feature_names),
                "score_outputs": list(LTF_SCORE_OUTPUTS) if self._output_type == "behaviour_scores" else [],
                "htf_directional_threshold": self._htf_directional_threshold,
                "htf_directional_margin": self._htf_directional_margin,
            }
            with open(path, "wb") as f:
                pickle.dump(payload, f)
            logger.info("RegimeClassifier[mode=%s] saved to %s", self._mode, path)
        except Exception as exc:
            logger.error("RegimeClassifier.save failed: %s", exc)

    def load(self, path: str) -> None:
        try:
            import torch
            with open(path, "rb") as f:
                payload = pickle.load(f)

            n_feat     = payload["n_features"]
            n_cls      = payload.get("n_classes", self._n_output_classes)
            saved_mode = payload.get("mode", self._mode)
            saved_output_type = payload.get("output_type", "classification")
            saved_feature_names = payload.get("feature_names")
            self._htf_directional_threshold = float(
                payload.get("htf_directional_threshold", self._htf_directional_threshold)
            )
            self._htf_directional_margin = float(
                payload.get("htf_directional_margin", self._htf_directional_margin)
            )
            state_dict = payload["state_dict"]

            # Detect stale/incompatible artifacts. Old regime_ltf.pkl files were
            # 4-class softmax classifiers; the current LTF model is a 5-output
            # score head and must cold-start instead of silently loading them.
            if n_cls != self._n_output_classes or saved_output_type != self._output_type:
                raise ModelNotTrainedError(
                    "RegimeClassifier.load: incompatible artifact saved_mode=%s saved_output=%s "
                    "saved_n=%d expected_mode=%s expected_output=%s expected_n=%d. "
                    "Delete the stale file and retrain regime weights: %s"
                    % (
                        saved_mode,
                        saved_output_type,
                        n_cls,
                        self._mode,
                        self._output_type,
                        self._n_output_classes,
                        path,
                    )
                )
            if saved_mode != self._mode:
                raise ModelNotTrainedError(
                    "RegimeClassifier.load: mode mismatch saved=%s current=%s "
                    "for %s. Delete the stale file and retrain regime weights."
                    % (saved_mode, self._mode, path)
                )
            if list(saved_feature_names or []) != list(self._feature_names):
                raise ModelNotTrainedError(
                    "RegimeClassifier.load: feature contract mismatch for %s. "
                    "saved=%s expected=%s. Delete the stale file and retrain regime weights."
                    % (path, list(saved_feature_names or []), list(self._feature_names))
                )
            if n_feat != len(self._feature_names):
                raise ModelNotTrainedError(
                    "RegimeClassifier.load: saved feature count %d does not match expected contract %d "
                    "for %s. Delete the stale file and retrain regime weights."
                    % (n_feat, len(self._feature_names), path)
                )

            m = _build_mlp(n_feat, n_cls)
            m.load_state_dict(state_dict)
            m = m.to(DEVICE)
            m.eval()

            if DEVICE.type == "cuda" and torch.cuda.device_count() > 1:
                m = torch.nn.DataParallel(m)

            self._model      = m
            self._n_features = n_feat
            self._n_classes  = n_cls   # store loaded value for mismatch detection in _fit
            self._loaded     = True
            logger.info("RegimeClassifier[mode=%s] loaded from %s (device=%s, features=%d, n_classes=%d)",
                        self._mode, path, DEVICE, n_feat, n_cls)
        except Exception as exc:
            logger.error("RegimeClassifier.load failed: %s", exc)
            raise
