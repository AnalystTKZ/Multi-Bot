"""
regime_classifier.py — GPU-native PyTorch MLP hierarchical regime classifier.

Hierarchical market structure framework:
  HTF classifier (4H) — "What is overall direction?" (mode="htf_bias")
    3 classes: 0=BIAS_UP, 1=BIAS_DOWN, 2=BIAS_NEUTRAL
  LTF classifier (1H) — "How is price behaving NOW?" (mode="ltf_behaviour")
    4 classes: 0=TRENDING, 1=RANGING, 2=CONSOLIDATING, 3=VOLATILE

Architecture: N_FEATURES → 128 → 64 → N_CLASSES  (BN + Dropout + residual skip)
DataParallel across both T4 GPUs during training and batch inference.
3-bar hysteresis: regime must persist 3 bars before switching.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from models.base_model import BaseModel
from services.feature_engine import REGIME_FEATURES, REGIME_4H_FEATURES, REGIME_1H_FEATURES

logger = logging.getLogger(__name__)

# ── New hierarchical class definitions ───────────────────────────────────────
HTF_CLASSES = ["BIAS_UP", "BIAS_DOWN", "BIAS_NEUTRAL"]          # 3-class HTF bias
LTF_CLASSES = ["TRENDING", "RANGING", "CONSOLIDATING", "VOLATILE"]  # 4-class LTF behaviour

# CLASSES kept as LTF_CLASSES for backward compat (most code paths use LTF or check by name)
CLASSES = LTF_CLASSES
N_FEATURES  = len(REGIME_FEATURES)   # full matrix width for _build_feature_matrix
N_CLASSES   = len(CLASSES)           # default (LTF, 4-class)

# Per-TF feature subsets: indices into the full REGIME_FEATURES column order.
# _build_feature_matrix always builds the full matrix; classifiers index these columns.
_FULL_NAMES = REGIME_FEATURES
_4H_COL_IDX = [_FULL_NAMES.index(f) for f in REGIME_4H_FEATURES if f in _FULL_NAMES]
_1H_COL_IDX = [_FULL_NAMES.index(f) for f in REGIME_1H_FEATURES if f in _FULL_NAMES]

# Mapping: timeframe → (column indices, feature names)
_TF_FEATURE_MAP: dict = {
    "4H": (_4H_COL_IDX, REGIME_4H_FEATURES),
    "1H": (_1H_COL_IDX, REGIME_1H_FEATURES),
    None: (list(range(N_FEATURES)), REGIME_FEATURES),
}
_MODEL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHT_PATH = os.path.join(_MODEL_ROOT, "weights", "regime_classifier.pkl")

# Per-mode weight paths for the hierarchical cascade.
# regime_htf.pkl: 3-class HTF bias (BIAS_UP/DOWN/NEUTRAL) — trained on 4H data.
# regime_ltf.pkl: 4-class LTF behaviour (TRENDING/RANGING/CONSOLIDATING/VOLATILE) — trained on 1H data.
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
          "ltf_behaviour" → 4-class (TRENDING/RANGING/CONSOLIDATING/VOLATILE) trained on 1H data.
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
            self._current_regime_id: int = 2   # default BIAS_NEUTRAL
        else:
            self._class_list = LTF_CLASSES
            self._n_output_classes = len(LTF_CLASSES)  # 4
            self._current_regime_id: int = 1   # default RANGING

        # Route weight file: per-TF/mode if specified, else legacy path
        self.weight_path = self._TF_TO_PATH.get(self._timeframe, WEIGHT_PATH)
        # Feature column indices and count for this TF's classifier
        _col_idx, _feat_names = _TF_FEATURE_MAP.get(self._timeframe, _TF_FEATURE_MAP[None])
        self._col_idx   = _col_idx
        self._n_features = len(_col_idx)
        logger.debug("RegimeClassifier[%s mode=%s]: %d features, %d classes, weight=%s",
                     self._timeframe or "default", self._mode, self._n_features,
                     self._n_output_classes, self.weight_path)
        os.makedirs(os.path.join(_MODEL_ROOT, "weights"), exist_ok=True)
        if self.is_trained:
            try:
                self.load(self.weight_path)
            except Exception as exc:
                logger.warning("RegimeClassifier[%s mode=%s]: initial load failed: %s",
                               self._timeframe or "default", self._mode, exc)

    # ── Predict ───────────────────────────────────────────────────────────────

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

        htf = dict(df_htf) if df_htf else {}
        if df_h4 is not None and "4H" not in htf:
            htf["4H"] = df_h4

        try:
            import torch
            from services.feature_engine import FeatureEngine, REGIME_FEATURES
            fe = FeatureEngine()
            feat = fe.get_regime_features(df, df_htf=htf, symbol=symbol)
            # Slice to TF-specific columns if model was built on a feature subset
            if len(feat) == N_FEATURES and len(self._col_idx) < N_FEATURES:
                feat = feat[self._col_idx]
            x = torch.tensor(feat.reshape(1, -1), dtype=torch.float32).to(DEVICE)

            # Unwrap DataParallel — DP cannot split a batch of 1 across 2 GPUs
            _infer_m = self._model.module if isinstance(self._model, torch.nn.DataParallel) else self._model
            _infer_m.eval()
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                    logits = _infer_m(x)
                proba_t = torch.softmax(logits.float(), dim=1)[0]
            proba = proba_t.cpu().numpy().tolist()
            raw_id = int(np.argmax(proba))

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
                "regime_confidence":  float(max(proba)),
            }
        except Exception as exc:
            logger.error("RegimeClassifier.predict failed: %s", exc)
            raise

    def predict_batch(self, X: np.ndarray, batch_size: int = 4096) -> tuple:
        """
        CPU inference on a pre-built feature matrix X (N, F_full or F_tf).
        If X has the full REGIME_FEATURES width, slices TF-specific columns first.
        Returns (labels: np.ndarray int32, confidences: np.ndarray float32).

        Small MLP (input→128→64→classes): CPU is faster than GPU for this because
        the H2D transfer and kernel launch overhead dominates at these batch sizes.
        GPU is reserved for the GRU which has large recurrent state and benefits
        from CUDA parallelism.
        """
        _default_id = 2 if self._mode == "htf_bias" else 1  # BIAS_NEUTRAL or RANGING
        _default_conf = 1.0 / self._n_output_classes
        if not self.is_trained or self._model is None:
            n = len(X)
            return np.full(n, _default_id, dtype=np.int32), np.full(n, _default_conf, dtype=np.float32)
        try:
            import torch
            # Slice to TF-specific columns if full matrix provided
            if X.shape[1] == N_FEATURES and len(self._col_idx) < N_FEATURES:
                X = X[:, self._col_idx]
            # Unwrap DataParallel and move to CPU for inference — no kernel launch overhead
            _raw = self._model.module if isinstance(self._model, torch.nn.DataParallel) else self._model
            _raw.eval()
            _cpu_m = _raw.to("cpu")
            all_labels = []
            all_conf   = []
            with torch.no_grad():
                for s in range(0, len(X), batch_size):
                    xb = torch.from_numpy(X[s: s + batch_size])  # stays on CPU
                    logits = _cpu_m(xb).float()
                    proba = torch.softmax(logits, dim=1).numpy()
                    all_labels.append(proba.argmax(axis=1).astype(np.int32))
                    all_conf.append(proba.max(axis=1).astype(np.float32))
            # Move model back to training device so subsequent train() calls work
            _raw.to(DEVICE)
            return np.concatenate(all_labels), np.concatenate(all_conf)
        except Exception as exc:
            logger.error("RegimeClassifier.predict_batch failed: %s", exc)
            n = len(X)
            return np.full(n, _default_id, dtype=np.int32), np.full(n, _default_conf, dtype=np.float32)

    # ── Labels ────────────────────────────────────────────────────────────────

    def create_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Institutional-grade regime labeling via unsupervised clustering.

        Step 1 — build 4 continuous, timeframe-agnostic features:
          efficiency_ratio  = |net n-bar move| / sum(|bar moves|)  [0→1, 1=clean trend]
          volatility        = ATR / close (relative, not pip-units)
          drift             = (close - close[n]) / n / close       (signed direction)
          compression       = (max - min) / ATR over n bars

        Step 2 — GaussianMixture (k=4) on z-scored features.
          GMM is preferred over KMeans: handles elongated clusters (trending
          regimes stretch along the drift axis) and gives soft probabilities.

        Step 3 — map clusters → regime labels by their feature profile:
          high efficiency + positive drift → TRENDING (0)
          high efficiency + negative drift → TRENDING (1)
          low efficiency + low vol         → RANGING (2)
          high vol + low efficiency        → VOLATILE (3)

        Result: balanced classes (~20-30% each), stable labels, timeframe-agnostic.
        """
        from indicators.market_structure import compute_atr
        try:
            from sklearn.mixture import GaussianMixture
            from sklearn.preprocessing import StandardScaler
            _has_gmm = True
        except ImportError:
            _has_gmm = False

        close = df["close"]
        # Scale lookback to the timeframe: 4H needs ~50 bars to see a full trend;
        # 14 bars at 4H is only 2.3 days — not enough to separate regimes.
        n_bar = self._TF_NBAR.get(self._timeframe, self._DEFAULT_NBAR)

        atr = compute_atr(df, n_bar)

        # ── Feature 1: efficiency ratio ───────────────────────────────────────
        abs_moves   = np.abs(close.diff()).rolling(n_bar, min_periods=n_bar).sum()
        net_move    = np.abs(close - close.shift(n_bar))
        eff_ratio   = (net_move / (abs_moves + 1e-9)).clip(0, 1)

        # ── Feature 2: relative volatility ───────────────────────────────────
        rel_vol = atr / (close + 1e-9)

        # ── Feature 3: drift (signed) ─────────────────────────────────────────
        drift = (close - close.shift(n_bar)) / (n_bar * close + 1e-9)

        # Use the shared feature extractor (includes atr_pctile as 6th feature)
        feat_df, _ = self._extract_gmm_features(df, n_bar=n_bar)

        if len(feat_df) < 50 or not _has_gmm:
            # Fallback: simple rule-based labels
            adx_s = df["adx_14"] if "adx_14" in df.columns else atr * 0
            vol_thresh = float(np.nanpercentile(rel_vol.fillna(0).values, 80))
            ema_stack = df.get("ema_stack", pd.Series(0, index=df.index))
            trend_up = (adx_s > 25) & (ema_stack == 2)
            trend_dn = (adx_s > 25) & (ema_stack == -2)
            volatile = (rel_vol > vol_thresh) & ~trend_up & ~trend_dn
            labels = pd.Series(2, index=df.index)
            labels[volatile] = 3
            labels[trend_dn] = 1
            labels[trend_up] = 0
            return labels.astype(int)

        # ── GMM clustering ────────────────────────────────────────────────────
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(feat_df.values)

        gmm = GaussianMixture(n_components=5, covariance_type="full",
                               random_state=42, max_iter=200)
        cluster_ids = gmm.fit_predict(X_scaled)

        # ── Map clusters → regime labels (rank-based — guaranteed all 5 classes) ──
        centroids = scaler.inverse_transform(gmm.means_)
        remaining = list(range(5))
        cluster_labels: dict[int, int] = {}

        vol_c = max(remaining, key=lambda c: centroids[c, 1] - centroids[c, 0])
        cluster_labels[vol_c] = 3
        remaining.remove(vol_c)

        tu = max(remaining, key=lambda c: centroids[c, 2])
        cluster_labels[tu] = 0
        remaining.remove(tu)

        td = min(remaining, key=lambda c: centroids[c, 2])
        cluster_labels[td] = 1
        remaining.remove(td)

        # CONSOLIDATING (4): lowest atr_pctile + lowest autocorr (pre-breakout compression)
        consol = min(remaining, key=lambda c: centroids[c, 5] + max(centroids[c, 6], 0))
        cluster_labels[consol] = 4
        remaining.remove(consol)

        # RANGING (2): the remainder — moderate vol, near-zero drift, stable ATR
        cluster_labels[remaining[0]] = 2

        labels = pd.Series(2, index=df.index, dtype=int)
        labels.loc[feat_df.index] = [cluster_labels[int(c)] for c in cluster_ids]
        return labels.astype(int)

    # Lookback window per timeframe — must span roughly one full regime cycle.
    # 4H: 50 bars ≈ 2.5 weeks (trend impulse); 1H: 24 bars ≈ 1 trading day;
    # 15M/default: 14 bars ≈ 3.5 hours. Using 14 everywhere collapses all
    # 4H distributions to nearly identical centroids → poor GMM separation.
    _TF_NBAR: dict = {"4H": 50, "1H": 24, "15M": 14, "5M": 10}
    _DEFAULT_NBAR = 14

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
    def create_rule_labels(
        df: pd.DataFrame,
        timeframe: str = "4H",
        mode: str = "ltf_behaviour",
        return_confidence: bool = False,
    ):
        """
        Mode-aware rule-based regime labels with per-bar confidence scores.

        mode="htf_bias" (4H): 3-class direction labels
          - BIAS_UP (0):     ADX>20 AND ema_stack >= 1 AND drift > 0
          - BIAS_DOWN (1):   ADX>20 AND ema_stack <= -1 AND drift < 0
          - BIAS_NEUTRAL (2): everything else
          Persistence: 20 bars (4H) for BIAS_UP/DOWN; 5 bars for BIAS_NEUTRAL.

        mode="ltf_behaviour" (1H): 4-class behaviour labels (direction-agnostic)
          Priority order:
          1. VOLATILE (3):       atr_pctile >= 80th pctile  (expansion/stress)
          2. TRENDING (0):       ADX>25 AND abs(ema_stack)>=1 AND abs(drift)>0 AND not volatile
          3. CONSOLIDATING (2):  atr_pctile<=25th pctile AND atr_slope<0 AND not trending/volatile
          4. RANGING (1):        everything else — sideways oscillation
          Persistence: TRENDING=20 bars, VOLATILE=10 bars, RANGING/CONSOLIDATING=5 bars.

        Bars with confidence < 0.4 are "ambiguous" — the MLP trains with reduced
        sample weight so it learns uncertainty on those bars.

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
        drift_p80 = float(drift.abs().quantile(0.80)) + 1e-9

        atr = compute_atr(df, n_bar)
        _hist = n_bar * 3
        atr_pctile = atr.rolling(_hist, min_periods=n_bar).apply(
            lambda x: float(np.searchsorted(np.sort(x[:-1]), x[-1])) / max(len(x) - 1, 1)
            if len(x) > 1 else 0.5, raw=True
        ).clip(0.0, 1.0).fillna(0.5)

        atr_slope = atr.rolling(n_bar, min_periods=max(2, n_bar // 2)).apply(
            lambda x: (x[-1] - x[0]) / (x[0] + 1e-9) if len(x) > 1 else 0.0, raw=True
        ).fillna(0.0)

        # ── HTF BIAS mode (3-class: direction-focused) ────────────────────────
        if mode == "htf_bias":
            labels = pd.Series(2, index=df.index, dtype=int)  # default BIAS_NEUTRAL

            # BIAS_UP: ADX>25 AND ema_stack>=1 AND drift>drift_p40
            # Raised from ADX>20 to ADX>25 — weak-trend bars (20-25 ADX) are genuinely
            # ambiguous and were causing the model to memorise noise instead of structure.
            # drift_p40 filter ensures the move has meaningful directional commitment.
            drift_p40 = float(drift.abs().quantile(0.40)) + 1e-9
            bu_mask = (adx > 25) & (ema_stack >= 1) & (drift > drift_p40)
            labels[bu_mask] = 0
            adx_conf_bu   = ((adx - 25) / 25.0).clip(0.0, 1.0)
            stack_conf_bu = np.where(ema_stack >= 2, 1.0, 0.7)
            drift_conf_bu = (drift.abs() / drift_p80).clip(0.0, 1.0)
            bu_conf = (adx_conf_bu * stack_conf_bu * drift_conf_bu).astype(np.float32)
            conf[bu_mask] = (0.5 + 0.5 * pd.Series(bu_conf, index=df.index)[bu_mask]).astype(np.float32)

            # BIAS_DOWN: ADX>25 AND ema_stack<=-1 AND drift<-drift_p40
            bd_mask = (adx > 25) & (ema_stack <= -1) & (drift < -drift_p40)
            labels[bd_mask] = 1
            adx_conf_bd   = ((adx - 25) / 25.0).clip(0.0, 1.0)
            stack_conf_bd = np.where(ema_stack <= -2, 1.0, 0.7)
            drift_conf_bd = (drift.abs() / drift_p80).clip(0.0, 1.0)
            bd_conf = (adx_conf_bd * stack_conf_bd * drift_conf_bd).astype(np.float32)
            conf[bd_mask] = (0.5 + 0.5 * pd.Series(bd_conf, index=df.index)[bd_mask]).astype(np.float32)

            # BIAS_NEUTRAL: everything else — high confidence when clearly non-directional
            neutral_mask = ~bu_mask & ~bd_mask
            adx_neutral_conf = (1.0 - (adx / 25.0).clip(0.0, 1.0))
            neutral_conf     = adx_neutral_conf.clip(0.2, 1.0)  # floor 0.2 so NEUTRAL bars aren't suppressed
            conf[neutral_mask] = neutral_conf[neutral_mask].astype(np.float32)

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
            vol_thresh    = float(atr_pctile.quantile(0.80))
            consol_thresh = float(atr_pctile.quantile(0.25))

            # VOLATILE (3): ATR expanding — chaotic, unpredictable
            volatile_mask = atr_pctile >= vol_thresh
            labels[volatile_mask] = 3
            vol_conf = ((atr_pctile - vol_thresh) / (1.0 - vol_thresh + 1e-9)).clip(0.0, 1.0)
            conf[volatile_mask] = (0.5 + 0.5 * vol_conf[volatile_mask]).astype(np.float32)

            # TRENDING (0): directional momentum — direction-agnostic (we know direction from HTF)
            trend_mask = (adx > 25) & (ema_stack.abs() >= 1) & (drift.abs() > 0) & ~volatile_mask
            labels[trend_mask] = 0
            adx_conf_t   = ((adx - 25) / 25.0).clip(0.0, 1.0)
            stack_conf_t = np.where(ema_stack.abs() >= 2, 1.0, 0.7)
            drift_conf_t = (drift.abs() / drift_p80).clip(0.0, 1.0)
            t_conf = (adx_conf_t * stack_conf_t * drift_conf_t).astype(np.float32)
            conf[trend_mask] = (0.5 + 0.5 * pd.Series(t_conf, index=df.index)[trend_mask]).astype(np.float32)

            # CONSOLIDATING (2): ATR at multi-period low AND falling — pre-breakout compression
            consol_mask = (atr_pctile <= consol_thresh) & (atr_slope < 0) & ~volatile_mask & ~trend_mask
            labels[consol_mask] = 2
            consol_atr_conf   = (1.0 - (atr_pctile / (consol_thresh + 1e-9)).clip(0.0, 1.0))
            consol_slope_conf = (-atr_slope).clip(0.0, 0.5) / 0.5
            consol_conf = (0.5 * consol_atr_conf + 0.5 * consol_slope_conf).clip(0.1, 1.0)
            conf[consol_mask] = (0.5 + 0.5 * consol_conf[consol_mask]).astype(np.float32)

            # RANGING (1): explicit definition — sideways oscillation in the middle band.
            # ADX<25 (no meaningful trend), atr_pctile in the mid-band (not volatile, not consolidating),
            # and abs(drift) below the 60th percentile (no committed direction).
            # Being explicit prevents the model from learning "RANGING = garbage bin of everything else".
            drift_p60 = float(drift.abs().quantile(0.60)) + 1e-9
            ranging_mask = (
                ~trend_mask & ~volatile_mask & ~consol_mask
                & (adx < 25)
                & (atr_pctile > consol_thresh) & (atr_pctile < vol_thresh)
                & (drift.abs() < drift_p60)
            )
            labels[ranging_mask] = 1
            adx_range_conf = (1.0 - (adx / 25.0).clip(0.0, 1.0))
            atr_range_conf = (1.0 - (atr_pctile / vol_thresh).clip(0.0, 1.0))
            ranging_conf   = (0.5 * adx_range_conf + 0.5 * atr_range_conf).clip(0.3, 1.0)
            conf[ranging_mask] = ranging_conf[ranging_mask].astype(np.float32)

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
    def _build_feature_matrix(df: pd.DataFrame, htf_full: dict,
                               symbol: Optional[str]) -> np.ndarray:
        """
        Vectorised feature extraction: computes all indicator series for the full df
        at once, then assembles the (N, n_features) matrix in a single numpy stack.
        O(N) — no per-bar Python loop.

        Returns float32 array shape (N, N_FEATURES) aligned to df.index.
        """
        from indicators.market_structure import (
            compute_adx, compute_atr, compute_ema_stack_score,
            compute_bollinger_bands, detect_break_of_structure,
            detect_liquidity_sweeps, detect_sr_zones,
        )
        from services.feature_engine import FeatureEngine, REGIME_FEATURES, INDEX_NAMES

        n = len(df)
        n_feat = len(REGIME_FEATURES)
        X = np.zeros((n, n_feat), dtype=np.float32)

        close = df["close"].values
        # ── Base structural features (indices 0–7) ────────────────────────────
        adx = df["adx_14"].values if "adx_14" in df.columns else compute_adx(df, 14).values
        X[:, 0] = np.clip(np.nan_to_num(adx), 0, 100)

        stk = df["ema_stack"].values if "ema_stack" in df.columns else compute_ema_stack_score(df).values
        X[:, 1] = np.clip(np.nan_to_num(stk), -2, 2)

        atr = compute_atr(df, 14).values
        X[:, 2] = np.clip(np.nan_to_num(atr / (close + 1e-9) * 1000), 0, 10)

        if "bb_width" in df.columns:
            bb_w = df["bb_width"].values
        else:
            bb_u, bb_m, bb_l = compute_bollinger_bands(df["close"])
            bb_w = ((bb_u - bb_l) / (bb_m + 1e-9)).values
        X[:, 3] = np.clip(np.nan_to_num(bb_w), 0, 0.1)

        ret = pd.Series(close).pct_change().values
        rv  = pd.Series(ret).rolling(20).std().values * 100
        X[:, 4] = np.clip(np.nan_to_num(rv), 0, 5)

        if hasattr(df.index, "hour"):
            h = df.index.hour
            sc = np.where((h >= 2) & (h < 7), 1,
                 np.where((h >= 7) & (h < 12), 2,
                 np.where(h == 12, 4,
                 np.where((h >= 13) & (h < 18), 3, 0))))
            X[:, 5] = sc.astype(np.float32)

        # BOS and sweep counts — causal rolling sums over 24 bars.
        # Reads pre-computed bos_bull/bos_bear and sweep_bull/sweep_bear columns
        # if present; falls back to zero if the columns are absent so inference
        # remains safe on DataFrames that lack these indicators.
        try:
            _bos_window = 24
            if "bos_bull" in df.columns or "bos_bear" in df.columns:
                _bos_bull = df["bos_bull"].fillna(0) if "bos_bull" in df.columns else pd.Series(0, index=df.index)
                _bos_bear = df["bos_bear"].fillna(0) if "bos_bear" in df.columns else pd.Series(0, index=df.index)
                _bos_count = (_bos_bull + _bos_bear).rolling(_bos_window, min_periods=1).sum()
                X[:, 6] = np.clip(np.nan_to_num(_bos_count.values.astype(np.float32), nan=0.0), 0, 20)
            if "sweep_bull" in df.columns or "sweep_bear" in df.columns:
                _sw_bull = df["sweep_bull"].fillna(0) if "sweep_bull" in df.columns else pd.Series(0, index=df.index)
                _sw_bear = df["sweep_bear"].fillna(0) if "sweep_bear" in df.columns else pd.Series(0, index=df.index)
                _sw_count = (_sw_bull + _sw_bear).rolling(_bos_window, min_periods=1).sum()
                X[:, 7] = np.clip(np.nan_to_num(_sw_count.values.astype(np.float32), nan=0.0), 0, 20)
        except Exception:
            pass  # columns absent — X[:, 6] and X[:, 7] remain zero (safe fallback)

        # ── MTF features (indices 8–27): 5 TFs × 4 features each ─────────────
        _tf_map = {"5M": (8, "5M"), "15M": (12, "15M"), "1H": (16, "1H"),
                   "4H": (20, "4H"), "1D": (24, "1D")}
        for tf, (offset, tf_key) in _tf_map.items():
            tf_df = htf_full.get(tf_key)
            if tf_df is None:
                tf_df = htf_full.get(tf)
            if tf_df is None or len(tf_df) < 14:
                continue
            try:
                tf_adx = tf_df["adx_14"].values if "adx_14" in tf_df.columns else compute_adx(tf_df, 14).values
                tf_stk = tf_df["ema_stack"].values if "ema_stack" in tf_df.columns else compute_ema_stack_score(tf_df).values
                tf_atr = compute_atr(tf_df, 14).values
                tf_c   = tf_df["close"].values
                if "bb_width" in tf_df.columns:
                    tf_bbw = tf_df["bb_width"].values
                else:
                    tf_bu, tf_bm, tf_bl = compute_bollinger_bands(tf_df["close"])
                    tf_bbw = ((tf_bu - tf_bl) / (tf_bm + 1e-9)).values
                # Forward-fill HTF values onto base df index
                tf_series = {
                    offset:     pd.Series(np.clip(np.nan_to_num(tf_adx), 0, 100), index=tf_df.index),
                    offset + 1: pd.Series(np.clip(np.nan_to_num(tf_stk), -2, 2), index=tf_df.index),
                    offset + 2: pd.Series(np.clip(np.nan_to_num(tf_atr / (tf_c + 1e-9) * 1000), 0, 10), index=tf_df.index),
                    offset + 3: pd.Series(np.clip(np.nan_to_num(tf_bbw), 0, 0.1), index=tf_df.index),
                }
                for col_idx, s in tf_series.items():
                    aligned = s.reindex(df.index, method="ffill").fillna(0).values
                    X[:, col_idx] = aligned.astype(np.float32)
            except Exception as exc:
                raise RuntimeError(f"_build_feature_matrix: MTF feature extraction failed for tf={tf}: {exc}") from exc

        # ── S/R zone features (indices 28–33) ─────────────────────────────────
        # detect_sr_zones uses rolling(center=True) internally — centred window
        # reads swing_n future bars per bar, which is lookahead. Until the
        # indicator is made strictly causal these columns stay zero so they
        # cannot leak into training or inference.
        # X[:, 28:34] already initialised to 0 above.

        # ── Regime dynamics (indices 34–35) ──────────────────────────────────
        # vol_slope: Δ(ATR/close) over 14 bars — positive = expanding, negative = contracting
        try:
            atr_series = compute_atr(df, 14)
            rel_vol = atr_series / (df["close"] + 1e-9)
            vol_slope = rel_vol.diff(14)  # change over 14 bars
            X[:, 34] = np.clip(np.nan_to_num(vol_slope.values * 1000, nan=0.0), -5, 5)
        except Exception as exc:
            raise RuntimeError(f"_build_feature_matrix: vol_slope failed: {exc}") from exc

        # regime_duration: bars since last close-direction flip (fully vectorised, O(N))
        # At each flip the counter resets; between flips it counts up.
        try:
            direction = np.sign(np.diff(df["close"].values, prepend=df["close"].values[0]))
            flip_mask = np.concatenate(([True], direction[1:] != direction[:-1]))
            # flip_positions[i] = index of the most recent flip at or before bar i
            flip_indices = np.where(flip_mask)[0].astype(np.int64)
            # For each bar, find which flip group it belongs to using searchsorted
            bar_indices  = np.arange(n, dtype=np.int64)
            group_starts = flip_indices[np.searchsorted(flip_indices, bar_indices, side="right") - 1]
            duration     = (bar_indices - group_starts).astype(np.float32)
            X[:, 35] = np.clip(duration, 0, 50) / 50.0
        except Exception as exc:
            raise RuntimeError(f"_build_feature_matrix: regime_duration failed: {exc}") from exc

        # ── ATR percentile (index 36) ─────────────────────────────────────────
        # Mirrors create_rule_labels exactly: rolling(n_bar*3) searchsorted rank.
        # n_bar=14, window=42 — same as _hist = n_bar * 3 in the label path.
        try:
            _atr_feat = compute_atr(df, 14)
            _atr_hist_window = 14 * 3  # 42 bars — matches _hist = n_bar * 3
            from services.feature_engine import _vec_atr_pctile
            X[:, 36] = _vec_atr_pctile(
                _atr_feat.to_numpy(dtype=np.float64),
                window=_atr_hist_window, min_periods=14,
            )
        except Exception as exc:
            raise RuntimeError(f"_build_feature_matrix: atr_pctile failed: {exc}") from exc

        # ── Time-series discriminators (indices 45–47) ───────────────────────
        # efficiency_ratio, autocorr_lag1, hurst_proxy — same features used
        # for GMM labeling but never given to the MLP. Without them, the MLP
        # cannot separate RANGING (autocorr≈0, eff≈0.2) from TRENDING (autocorr>0,
        # eff>0.7) because ADX and ATR look identical for both at the boundary.
        try:
            _n_bar = 14  # default lookback — matches _DEFAULT_NBAR
            _close = df["close"]
            _log_ret = np.log(_close / _close.shift(1))
            _abs_moves = np.abs(_close.diff()).rolling(_n_bar, min_periods=_n_bar).sum()
            _net_move  = np.abs(_close - _close.shift(_n_bar))
            _eff_ratio = (_net_move / (_abs_moves + 1e-9)).clip(0, 1)
            X[:, 45] = np.nan_to_num(_eff_ratio.values.astype(np.float32), nan=0.5)

            from services.feature_engine import _vec_autocorr
            _autocorr_arr = _vec_autocorr(
                np.nan_to_num(np.asarray(_log_ret, dtype=np.float64), nan=0.0),
                window=_n_bar,
            )
            X[:, 46] = _autocorr_arr

            _hi_n    = df["high"].rolling(_n_bar, min_periods=_n_bar).max()
            _lo_n    = df["low"].rolling(_n_bar, min_periods=_n_bar).min()
            _range_n = (_hi_n - _lo_n).clip(1e-9)
            _hi_h    = df["high"].rolling(max(2, _n_bar // 2), min_periods=2).max()
            _lo_h    = df["low"].rolling(max(2, _n_bar // 2), min_periods=2).min()
            _range_h = (_hi_h - _lo_h).clip(1e-9)
            # Normalise Hurst proxy to [0,1]: raw range is [0.2,3.0] → (val-0.2)/2.8
            _hurst_raw = (_range_n / _range_h / (2 ** 0.5)).clip(0.2, 3.0)
            _hurst_norm = ((_hurst_raw - 0.2) / 2.8).clip(0.0, 1.0)
            X[:, 47] = np.nan_to_num(_hurst_norm.values.astype(np.float32), nan=0.5)
        except Exception as exc:
            raise RuntimeError(f"_build_feature_matrix: ts_discriminators failed: {exc}") from exc

        # ── Macro features (indices 48–66) ────────────────────────────────────
        try:
            fe = FeatureEngine()
            macro_df = fe._build_macro_frame(df.index, symbol)
            base_macro = 48  # after 8 base + 5×4 MTF + 6 S/R + 2 regime dynamics + 3 TS discriminators
            for i, name in enumerate(INDEX_NAMES):
                col = f"idx_{name}_ret"
                if col in macro_df.columns:
                    X[:, base_macro + i] = np.clip(macro_df[col].fillna(0).values * 100, -5, 5)
            X[:, base_macro + len(INDEX_NAMES)]     = np.clip(macro_df["macro_vix_level"].fillna(0).values, 0, 2)
            X[:, base_macro + len(INDEX_NAMES) + 1] = np.clip(macro_df["macro_yield_spread"].fillna(0).values, -0.2, 0.4)
        except Exception as exc:
            raise RuntimeError(f"_build_feature_matrix: macro features failed: {exc}") from exc

        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def train_on_arrays(self, X: np.ndarray, y: np.ndarray,
                        sample_weight: Optional[np.ndarray] = None) -> dict:
        """
        Train directly on pre-built feature matrix X (N, F_full) and label array y (N,).
        sample_weight: float32 array (N,) in [0, 1] — confidence per bar from rule labeling.
          High-confidence bars (strong ADX + full stack + clear drift) get weight 1.0.
          Ambiguous bars (borderline ADX, partial stack, weak drift) get reduced weight.
          The MLP learns uncertainty on ambiguous bars rather than memorising a hard label.
        """
        if len(self._col_idx) < N_FEATURES:
            X = X[:, self._col_idx]
            if sample_weight is not None and len(sample_weight) == len(y):
                pass  # weight aligns to rows, not features — no slicing needed
        return self._fit(X, y, sample_weight=sample_weight)

    def _fit(self, X: np.ndarray, y: np.ndarray,
             sample_weight: Optional[np.ndarray] = None) -> dict:
        """Core GPU training loop. X: (N, F) float32, y: (N,) int64.

        sample_weight: optional (N,) float32 — per-bar confidence from rule labeling.
          Implemented as weighted CrossEntropyLoss (reduction='none' × weight).
          Bars with weight < 0.4 (ambiguous) also get softer label targets via
          per-bar label smoothing: target = weight * one_hot + (1-weight)/N_CLASSES.
          Entropy regularisation (λ=0.05) penalises overconfident predictions on
          ALL bars — encourages the model to output high uncertainty on random/ambiguous
          market conditions rather than always committing to one regime.
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

            # ── Temporal split ────────────────────────────────────────────────
            split      = int(len(X) * 0.8)
            X_tr, X_va = X[:split],  X[split:]
            y_tr, y_va = y[:split],  y[split:]

            # Sample weights: confidence per bar (rule strength).
            # Uniform 1.0 if not provided (e.g. legacy callers).
            if sample_weight is not None and len(sample_weight) == len(X):
                sw_tr = sample_weight[:split].astype(np.float32)
                _ambig_pct = float((sw_tr < 0.4).mean()) * 100
                logger.info("RegimeClassifier: sample weights — mean=%.3f  ambiguous(<0.4)=%.1f%%",
                            float(sw_tr.mean()), _ambig_pct)
            else:
                sw_tr = np.ones(len(X_tr), dtype=np.float32)

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
            _warm_start = (self._model is not None and not _feature_mismatch and not _class_mismatch)
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
            class_w = counts.sum() / (_n_cls * counts)
            # Extra boost for the hardest-to-learn classes:
            # LTF RANGING (1) collapses to 0% — 2× boost on top of inverse-freq weight.
            # HTF BIAS_NEUTRAL (2) stalls at ~33% — 1.5× boost.
            if self._mode == "ltf_behaviour":
                class_w[1] *= 3.0   # RANGING
            elif self._mode == "htf_bias":
                class_w[2] *= 4.0   # BIAS_NEUTRAL: raised 1.5→4.0 to overcome low bar_w suppression
            class_w = torch.tensor(class_w, dtype=torch.float32).to(DEVICE)

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
            # Base CE with class weights (no label_smoothing — we do per-bar soft targets below)
            _base_ce = nn.CrossEntropyLoss(weight=class_w, reduction="none")
            # Entropy regularisation coefficient: penalises overconfident outputs on ALL bars.
            # Raised 0.05 → 0.10: the HTF model was hitting proba=1.0 on BIAS_UP/DOWN by
            # epoch 20 (pure memorisation). Higher λ forces the model to spread probability
            # mass, which also naturally reduces the BIAS_NEUTRAL / RANGING collapse.
            _entropy_lambda = 0.10

            def _hybrid_loss(logits: "torch.Tensor", labels: "torch.Tensor",
                             bar_weights: "torch.Tensor") -> "torch.Tensor":
                """
                Weighted CE with per-bar soft targets + entropy regularisation.

                Soft targets: for bar i with confidence w_i:
                  target[i, true_class] = w_i * (1 - ε) + ε/4
                  target[i, other]      = (1 - w_i) * ε/4 + ε/4
                  where ε = 0.1 (base smoothing).
                This means high-confidence bars get near-hard targets; ambiguous
                bars get flatter targets — the model learns "I'm not sure here".

                Entropy term: -λ × mean(H(p)) where H(p) = -Σ p log p.
                Subtracting entropy from the loss penalises low-entropy (overconfident)
                predictions, pushing the model to spread probability mass when uncertain.
                """
                logits_f = logits.float()
                eps = 0.1
                n_c = logits_f.shape[1]

                # Build per-bar soft targets
                one_hot = torch.zeros_like(logits_f).scatter_(1, labels.unsqueeze(1), 1.0)
                w = bar_weights.unsqueeze(1)                      # (B, 1)
                # High-confidence bar → near one-hot; low-confidence → flat
                smooth_targets = w * (one_hot * (1 - eps) + eps / n_c) + \
                                 (1 - w) * torch.full_like(logits_f, 1.0 / n_c)

                # KL(soft_target || softmax(logits)) = soft CE - H(soft_target)
                log_probs = torch.log_softmax(logits_f, dim=1)
                soft_ce   = -(smooth_targets * log_probs).sum(dim=1)

                # Class-imbalance re-weighting applied INDEPENDENTLY of bar confidence.
                # Multiplying cw_per_bar × bar_weights suppresses minority classes
                # (RANGING, BIAS_NEUTRAL) because they are inherently low-confidence,
                # causing the model to collapse and never predict them.
                # Fix: class weight and bar confidence weight are additive influences,
                # not multiplicative. Class weight ensures rare classes are learned;
                # bar weight scales each bar's contribution to training stability.
                cw_per_bar = class_w[labels]
                # Hard floor on bar_weights so ambiguous bars still contribute.
                # Lowered from 0.4 → 0.1: RANGING bars have legitimately low confidence
                # (they're defined as "not trending/volatile/consolidating") but were
                # over-weighted at 0.4, drowning out high-confidence TRENDING/VOLATILE
                # gradient signal. 0.1 keeps ambiguous bars in training without
                # overwhelming the loss.
                bar_w_floored = torch.clamp(bar_weights, min=0.1)
                # Detach class weight from bar confidence: use an additive blend instead
                # of pure multiplication. Pure product cw × bar_w crushes NEUTRAL bars
                # (low-confidence by definition) even after raising the class_w multiplier.
                # Additive blend: 0.5*(cw_norm) + 0.5*(bar_w_floored) means a NEUTRAL bar
                # with bar_w=0.1 still gets ~50% of its class weight contribution rather
                # than 10%. cw_norm is normalised to have the same mean as bar_w_floored
                # so the two components are on the same scale.
                cw_norm = cw_per_bar / (cw_per_bar.mean() + 1e-9) * bar_w_floored.mean()
                effective_w = 0.5 * cw_norm + 0.5 * bar_w_floored
                weighted_ce = (soft_ce * effective_w).mean()

                # Entropy regularisation: encourages uncertainty on ambiguous bars
                probs   = torch.softmax(logits_f, dim=1)
                entropy = -(probs * (probs + 1e-9).log()).sum(dim=1)  # H(p) per bar
                entropy_term = _entropy_lambda * entropy.mean()

                # Maximise entropy = subtract it from loss
                return weighted_ce - entropy_term

            # ── Optimiser + scheduler ─────────────────────────────────────────
            # Fine-tuning uses a lower LR to preserve learned structure.
            # Cold start gets full LR; warm start gets 5× lower to avoid
            # blowing away weights that took 7 years of data to learn.
            _base_lr  = 3e-4
            _train_lr = _base_lr / 5.0 if _warm_start else _base_lr
            # weight_decay raised 5e-2 → 1e-1: train/val gap was ~0.35 (HTF) and ~0.37 (LTF)
            # indicating the small MLP was memorising label noise. Stronger L2 penalty
            # keeps weights closer to zero and forces the model to learn general patterns.
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
            best_val_loss = float("inf")
            patience, no_improve = 10, 0   # rule labels are stable — allow longer runs
            min_epochs_before_stop = 5    # always train at least 5 epochs
            best_state = None

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
                va_loss  = 0.0
                n_correct = 0
                with torch.no_grad():
                    val_bs = batch_size * 2
                    for v_s in range(0, n_va, val_bs):
                        xb = X_va_gpu[v_s: v_s + val_bs]
                        yb = y_va_gpu[v_s: v_s + val_bs]
                        with torch.amp.autocast("cuda", enabled=use_amp):
                            logits = self._model(xb)
                        logits = logits.float()
                        # Val loss: uniform weights (no rule confidence on val — measure true accuracy)
                        va_loss   += _base_ce(logits, yb).mean().item() * len(xb)
                        n_correct += (logits.argmax(1) == yb).sum().item()
                va_loss /= max(1, n_va)
                val_acc  = n_correct / max(1, n_va)

                # Per-class accuracy every 5 epochs — exposes collapse to majority class
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    all_va_preds, all_va_true = [], []
                    with torch.no_grad():
                        for v_s in range(0, n_va, batch_size * 2):
                            xb = X_va_gpu[v_s: v_s + batch_size * 2]
                            yb = y_va_gpu[v_s: v_s + batch_size * 2]
                            with torch.amp.autocast("cuda", enabled=use_amp):
                                preds_b = self._model(xb).float().argmax(1)
                            all_va_preds.extend(preds_b.cpu().numpy())
                            all_va_true.extend(yb.cpu().numpy())
                    va_p = np.array(all_va_preds)
                    va_t = np.array(all_va_true)
                    per_class = {_classes[c]: round(float((va_p[va_t == c] == c).mean()), 3)
                                 if (va_t == c).sum() > 0 else 0.0 for c in range(_n_cls)}
                    logger.info("Regime epoch %2d/50 — tr=%.4f va=%.4f acc=%.3f per_class=%s",
                                epoch + 1, tr_loss, va_loss, val_acc, per_class)
                else:
                    logger.info("Regime epoch %2d/50 — tr=%.4f va=%.4f acc=%.3f",
                                epoch + 1, tr_loss, va_loss, val_acc)

                if va_loss < best_val_loss:
                    best_val_loss = va_loss
                    no_improve    = 0
                    # Save best weights (unwrap DataParallel)
                    m = self._model.module if isinstance(
                        self._model, torch.nn.DataParallel) else self._model
                    best_state = {k: v.cpu().clone() for k, v in m.state_dict().items()}
                else:
                    no_improve += 1
                    if no_improve >= patience and epoch + 1 >= min_epochs_before_stop:
                        logger.info("Regime early stop at epoch %d (no_improve=%d)",
                                    epoch + 1, no_improve)
                        break

            # Restore best weights
            if best_state is not None:
                m = self._model.module if isinstance(
                    self._model, torch.nn.DataParallel) else self._model
                m.load_state_dict(best_state)

            # Final accuracy on val set (reuse GPU tensors already resident)
            self._model.eval()
            all_preds = []
            with torch.no_grad():
                val_bs = batch_size * 2
                for v_s in range(0, n_va, val_bs):
                    xb = X_va_gpu[v_s: v_s + val_bs]
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        preds = self._model(xb).float().argmax(1).cpu().numpy()
                    all_preds.extend(preds)
            accuracy = float(np.mean(np.array(all_preds) == y_va))
            del X_tr_gpu, y_tr_gpu, sw_tr_gpu, X_va_gpu, y_va_gpu, tr_idx_t
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            if accuracy < 0.65:
                logger.warning("RegimeClassifier accuracy %.2f < 0.65 threshold", accuracy)

            self.save(self.weight_path)
            logger.info("RegimeClassifier[%s] saved to %s",
                        self._timeframe or "default", self.weight_path)
            return {
                "accuracy":  accuracy,
                "n_train":   len(X_tr),
                "n_val":     len(X_va),
                "val_loss":  round(best_val_loss, 6),
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

            labels_series = self.create_labels(df)
            n = len(df)
            X_all = self._build_feature_matrix(df, htf_full, symbol)

            MAX_ROWS = 100_000
            step = max(1, (n - 50) // MAX_ROWS)
            idx  = np.arange(50, n, step)
            X    = X_all[idx]
            y    = labels_series.iloc[idx].values.astype(np.int64)
            logger.info("RegimeClassifier: vectorised extraction — %d rows (step=%d)", len(X), step)
            return self._fit(X, y)
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
            state_dict = payload["state_dict"]

            # Detect n_classes mismatch (e.g. old 5-class pkl vs new 3/4-class mode)
            if n_cls != self._n_output_classes:
                logger.warning(
                    "RegimeClassifier.load: n_classes mismatch saved=%d expected=%d (mode=%s) "
                    "— will cold-start on first training call",
                    n_cls, self._n_output_classes, self._mode,
                )
                # Load the model with saved dims so it doesn't crash, but mark stale
                # The _fit() mismatch detection will cold-start it on next train call
            else:
                if saved_mode != self._mode:
                    logger.warning(
                        "RegimeClassifier.load: mode mismatch saved=%s current=%s "
                        "— predictions may be incorrect until retrained",
                        saved_mode, self._mode,
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
