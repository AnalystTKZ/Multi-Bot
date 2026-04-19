"""
regime_classifier.py — GPU-native PyTorch MLP 4-class market regime classifier.

Multi-TF cascade architecture:
  4H classifier → market bias (TRENDING_UP/DOWN/RANGING/VOLATILE)
  1H classifier → intraday structure refinement
  15M/5M entry precision is handled by the GRU which receives both as context.

Classes: 0=TRENDING_UP, 1=TRENDING_DOWN, 2=RANGING, 3=VOLATILE
Architecture: N_FEATURES → 128 → 64 → 4  (BN + Dropout + residual skip)
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

CLASSES = ["TRENDING_UP", "TRENDING_DOWN", "RANGING", "VOLATILE", "CONSOLIDATION"]
N_FEATURES  = len(REGIME_FEATURES)   # full matrix width for _build_feature_matrix
N_CLASSES   = len(CLASSES)

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

# Per-TF weight paths for the 4H→1H cascade.
# regime_4h.pkl: bias layer — trained on 4H data across all symbols/groups.
# regime_1h.pkl: structure layer — trained on 1H data, refines 4H bias.
WEIGHT_PATH_4H = os.path.join(_MODEL_ROOT, "weights", "regime_4h.pkl")
WEIGHT_PATH_1H = os.path.join(_MODEL_ROOT, "weights", "regime_1h.pkl")


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
        return torch.device("cuda")
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
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
    GPU-native PyTorch MLP 4-class regime classifier.

    timeframe: "4H" (bias layer) | "1H" (structure layer) | None (legacy default).
    Each timeframe trains and saves to its own weight file so both can coexist.
    DataParallel is used across all available GPUs for both training and batch predict.
    """

    weight_path = WEIGHT_PATH

    _TF_TO_PATH = {
        "4H": WEIGHT_PATH_4H,
        "1H": WEIGHT_PATH_1H,
    }

    def __init__(self, timeframe: Optional[str] = None):
        super().__init__()
        self._model = None
        self._hysteresis_buffer: List[int] = []
        self._current_regime_id: int = 2   # default RANGING
        self._timeframe = timeframe.upper() if timeframe else None
        # Route weight file: per-TF if specified, else legacy path
        self.weight_path = self._TF_TO_PATH.get(self._timeframe, WEIGHT_PATH)
        # Feature column indices and count for this TF's classifier
        _col_idx, _feat_names = _TF_FEATURE_MAP.get(self._timeframe, _TF_FEATURE_MAP[None])
        self._col_idx   = _col_idx
        self._n_features = len(_col_idx)
        logger.debug("RegimeClassifier[%s]: %d features, weight=%s",
                     self._timeframe or "default", self._n_features, self.weight_path)
        os.makedirs(os.path.join(_MODEL_ROOT, "weights"), exist_ok=True)
        if self.is_trained:
            try:
                self.load(self.weight_path)
            except Exception as exc:
                logger.warning("RegimeClassifier[%s]: initial load failed: %s",
                               self._timeframe or "default", exc)

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
            x = torch.tensor(feat.reshape(1, -1), dtype=torch.float32).to(DEVICE)

            self._model.eval()
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                    logits = self._model(x)
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
                "regime":             CLASSES[self._current_regime_id],
                "regime_id":          self._current_regime_id,
                "proba":              proba,
                "regime_confidence":  float(max(proba)),
            }
        except Exception as exc:
            logger.error("RegimeClassifier.predict failed: %s", exc)
            raise

    def predict_batch(self, X: np.ndarray, batch_size: int = 4096) -> tuple:
        """
        GPU-batched inference on a pre-built feature matrix X (N, F_full or F_tf).
        If X has the full REGIME_FEATURES width, slices TF-specific columns first.
        Returns (labels: np.ndarray int32, confidences: np.ndarray float32).
        DataParallel splits batches across both T4 GPUs automatically.
        """
        if not self.is_trained or self._model is None:
            n = len(X)
            return np.full(n, 2, dtype=np.int32), np.full(n, 0.25, dtype=np.float32)
        try:
            import torch
            # Slice to TF-specific columns if full matrix provided
            if X.shape[1] == N_FEATURES and len(self._col_idx) < N_FEATURES:
                X = X[:, self._col_idx]
            self._model.eval()
            all_labels = []
            all_conf   = []
            with torch.no_grad():
                for s in range(0, len(X), batch_size):
                    xb = torch.from_numpy(X[s: s + batch_size]).to(DEVICE)
                    with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                        logits = self._model(xb).float()
                    proba = torch.softmax(logits, dim=1).cpu().numpy()
                    all_labels.append(proba.argmax(axis=1).astype(np.int32))
                    all_conf.append(proba.max(axis=1).astype(np.float32))
            return np.concatenate(all_labels), np.concatenate(all_conf)
        except Exception as exc:
            logger.error("RegimeClassifier.predict_batch failed: %s", exc)
            n = len(X)
            return np.full(n, 2, dtype=np.int32), np.full(n, 0.25, dtype=np.float32)

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
          high efficiency + positive drift → TRENDING_UP (0)
          high efficiency + negative drift → TRENDING_DOWN (1)
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

        # CONSOLIDATION (4): lowest atr_pctile + lowest autocorr (pre-breakout compression)
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
    def fit_global_gmm(dfs: list[pd.DataFrame], timeframe: str = None) -> tuple:
        """
        Fit one GMM on combined features from all dfs and return (gmm, scaler, cluster_labels).
        Call once before labeling — guarantees consistent regime semantics across all symbols/TFs.
        timeframe: used to pick the correct lookback window ("4H", "1H", "15M", etc.).
        """
        try:
            from sklearn.mixture import GaussianMixture
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            return None, None, None

        n_bar = RegimeClassifier._TF_NBAR.get(timeframe, RegimeClassifier._DEFAULT_NBAR)
        logger.info("GMM fit: timeframe=%s → n_bar=%d", timeframe or "default", n_bar)

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

        gmm = GaussianMixture(n_components=4, covariance_type="full",
                              random_state=42, max_iter=300)
        gmm.fit(X_scaled)

        centroids = scaler.inverse_transform(gmm.means_)
        # centroids cols: [eff, vol, drift, comp, vol_slope, atr_pctile, autocorr, hurst_proxy]
        #
        # Rank-based assignment — guaranteed to assign exactly one cluster per class
        # regardless of centroid magnitudes. Constraint-based logic fails for small or
        # correlated groups (e.g. forex crosses) where all drift centroids are near zero.
        #
        # Assignment order (greedy, most-distinguishable first):
        #   1. VOLATILE    — highest (vol - eff): high noise, low directionality
        #   2. TRENDING_UP — highest drift among remaining (most positive momentum)
        #   3. TRENDING_DOWN — lowest drift among remaining (most negative momentum)
        #   4. RANGING     — the leftover cluster (low vol, low drift, low autocorr)
        remaining = list(range(4))
        cluster_labels: dict[int, int] = {}

        # 1. VOLATILE: highest (vol - eff) — chaotic expansion, low directional efficiency
        vol_c = max(remaining, key=lambda c: centroids[c, 1] - centroids[c, 0])
        cluster_labels[vol_c] = 3
        remaining.remove(vol_c)

        # 2. TRENDING_UP: highest signed drift among remaining 3
        tu_c = max(remaining, key=lambda c: centroids[c, 2])
        cluster_labels[tu_c] = 0
        remaining.remove(tu_c)

        # 3. TRENDING_DOWN: lowest signed drift among remaining 2
        td_c = min(remaining, key=lambda c: centroids[c, 2])
        cluster_labels[td_c] = 1
        remaining.remove(td_c)

        # 4. CONSOLIDATION (4): lowest atr_pctile + lowest autocorr among remaining 2
        consol = min(remaining, key=lambda c: centroids[c, 5] + max(centroids[c, 6], 0))
        cluster_labels[consol] = 4
        remaining.remove(consol)

        # 5. RANGING (2): the last cluster — moderate vol, near-zero drift, stable ATR
        cluster_labels[remaining[0]] = 2

        dist = {CLASSES[v]: 0 for v in range(N_CLASSES)}
        for v in cluster_labels.values():
            dist[CLASSES[v]] += 1
        logger.info("GMM fitted on %d samples — cluster→regime: %s dist: %s",
                    len(X_all), cluster_labels, dist)
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
        return_confidence: bool = False,
    ):
        """
        Rule-based regime labels with per-bar confidence scores.

        Rules (priority order):
          1. VOLATILE  (3): atr_pctile ≥ 80th pctile  (expansion / stress)
          2. TRENDING_UP  (0): adx > 25 AND ema_stack ≥ 1 AND drift > 0
          3. TRENDING_DOWN(1): adx > 25 AND ema_stack ≤ -1 AND drift < 0
          4. RANGING (2): everything else

        Confidence [0→1] encodes how cleanly the rule fired:
          - Trending: scales with ADX strength (25→50 maps to 0.5→1.0),
            full stack (±2) vs partial (±1) multiplier 1.0 vs 0.7,
            and drift magnitude relative to its 80th percentile.
          - Volatile: scales with atr_pctile above threshold (0.8→1.0 maps to 0.5→1.0).
          - Ranging: confidence is low when near a trend/volatile boundary
            (ADX 20-25, or atr_pctile 0.65-0.80) and higher deep inside range.

        Bars with confidence < 0.4 are "ambiguous" — the MLP will be trained
        with reduced sample weight so it learns uncertainty on those bars rather
        than memorising a noisy hard label.

        Returns:
          labels pd.Series[int]                        (always)
          confidence pd.Series[float32]  (only if return_confidence=True)
        """
        from indicators.market_structure import compute_atr, compute_adx, compute_ema_stack_score

        n_bar = RegimeClassifier._TF_NBAR.get(
            timeframe.upper() if timeframe else "4H", RegimeClassifier._DEFAULT_NBAR
        )
        close = df["close"]
        labels = pd.Series(2, index=df.index, dtype=int)
        conf   = pd.Series(0.5, index=df.index, dtype=np.float32)

        adx       = df["adx_14"]    if "adx_14"    in df.columns else compute_adx(df, 14)
        ema_stack = df["ema_stack"] if "ema_stack" in df.columns else compute_ema_stack_score(df)

        drift = (close - close.shift(n_bar)) / (n_bar * close.shift(n_bar) + 1e-9)

        atr = compute_atr(df, n_bar)
        _hist = n_bar * 3
        atr_pctile = atr.rolling(_hist, min_periods=n_bar).apply(
            lambda x: float(np.searchsorted(np.sort(x[:-1]), x[-1])) / max(len(x) - 1, 1)
            if len(x) > 1 else 0.5, raw=True
        ).clip(0.0, 1.0).fillna(0.5)

        drift_p80 = float(drift.abs().quantile(0.80)) + 1e-9
        vol_thresh = float(atr_pctile.quantile(0.80))

        # ── VOLATILE ─────────────────────────────────────────────────────────────
        volatile_mask = atr_pctile >= vol_thresh
        labels[volatile_mask] = 3
        # confidence scales linearly from 0.5 at threshold to 1.0 at atr_pctile=1.0
        vol_conf = ((atr_pctile - vol_thresh) / (1.0 - vol_thresh + 1e-9)).clip(0.0, 1.0)
        conf[volatile_mask] = (0.5 + 0.5 * vol_conf[volatile_mask]).astype(np.float32)

        # ── TRENDING_DOWN ─────────────────────────────────────────────────────────
        td_mask = (adx > 25) & (ema_stack <= -1) & (drift < 0) & ~volatile_mask
        labels[td_mask] = 1
        adx_conf_td  = ((adx - 25) / 25.0).clip(0.0, 1.0)                   # 25→50 maps 0→1
        stack_conf_td = np.where(ema_stack <= -2, 1.0, 0.7)                  # full stack bonus
        drift_conf_td = (drift.abs() / drift_p80).clip(0.0, 1.0)
        td_conf = (adx_conf_td * stack_conf_td * drift_conf_td).astype(np.float32)
        conf[td_mask] = (0.5 + 0.5 * pd.Series(td_conf, index=df.index)[td_mask]).astype(np.float32)

        # ── TRENDING_UP ───────────────────────────────────────────────────────────
        tu_mask = (adx > 25) & (ema_stack >= 1) & (drift > 0) & ~volatile_mask
        labels[tu_mask] = 0
        adx_conf_tu   = ((adx - 25) / 25.0).clip(0.0, 1.0)
        stack_conf_tu = np.where(ema_stack >= 2, 1.0, 0.7)
        drift_conf_tu = (drift.abs() / drift_p80).clip(0.0, 1.0)
        tu_conf = (adx_conf_tu * stack_conf_tu * drift_conf_tu).astype(np.float32)
        conf[tu_mask] = (0.5 + 0.5 * pd.Series(tu_conf, index=df.index)[tu_mask]).astype(np.float32)

        # ── CONSOLIDATION — pre-breakout compression: very low ATR + negative drift ─
        # atr_pctile at multi-week low (bottom 20%) AND ATR falling (vol_slope < 0)
        _hist_consol = n_bar * 3
        atr_slope = atr.rolling(n_bar, min_periods=max(2, n_bar // 2)).apply(
            lambda x: (x[-1] - x[0]) / (x[0] + 1e-9) if len(x) > 1 else 0.0, raw=True
        ).fillna(0.0)
        consol_vol_thresh = float(atr_pctile.quantile(0.25))  # bottom quartile only
        consol_mask = (atr_pctile <= consol_vol_thresh) & (atr_slope < 0) & ~volatile_mask & ~tu_mask & ~td_mask
        labels[consol_mask] = 4
        # confidence: how deep into low-ATR territory + how negative the slope is
        consol_atr_conf   = (1.0 - (atr_pctile / (consol_vol_thresh + 1e-9)).clip(0.0, 1.0))
        consol_slope_conf = (-atr_slope).clip(0.0, 0.5) / 0.5
        consol_conf = (0.5 * consol_atr_conf + 0.5 * consol_slope_conf).clip(0.1, 1.0)
        conf[consol_mask] = (0.5 + 0.5 * consol_conf[consol_mask]).astype(np.float32)

        # ── RANGING — confidence based on distance from trend/volatile boundaries ─
        ranging_mask = ~tu_mask & ~td_mask & ~volatile_mask & ~consol_mask
        # Deep range: ADX low AND atr_pctile away from breakout threshold → high conf
        adx_range_conf   = (1.0 - (adx / 25.0).clip(0.0, 1.0))              # low ADX = confident ranging
        atr_range_conf   = (1.0 - (atr_pctile / vol_thresh).clip(0.0, 1.0)) # low atr_pctile = confident ranging
        ranging_conf     = (0.5 * adx_range_conf + 0.5 * atr_range_conf).clip(0.1, 1.0)
        conf[ranging_mask] = ranging_conf[ranging_mask].astype(np.float32)

        # ── Minimum persistence filter ────────────────────────────────────────
        # Regimes that last fewer than MIN_PERSIST bars are almost certainly
        # noise — the classifier cannot learn a reliable signal from them.
        # Zero-weight these runs so the MLP treats them as ambiguous.
        # Threshold: 20 bars at 4H ≈ 80 hours; 48 bars at 1H ≈ 48 hours.
        _tf = (timeframe or "4H").upper()
        _min_persist = {"5M": 288, "15M": 96, "1H": 48, "4H": 20, "1D": 5}.get(_tf, 20)
        _runs = (labels != labels.shift()).cumsum()
        _run_len = _runs.map(_runs.value_counts())
        _short_run_mask = _run_len < _min_persist
        conf[_short_run_mask] = 0.0   # zero weight → treated as ambiguous by trainer

        dist = {CLASSES[c]: int((labels == c).sum()) for c in range(N_CLASSES)}
        ambiguous = int((conf < 0.4).sum())
        logger.info("Rule labels [%s]: %s  ambiguous(conf<0.4)=%d (total=%d)  short_runs_zeroed=%d",
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

        # BOS and sweep counts — detect_break_of_structure uses rolling(center=True)
        # which reads swing_n future bars. Until these indicators are made strictly
        # causal, indices 6–7 stay zero to avoid lookahead contamination.
        # X[:, 6] and X[:, 7] already initialised to 0 above.

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

        # ── Macro features (indices 36–54) ────────────────────────────────────
        try:
            fe = FeatureEngine()
            macro_df = fe._build_macro_frame(df.index, symbol)
            base_macro = 36  # after 8 base + 5×4 MTF + 6 S/R + 2 regime dynamics
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
          per-bar label smoothing: target = weight * one_hot + (1-weight)/4.
          Entropy regularisation (λ=0.05) penalises overconfident predictions on
          ALL bars — encourages the model to output high uncertainty on random/ambiguous
          market conditions rather than always committing to one regime.
        """
        try:
            import torch
            import torch.nn as nn

            if len(X) < 100:
                return {"error": f"Insufficient data ({len(X)} rows)"}

            class_counts = {
                CLASSES[int(c)]: int(cnt)
                for c, cnt in zip(*np.unique(y, return_counts=True))
            }
            logger.info("RegimeClassifier: %d samples, classes=%s, device=%s",
                        len(y), class_counts, DEVICE)

            # Sanity check: all 4 classes must be present with at least 1% of samples.
            # If a class is missing the model will never predict it, and val accuracy
            # will reflect label noise rather than model quality.
            min_class_pct = min(class_counts.get(c, 0) for c in CLASSES) / max(len(y), 1)
            if len(class_counts) < N_CLASSES:
                missing = [c for c in CLASSES if c not in class_counts]
                logger.error(
                    "RegimeClassifier: MISSING CLASSES %s — label generation is broken. "
                    "All 4 classes must be present. Check GMM thresholds and create_labels().",
                    missing
                )
                return {"error": f"Missing classes: {missing}"}
            if min_class_pct < 0.01:
                rare = {c: v for c, v in class_counts.items() if v / len(y) < 0.01}
                logger.warning(
                    "RegimeClassifier: classes with <1%% of samples: %s — "
                    "model may collapse to majority class", rare
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
            if _feature_mismatch:
                logger.warning("RegimeClassifier: feature count changed %d→%d, resetting",
                               self._n_features, n_feat)
            _warm_start = (self._model is not None and not _feature_mismatch)
            if not _warm_start:
                # Cold start: fresh random init
                self._model      = _build_mlp(n_feat, N_CLASSES).to(DEVICE)
                self._n_features = n_feat
                logger.info("RegimeClassifier: cold start (no existing weights)")
            else:
                # Warm start: continue from loaded weights — preserves learned structure
                logger.info("RegimeClassifier: warm start from existing weights")
            if DEVICE.type == "cuda" and torch.cuda.device_count() > 1:
                if not isinstance(self._model, torch.nn.DataParallel):
                    self._model = torch.nn.DataParallel(self._model)
                logger.info("RegimeClassifier: DataParallel across %d GPUs",
                            torch.cuda.device_count())

            # ── Class weights (handle imbalance) ─────────────────────────────
            counts   = np.bincount(y_tr, minlength=N_CLASSES).astype(np.float32)
            counts   = np.where(counts == 0, 1.0, counts)
            class_w  = torch.tensor(counts.sum() / (N_CLASSES * counts),
                                    dtype=torch.float32).to(DEVICE)

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
            # λ=0.05 means ~5% of the loss budget pushes the model toward output uncertainty
            # on genuinely ambiguous bars. This is the "leave room for randomness" mechanism.
            _entropy_lambda = 0.05

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

                # Class-imbalance re-weighting (apply class_w to the hard label's class)
                cw_per_bar = class_w[labels]
                weighted_ce = (soft_ce * cw_per_bar * bar_weights).mean()

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
            optimiser = torch.optim.AdamW(self._model.parameters(),
                                          lr=_train_lr, weight_decay=5e-2)
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
                    per_class = {CLASSES[c]: round(float((va_p[va_t == c] == c).mean()), 3)
                                 if (va_t == c).sum() > 0 else 0.0 for c in range(N_CLASSES)}
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
                "n_classes":  N_CLASSES,
            }
            with open(path, "wb") as f:
                pickle.dump(payload, f)
            logger.info("RegimeClassifier saved to %s", path)
        except Exception as exc:
            logger.error("RegimeClassifier.save failed: %s", exc)

    def load(self, path: str) -> None:
        try:
            import torch
            with open(path, "rb") as f:
                payload = pickle.load(f)

            n_feat     = payload["n_features"]
            n_cls      = payload.get("n_classes", N_CLASSES)
            state_dict = payload["state_dict"]

            m = _build_mlp(n_feat, n_cls)
            m.load_state_dict(state_dict)
            m = m.to(DEVICE)
            m.eval()

            if DEVICE.type == "cuda" and torch.cuda.device_count() > 1:
                m = torch.nn.DataParallel(m)

            self._model      = m
            self._n_features = n_feat
            self._loaded     = True
            logger.info("RegimeClassifier loaded from %s (device=%s, features=%d)",
                        path, DEVICE, n_feat)
        except Exception as exc:
            logger.error("RegimeClassifier.load failed: %s", exc)
            raise
