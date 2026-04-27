"""
quality_scorer.py — GPU-native PyTorch MLP Expected Value (EV) predictor.

Replaces binary P(win) classifier with a continuous EV regressor:
  EV label = realized_rr  for winning trades (hit TP)
  EV label = -1.0          for losing trades (hit SL)

This lets the model directly predict trade profitability rather than just win
probability. A trade with P(win)=0.6 but RR=1.0 (EV≈0.2) is correctly ranked
below a trade with P(win)=0.45 but RR=3.0 (EV≈0.35).

Architecture: N → 64 → 32 → 1  (BN + GELU + Dropout), identity output (unbounded float)
Loss: Huber (robust to outlier large-R trades)
Output: float EV ∈ [-2, +5] approximately. Callers also receive quality_score = sigmoid(EV)
        for backward compatibility with guards expecting [0,1].
"""

from __future__ import annotations

import copy
import json
import logging
import os
import pickle
import threading
from typing import Dict

import numpy as np
import pandas as pd

from models.base_model import BaseModel

logger = logging.getLogger(__name__)

N_FEATURES = 20  # matches len(QUALITY_FEATURES) in feature_engine.py
_MODEL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHT_PATH = os.path.join(_MODEL_ROOT, "weights", "quality_scorer.pkl")


# ── Device ────────────────────────────────────────────────────────────────────

def _get_device():
    import torch
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        del os.environ["CUDA_VISIBLE_DEVICES"]
    if torch.cuda.is_available():
        logger.info("QualityScorer: CUDA available — using GPU")
        torch.backends.cudnn.benchmark        = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        return torch.device("cuda")
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") and not os.environ.get("INFERENCE_ONLY"):
        raise RuntimeError(
            "QualityScorer: CUDA not available on Kaggle — "
            "enable GPU accelerator in notebook settings."
        )
    logger.warning("QualityScorer: CUDA unavailable — using CPU")
    import torch as _t
    return _t.device("cpu")


DEVICE = _get_device()


def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).lower() in ("1", "true", "yes", "on")


def _default_allowed_splits() -> set[str] | None:
    if _env_flag("ALLOW_NONTRAIN_JOURNAL_TRAINING"):
        return None
    raw = os.getenv("JOURNAL_ALLOWED_SPLITS", "train,live,paper,production")
    return {s.strip().lower() for s in raw.split(",") if s.strip()}


def _record_allowed_by_split(record: dict, allowed_splits: set[str] | None) -> bool:
    if allowed_splits is None:
        return True
    split = str(record.get("source_split", "")).strip().lower()
    if split:
        return split in allowed_splits
    source = str(record.get("source", "")).strip().lower()
    if source.startswith("backtest_round_"):
        return False
    return _env_flag("ALLOW_UNTAGGED_JOURNAL_TRAINING")


# ── Architecture ──────────────────────────────────────────────────────────────

def _build_mlp(n_features: int = N_FEATURES):
    """N → 128 → 64 → 32 → 1  with BatchNorm + GELU + Dropout. Identity output (EV regressor)."""
    import torch.nn as nn

    class _QualityMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.BatchNorm1d(n_features),
                nn.Linear(n_features, 128),
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Dropout(0.25),
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1),   # unbounded EV output (no sigmoid)
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)   # (batch,) — raw EV float

    return _QualityMLP()


class ModelNotTrainedError(RuntimeError):
    pass


# ── Scorer ────────────────────────────────────────────────────────────────────

class QualityScorer(BaseModel):
    """GPU-native PyTorch MLP trade quality scorer."""

    weight_path = WEIGHT_PATH

    def __init__(self):
        super().__init__()
        self._model        = None
        self._n_features   = N_FEATURES
        self._feature_order = None
        self._training_metadata = {}
        self._inference_lock = threading.RLock()
        os.makedirs(os.path.join(_MODEL_ROOT, "weights"), exist_ok=True)
        if self.is_trained:
            try:
                self.load(WEIGHT_PATH)
            except Exception as exc:
                logger.warning("QualityScorer: initial load failed: %s", exc)

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, features: dict) -> dict:
        """
        features keys = FeatureEngine.QUALITY_FEATURES (exact order enforced internally).
        Returns dict with:
          "ev"            : float — raw expected value prediction (unbounded)
          "quality_score" : float in [0,1] — sigmoid(ev) for backward compatibility
        Raises ModelNotTrainedError if weights not present.
        """
        if not self.is_trained or self._model is None:
            raise ModelNotTrainedError(
                "QualityScorer has no trained weights. "
                "Run: python scripts/retrain_incremental.py --model quality"
            )
        self.reload_if_updated()

        try:
            import torch
            from services.feature_engine import QUALITY_FEATURES
            order = self._feature_order or QUALITY_FEATURES
            arr = np.array(
                [float(features.get(k, 0.0)) for k in order],
                dtype=np.float32,
            ).reshape(1, -1)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            x = torch.from_numpy(arr).to(DEVICE)

            # Unwrap DataParallel for single-sample inference — DP can't split batch=1
            _infer_m = self._model.module if isinstance(self._model, torch.nn.DataParallel) else self._model
            _infer_m.eval()
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                    ev_raw = _infer_m(x)
                ev = float(ev_raw.float()[0].item())
                quality_score = float(torch.sigmoid(ev_raw.float())[0].item())
            return {"ev": ev, "quality_score": float(np.clip(quality_score, 0.0, 1.0))}
        except Exception as exc:
            logger.error("QualityScorer.predict failed: %s", exc)
            raise

    def get_cpu_inference_fn(self):
        """
        Returns a fast CPU inference callable for single-sample use in tight loops.
        Avoids GPU kernel launch overhead (~500 µs) for batch=1 forward passes.
        The 20→128→64→32→1 MLP runs in ~3–5 µs on CPU vs ~500 µs GPU launch overhead.

        Usage:
            infer = qs_model.get_cpu_inference_fn()
            ev, qs = infer(feat_array_20)   # feat_array_20: shape (20,) float32
        """
        import torch
        if not self.is_trained or self._model is None:
            raise ModelNotTrainedError("QualityScorer has no trained weights.")
        self.reload_if_updated()
        with self._inference_lock:
            _raw = self._model.module if isinstance(self._model, torch.nn.DataParallel) else self._model
            _cpu_m = copy.deepcopy(_raw).to("cpu").eval()

        def _infer(feat: np.ndarray) -> tuple:
            arr = np.asarray(feat, dtype=np.float32).reshape(1, -1)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            x = torch.from_numpy(arr)
            with torch.no_grad():
                ev_raw = _cpu_m(x)
            ev = float(ev_raw.float().item())
            qs = float(torch.sigmoid(ev_raw.float()).item())
            return ev, float(np.clip(qs, 0.0, 1.0))

        return _infer

    def predict_batch(self, feature_matrix: np.ndarray) -> tuple:
        """
        Batch inference on CPU — small MLP (input→128→64→32→1) runs faster on CPU
        than GPU due to H2D transfer overhead at the batch sizes used here.
        feature_matrix: (N, n_features) float32 array, columns in QUALITY_FEATURES order.
        Returns (ev_array, quality_score_array) each shape (N,) float32.
        """
        if not self.is_trained or self._model is None:
            raise ModelNotTrainedError("QualityScorer has no trained weights.")
        self.reload_if_updated()

        import torch
        arr = np.asarray(feature_matrix, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.from_numpy(arr)  # stays on CPU

        with self._inference_lock:
            _raw = self._model.module if isinstance(self._model, torch.nn.DataParallel) else self._model
            _cpu_m = copy.deepcopy(_raw).to("cpu").eval()
            with torch.no_grad():
                ev_raw = _cpu_m(x)
                ev_arr = ev_raw.float().squeeze(-1).numpy()
                qs_arr = torch.sigmoid(ev_raw.float()).squeeze(-1).numpy()
            return ev_arr.astype(np.float32), np.clip(qs_arr, 0.0, 1.0).astype(np.float32)

    # ── Labels ────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_ev_label(r: dict) -> float | None:
        """
        Compute the EV label for one journal record using the documented tiered
        outcome mapping. This intentionally rewards full TP2 more than partial
        trailing exits while keeping SL outcomes at -1R.
        """
        exit_reason = str(r.get("exit_reason", "")).lower()
        rr_planned  = float(r.get("rr_ratio", 1.5))

        # Tiered EV labels — model learns to aim for tp2 (highest reward):
        #   tp2         → full TP hit: best outcome, full rr_ratio
        #   tp1         → TP1 hit, held for more: good, 0.75× rr
        #   be_or_trail → TP1 hit then trailed/stopped: partial win, 0.4× rr
        #   sl_*        → stopped out: -1.0R
        #   time_exit   → timed out: +0.2×rr_planned if pnl>0, -0.5 if pnl<0, 0.0 if flat
        pnl = float(r.get("pnl", 0.0))
        if exit_reason == "tp2":
            return float(np.clip(rr_planned, 0.1, 10.0))
        if exit_reason == "tp1":
            return float(np.clip(rr_planned * 0.75, 0.1, 10.0))
        if exit_reason == "be_or_trail":
            return float(np.clip(rr_planned * 0.4, 0.1, 10.0))
        if "sl" in exit_reason:
            return -1.0
        if "tp" in exit_reason:
            return float(np.clip(rr_planned, 0.1, 10.0))
        if "time" in exit_reason:
            if pnl > 0:
                return float(np.clip(rr_planned * 0.2, 0.01, 10.0))
            if pnl < 0:
                return -0.5
            return 0.0
        return None

    def create_labels(
        self,
        journal_path: str,
        allowed_splits: set[str] | None = None,
    ) -> pd.DataFrame:
        """
        Read trade_journal_detailed.jsonl.
        EV label = documented tiered R-multiple mapping.
        Production training excludes validation/test backtest journals unless
        ALLOW_NONTRAIN_JOURNAL_TRAINING=1 is set.
        """
        allowed_splits = _default_allowed_splits() if allowed_splits is None else allowed_splits
        records = []
        skipped_split = 0
        try:
            with open(journal_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not _record_allowed_by_split(rec, allowed_splits):
                        skipped_split += 1
                        continue
                    records.append(rec)
        except FileNotFoundError:
            logger.warning("QualityScorer: journal not found at %s", journal_path)
            return pd.DataFrame()
        if skipped_split:
            logger.info(
                "QualityScorer: skipped %d journal records outside allowed splits %s",
                skipped_split,
                "ALL" if allowed_splits is None else sorted(allowed_splits),
            )

        # Per-strategy rolling outcome history for win-rate features (no lookahead).
        # Populated as we iterate records in chronological order — each row gets
        # the win rates from *prior* trades only, then we append the current outcome.
        from collections import defaultdict
        _outcomes: dict = defaultdict(list)  # trader_id → list of bool (win=True)

        rows = []
        for r in records:
            ev_label = self._compute_ev_label(r)
            if ev_label is None:
                continue

            rr = float(r.get("rr_ratio", 1.5))
            ml_scores = r.get("ml_model_scores", {}) or {}
            sig_meta  = r.get("signal_metadata",  {}) or {}
            trader_id = str(r.get("trader", ""))

            # Rolling win rates from prior trades only (no lookahead)
            _hist = _outcomes[trader_id]

            def _roll_wr(n: int) -> float:
                if not _hist:
                    return 0.5
                recent = _hist[-n:]
                return float(np.mean(recent))

            wr_5  = _roll_wr(5)
            wr_20 = _roll_wr(20)
            wr_50 = _roll_wr(50)

            # GRU/signal agreement: 1.0 if GRU bull direction agrees with signal side
            p_bull = float(ml_scores.get("p_bull", 0.5))
            side   = str(r.get("side", "")).lower()
            gru_bull = p_bull > 0.5
            if side == "buy":
                _agree = 1.0 if gru_bull else 0.0
            elif side == "sell":
                _agree = 1.0 if not gru_bull else 0.0
            else:
                _agree = 0.5

            rows.append({
                "strategy_id":          trader_id,
                "signal_direction":     1 if side == "buy" else 0,
                "rr_ratio":             rr,
                "p_bull_gru":           p_bull,
                "p_bear_gru":           float(ml_scores.get("p_bear", 0.5)),
                "regime_class":         ml_scores.get("regime", "RANGING"),
                "sentiment_score":      float(ml_scores.get("sentiment_score", 0.0)),
                # Read real bar-derived values from signal_metadata; fall back to
                # ml_model_scores for fields that some recorders put there, and
                # only use the neutral default when both are absent.
                "adx_at_signal":        float(
                    sig_meta.get("adx_at_signal",
                    sig_meta.get("adx",
                    ml_scores.get("adx_at_signal", 20.0)))),
                "atr_ratio_at_signal":  float(
                    sig_meta.get("atr_ratio_at_signal",
                    sig_meta.get("atr_ratio",
                    ml_scores.get("atr_ratio_at_signal", 1.0)))),
                "volume_ratio":         float(
                    sig_meta.get("volume_ratio",
                    ml_scores.get("volume_ratio", 1.0))),
                "spread_at_signal":     float(
                    sig_meta.get("spread_at_signal",
                    sig_meta.get("spread",
                    ml_scores.get("spread_at_signal", 1.0)))),
                "session_at_signal":    r.get("session", "NY"),
                "news_in_30min":        int(
                    sig_meta.get("news_in_30min",
                    ml_scores.get("news_in_30min", 0))),
                "strategy_win_rate_20": wr_20,
                "gru_uncertainty":      float(ml_scores.get("expected_variance", 0.1)),
                "regime_duration":      float(ml_scores.get("regime_duration", 0.5)),
                "vol_slope_at_signal":  float(ml_scores.get("vol_slope", 0.0)),
                # New features (indices 17–19) — see feature_engine.QUALITY_FEATURES
                "strategy_win_rate_5":  wr_5,
                "strategy_win_rate_50": wr_50,
                "gru_signal_agreement": _agree,
                "label":                ev_label,
            })

            # Append outcome *after* recording the row (strict no-lookahead)
            _outcomes[trader_id].append(ev_label > 0)

        return pd.DataFrame(rows)

    # ── Train ─────────────────────────────────────────────────────────────────

    def train(
        self,
        journal_path: str,
        allowed_splits: set[str] | None = None,
    ) -> dict:
        """Train EV regressor from journal. Returns metrics dict."""
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
            from services.feature_engine import QUALITY_FEATURES

            labeled_df = self.create_labels(journal_path, allowed_splits=allowed_splits)
            if labeled_df is None or len(labeled_df) < 20:
                return {"error": "Insufficient journal data"}
            effective_splits = _default_allowed_splits() if allowed_splits is None else allowed_splits
            self._training_metadata = {
                "journal_path": journal_path,
                "allowed_splits": "ALL" if effective_splits is None else sorted(effective_splits),
                "n_labeled_rows": int(len(labeled_df)),
            }

            feature_cols = [c for c in QUALITY_FEATURES if c in labeled_df.columns]
            self._feature_order = feature_cols

            df_enc = labeled_df[feature_cols].copy()
            for col in df_enc.select_dtypes(include="object").columns:
                df_enc[col] = pd.Categorical(df_enc[col]).codes.astype(np.float32)
            X = df_enc.values.astype(np.float32)
            y = labeled_df["label"].values.astype(np.float32)   # EV labels: rr for wins, -1 for losses

            ev_stats = {"mean": float(np.mean(y)), "std": float(np.std(y)),
                        "n_pos": int(np.sum(y > 0)), "n_neg": int(np.sum(y <= 0))}
            logger.info("QualityScorer: %d samples, EV stats=%s, device=%s",
                        len(y), ev_stats, DEVICE)

            # Normalise EV labels to [-1, +1] range so losers and winners occupy
            # symmetric scales. Without this, winners cluster at +0.5–+3.0 while
            # losers spike at -1.0, causing the model to predict negative EV for
            # almost all bars (since 80% of trades lose).
            # Wins: scale by median winner RR so +1.0 = "typical win".
            # Losses stay at -1.0 (already 1R loss by definition).
            pos_mask = y > 0
            if pos_mask.sum() > 5:
                median_win = float(np.median(y[pos_mask]))
                if median_win > 0.1:
                    y = y.copy()
                    y[pos_mask] = np.clip(y[pos_mask] / median_win, 0.0, 3.0)
                    logger.info("QualityScorer: normalised win labels by median_win=%.3f — EV range now [-1, +3]", median_win)

            # Temporal split
            split      = int(len(X) * 0.8)
            X_tr, X_va = X[:split],  X[split:]
            y_tr, y_va = y[:split],  y[split:]

            if len(X_tr) < 10 or len(X_va) < 5:
                return {"error": "Not enough data after split"}

            n_feat = X.shape[1]
            _warm_start = (self._model is not None and self._n_features == n_feat)
            if not _warm_start:
                _m = _build_mlp(n_feat).to(DEVICE)
                _n_gpu = torch.cuda.device_count() if DEVICE.type == "cuda" else 0
                if _n_gpu > 1:
                    _m = torch.nn.DataParallel(_m)
                    logger.info("QualityScorer: DataParallel across %d GPUs", _n_gpu)
                self._model      = _m
                self._n_features = n_feat
                logger.info("QualityScorer: cold start")
            else:
                logger.info("QualityScorer: warm start from existing weights")

            tr_ds    = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
            va_ds    = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))
            _pin     = DEVICE.type == "cuda"
            _workers = 2 if DEVICE.type == "cuda" else 0
            _bs      = min(512, len(tr_ds)) if DEVICE.type == "cuda" else min(256, len(tr_ds))
            tr_dl    = DataLoader(tr_ds, batch_size=_bs, shuffle=True,
                                  num_workers=_workers, pin_memory=_pin,
                                  persistent_workers=(_workers > 0))
            va_dl    = DataLoader(va_ds, batch_size=_bs, shuffle=False,
                                  num_workers=_workers, pin_memory=_pin,
                                  persistent_workers=(_workers > 0))

            # Class-weighted Huber: balance gradients between winning and losing trades.
            # pos_weight = n_neg / n_pos — amplifies winner gradient when losses dominate,
            # attenuates it when winners dominate (WR > 50%). Do NOT clamp to 1.0 minimum:
            # when WR > 50% (n_pos > n_neg), pos_weight < 1.0 is correct — it down-weights
            # winner predictions so the model allocates more capacity to separating marginal
            # losers from strong winners, rather than predicting every trade as positive EV.
            _n_pos = float(np.sum(y_tr > 0))
            _n_neg = float(np.sum(y_tr <= 0))
            _pos_weight = (_n_neg / max(_n_pos, 1.0))
            _pos_weight = float(np.clip(_pos_weight, 0.25, 8.0))  # allow < 1 when WR > 50%
            logger.info("QualityScorer: pos_weight=%.2f (n_pos=%d n_neg=%d)", _pos_weight, int(_n_pos), int(_n_neg))

            import torch as _torch_qs
            def _weighted_huber(pred, target, delta=1.0):
                w = _torch_qs.where(target > 0,
                                    _torch_qs.tensor(_pos_weight, device=pred.device),
                                    _torch_qs.ones(1, device=pred.device))
                err = pred - target
                abs_err = err.abs()
                huber = _torch_qs.where(abs_err < delta,
                                        0.5 * err ** 2,
                                        delta * (abs_err - 0.5 * delta))
                return (w * huber).mean()

            criterion = _weighted_huber
            # Fine-tuning uses 5× lower LR — preserves learned EV structure.
            _base_lr  = 1e-3
            _train_lr = _base_lr / 5.0 if _warm_start else _base_lr
            optimiser = torch.optim.AdamW(self._model.parameters(),
                                          lr=_train_lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=100)
            use_amp   = DEVICE.type == "cuda"
            scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)

            best_val_loss = float("inf")
            patience, no_improve = 10, 0
            best_state = None

            for epoch in range(100):
                self._model.train()
                for xb, yb in tr_dl:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    optimiser.zero_grad()
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        ev_pred = self._model(xb)
                    loss = criterion(ev_pred.float(), yb)
                    scaler.scale(loss).backward()
                    scaler.step(optimiser)
                    scaler.update()
                scheduler.step()

                self._model.eval()
                va_loss = 0.0
                with torch.no_grad():
                    for xb, yb in va_dl:
                        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                        with torch.amp.autocast("cuda", enabled=use_amp):
                            ev_pred = self._model(xb)
                        va_loss += criterion(ev_pred.float(), yb).item() * len(xb)
                va_loss /= max(1, len(va_ds))

                if epoch % 10 == 0 or epoch < 5:
                    logger.info("Quality epoch %3d/100 — va_huber=%.4f",
                                epoch + 1, va_loss)

                if va_loss < best_val_loss:
                    best_val_loss = va_loss
                    no_improve    = 0
                    _snap = self._model.module if isinstance(self._model, torch.nn.DataParallel) else self._model
                    best_state    = {k: v.cpu().clone() for k, v in _snap.state_dict().items()}
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        logger.info("Quality early stop at epoch %d", epoch + 1)
                        break

            if best_state is not None:
                _restore = self._model.module if isinstance(self._model, torch.nn.DataParallel) else self._model
                _restore.load_state_dict(best_state)

            # Final metrics: MAE and EV direction accuracy (predicted positive → actual positive)
            self._model.eval()
            all_preds_ev, all_true_ev = [], []
            with torch.no_grad():
                for xb, yb in va_dl:
                    xb = xb.to(DEVICE)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        ev_out = self._model(xb).float().cpu().numpy()
                    all_preds_ev.extend(ev_out)
                    all_true_ev.extend(yb.numpy())
            preds_arr = np.array(all_preds_ev)
            true_arr  = np.array(all_true_ev)
            mae = float(np.mean(np.abs(preds_arr - true_arr)))
            # Direction accuracy: predicted EV > 0 matches actual EV > 0
            dir_acc = float(np.mean((preds_arr > 0) == (true_arr > 0)))
            logger.info("QualityScorer EV model: MAE=%.3f dir_acc=%.3f n_val=%d",
                        mae, dir_acc, len(true_arr))

            self.save(WEIGHT_PATH)
            return {"mae": mae, "dir_accuracy": dir_acc, "n_samples": len(X), "ev_stats": ev_stats}

        except Exception as exc:
            logger.error("QualityScorer.train failed: %s", exc)
            raise

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        if self._model is None:
            return
        try:
            import torch
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            import torch
            _m = self._model.module if isinstance(self._model, torch.nn.DataParallel) else self._model
            payload = {
                "state_dict":    {k: v.cpu() for k, v in _m.state_dict().items()},
                "n_features":    self._n_features,
                "feature_order": self._feature_order,
                "training_metadata": self._training_metadata,
            }
            with open(path, "wb") as f:
                pickle.dump(payload, f)
            logger.info("QualityScorer saved to %s", path)
        except Exception as exc:
            logger.error("QualityScorer.save failed: %s", exc)

    def load(self, path: str) -> None:
        try:
            import torch
            with open(path, "rb") as f:
                payload = pickle.load(f)

            n_feat     = payload["n_features"]
            state_dict = payload["state_dict"]

            # Architecture mismatch guard: if the saved weights have a different
            # n_features than the current N_FEATURES (e.g. old 17-dim vs new 20-dim),
            # the pkl is stale and cannot be loaded into the current model. Delete it
            # so the next training run starts cold with the correct architecture.
            if n_feat != N_FEATURES:
                logger.warning(
                    "QualityScorer: stale weights at %s have n_features=%d but "
                    "current model expects %d — deleting and starting cold.",
                    path, n_feat, N_FEATURES,
                )
                os.remove(path)
                return

            import torch
            m = _build_mlp(n_feat)
            m.load_state_dict(state_dict)
            m = m.to(DEVICE)
            _n_gpu = torch.cuda.device_count() if DEVICE.type == "cuda" else 0
            if _n_gpu > 1:
                m = torch.nn.DataParallel(m)
            m.eval()

            self._model         = m
            self._n_features    = n_feat
            self._feature_order = payload.get("feature_order")
            self._loaded        = True
            logger.info("QualityScorer loaded from %s (device=%s)", path, DEVICE)
        except Exception as exc:
            logger.error("QualityScorer.load failed: %s", exc)
            raise
