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

import json
import logging
import os
import pickle
from typing import Dict

import numpy as np
import pandas as pd

from models.base_model import BaseModel

logger = logging.getLogger(__name__)

N_FEATURES = 14
_MODEL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHT_PATH = os.path.join(_MODEL_ROOT, "weights", "quality_scorer.pkl")


# ── Device ────────────────────────────────────────────────────────────────────

def _get_device():
    import torch
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        del os.environ["CUDA_VISIBLE_DEVICES"]
    if torch.cuda.is_available():
        logger.info("QualityScorer: CUDA available — using GPU")
        return torch.device("cuda")
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        raise RuntimeError(
            "QualityScorer: CUDA not available on Kaggle — "
            "enable GPU accelerator in notebook settings."
        )
    logger.warning("QualityScorer: CUDA unavailable — using CPU")
    import torch as _t
    return _t.device("cpu")


DEVICE = _get_device()


# ── Architecture ──────────────────────────────────────────────────────────────

def _build_mlp(n_features: int = N_FEATURES):
    """N → 64 → 32 → 1  with BatchNorm + GELU + Dropout. Identity output (EV regressor)."""
    import torch.nn as nn

    class _QualityMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.BatchNorm1d(n_features),
                nn.Linear(n_features, 64),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Dropout(0.3),
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

            self._model.eval()
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                    ev_raw = self._model(x)
                ev = float(ev_raw.float()[0].item())
                quality_score = float(torch.sigmoid(ev_raw.float())[0].item())
            return {"ev": ev, "quality_score": float(np.clip(quality_score, 0.0, 1.0))}
        except Exception as exc:
            logger.error("QualityScorer.predict failed: %s", exc)
            raise

    # ── Labels ────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_ev_label(r: dict) -> float | None:
        """
        Compute the EV label for one journal record as realized_return / entry_risk.

        Realized return = actual PnL in price units (from journal "pnl" field).
        Entry risk      = |entry - stop_loss| in price units.

        This gives a continuously-scaled label in R-multiples:
          +1.5  → trade won 1.5R
          -1.0  → trade lost exactly 1R (hit SL)
          +0.7  → partial TP, exited early at 0.7R

        Falls back to ±planned_rr when pnl/entry/sl are missing (backtest records
        don't always store all fields).
        """
        exit_reason = str(r.get("exit_reason", "")).lower()
        is_win  = "tp" in exit_reason
        is_loss = "sl" in exit_reason
        if not is_win and not is_loss:
            return None

        rr_planned = float(r.get("rr_ratio", 1.5))

        # Use rr_ratio directly as the EV label. This is already in R-multiples
        # and is consistent across all symbols/sizes/currencies.
        # Wins: +rr_ratio (e.g. +1.5 for a 1.5R winner, +2.0 for TP2 hit).
        # Losses: -1.0 (lost exactly 1R regardless of pip size or lot size).
        # This avoids the pnl/risk_staked calculation which produces out-of-range
        # values when pnl is in account currency and entry_risk is in price units.
        if is_win:
            return float(np.clip(rr_planned, 0.1, 10.0))
        return -1.0

    def create_labels(self, journal_path: str) -> pd.DataFrame:
        """
        Read trade_journal_detailed.jsonl.
        EV label = realized_pnl / entry_risk (in R-multiples).
        Positive → winner, negative → loser. Continuously scaled.
        """
        records = []
        try:
            with open(journal_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            logger.warning("QualityScorer: journal not found at %s", journal_path)
            return pd.DataFrame()

        rows = []
        for r in records:
            ev_label = self._compute_ev_label(r)
            if ev_label is None:
                continue

            rr = float(r.get("rr_ratio", 1.5))
            ml_scores = r.get("ml_model_scores", {})
            rows.append({
                "strategy_id":          r.get("trader", ""),
                "signal_direction":     1 if r.get("side") == "buy" else 0,
                "rr_ratio":             rr,
                "p_bull_gru":           float(ml_scores.get("p_bull", 0.5)),
                "p_bear_gru":           float(ml_scores.get("p_bear", 0.5)),
                "regime_class":         ml_scores.get("regime", "RANGING"),
                "sentiment_score":      float(ml_scores.get("sentiment_score", 0.0)),
                "adx_at_signal":        20.0,
                "atr_ratio_at_signal":  1.0,
                "volume_ratio":         1.0,
                "spread_at_signal":     1.0,
                "session_at_signal":    r.get("session", "NY"),
                "news_in_30min":        0,
                "strategy_win_rate_20": 0.5,
                "gru_uncertainty":      float(ml_scores.get("expected_variance", 0.1)),
                "regime_duration":      float(ml_scores.get("regime_duration", 0.5)),
                "vol_slope_at_signal":  float(ml_scores.get("vol_slope", 0.0)),
                "label":                ev_label,
            })

        return pd.DataFrame(rows)

    # ── Train ─────────────────────────────────────────────────────────────────

    def train(self, journal_path: str) -> dict:
        """Train EV regressor from journal. Returns metrics dict."""
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
            from services.feature_engine import QUALITY_FEATURES

            labeled_df = self.create_labels(journal_path)
            if labeled_df is None or len(labeled_df) < 20:
                return {"error": "Insufficient journal data"}

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
                self._model      = _build_mlp(n_feat).to(DEVICE)
                self._n_features = n_feat
                logger.info("QualityScorer: cold start")
            else:
                logger.info("QualityScorer: warm start from existing weights")

            _pin     = DEVICE.type == "cuda"
            _workers = 2 if DEVICE.type == "cuda" else 0
            tr_ds    = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
            va_ds    = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))
            tr_dl    = DataLoader(tr_ds, batch_size=min(256, len(tr_ds)), shuffle=True,
                                  num_workers=_workers, pin_memory=_pin,
                                  persistent_workers=(_workers > 0))
            va_dl    = DataLoader(va_ds, batch_size=min(256, len(va_ds)), shuffle=False,
                                  num_workers=_workers, pin_memory=_pin,
                                  persistent_workers=(_workers > 0))

            # Class-weighted Huber: amplify gradient on winning trades to counteract
            # the ~18%/82% positive/negative imbalance in the journal.
            _n_pos = float(np.sum(y_tr > 0))
            _n_neg = float(np.sum(y_tr <= 0))
            _pos_weight = (_n_neg / max(_n_pos, 1.0))  # e.g. ~4.6× for 18/82 split
            _pos_weight = float(np.clip(_pos_weight, 1.0, 8.0))  # cap at 8× to avoid instability
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
                    best_state    = {k: v.cpu().clone()
                                     for k, v in self._model.state_dict().items()}
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        logger.info("Quality early stop at epoch %d", epoch + 1)
                        break

            if best_state is not None:
                self._model.load_state_dict(best_state)

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
            payload = {
                "state_dict":    {k: v.cpu() for k, v in self._model.state_dict().items()},
                "n_features":    self._n_features,
                "feature_order": self._feature_order,
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

            m = _build_mlp(n_feat)
            m.load_state_dict(state_dict)
            m = m.to(DEVICE)
            m.eval()

            self._model         = m
            self._n_features    = n_feat
            self._feature_order = payload.get("feature_order")
            self._loaded        = True
            logger.info("QualityScorer loaded from %s (device=%s)", path, DEVICE)
        except Exception as exc:
            logger.error("QualityScorer.load failed: %s", exc)
            raise
