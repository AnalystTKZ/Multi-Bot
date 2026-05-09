"""
unified_direction_regime.py - combined direction and HTF regime predictor.

Parallel simplification path for simplify-v2. It preserves the live/backtest
prediction dictionary consumed by SignalPipeline while replacing separate
GRU-LSTM + HTF/LTF regime inference with one PyTorch forward pass.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from models.base_model import BaseModel
from services.feature_engine import SEQUENCE_FEATURES

logger = logging.getLogger(__name__)

SEQUENCE_LENGTH = 30
N_FEATURES = len(SEQUENCE_FEATURES)
REGIME_CLASSES = ["BIAS_UP", "BIAS_DOWN", "BIAS_NEUTRAL"]

_MODEL_ROOT = Path(__file__).resolve().parent.parent
WEIGHT_DIR = _MODEL_ROOT / "weights" / "unified_direction_regime"
WEIGHT_FILE = WEIGHT_DIR / "model.pt"
MANIFEST_FILE = WEIGHT_DIR / "manifest.json"


class ModelNotTrainedError(RuntimeError):
    """Raised when unified model weights are unavailable."""


def _get_device():
    import torch

    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        del os.environ["CUDA_VISIBLE_DEVICES"]
    if torch.cuda.is_available():
        logger.info("UnifiedDirectionRegime: using CUDA")
        return torch.device("cuda")
    logger.info("UnifiedDirectionRegime: using CPU")
    return torch.device("cpu")


DEVICE = _get_device()


def _build_torch_model():
    import torch.nn as nn

    class _UnifiedDirectionRegimeNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=N_FEATURES,
                hidden_size=64,
                num_layers=2,
                batch_first=True,
                dropout=0.2,
            )
            self.shared = nn.Sequential(
                nn.LayerNorm(64),
                nn.Linear(64, 64),
                nn.GELU(),
                nn.Dropout(0.2),
            )
            self.direction_head = nn.Linear(64, 2)
            self.regime_head = nn.Linear(64, len(REGIME_CLASSES))
            self.move_head = nn.Linear(64, 1)
            self.variance_head = nn.Linear(64, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            shared = self.shared(out[:, -1, :])
            direction_logits = self.direction_head(shared)
            regime_logits = self.regime_head(shared)
            expected_move = self.move_head(shared).squeeze(-1)
            log_variance = self.variance_head(shared).squeeze(-1)
            return direction_logits, regime_logits, expected_move, log_variance

    return _UnifiedDirectionRegimeNet()


def _ensure_sequence_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in SEQUENCE_FEATURES if col not in df.columns]
    if missing:
        raise ValueError(
            "UnifiedDirectionRegime training data is missing sequence columns: "
            + ", ".join(missing[:12])
            + (" ..." if len(missing) > 12 else "")
        )
    out = df[SEQUENCE_FEATURES].astype(np.float32)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _direction_targets(df: pd.DataFrame, horizon: int) -> pd.Series:
    future = df["close"].shift(-horizon)
    returns = (future - df["close"]) / (df["close"].abs() + 1e-9)
    return (returns > 0.0).astype(np.int64)


def _regime_targets(df: pd.DataFrame, horizon: int) -> pd.Series:
    future = df["close"].shift(-horizon)
    returns = (future - df["close"]) / (df["close"].abs() + 1e-9)
    rolling_vol = df["close"].pct_change().rolling(50, min_periods=10).std().fillna(0.0)
    neutral_band = (rolling_vol * np.sqrt(max(horizon, 1)) * 0.25).clip(lower=0.0002)
    out = pd.Series(2, index=df.index, dtype=np.int64)
    out[returns > neutral_band] = 0
    out[returns < -neutral_band] = 1
    return out


def _windowed_dataset(df: pd.DataFrame, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = _ensure_sequence_columns(df).to_numpy(dtype=np.float32)
    y_dir = _direction_targets(df, horizon).to_numpy(dtype=np.int64)
    y_regime = _regime_targets(df, horizon).to_numpy(dtype=np.int64)

    max_start = len(df) - SEQUENCE_LENGTH - horizon + 1
    if max_start <= 0:
        raise ValueError("not enough rows to build unified model sequences")

    X = np.empty((max_start, SEQUENCE_LENGTH, N_FEATURES), dtype=np.float32)
    for i in range(max_start):
        X[i] = features[i : i + SEQUENCE_LENGTH]
    target_pos = np.arange(SEQUENCE_LENGTH - 1, SEQUENCE_LENGTH - 1 + max_start)
    return X, y_dir[target_pos], y_regime[target_pos]


def _windowed_dataset_by_symbol(df: pd.DataFrame, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build windows without crossing symbol boundaries when a symbol column exists."""
    if "symbol" not in df.columns:
        return _windowed_dataset(df, horizon)
    xs: list[np.ndarray] = []
    y_dirs: list[np.ndarray] = []
    y_regimes: list[np.ndarray] = []
    for _symbol, group in df.groupby("symbol", sort=False):
        group = group.sort_index()
        if len(group) < SEQUENCE_LENGTH + horizon:
            continue
        x, yd, yr = _windowed_dataset(group, horizon)
        xs.append(x)
        y_dirs.append(yd)
        y_regimes.append(yr)
    if not xs:
        raise ValueError("not enough per-symbol rows to build unified model sequences")
    return np.concatenate(xs), np.concatenate(y_dirs), np.concatenate(y_regimes)


def _trade_regime_from_outputs(regime: str, regime_conf: float, variance: float) -> str:
    if variance > float(os.getenv("UNIFIED_EXTREME_VARIANCE", "0.25")):
        return "NO_TRADE_EXTREME_VOL"
    if regime_conf < float(os.getenv("UNIFIED_MIN_REGIME_CONF", "0.45")):
        return "UNCERTAIN"
    if regime == "BIAS_NEUTRAL":
        return "UNCERTAIN"
    return "TRADEABLE_TREND"


class UnifiedDirectionRegimePredictor(BaseModel):
    """
    Single-call predictor for p_bull/p_bear and HTF directional bias.

    Compatible with SignalPipeline._compute_ml_signal through these keys:
    p_bull, p_bear, expected_move, expected_variance, regime, regime_id,
    regime_conf, regime_ltf, regime_ltf_id, trade_regime, regime_scores.
    """

    weight_path = str(WEIGHT_FILE)

    def __init__(self, load_existing: bool = True):
        super().__init__()
        self._model = None
        self._temperature = 1.0
        WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
        if load_existing and self.is_trained:
            self.load(self.weight_path)

    @property
    def is_trained(self) -> bool:
        return WEIGHT_FILE.exists() and WEIGHT_FILE.stat().st_size > 0

    def build_model(self) -> None:
        self._model = _build_torch_model().to(DEVICE)

    def predict(
        self,
        df: Optional[pd.DataFrame],
        symbol: Optional[str] = None,
        df_htf: Optional[dict] = None,
    ) -> Dict[str, float | int | str | list | dict]:
        if not self.is_trained or self._model is None:
            raise ModelNotTrainedError(
                "UnifiedDirectionRegimePredictor has no trained weights. "
                "Run: python pipeline/step7_train_unified.py"
            )
        if df is None:
            raise ValueError("UnifiedDirectionRegimePredictor.predict: df cannot be None")

        self.reload_if_updated()

        import torch
        from services.feature_engine import FeatureEngine

        seq = FeatureEngine().get_sequence(
            df,
            length=SEQUENCE_LENGTH,
            df_htf=df_htf,
            symbol=symbol,
        )
        x = torch.tensor(seq[np.newaxis, ...], dtype=torch.float32, device=DEVICE)
        self._model.eval()
        with torch.no_grad():
            dir_logits, regime_logits, move_pred, log_variance = self._model(x)
            direction_prob = torch.softmax(dir_logits[0] / self._temperature, dim=0)
            regime_prob = torch.softmax(regime_logits[0], dim=0)
            p_bull = float(direction_prob[0].item())
            p_bear = float(direction_prob[1].item())
            regime_id = int(torch.argmax(regime_prob).item())
            regime = REGIME_CLASSES[regime_id]
            regime_conf = float(regime_prob[regime_id].item())
            expected_move = float(torch.relu(move_pred[0]).clamp(0.0, 1.0).item())
            expected_variance = float(
                torch.nn.functional.softplus(log_variance)[0].clamp(1e-6, 1.0).item()
            )

        trade_regime = _trade_regime_from_outputs(regime, regime_conf, expected_variance)
        trend_score = max(p_bull, p_bear)
        chop_score = max(0.0, 1.0 - abs(p_bull - p_bear) * 2.0)
        regime_scores = {
            "trend_score": float(trend_score),
            "range_score": 0.0,
            "chop_score": float(chop_score),
            "volatility_percentile": float(np.clip(expected_variance, 0.0, 1.0)),
            "consolidation_score": float(1.0 if regime == "BIAS_NEUTRAL" else 0.0),
        }

        return {
            "p_bull": p_bull,
            "p_bear": p_bear,
            "p_bull_gru": p_bull,
            "p_bear_gru": p_bear,
            "entry_depth": float(np.clip(expected_move, 0.0, 1.0)),
            "expected_move": expected_move,
            "expected_variance": expected_variance,
            "regime": regime,
            "regime_id": regime_id,
            "regime_proba": regime_prob.detach().cpu().numpy().astype(float).tolist(),
            "regime_conf": regime_conf,
            "regime_ltf": "TRENDING" if trade_regime == "TRADEABLE_TREND" else "CONSOLIDATING",
            "regime_ltf_id": 0 if trade_regime == "TRADEABLE_TREND" else 2,
            "regime_ltf_conf": regime_scores,
            "trade_regime": trade_regime,
            "regime_scores": regime_scores,
            **regime_scores,
        }

    def train_from_frames(
        self,
        train_df: pd.DataFrame,
        validation_df: Optional[pd.DataFrame] = None,
        *,
        horizon: int = 4,
        epochs: int = 8,
        batch_size: int = 512,
        learning_rate: float = 3e-4,
    ) -> dict:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        if self._model is None:
            self.build_model()

        X_train, y_dir_train, y_reg_train = _windowed_dataset_by_symbol(train_df, horizon)
        if validation_df is not None and len(validation_df) >= SEQUENCE_LENGTH + horizon:
            X_val, y_dir_val, y_reg_val = _windowed_dataset_by_symbol(validation_df, horizon)
        else:
            split = max(1, int(len(X_train) * 0.8))
            X_val, y_dir_val, y_reg_val = X_train[split:], y_dir_train[split:], y_reg_train[split:]
            X_train, y_dir_train, y_reg_train = X_train[:split], y_dir_train[:split], y_reg_train[:split]

        train_ds = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_dir_train),
            torch.from_numpy(y_reg_train),
        )
        val_ds = TensorDataset(
            torch.from_numpy(X_val),
            torch.from_numpy(y_dir_val),
            torch.from_numpy(y_reg_val),
        )
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        opt = torch.optim.AdamW(self._model.parameters(), lr=learning_rate, weight_decay=1e-3)
        ce = nn.CrossEntropyLoss()
        history = {"train_loss": [], "val_loss": [], "val_direction_accuracy": []}
        best_val = float("inf")

        for epoch in range(int(epochs)):
            self._model.train()
            train_loss = 0.0
            for xb, y_dir, y_reg in train_dl:
                xb = xb.to(DEVICE)
                y_dir = y_dir.to(DEVICE)
                y_reg = y_reg.to(DEVICE)
                opt.zero_grad(set_to_none=True)
                dir_logits, reg_logits, _move, log_variance = self._model(xb)
                loss = ce(dir_logits, y_dir) + 0.6 * ce(reg_logits, y_reg)
                loss = loss + 0.02 * torch.nn.functional.softplus(log_variance).mean()
                loss.backward()
                opt.step()
                train_loss += float(loss.item()) * len(xb)
            train_loss /= max(1, len(train_ds))

            self._model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for xb, y_dir, y_reg in val_dl:
                    xb = xb.to(DEVICE)
                    y_dir = y_dir.to(DEVICE)
                    y_reg = y_reg.to(DEVICE)
                    dir_logits, reg_logits, _move, log_variance = self._model(xb)
                    loss = ce(dir_logits, y_dir) + 0.6 * ce(reg_logits, y_reg)
                    loss = loss + 0.02 * torch.nn.functional.softplus(log_variance).mean()
                    val_loss += float(loss.item()) * len(xb)
                    correct += int((torch.argmax(dir_logits, dim=1) == y_dir).sum().item())
                    total += int(len(xb))
            val_loss /= max(1, len(val_ds))
            val_acc = correct / max(1, total)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_direction_accuracy"].append(val_acc)
            logger.info(
                "Unified epoch %d/%d train=%.4f val=%.4f dir_acc=%.3f",
                epoch + 1,
                epochs,
                train_loss,
                val_loss,
                val_acc,
            )
            if val_loss < best_val:
                best_val = val_loss
                self.save(str(WEIGHT_DIR))

        history["best_val_loss"] = best_val
        history["train_sequences"] = int(len(train_ds))
        history["validation_sequences"] = int(len(val_ds))
        return history

    def save(self, path: str) -> None:
        import torch

        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        if self._model is None:
            raise RuntimeError("cannot save unified model before build/train")
        payload = {
            "model_state_dict": self._model.state_dict(),
            "feature_names": list(SEQUENCE_FEATURES),
            "sequence_length": SEQUENCE_LENGTH,
            "regime_classes": list(REGIME_CLASSES),
            "temperature": float(self._temperature),
        }
        torch.save(payload, out_dir / "model.pt")
        manifest = {
            "model": "UnifiedDirectionRegimePredictor",
            "weight_file": str(out_dir / "model.pt"),
            "sequence_length": SEQUENCE_LENGTH,
            "n_features": N_FEATURES,
            "feature_names": list(SEQUENCE_FEATURES),
            "regime_classes": list(REGIME_CLASSES),
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self.weight_path = str(out_dir / "model.pt")
        self._last_mtime = os.path.getmtime(self.weight_path)
        self._loaded = True

    def load(self, path: str) -> None:
        import torch

        weight_path = Path(path)
        if weight_path.is_dir():
            weight_path = weight_path / "model.pt"
        if not weight_path.exists():
            raise ModelNotTrainedError(f"unified model weights not found: {weight_path}")
        payload = torch.load(weight_path, map_location=DEVICE)
        feature_names = payload.get("feature_names", [])
        if list(feature_names) != list(SEQUENCE_FEATURES):
            raise ValueError("unified model feature contract does not match SEQUENCE_FEATURES")
        self.build_model()
        self._model.load_state_dict(payload["model_state_dict"])
        self._temperature = float(payload.get("temperature", 1.0))
        self.weight_path = str(weight_path)
        self._last_mtime = os.path.getmtime(self.weight_path)
        self._loaded = True
