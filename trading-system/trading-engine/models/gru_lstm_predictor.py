"""
gru_lstm_predictor.py — multi-head GRU-LSTM market state predictor (PyTorch).

Architecture: shared GRU-LSTM encoder → direction / magnitude / variance heads.
Outputs preserve the existing public interface (`p_bull`, `p_bear`, `entry_depth`)
while also exposing richer predictions (`expected_move`, `expected_volatility`,
`expected_variance`).

Weights saved as standard PyTorch state_dict (.pt).
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from models.base_model import BaseModel
from services.feature_engine import SEQUENCE_FEATURES

logger = logging.getLogger(__name__)

SEQUENCE_LENGTH = 30
N_FEATURES = len(SEQUENCE_FEATURES)
_MODEL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # trading-engine/
WEIGHT_DIR  = os.path.join(_MODEL_ROOT, "weights", "gru_lstm") + os.sep
WEIGHT_FILE = os.path.join(_MODEL_ROOT, "weights", "gru_lstm", "model.pt")
MAX_INFERENCE_VARIANCE = float(os.getenv("GRU_MAX_INFERENCE_VARIANCE", "1.0"))
GRU_LABEL_TARGET_R = float(os.getenv("GRU_LABEL_TARGET_ATR", "2.5")) / max(
    float(os.getenv("GRU_LABEL_STOP_ATR", "1.5")),
    1e-6,
)

# Symbol group IDs for per-group specialisation: 0=gold, 1=dollar-pairs, 2=JPY-pairs, 3=unknown
SYMBOL_GROUPS: dict = {
    "XAUUSD": 0,
    "EURUSD": 1, "GBPUSD": 1, "AUDUSD": 1, "USDCAD": 1, "USDCHF": 1, "NZDUSD": 1,
    "USDJPY": 2, "GBPJPY": 2, "EURJPY": 2, "CADJPY": 2, "AUDJPY": 2, "CHFJPY": 2,
}
N_GROUPS = 3          # gold / dollar / JPY
GROUP_EMBED_DIM = 8   # learned per-group bias concatenated to shared representation


def _calibrated_variance(log_variance_pred):
    """Match inference variance scale to the clamped training objective."""
    import torch

    return torch.clamp(
        torch.nn.functional.softplus(log_variance_pred) + 1e-6,
        min=1e-4,
        max=MAX_INFERENCE_VARIANCE,
    )


def _get_device():
    """
    Return best available device.
    Ensures CUDA_VISIBLE_DEVICES is not masking GPUs, then selects cuda:0.
    Falls back to CPU if CUDA is unavailable. Kaggle CPU training is slow, but
    it should still cold-start/warm-start and produce artifacts.
    """
    import os
    import torch

    # Remove any mask that hides GPUs (e.g. CUDA_VISIBLE_DEVICES="")
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        del os.environ["CUDA_VISIBLE_DEVICES"]

    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        logger.info("GRU: %d CUDA device(s) available — using GPU", n)
        for i in range(n):
            logger.info("  GPU %d: %s (%.1f GB)",
                        i, torch.cuda.get_device_name(i),
                        torch.cuda.get_device_properties(i).total_memory / 1e9)
        torch.backends.cudnn.benchmark        = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        return torch.device("cuda")

    logger.warning(
        "GRU: CUDA unavailable%s — using CPU (training will be slow)",
        " on Kaggle" if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") else "",
    )
    return torch.device("cpu")


DEVICE = _get_device()


class ModelNotTrainedError(RuntimeError):
    """Raised when a model is used before being trained."""


# ── PyTorch model definition ──────────────────────────────────────────────────

def _build_torch_model():
    import torch
    import torch.nn as nn

    class _MultiHeadGRULSTM(nn.Module):
        """
        GRU(64,2) → LSTM(128,2) → MultiHeadAttention(128,4) → mean-pool → shared(64)
        → group_embed(8) → concat(72)
        → [direction logit, r_long regression, r_short regression, log-variance].

        Side-conditioned R heads: head_r_long is trained on realized long-side R,
        head_r_short on realized short-side R. At inference the selected side's
        head is used as the signal-quality gate.

        Group embedding (gold / dollar / JPY): 8-dim learnable bias concatenated
        to the 64-dim shared repr gives each symbol group its own head calibration
        without sacrificing shared GRU weights.
        """
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(
                input_size=N_FEATURES,
                hidden_size=64,
                num_layers=2,
                batch_first=True,
                dropout=0.3,
            )
            self.drop1 = nn.Dropout(0.3)
            self.lstm = nn.LSTM(
                input_size=64,
                hidden_size=128,
                num_layers=2,
                batch_first=True,
                dropout=0.3,
            )
            # Self-attention over the full sequence.
            # 4 heads × 32 dims each = 128 total embed_dim.
            self.attn  = nn.MultiheadAttention(
                embed_dim=128, num_heads=4, dropout=0.1, batch_first=True
            )
            self.drop2 = nn.Dropout(0.3)
            self.shared = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
            )
            # Symbol-group embedding: N_GROUPS+1 rows (last = unknown/default)
            self.group_embed = nn.Embedding(N_GROUPS + 1, GROUP_EMBED_DIM)
            nn.init.zeros_(self.group_embed.weight)  # start as no-op; learn offsets

            _h = 64 + GROUP_EMBED_DIM  # 72
            self.head_dir     = nn.Linear(_h, 1)
            self.head_r_long  = nn.Linear(_h, 1)   # long-side realized R
            self.head_r_short = nn.Linear(_h, 1)   # short-side realized R
            self.head_var     = nn.Linear(_h, 1)

        def forward(self, x, group_id=None):
            out, _ = self.gru(x)                       # (B, T, 64)
            out    = self.drop1(out)
            out, _ = self.lstm(out)                    # (B, T, 128)
            attn_out, _ = self.attn(out, out, out)     # (B, T, 128)
            out    = self.drop2(attn_out.mean(dim=1))  # (B, 128) — mean-pool timesteps
            shared = self.shared(out)                  # (B, 64)

            if group_id is None:
                group_id = shared.new_zeros(shared.shape[0], dtype=torch.long) + N_GROUPS
            emb = self.group_embed(group_id)           # (B, 8)
            h   = torch.cat([shared, emb], dim=-1)     # (B, 72)

            dir_logits   = self.head_dir(h).squeeze(-1)
            r_long       = self.head_r_long(h).squeeze(-1)
            r_short      = self.head_r_short(h).squeeze(-1)
            log_variance = self.head_var(h).squeeze(-1)
            return dir_logits, r_long, r_short, log_variance

        def encode(self, x):
            """Return the 64-dim shared embedding (no group offset) for similarity search."""
            out, _ = self.gru(x)
            out    = self.drop1(out)
            out, _ = self.lstm(out)
            attn_out, _ = self.attn(out, out, out)
            out = self.drop2(attn_out.mean(dim=1))
            return self.shared(out)  # (B, 64)

    return _MultiHeadGRULSTM()


# ── Predictor ─────────────────────────────────────────────────────────────────

class GRULSTMPredictor(BaseModel):
    """
    GRU-LSTM direction + entry-depth predictor (PyTorch backend).
    Raises ModelNotTrainedError if weights are not present.
    Run: python scripts/retrain_incremental.py --model gru
    """

    weight_path = WEIGHT_DIR

    def __init__(self):
        super().__init__()
        self._model = None
        self._temperature: float = 1.0
        self._isotonic = None   # IsotonicRegression calibrator — loaded from isotonic.pkl if present
        os.makedirs(WEIGHT_DIR, exist_ok=True)
        if self.is_trained:
            try:
                self.load(WEIGHT_DIR)
            except Exception as exc:
                logger.warning("GRULSTMPredictor: initial load failed: %s", exc)

    @property
    def is_trained(self) -> bool:
        return os.path.exists(WEIGHT_FILE)

    def build_model(self) -> None:
        """Build PyTorch architecture. Called once before training."""
        try:
            import torch
            m = _build_torch_model().to(DEVICE)
            n_gpu = torch.cuda.device_count() if DEVICE.type == "cuda" else 0
            if n_gpu > 1:
                m = torch.nn.DataParallel(m, device_ids=list(range(n_gpu)))
                logger.info("GRULSTMPredictor: DataParallel across %d GPUs %s",
                            n_gpu, [torch.cuda.get_device_name(i) for i in range(n_gpu)])
            elif n_gpu == 1:
                logger.info("GRULSTMPredictor: single GPU — %s", torch.cuda.get_device_name(0))
            self._model = m
            logger.info("GRULSTMPredictor: model built (PyTorch, device=%s)", DEVICE)
        except ImportError:
            logger.error("PyTorch not available — GRULSTMPredictor disabled")
            self._model = None

    def predict(self, df: Optional[pd.DataFrame], symbol: Optional[str] = None,
                df_htf: Optional[dict] = None) -> Dict[str, float]:
        """
        Returns dict with p_bull, p_bear, entry_depth plus probabilistic move outputs.
        df_htf: full {tf: DataFrame} dict for MTF cross-TF sequence features.
                Keys: "5M", "1H", "4H", "1D". Missing keys raise.
        Raises ModelNotTrainedError if weights not available.
        Run: python scripts/retrain_incremental.py --model gru
        """
        if not self.is_trained or self._model is None:
            raise ModelNotTrainedError(
                "GRULSTMPredictor has no trained weights. "
                "Run: python scripts/retrain_incremental.py --model gru"
            )
        if df is None:
            raise ValueError("GRULSTMPredictor.predict: df cannot be None")

        self.reload_if_updated()

        try:
            import torch
            from services.feature_engine import FeatureEngine
            fe = FeatureEngine()
            seq = fe.get_sequence(df, length=SEQUENCE_LENGTH, df_htf=df_htf, symbol=symbol)  # (30, N)
            x = torch.tensor(seq[np.newaxis, ...], dtype=torch.float32).to(DEVICE)  # (1, 30, N_FEATURES)

            group_id = SYMBOL_GROUPS.get(symbol, N_GROUPS)
            group_id_t = torch.tensor([group_id], dtype=torch.long).to(DEVICE)

            self._model.eval()
            with torch.no_grad():
                dir_logits, r_long_pred, r_short_pred, log_variance_pred = self._model(x, group_id_t)
                p_bull_raw = float(torch.sigmoid(dir_logits[0] / self._temperature).item())
                expected_r_long  = float(np.clip(r_long_pred[0].item(),  -1.0, GRU_LABEL_TARGET_R))
                expected_r_short = float(np.clip(r_short_pred[0].item(), -1.0, GRU_LABEL_TARGET_R))
                # Side-conditioned R: use the head that matches the predicted direction
                expected_r_gru = expected_r_long if p_bull_raw >= 0.5 else expected_r_short
                expected_move = float(np.clip(max(expected_r_gru, 0.0) / max(GRU_LABEL_TARGET_R, 1e-6), 0.0, 1.0))
                expected_variance = float(_calibrated_variance(log_variance_pred)[0].item())
                expected_volatility = float(np.sqrt(expected_variance))
                entry_depth = expected_move

            # Apply isotonic calibration if available — reduces ECE from 0.56 to <0.10
            if self._isotonic is not None:
                p_bull = float(np.clip(self._isotonic.predict([p_bull_raw])[0], 0.0, 1.0))
            else:
                p_bull = p_bull_raw

            return {
                "p_bull": p_bull,
                "p_bear": float(np.clip(1.0 - p_bull, 0.0, 1.0)),
                "entry_depth": entry_depth,
                "expected_move": expected_move,
                "expected_r_gru": expected_r_gru,
                "expected_r_long": expected_r_long,
                "expected_r_short": expected_r_short,
                "expected_volatility": expected_volatility,
                "expected_variance": expected_variance,
            }
        except Exception as exc:
            logger.error("GRULSTMPredictor.predict failed: %s", exc)
            raise

    def get_embedding(
        self,
        df: Optional[pd.DataFrame],
        symbol: Optional[str] = None,
        df_htf: Optional[dict] = None,
    ) -> Optional[np.ndarray]:
        """
        Return the 64-dim shared-layer embedding for the most recent bar in df.

        This is the output of the GRU→LSTM→shared path before the three output
        heads — a dense representation of the current market state suitable for
        vector similarity search.

        Returns np.ndarray of shape (64,) float32, or None if model not trained.
        """
        if not self.is_trained or self._model is None:
            return None
        if df is None:
            return None

        try:
            import torch
            from services.feature_engine import FeatureEngine
            fe = FeatureEngine()
            seq = fe.get_sequence(df, length=SEQUENCE_LENGTH, df_htf=df_htf, symbol=symbol)
            x = torch.tensor(seq[np.newaxis, ...], dtype=torch.float32).to(DEVICE)

            # Unwrap DataParallel to access .encode() directly
            m = self._model.module if isinstance(self._model, torch.nn.DataParallel) else self._model
            m.eval()
            with torch.no_grad():
                emb = m.encode(x)  # (1, 64)
            return emb.cpu().numpy().astype(np.float32).flatten()
        except Exception as exc:
            logger.warning("GRULSTMPredictor.get_embedding failed: %s", exc)
            return None

    def get_embedding_batch(
        self,
        sequences: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Compute 64-dim embeddings for a pre-built batch of sequences.

        sequences : (N, SEQUENCE_LENGTH, N_FEATURES) float32 array
        Returns   : (N, 64) float32 array, or None if model not trained.

        Used by diagnostics and optional downstream embedding consumers.
        """
        if not self.is_trained or self._model is None:
            return None

        try:
            import torch
            from torch.utils.data import DataLoader, TensorDataset

            x = torch.tensor(sequences, dtype=torch.float32)
            ds = TensorDataset(x)
            dl = DataLoader(ds, batch_size=512, shuffle=False,
                            pin_memory=(DEVICE.type == "cuda"),
                            num_workers=0)

            m = self._model.module if isinstance(self._model, torch.nn.DataParallel) else self._model
            m.eval()
            embeddings = []
            with torch.no_grad():
                for (batch,) in dl:
                    batch = batch.to(DEVICE)
                    emb = m.encode(batch)  # (B, 64)
                    embeddings.append(emb.cpu().numpy())
            return np.concatenate(embeddings, axis=0).astype(np.float32)
        except Exception as exc:
            logger.warning("GRULSTMPredictor.get_embedding_batch failed: %s", exc)
            return None

    def fit_temperature(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """
        Fit a scalar temperature T that minimises binary cross-entropy (NLL) on
        the calibration set, then save it as `temperature.pt` alongside `model.pt`.

        Also fits isotonic regression as a post-hoc probability calibrator and
        saves it as `isotonic.pkl` — used in predict() when present. Temperature
        scaling alone produced ECE=0.56 on the training run; isotonic regression
        is a non-parametric monotone transform and typically achieves ECE<0.10.

        logits : (N,) float32 — raw direction logits (before sigmoid)
        labels : (N,) float32 — binary labels (0.0 or 1.0); NaN rows are ignored

        Returns the fitted temperature T.
        """
        import torch
        from scipy.optimize import minimize_scalar

        logits = np.asarray(logits, dtype=np.float64).ravel()
        labels = np.asarray(labels, dtype=np.float64).ravel()

        # Drop NaN labels (dead-zone bars)
        mask = ~np.isnan(labels)
        logits = logits[mask]
        labels = labels[mask]

        if len(logits) == 0:
            logger.warning("fit_temperature: no valid samples after NaN mask — keeping T=1.0")
            return self._temperature

        def _nll(T: float) -> float:
            """Binary cross-entropy under temperature T (scalar, minimised)."""
            T = max(T, 1e-6)
            scaled = logits / T
            loss = np.logaddexp(0.0, scaled) - labels * scaled
            return float(np.mean(loss))

        result = minimize_scalar(_nll, bounds=(0.05, 10.0), method="bounded")
        T_opt = float(result.x)

        self._temperature = T_opt
        logger.info("fit_temperature: T=%.4f  (NLL before=%.4f, after=%.4f)",
                    T_opt, _nll(1.0), _nll(T_opt))

        # Save temperature sidecar
        _temp_file = os.path.join(WEIGHT_DIR, "temperature.pt")
        try:
            torch.save(torch.tensor(T_opt, dtype=torch.float32), _temp_file)
            logger.info("fit_temperature: saved %s", _temp_file)
        except Exception as exc:
            logger.error("fit_temperature: could not save %s: %s", _temp_file, exc)

        # Fit isotonic regression on temperature-scaled probabilities.
        # This is a non-parametric monotone calibrator — it learns the exact
        # mapping from model probability to empirical frequency on the cal set.
        # Saved as isotonic.pkl alongside model.pt and loaded in predict().
        try:
            from sklearn.isotonic import IsotonicRegression
            import pickle as _pkl
            probs_scaled = 1.0 / (1.0 + np.exp(-logits / T_opt))
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(probs_scaled, labels)
            _iso_file = os.path.join(WEIGHT_DIR, "isotonic.pkl")
            with open(_iso_file, "wb") as fh:
                _pkl.dump(iso, fh, protocol=4)
            self._isotonic = iso
            logger.info("fit_temperature: isotonic calibrator fitted and saved to %s", _iso_file)
        except ImportError:
            logger.warning("fit_temperature: sklearn not available — isotonic calibration skipped")
        except Exception as exc:
            logger.error("fit_temperature: isotonic calibration failed: %s", exc)

        return T_opt

    def train(
        self,
        df: pd.DataFrame,
        labels: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 512,
        validation_split: float = 0.2,
        symbol: Optional[str] = None,
        df_htf: Optional[dict] = None,
        grad_accum_steps: int = 4,
    ) -> dict:
        """
        Trains with strict temporal split (last 20% = validation, no shuffle).
        labels columns: direction_up, move_magnitude, volatility_target, entry_depth
        df_htf: full {tf: DataFrame} dict for MTF cross-TF sequence features.
                Must be passed so training features match inference features exactly.
        grad_accum_steps: accumulate gradients over N micro-batches before stepping
                          (effective batch = batch_size × grad_accum_steps).
        Returns history dict with train_loss and val_loss lists.
        """
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader

            from services.feature_engine import FeatureEngine, SEQUENCE_FEATURES
            fe = FeatureEngine()

            if self._model is None:
                self.build_model()
            if self._model is None:
                return {"error": "PyTorch not available"}

            feat_df = fe._build_sequence_df(df, df_htf, symbol=symbol)
            feat_arr = feat_df[SEQUENCE_FEATURES].to_numpy(dtype=np.float32, copy=False)
            feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
            del feat_df

            n_seq = len(feat_arr) - SEQUENCE_LENGTH
            if n_seq <= 0:
                return {"error": "Not enough rows for sequence training"}

            label_cols = labels.iloc[SEQUENCE_LENGTH:]
            y_dir     = label_cols.get("direction_up",      pd.Series(np.nan, index=label_cols.index)).values.astype(np.float32)
            y_r_long  = label_cols.get("realized_r_long",   pd.Series(np.nan, index=label_cols.index)).values.astype(np.float32)
            y_r_short = label_cols.get("realized_r_short",  pd.Series(np.nan, index=label_cols.index)).values.astype(np.float32)
            y_vol     = label_cols.get("volatility_target", pd.Series(0.0,    index=label_cols.index)).values.astype(np.float32)
            targets   = np.column_stack([y_dir, y_r_long, y_r_short, y_vol]).astype(np.float32)

            if len(targets) != n_seq:
                n_seq = min(n_seq, len(targets))
                targets = targets[:n_seq]

            split = int(n_seq * (1 - validation_split))
            if split <= 0 or split >= n_seq:
                return {"error": "Not enough sequences after validation split"}

            pos_rate = float(np.nanmean(targets[:, 0]))
            avg_r_long = float(np.nanmean(targets[:, 1]))
            avg_r_short = float(np.nanmean(targets[:, 2]))
            avg_vol = float(np.nanmean(targets[:, 3]))
            logger.info(
                "GRU targets samples=%d long_side_rate=%.4f avg_r_long=%.4f avg_r_short=%.4f avg_vol=%.6f",
                n_seq, pos_rate, avg_r_long, avg_r_short, avg_vol,
            )

            group_id_value = int(SYMBOL_GROUPS.get(symbol, N_GROUPS))

            class _SequenceDataset(torch.utils.data.Dataset):
                def __init__(self, features: np.ndarray, labels_arr: np.ndarray, start: int, end: int, group_id: int):
                    self._features = features
                    self._labels = labels_arr
                    self._start = start
                    self._end = end
                    self._group_id = int(group_id)

                def __len__(self) -> int:
                    return max(0, self._end - self._start)

                def __getitem__(self, idx: int):
                    i = self._start + idx
                    x = self._features[i:i + SEQUENCE_LENGTH]
                    y_item = self._labels[i]
                    return torch.from_numpy(x), torch.from_numpy(y_item), torch.tensor(self._group_id, dtype=torch.long)

            train_ds = _SequenceDataset(feat_arr, targets, 0, split, group_id_value)
            val_ds = _SequenceDataset(feat_arr, targets, split, n_seq, group_id_value)
            _pin = DEVICE.type == "cuda"
            _workers = 4 if DEVICE.type == "cuda" else 0
            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=_workers, pin_memory=_pin, persistent_workers=(_workers > 0))
            val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=_workers, pin_memory=_pin, persistent_workers=(_workers > 0))

            optimiser = torch.optim.Adam(self._model.parameters(), lr=1e-3, weight_decay=1e-4)
            # pos_weight from non-NaN direction labels only (dead-zone bars are NaN)
            valid_dirs = targets[:split, 0]
            valid_dirs = valid_dirs[~np.isnan(valid_dirs)]
            n_pos_v = float(np.sum(valid_dirs > 0.5))
            n_neg_v = float(len(valid_dirs) - n_pos_v)
            pos_weight_value = n_neg_v / max(n_pos_v, 1.0)
            pos_weight = torch.tensor([max(pos_weight_value, 1e-3)], dtype=torch.float32).to(DEVICE)
            criterion_dir = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            criterion_mag = nn.SmoothL1Loss()

            # Mixed precision — enabled on CUDA (T4 supports FP16 natively)
            use_amp = DEVICE.type == "cuda"
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

            best_val = float("inf")
            best_dir_acc = 0.0
            best_val_r_mae = float("inf")
            best_positive_r_acc = 0.0
            patience, no_improve = 5, 0
            history = {"train_loss": [], "val_loss": [], "val_direction_accuracy": []}

            def _compute_loss(dir_logits, r_long_pred, r_short_pred, log_variance_pred, yb):
                """
                Multi-head loss with dead-zone masking on direction.
                yb[:, 0] = direction_up (NaN in dead zone → excluded from BCE).
                yb[:, 1] = realized_r_long  (NaN where not labeled long).
                yb[:, 2] = realized_r_short (NaN where not labeled short).
                yb[:, 3] = volatility_target.
                λ: dir=2.0, r_long=0.5, r_short=0.5, vol=0.3.
                """
                y_dir     = yb[:, 0]
                y_r_long  = yb[:, 1]
                y_r_short = yb[:, 2]
                y_vol     = yb[:, 3]

                # Direction: mask NaN dead-zone bars (smoothed labels are 0.05/0.95)
                dir_mask = ~torch.isnan(y_dir)
                if dir_mask.sum() > 0:
                    loss_dir = criterion_dir(dir_logits[dir_mask], y_dir[dir_mask])
                else:
                    loss_dir = torch.tensor(0.0, device=dir_logits.device)

                # Side-conditioned R: each head learns that side's realized R on
                # every valid bar, including losing/unchosen sides. This makes the
                # inference gate answer "what happens if I buy/sell here?"
                long_mask = ~torch.isnan(y_r_long)
                if long_mask.sum() > 0:
                    loss_r_long = criterion_mag(r_long_pred[long_mask], y_r_long[long_mask])
                else:
                    loss_r_long = torch.tensor(0.0, device=r_long_pred.device)

                short_mask = ~torch.isnan(y_r_short)
                if short_mask.sum() > 0:
                    loss_r_short = criterion_mag(r_short_pred[short_mask], y_r_short[short_mask])
                else:
                    loss_r_short = torch.tensor(0.0, device=r_short_pred.device)

                # Volatility: Gaussian NLL (heteroscedastic)
                vol_mask = ~torch.isnan(y_vol)
                if vol_mask.sum() > 0:
                    pv = torch.clamp(torch.nn.functional.softplus(log_variance_pred[vol_mask]) + 1e-6,
                                     min=1e-4, max=1.0)
                    se = torch.square(y_vol[vol_mask] - torch.sqrt(pv))
                    loss_vol = torch.clamp(torch.mean((se / pv) + torch.log(pv)), min=0.0)
                else:
                    loss_vol = torch.tensor(0.0, device=log_variance_pred.device)

                return 2.0 * loss_dir + 0.5 * loss_r_long + 0.5 * loss_r_short + 0.3 * loss_vol

            for epoch in range(epochs):
                self._model.train()
                train_loss = 0.0
                optimiser.zero_grad()
                for step, (xb, yb, gb) in enumerate(train_dl):
                    xb, yb, gb = xb.to(DEVICE), yb.to(DEVICE), gb.to(DEVICE)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        dir_logits, r_long_pred, r_short_pred, log_variance_pred = self._model(xb, gb)
                    dir_logits = dir_logits.float()
                    r_long_pred = r_long_pred.float()
                    r_short_pred = r_short_pred.float()
                    log_variance_pred = log_variance_pred.float()
                    loss = _compute_loss(dir_logits, r_long_pred, r_short_pred, log_variance_pred, yb) / grad_accum_steps
                    scaler.scale(loss).backward()
                    if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_dl):
                        scaler.step(optimiser)
                        scaler.update()
                        optimiser.zero_grad()
                    train_loss += loss.item() * grad_accum_steps * len(xb)
                train_loss /= max(1, len(train_ds))

                self._model.eval()
                val_loss = 0.0
                val_dir_correct = 0
                val_dir_total = 0
                val_r_abs = 0.0
                val_r_total = 0
                val_side_r_sign_correct = 0
                val_side_r_sign_total = 0
                with torch.no_grad():
                    for xb, yb, gb in val_dl:
                        xb, yb, gb = xb.to(DEVICE), yb.to(DEVICE), gb.to(DEVICE)
                        with torch.amp.autocast("cuda", enabled=use_amp):
                            dir_logits, r_long_pred, r_short_pred, log_variance_pred = self._model(xb, gb)
                        dir_logits = dir_logits.float()
                        r_long_pred = r_long_pred.float()
                        r_short_pred = r_short_pred.float()
                        log_variance_pred = log_variance_pred.float()
                        batch_loss = _compute_loss(dir_logits, r_long_pred, r_short_pred, log_variance_pred, yb)
                        val_loss += batch_loss.item() * len(xb)
                        dir_mask = ~torch.isnan(yb[:, 0])
                        if dir_mask.sum() > 0:
                            probs = torch.sigmoid(dir_logits[dir_mask])
                            pred_up = probs >= 0.5
                            true_up = yb[:, 0][dir_mask] > 0.5
                            val_dir_correct += int((pred_up == true_up).sum().item())
                            val_dir_total += int(dir_mask.sum().item())
                        # Side-conditioned R MAE/sign accuracy across both entry sides.
                        long_mask = ~torch.isnan(yb[:, 1])
                        short_mask = ~torch.isnan(yb[:, 2])
                        if long_mask.sum() > 0:
                            val_r_abs += float(torch.abs(r_long_pred[long_mask] - yb[:, 1][long_mask]).sum().item())
                            val_r_total += int(long_mask.sum().item())
                            val_side_r_sign_correct += int(((r_long_pred[long_mask] > 0) == (yb[:, 1][long_mask] > 0)).sum().item())
                            val_side_r_sign_total += int(long_mask.sum().item())
                        if short_mask.sum() > 0:
                            val_r_abs += float(torch.abs(r_short_pred[short_mask] - yb[:, 2][short_mask]).sum().item())
                            val_r_total += int(short_mask.sum().item())
                            val_side_r_sign_correct += int(((r_short_pred[short_mask] > 0) == (yb[:, 2][short_mask] > 0)).sum().item())
                            val_side_r_sign_total += int(short_mask.sum().item())
                val_loss /= max(1, len(val_ds))
                val_dir_acc = float(val_dir_correct / max(val_dir_total, 1))
                val_r_mae = float(val_r_abs / max(val_r_total, 1))
                val_positive_r_acc = float(val_side_r_sign_correct / max(val_side_r_sign_total, 1))

                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["val_direction_accuracy"].append(val_dir_acc)
                history.setdefault("val_r_mae", []).append(val_r_mae)
                history.setdefault("val_positive_r_accuracy", []).append(val_positive_r_acc)
                logger.info(
                    "GRU epoch %d/%d — train=%.4f val=%.4f r_mae=%.3f pos_r_acc=%.3f side_acc=%.3f r_n=%d",
                    epoch + 1, epochs, train_loss, val_loss, val_r_mae,
                    val_positive_r_acc, val_dir_acc, val_r_total,
                )

                if val_loss < best_val:
                    best_val = val_loss
                    best_dir_acc = val_dir_acc
                    best_val_r_mae = val_r_mae
                    best_positive_r_acc = val_positive_r_acc
                    no_improve = 0
                    self.save(WEIGHT_DIR)
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        logger.info("GRU early stop at epoch %d", epoch + 1)
                        break

            max_r_mae = float(os.getenv("GRU_MAX_VAL_R_MAE", "0.75"))
            if best_val_r_mae > max_r_mae:
                warning = (
                    f"GRU validation R-MAE above floor: "
                    f"best_val_r_mae={best_val_r_mae:.3f} max={max_r_mae:.3f}"
                )
                logger.warning("%s. Keeping saved best weights so the pipeline can progress.", warning)
                history.setdefault("warnings", []).append(warning)
                history["status"] = "complete_with_warnings"
            history["best_val_direction_accuracy"] = best_dir_acc
            history["best_val_r_mae"] = best_val_r_mae
            history["best_val_positive_r_accuracy"] = best_positive_r_acc
            return history
        except Exception as exc:
            logger.error("GRULSTMPredictor.train failed: %s", exc)
            return {"error": str(exc)}

    def train_multi(
        self,
        segments: list,
        epochs: int = 50,
        batch_size: int = 512,
        validation_split: float = 0.2,
        grad_accum_steps: int = 4,
        max_sequences_per_tf: int = 600_000,
    ) -> dict:
        """
        Train on multiple 15M execution segments in one combined pass.

        Sequences are boundary-safe: __getitem__ never crosses symbol boundaries.
        shuffle=True on the DataLoader mixes all symbols each epoch.

        segments: list of dicts with keys: df, labels, df_htf, symbol, timeframe
        Returns combined history dict.
        """
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, Sampler
            from services.feature_engine import FeatureEngine, SEQUENCE_FEATURES
            fe = FeatureEngine()

            if self._model is None:
                self.build_model()
            if self._model is None:
                return {"error": "PyTorch not available"}

            # Group by TF for logging, but train all TFs together in one pass.
            # Training sequentially per TF causes the last TF (4H, smallest dataset)
            # to overwrite weights from 5M/15M. Interleaving all TFs in one dataset
            # lets the model generalise across timeframes simultaneously.
            from collections import defaultdict
            tf_groups: dict = defaultdict(list)
            for seg in segments:
                tf_groups[seg["timeframe"]].append(seg)

            combined_history: dict = {"train_loss": [], "val_loss": [], "groups_trained": 0}

            # Use a single group containing ALL segments across all TFs
            all_tfs = list(tf_groups.keys())
            all_segs_flat = [seg for segs in tf_groups.values() for seg in segs]
            logger.info("train_multi: training ALL %d segments across TFs %s in one combined pass",
                        len(all_segs_flat), all_tfs)
            # Iterate once — the loop body processes this single combined group
            tf_items = [("ALL", all_segs_flat)]

            for tf, group in tf_items:
                logger.info("train_multi: building combined dataset for TF=%s (%d segments)", tf, len(group))

                # Build per-segment (feat_arr, tgt_arr, group_id) serially.
                seg_feats: list  = []
                seg_tgts: list   = []
                seg_groups: list = []   # symbol group ID per segment
                segment_errors: list[str] = []

                for seg in group:
                    try:
                        feat_df = fe._build_sequence_df(
                            seg["df"], seg.get("df_htf"), symbol=seg.get("symbol")
                        )
                        feat_arr = feat_df[SEQUENCE_FEATURES].to_numpy(dtype=np.float32, copy=False)
                        feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)
                        del feat_df
                        lbl   = seg["labels"]
                        n_seq = len(feat_arr) - SEQUENCE_LENGTH
                        if n_seq <= 0:
                            segment_errors.append(f"{seg.get('symbol')}/{seg.get('timeframe')}: no complete sequences")
                            continue
                        y_dir     = lbl.get("direction_up",      pd.Series(np.nan, index=lbl.index)).values.astype(np.float32)[SEQUENCE_LENGTH:]
                        y_r_long  = lbl.get("realized_r_long",   pd.Series(np.nan, index=lbl.index)).values.astype(np.float32)[SEQUENCE_LENGTH:]
                        y_r_short = lbl.get("realized_r_short",  pd.Series(np.nan, index=lbl.index)).values.astype(np.float32)[SEQUENCE_LENGTH:]
                        y_vol     = lbl.get("volatility_target", pd.Series(0.0,    index=lbl.index)).values.astype(np.float32)[SEQUENCE_LENGTH:]
                        tgt       = np.column_stack([y_dir, y_r_long, y_r_short, y_vol]).astype(np.float32)
                        n_seq = min(n_seq, len(tgt))
                        seg_feats.append(feat_arr[:n_seq + SEQUENCE_LENGTH].copy())
                        seg_tgts.append(tgt[:n_seq].copy())
                        seg_groups.append(SYMBOL_GROUPS.get(seg.get("symbol"), N_GROUPS))
                        del feat_arr, tgt
                    except Exception as exc:
                        segment_errors.append(f"{seg.get('symbol')}/{seg.get('timeframe')}: {exc}")

                if segment_errors:
                    raise RuntimeError(
                        "train_multi segment build failed; refusing partial GRU training: "
                        + "; ".join(segment_errors[:8])
                    )

                if not seg_feats:
                    raise RuntimeError(f"train_multi: no valid segments for TF={tf}")

                total_seq = sum(len(t) for t in seg_tgts)
                logger.info("train_multi TF=%s: %d sequences across %d segments",
                            tf, total_seq, len(seg_feats))

                # Enforce cap — trim largest segments proportionally
                if total_seq > max_sequences_per_tf:
                    ratio = max_sequences_per_tf / total_seq
                    seg_feats_new, seg_tgts_new = [], []
                    for sf, st in zip(seg_feats, seg_tgts):
                        keep = max(SEQUENCE_LENGTH + 1, int(len(st) * ratio))
                        seg_feats_new.append(sf[:keep + SEQUENCE_LENGTH])
                        seg_tgts_new.append(st[:keep])
                    seg_feats, seg_tgts = seg_feats_new, seg_tgts_new
                    total_seq = sum(len(t) for t in seg_tgts)

                # Count train/val sizes first (no allocation yet)
                n_train, n_val = 0, 0
                n_feat = seg_feats[0].shape[1] if seg_feats else 0
                for sf, st in zip(seg_feats, seg_tgts):
                    n = len(st)
                    sp = int(n * (1 - validation_split))
                    if sp <= 0 or sp >= n:
                        continue
                    n_train += sp
                    n_val   += n - sp

                if n_train == 0:
                    del seg_feats, seg_tgts
                    continue

                # RAM budget check before allocation.
                # Each row = SEQUENCE_LENGTH × n_feat × 4 bytes.
                # Pre-allocate only if estimated peak usage < 20 GB (leaves ~10 GB headroom).
                import gc as _gc
                import numpy.lib.stride_tricks as _st
                row_bytes   = SEQUENCE_LENGTH * n_feat * 4
                numpy_bytes = (n_train + n_val) * row_bytes
                # After pinning, torch makes another copy → peak ≈ 2× numpy allocation
                peak_est_mb = numpy_bytes * 2 / 1e6
                logger.info("train_multi TF=%s: estimated peak RAM = %.0f MB "
                            "(train=%d val=%d n_feat=%d seq_len=%d)",
                            tf, peak_est_mb, n_train, n_val, n_feat, SEQUENCE_LENGTH)
                if peak_est_mb > 20_000:
                    # Trim to fit: recompute n_train/n_val for 20 GB budget
                    max_rows  = int(20_000 * 1e6 / (row_bytes * 2))
                    n_train   = min(n_train, int(max_rows * 0.8))
                    n_val     = min(n_val,   int(max_rows * 0.2))
                    logger.warning(
                        "train_multi TF=%s: trimming to fit RAM budget — "
                        "new train=%d val=%d (%.0f MB est)",
                        tf, n_train, n_val, (n_train + n_val) * row_bytes * 2 / 1e6
                    )
                _gc.collect()

                # Pre-allocate output arrays — one-shot, no intermediate copies
                X_train = np.empty((n_train, SEQUENCE_LENGTH, n_feat), dtype=np.float32)
                Y_train = np.empty((n_train, 4), dtype=np.float32)
                G_train = np.empty((n_train,),   dtype=np.int64)   # symbol group IDs
                X_val   = np.empty((n_val,   SEQUENCE_LENGTH, n_feat), dtype=np.float32)
                Y_val   = np.empty((n_val,   4), dtype=np.float32)
                G_val   = np.empty((n_val,),     dtype=np.int64)

                tr_off, va_off = 0, 0
                for sf, st, grp in zip(seg_feats, seg_tgts, seg_groups):
                    if tr_off >= n_train and va_off >= n_val:
                        break   # budget exhausted
                    n = len(st)
                    sp = int(n * (1 - validation_split))
                    if sp <= 0 or sp >= n:
                        continue
                    for start_idx, end_idx, X_out, Y_out, G_out, is_train in [
                        (0,  sp, X_train, Y_train, G_train, True),
                        (sp, n,  X_val,   Y_val,   G_val,   False),
                    ]:
                        seg_n = end_idx - start_idx
                        if seg_n <= 0:
                            continue
                        off      = tr_off if is_train else va_off
                        cap      = n_train if is_train else n_val
                        seg_n    = min(seg_n, cap - off)   # don't overflow pre-alloc
                        if seg_n <= 0:
                            continue
                        raw = _st.sliding_window_view(
                            sf[start_idx: start_idx + seg_n + SEQUENCE_LENGTH - 1],
                            (SEQUENCE_LENGTH, n_feat)
                        ).reshape(seg_n, SEQUENCE_LENGTH, n_feat)
                        X_out[off: off + seg_n] = raw
                        Y_out[off: off + seg_n] = st[start_idx: start_idx + seg_n]
                        G_out[off: off + seg_n] = grp   # same group for every row in segment
                        if is_train:
                            tr_off += seg_n
                        else:
                            va_off += seg_n
                        del raw

                # Trim pre-alloc arrays to actually filled rows (budget may have been generous)
                X_train = X_train[:tr_off]
                Y_train = Y_train[:tr_off]
                G_train = G_train[:tr_off]
                X_val   = X_val[:va_off]
                Y_val   = Y_val[:va_off]
                G_val   = G_val[:va_off]
                n_train, n_val = tr_off, va_off

                del seg_feats, seg_tgts, seg_groups
                _gc.collect()

                logger.info("train_multi TF=%s: train=%d val=%d (%.0f MB tensors)",
                            tf, n_train, n_val,
                            (X_train.nbytes + Y_train.nbytes + X_val.nbytes + Y_val.nbytes) / 1e6)

                # Pin to page-locked memory for fast H→D transfers
                X_train_t = torch.from_numpy(X_train).pin_memory()
                Y_train_t = torch.from_numpy(Y_train).pin_memory()
                G_train_t = torch.from_numpy(G_train).pin_memory()
                X_val_t   = torch.from_numpy(X_val).pin_memory()
                Y_val_t   = torch.from_numpy(Y_val).pin_memory()
                G_val_t   = torch.from_numpy(G_val).pin_memory()
                del X_train, Y_train, G_train, X_val, Y_val, G_val
                _gc.collect()

                # Pre-shuffle index array (re-shuffled each epoch) — no DataLoader needed
                train_idx = np.arange(n_train, dtype=np.int64)
                steps_per_epoch = max(1, (n_train + batch_size - 1) // batch_size)

                # Detect warm-start vs cold-start — determines LR and scheduler strategy.
                # Warm-start (loaded from prior weights): OneCycleLR with max_lr=3e-4 is too
                # aggressive — it ramps the LR during its warmup phase, overshooting the local
                # minimum immediately and causing val_loss to worsen from epoch 1 onward. The
                # model early-stops at epoch 5-6 before the LR ever decays enough to recover.
                # Fix: use CosineAnnealingLR with a low initial LR (3e-5) for fine-tuning.
                # Cold-start: keep OneCycleLR with max_lr=3e-4 — it's designed for training
                # from random init where aggressive LR warmup is beneficial.
                _warm_start = self._loaded  # BaseModel sets _loaded=True after load()
                if _warm_start:
                    # Fine-tuning: conservative LR, cosine decay, generous patience.
                    # 3e-5 is 10× lower than the cold-start peak — stays near the local minimum
                    # while allowing gradual adaptation to new data distribution.
                    _train_lr = 3e-5
                    _patience = 12  # longer patience: cosine decay needs more epochs to converge
                    _min_epochs_before_stop = 8
                    optimiser = torch.optim.AdamW(
                        self._model.parameters(), lr=_train_lr, weight_decay=1e-3
                    )
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimiser, T_max=epochs, eta_min=1e-6,
                    )
                    logger.info(
                        "train_multi TF=%s: warm-start detected — "
                        "using CosineAnnealingLR (lr=%.0e, patience=%d)",
                        tf, _train_lr, _patience,
                    )
                else:
                    # Cold-start: OneCycleLR with standard LR — good for random init.
                    # Do not allow early stopping during the OneCycle warmup/peak LR
                    # phase. The first pass can look flat around BCE~=0.69 until the
                    # LR starts decaying; stopping there leaves p_bull clustered near
                    # 0.5, which makes validation/test backtests reject every bar at
                    # the direction gate.
                    _train_lr = 3e-4
                    _patience = 25   # was 18 — model needs more epochs to converge
                    _min_epochs_before_stop = max(22, int(epochs * 0.45))
                    optimiser = torch.optim.AdamW(
                        self._model.parameters(), lr=_train_lr, weight_decay=1e-3
                    )
                    scheduler = torch.optim.lr_scheduler.OneCycleLR(
                        optimiser, max_lr=_train_lr, epochs=epochs,
                        steps_per_epoch=steps_per_epoch, pct_start=0.2,
                    )
                    logger.info(
                        "train_multi TF=%s: cold-start — "
                        "using OneCycleLR (max_lr=%.0e, patience=%d, min_epochs=%d)",
                        tf, _train_lr, _patience, _min_epochs_before_stop,
                    )

                # pos_weight from training direction labels
                valid_dir = Y_train_t[:, 0].numpy()
                valid_dir = valid_dir[~np.isnan(valid_dir)]
                n_pos_tm = float(np.sum(valid_dir > 0.5))
                n_neg_tm = float(len(valid_dir) - n_pos_tm)
                pos_w_val = n_neg_tm / max(n_pos_tm, 1.0)
                pos_weight    = torch.tensor([float(np.clip(pos_w_val, 0.5, 2.0))], dtype=torch.float32).to(DEVICE)
                criterion_dir = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
                criterion_mag = nn.SmoothL1Loss()
                use_amp = DEVICE.type == "cuda"
                scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

                def _loss_tm(dl, rl, rs, lv, yb):
                    """
                    yb[:, 0] = direction_up, yb[:, 1] = realized_r_long,
                    yb[:, 2] = realized_r_short, yb[:, 3] = volatility_target.
                    λ: dir=2.0 (was 1.0), r_long=0.5, r_short=0.5, vol=0.3.
                    """
                    y_dir, y_r_long, y_r_short, y_vol = yb[:, 0], yb[:, 1], yb[:, 2], yb[:, 3]
                    dir_mask = ~torch.isnan(y_dir)
                    loss_dir = criterion_dir(dl[dir_mask], y_dir[dir_mask]).mean() if dir_mask.sum() > 0 else dl.sum() * 0

                    # Side-conditioned: each head learns that side's realized R
                    # across all valid bars, not only bars where the side won.
                    long_mask = ~torch.isnan(y_r_long)
                    loss_r_long = criterion_mag(rl[long_mask], y_r_long[long_mask]) if long_mask.sum() > 0 else rl.sum() * 0
                    short_mask = ~torch.isnan(y_r_short)
                    loss_r_short = criterion_mag(rs[short_mask], y_r_short[short_mask]) if short_mask.sum() > 0 else rs.sum() * 0

                    vol_mask = ~torch.isnan(y_vol)
                    if vol_mask.sum() > 0:
                        # Clamp variance in [1e-4, 1.0] — prevents NLL going negative.
                        pv = torch.clamp(torch.nn.functional.softplus(lv[vol_mask]) + 1e-6,
                                         min=1e-4, max=1.0)
                        loss_vol = torch.clamp(
                            torch.mean((torch.square(y_vol[vol_mask] - torch.sqrt(pv)) / pv) + torch.log(pv)),
                            min=0.0)
                    else:
                        loss_vol = lv.sum() * 0
                    return 2.0 * loss_dir + 0.5 * loss_r_long + 0.5 * loss_r_short + 0.3 * loss_vol

                best_val = float("inf")
                best_dir_acc = 0.0
                best_val_r_mae = float("inf")
                best_positive_r_acc = 0.0
                patience, no_improve = _patience, 0

                # Move val set to GPU once — it's small enough (~600K × 30 × F ≈ 600MB)
                X_val_gpu = X_val_t.to(DEVICE, non_blocking=True)
                Y_val_gpu = Y_val_t.to(DEVICE, non_blocking=True)
                G_val_gpu = G_val_t.to(DEVICE, non_blocking=True)

                for epoch in range(epochs):
                    self._model.train()
                    train_loss = 0.0
                    # Re-shuffle index array each epoch
                    np.random.shuffle(train_idx)
                    optimiser.zero_grad()
                    for step in range(steps_per_epoch):
                        b_start = step * batch_size
                        b_end   = min(b_start + batch_size, n_train)
                        idx_b   = train_idx[b_start:b_end]
                        # Non-blocking H→D transfer from pinned memory
                        xb = X_train_t[idx_b].to(DEVICE, non_blocking=True)
                        yb = Y_train_t[idx_b].to(DEVICE, non_blocking=True)
                        gb = G_train_t[idx_b].to(DEVICE, non_blocking=True)
                        with torch.amp.autocast("cuda", enabled=use_amp):
                            dl, rl, rs, lv = self._model(xb, gb)
                        dl, rl, rs, lv = dl.float(), rl.float(), rs.float(), lv.float()
                        loss = _loss_tm(dl, rl, rs, lv, yb) / grad_accum_steps
                        scaler.scale(loss).backward()
                        if (step + 1) % grad_accum_steps == 0 or (step + 1) == steps_per_epoch:
                            scaler.unscale_(optimiser)
                            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                            scaler.step(optimiser)
                            scaler.update()
                            optimiser.zero_grad()
                            # OneCycleLR must step every optimizer step (not per-epoch).
                            # CosineAnnealingLR steps per epoch — handled after the inner loop.
                            if not _warm_start:
                                scheduler.step()
                        train_loss += loss.item() * grad_accum_steps * (b_end - b_start)
                    train_loss /= max(1, n_train)
                    # CosineAnnealingLR: step once per epoch (after all batches done)
                    if _warm_start:
                        scheduler.step()

                    self._model.eval()
                    val_loss = 0.0
                    val_dir_correct = 0
                    val_dir_total = 0
                    val_r_abs = 0.0
                    val_r_total = 0
                    val_side_r_sign_correct = 0
                    val_side_r_sign_total = 0
                    with torch.no_grad():
                        val_bs = batch_size * 2   # larger batches during eval (no backward pass)
                        for v_start in range(0, n_val, val_bs):
                            xb = X_val_gpu[v_start: v_start + val_bs]
                            yb = Y_val_gpu[v_start: v_start + val_bs]
                            gb = G_val_gpu[v_start: v_start + val_bs]
                            with torch.amp.autocast("cuda", enabled=use_amp):
                                dl, rl, rs, lv = self._model(xb, gb)
                            dl, rl, rs, lv = dl.float(), rl.float(), rs.float(), lv.float()
                            val_loss += _loss_tm(dl, rl, rs, lv, yb).item() * len(xb)
                            dir_mask = ~torch.isnan(yb[:, 0])
                            if dir_mask.sum() > 0:
                                probs = torch.sigmoid(dl[dir_mask])
                                pred_up = probs >= 0.5
                                true_up = yb[:, 0][dir_mask] > 0.5
                                val_dir_correct += int((pred_up == true_up).sum().item())
                                val_dir_total += int(dir_mask.sum().item())
                            # Side-conditioned R MAE/sign accuracy across both entry sides.
                            long_mask = ~torch.isnan(yb[:, 1])
                            short_mask = ~torch.isnan(yb[:, 2])
                            if long_mask.sum() > 0:
                                val_r_abs += float(torch.abs(rl[long_mask] - yb[:, 1][long_mask]).sum().item())
                                val_r_total += int(long_mask.sum().item())
                                val_side_r_sign_correct += int(((rl[long_mask] > 0) == (yb[:, 1][long_mask] > 0)).sum().item())
                                val_side_r_sign_total += int(long_mask.sum().item())
                            if short_mask.sum() > 0:
                                val_r_abs += float(torch.abs(rs[short_mask] - yb[:, 2][short_mask]).sum().item())
                                val_r_total += int(short_mask.sum().item())
                                val_side_r_sign_correct += int(((rs[short_mask] > 0) == (yb[:, 2][short_mask] > 0)).sum().item())
                                val_side_r_sign_total += int(short_mask.sum().item())
                    val_loss /= max(1, n_val)
                    val_dir_acc = float(val_dir_correct / max(val_dir_total, 1))
                    val_r_mae = float(val_r_abs / max(val_r_total, 1))
                    val_positive_r_acc = float(val_side_r_sign_correct / max(val_side_r_sign_total, 1))

                    combined_history["train_loss"].append(train_loss)
                    combined_history["val_loss"].append(val_loss)
                    combined_history.setdefault("val_direction_accuracy", []).append(val_dir_acc)
                    combined_history.setdefault("val_r_mae", []).append(val_r_mae)
                    combined_history.setdefault("val_positive_r_accuracy", []).append(val_positive_r_acc)
                    logger.info(
                        "train_multi TF=%s epoch %d/%d train=%.4f val=%.4f r_mae=%.3f pos_r_acc=%.3f side_acc=%.3f r_n=%d",
                        tf, epoch + 1, epochs, train_loss, val_loss, val_r_mae,
                        val_positive_r_acc, val_dir_acc, val_r_total,
                    )

                    if val_loss < best_val:
                        best_val = val_loss
                        best_dir_acc = val_dir_acc
                        best_val_r_mae = val_r_mae
                        best_positive_r_acc = val_positive_r_acc
                        no_improve = 0
                        self.save(WEIGHT_DIR)
                        logger.info("train_multi TF=%s: new best val=%.4f — saved", tf, best_val)
                    else:
                        no_improve += 1
                        if no_improve >= patience and (epoch + 1) >= _min_epochs_before_stop:
                            logger.info("train_multi TF=%s early stop at epoch %d", tf, epoch + 1)
                            break

                combined_history["groups_trained"] += 1
                del X_train_t, Y_train_t, G_train_t, X_val_t, Y_val_t, G_val_t
                del X_val_gpu, Y_val_gpu, G_val_gpu, train_idx
                import gc; gc.collect()
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
                max_r_mae = float(os.getenv("GRU_MAX_VAL_R_MAE", "0.75"))
                if best_val_r_mae > max_r_mae:
                    warning = (
                        f"GRU validation R-MAE above floor for TF={tf}: "
                        f"best_val_r_mae={best_val_r_mae:.3f} max={max_r_mae:.3f}"
                    )
                    logger.warning("%s. Keeping saved best weights so the pipeline can progress.", warning)
                    combined_history.setdefault("warnings", []).append(warning)
                    combined_history["status"] = "complete_with_warnings"
                combined_history["best_val_direction_accuracy"] = best_dir_acc
                combined_history["best_val_r_mae"] = best_val_r_mae
                combined_history["best_val_positive_r_accuracy"] = best_positive_r_acc

            return combined_history
        except Exception as exc:
            logger.error("GRULSTMPredictor.train_multi failed: %s", exc)
            return {"error": str(exc)}

    def create_labels(
        self,
        df: pd.DataFrame,
        horizon_bars: int = 12,
        atr_threshold: float = 0.3,
        volatility_window: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Institutional-grade multi-head labels — predict actionable trade outcome.

        horizon_bars=12 on 15M ≈ 3 hours ahead (enough signal, not 1-bar noise).

        direction_up:
            By default, derived from forward trade geometry rather than close-to-
            close direction. For each bar we simulate long and short stop/target
            outcomes over the horizon, choose the better side, and
            leave non-actionable bars as NaN so the BCE loss ignores them.

        move_magnitude:
            Signed best realized R-multiple. This keeps the regression head
            aligned with the backtest objective instead of raw direction size.

        volatility_target:
            Rolling std of forward 1-step log-returns — model learns when NOT
            to trade (high vol → reduce size / skip entry).

        efficiency_ratio:
            |net k-step move| / sum(|bar moves|) — 0=choppy, 1=clean trend.
            Auxiliary signal to separate trending from ranging conditions.

        CRITICAL: last horizon_bars rows have NaN — excluded from training.
        """
        from indicators.market_structure import compute_atr

        atr   = compute_atr(df, 14)
        close = df["close"]

        # k-step log return — retained for diagnostics/provenance.
        future_close = close.shift(-horizon_bars)
        log_ret_k    = np.log((future_close + 1e-9) / (close + 1e-9))

        # Forward-looking volatility: std of the next horizon_bars 1-step log returns.
        # The previous implementation used rolling() on shift(-1), which still
        # included mostly past returns at each timestamp. Reverse-rolling keeps
        # this target aligned with the same future window as direction_up.
        log_ret_1  = np.log((close + 1e-9) / (close.shift(1) + 1e-9))
        vol_target = (
            log_ret_1.shift(-1)
            .iloc[::-1]
            .rolling(window=horizon_bars, min_periods=horizon_bars)
            .std()
            .iloc[::-1]
            .reindex(df.index)
            .astype(np.float32)
        )

        f_high = pd.Series(df["high"].astype(float)).shift(-1).iloc[::-1].rolling(
            horizon_bars, min_periods=horizon_bars
        ).max().iloc[::-1].reindex(df.index)
        f_low = pd.Series(df["low"].astype(float)).shift(-1).iloc[::-1].rolling(
            horizon_bars, min_periods=horizon_bars
        ).min().iloc[::-1].reindex(df.index)
        up_exc_atr = ((f_high - close) / (atr + 1e-9)).clip(lower=0.0)
        down_exc_atr = ((close - f_low) / (atr + 1e-9)).clip(lower=0.0)
        terminal_atr = (future_close - close) / (atr + 1e-9)

        label_mode = os.getenv("GRU_LABEL_MODE", "trade_r").strip().lower()
        if label_mode in {"trade_r", "r_multiple", "trade_outcome"}:
            stop_atr = max(float(os.getenv("GRU_LABEL_STOP_ATR", "1.5")), 1e-6)
            target_atr = max(float(os.getenv("GRU_LABEL_TARGET_ATR", "2.5")), 1e-6)
            target_r = target_atr / stop_atr
            min_side_edge_r = float(os.getenv("GRU_LABEL_MIN_SIDE_EDGE_R", "0.10"))

            close_arr = close.to_numpy(dtype=np.float64, copy=False)
            high_arr = df["high"].astype(float).to_numpy(dtype=np.float64, copy=False)
            low_arr = df["low"].astype(float).to_numpy(dtype=np.float64, copy=False)
            atr_arr = atr.to_numpy(dtype=np.float64, copy=False)
            n_rows = len(df)

            long_tp = close_arr + target_atr * atr_arr
            long_sl = close_arr - stop_atr * atr_arr
            short_tp = close_arr - target_atr * atr_arr
            short_sl = close_arr + stop_atr * atr_arr
            long_r = np.full(n_rows, np.nan, dtype=np.float32)
            short_r = np.full(n_rows, np.nan, dtype=np.float32)
            long_done = np.zeros(n_rows, dtype=bool)
            short_done = np.zeros(n_rows, dtype=bool)
            valid_base = (
                np.isfinite(close_arr)
                & np.isfinite(atr_arr)
                & (atr_arr > 0.0)
                & (np.arange(n_rows) < (n_rows - horizon_bars))
            )

            for step_i in range(1, horizon_bars + 1):
                future_high = np.empty_like(high_arr)
                future_low = np.empty_like(low_arr)
                future_high[:-step_i] = high_arr[step_i:]
                future_high[-step_i:] = np.nan
                future_low[:-step_i] = low_arr[step_i:]
                future_low[-step_i:] = np.nan
                valid_step = valid_base & np.isfinite(future_high) & np.isfinite(future_low)

                long_stop_hit = valid_step & (future_low <= long_sl)
                long_target_hit = valid_step & (future_high >= long_tp)
                long_resolve = (~long_done) & (long_stop_hit | long_target_hit)
                # Conservative same-bar policy: if both TP and SL print inside
                # one candle, count the stop first because tick order is unknown.
                long_r[long_resolve & long_stop_hit] = -1.0
                long_r[long_resolve & ~long_stop_hit & long_target_hit] = target_r
                long_done |= long_resolve

                short_stop_hit = valid_step & (future_high >= short_sl)
                short_target_hit = valid_step & (future_low <= short_tp)
                short_resolve = (~short_done) & (short_stop_hit | short_target_hit)
                short_r[short_resolve & short_stop_hit] = -1.0
                short_r[short_resolve & ~short_stop_hit & short_target_hit] = target_r
                short_done |= short_resolve

            future_close_arr = future_close.to_numpy(dtype=np.float64, copy=False)
            unresolved = valid_base & np.isfinite(future_close_arr)
            terminal_long_r = np.clip((future_close_arr - close_arr) / (stop_atr * atr_arr + 1e-9), -1.0, target_r)
            terminal_short_r = np.clip((close_arr - future_close_arr) / (stop_atr * atr_arr + 1e-9), -1.0, target_r)
            long_r[unresolved & ~long_done] = terminal_long_r[unresolved & ~long_done]
            short_r[unresolved & ~short_done] = terminal_short_r[unresolved & ~short_done]

            smoothing = 0.05
            side_labelable = valid_base & np.isfinite(long_r) & np.isfinite(short_r)
            best_long = (
                side_labelable
                & ((long_r - short_r) >= min_side_edge_r)
            )
            best_short = (
                side_labelable
                & ((short_r - long_r) >= min_side_edge_r)
            )
            if not bool(np.any(best_long | best_short)):
                best_long = (
                    side_labelable
                    & (long_r >= short_r)
                )
                best_short = (
                    side_labelable
                    & (short_r > long_r)
                )
            direction_up = pd.Series(np.nan, index=df.index, dtype=np.float32)
            direction_up.iloc[np.where(best_long)[0]] = 1.0 - smoothing
            direction_up.iloc[np.where(best_short)[0]] = 0.0 + smoothing

            best_r_arr = np.where(long_r >= short_r, long_r, short_r).astype(np.float32)
            # move_magnitude = R for the LABELED side (not the always-positive best side).
            # This makes the regression target honest: model learns "what R do I get
            # if I enter in the direction I'm predicting", not "what is the best possible R".
            side_r_arr = np.where(
                best_long, long_r,
                np.where(best_short, short_r, np.nan)
            ).astype(np.float32)
            move_magnitude = pd.Series(
                np.clip(side_r_arr, -1.0, target_r),
                index=df.index,
                dtype=np.float32,
            )
            move_magnitude[~side_labelable] = np.nan

            positive_r_score = pd.Series(
                np.clip(np.maximum(best_r_arr, 0.0) / max(target_r, 1e-6), 0.0, 1.0),
                index=df.index,
                dtype=np.float32,
            )
            positive_r_score[~side_labelable] = np.nan

            abs_moves = np.abs(close.diff()).rolling(horizon_bars, min_periods=horizon_bars).sum().shift(-horizon_bars)
            net_move = np.abs(close.shift(-horizon_bars) - close)
            eff_ratio = (net_move / (abs_moves + 1e-9)).clip(0, 1).astype(np.float32)

            forward_range_atr = ((f_high - f_low) / (atr + 1e-9)).clip(lower=0.0)
            vol_target = (
                0.5 * vol_target.fillna(0.0).clip(lower=0.0) * 100.0
                + 0.5 * (forward_range_atr / 4.0)
            ).clip(0.0, 1.0).astype(np.float32)
            entry_depth = positive_r_score.clip(0.0, 1.0)

            label_df = pd.DataFrame(
                {
                    "direction_up": direction_up,
                    "future_return": log_ret_k.astype(np.float32),
                    "move_magnitude": move_magnitude,
                    "volatility_target": vol_target,
                    "efficiency_ratio": eff_ratio,
                    "entry_depth": entry_depth.astype(np.float32),
                    "realized_r_long": pd.Series(long_r, index=df.index, dtype=np.float32),
                    "realized_r_short": pd.Series(short_r, index=df.index, dtype=np.float32),
                    "realized_r_best": pd.Series(best_r_arr, index=df.index, dtype=np.float32),
                },
                index=df.index,
            )
            label_df.iloc[-horizon_bars:, :] = np.nan
            return label_df

        close_arr = close.to_numpy(dtype=np.float64, copy=False)
        high_arr = df["high"].astype(float).to_numpy(dtype=np.float64, copy=False)
        low_arr = df["low"].astype(float).to_numpy(dtype=np.float64, copy=False)
        atr_arr = atr.to_numpy(dtype=np.float64, copy=False)
        up_level = close_arr + atr_threshold * atr_arr
        down_level = close_arr - atr_threshold * atr_arr
        no_hit = horizon_bars + 1
        up_hit = np.full(len(df), no_hit, dtype=np.int16)
        down_hit = np.full(len(df), no_hit, dtype=np.int16)
        for step_i in range(1, horizon_bars + 1):
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

        terminal_arr = terminal_atr.to_numpy(dtype=np.float64, copy=False)
        smoothing    = 0.05
        direction_up = pd.Series(np.nan, index=df.index, dtype=np.float32)
        up_first = (
            (up_hit <= horizon_bars)
            & (
                (down_hit > up_hit)
                | ((down_hit == up_hit) & (terminal_arr > atr_threshold))
            )
        )
        down_first = (
            (down_hit <= horizon_bars)
            & (
                (up_hit > down_hit)
                | ((up_hit == down_hit) & (terminal_arr < -atr_threshold))
            )
        )
        fallback_up = (up_hit > horizon_bars) & (down_hit > horizon_bars) & (terminal_atr > atr_threshold)
        fallback_down = (up_hit > horizon_bars) & (down_hit > horizon_bars) & (terminal_atr < -atr_threshold)
        direction_up[up_first | fallback_up] = 1.0 - smoothing
        direction_up[down_first | fallback_down] = 0.0 + smoothing
        # Dead zone (no barrier and no terminal ATR edge): NaN → masked out in BCE loss

        dominant_exc = pd.concat([up_exc_atr, down_exc_atr], axis=1).max(axis=1)
        move_magnitude = (dominant_exc / 3.0).clip(0.0, 1.0).astype(np.float32)

        # Efficiency ratio: directional purity of the move
        abs_moves = np.abs(close.diff()).rolling(horizon_bars, min_periods=horizon_bars).sum().shift(-horizon_bars)
        net_move  = np.abs(close.shift(-horizon_bars) - close)
        eff_ratio = (net_move / (abs_moves + 1e-9)).clip(0, 1).astype(np.float32)

        forward_range_atr = ((f_high - f_low) / (atr + 1e-9)).clip(lower=0.0)
        vol_target = (0.5 * vol_target.fillna(0.0).clip(lower=0.0) * 100.0 + 0.5 * (forward_range_atr / 4.0)).clip(0.0, 1.0).astype(np.float32)

        entry_depth = move_magnitude.clip(0.0, 1.0)

        label_df = pd.DataFrame(
            {
                "direction_up":      direction_up,
                "future_return":     log_ret_k.astype(np.float32),
                "move_magnitude":    move_magnitude,
                "volatility_target": vol_target,
                "efficiency_ratio":  eff_ratio,
                "entry_depth":       entry_depth.astype(np.float32),
            },
            index=df.index,
        )
        label_df.iloc[-horizon_bars:, :] = np.nan
        return label_df

    def save(self, path: str) -> None:
        if self._model is not None:
            try:
                import torch
                from models.weights_manifest import WeightsManifest
                from services.feature_engine import SEQUENCE_FEATURES, REGIME_4H_FEATURES, REGIME_1H_FEATURES, QUALITY_FEATURES
                os.makedirs(WEIGHT_DIR, exist_ok=True)
                # Unwrap DataParallel before saving so weights are portable
                m = self._model.module if isinstance(self._model, torch.nn.DataParallel) else self._model
                torch.save(m.state_dict(), WEIGHT_FILE)
                self._last_mtime = os.path.getmtime(WEIGHT_FILE)
                WeightsManifest(WEIGHT_DIR).write(
                    gru_features=list(SEQUENCE_FEATURES),
                    regime_4h_features=list(REGIME_4H_FEATURES),
                    regime_1h_features=list(REGIME_1H_FEATURES),
                    quality_features=list(QUALITY_FEATURES),
                    gru_hidden=64,
                    gru_layers=2,
                )
                logger.info("GRULSTMPredictor saved to %s", WEIGHT_FILE)
            except Exception as exc:
                logger.error("GRULSTMPredictor.save failed: %s", exc)

    def load(self, path: str) -> None:
        try:
            import torch
            from models.weights_manifest import WeightsManifest
            from services.feature_engine import SEQUENCE_FEATURES, REGIME_4H_FEATURES, REGIME_1H_FEATURES, QUALITY_FEATURES

            # Guard: refuse to load if feature contract changed since weights were saved
            compat = WeightsManifest(WEIGHT_DIR).check(
                gru_features=list(SEQUENCE_FEATURES),
                regime_4h_features=list(REGIME_4H_FEATURES),
                regime_1h_features=list(REGIME_1H_FEATURES),
                quality_features=list(QUALITY_FEATURES),
            )
            if not compat:
                logger.warning(
                    "GRULSTMPredictor: stale weights detected (%s) — "
                    "deleting %s so retrain starts fresh", compat.reason, WEIGHT_FILE
                )
                WeightsManifest.delete_stale([WEIGHT_FILE], compat.reason)
                self._model = None
                return

            m = _build_torch_model()
            try:
                m.load_state_dict(torch.load(WEIGHT_FILE, map_location=DEVICE, weights_only=True))
            except RuntimeError as shape_exc:
                # Shape mismatch = weights were saved with a different N_FEATURES.
                # Delete and let retrain rebuild from scratch.
                logger.warning(
                    "GRULSTMPredictor: shape mismatch loading weights — "
                    "deleting stale %s and retraining from scratch. Detail: %s",
                    WEIGHT_FILE, shape_exc,
                )
                WeightsManifest.delete_stale([WEIGHT_FILE], f"shape mismatch: {shape_exc}")
                self._model = None
                return
            m = m.to(DEVICE)
            if DEVICE.type == "cuda" and torch.cuda.device_count() > 1:
                m = torch.nn.DataParallel(m)
            m.eval()
            self._model = m
            self._loaded = True
            # Load temperature sidecar if present
            _temp_file = os.path.join(WEIGHT_DIR, "temperature.pt")
            if os.path.exists(_temp_file):
                try:
                    self._temperature = float(torch.load(_temp_file, map_location="cpu", weights_only=True).item())
                    logger.info("GRULSTMPredictor: loaded temperature=%.4f from %s", self._temperature, _temp_file)
                except Exception as _te:
                    logger.warning("GRULSTMPredictor: could not load temperature.pt: %s", _te)
                    self._temperature = 1.0
            else:
                self._temperature = 1.0
            # Load isotonic calibrator sidecar if present
            _iso_file = os.path.join(WEIGHT_DIR, "isotonic.pkl")
            if os.path.exists(_iso_file):
                try:
                    import pickle as _pkl
                    with open(_iso_file, "rb") as fh:
                        self._isotonic = _pkl.load(fh)
                    logger.info("GRULSTMPredictor: loaded isotonic calibrator from %s", _iso_file)
                except Exception as _ie:
                    logger.warning("GRULSTMPredictor: could not load isotonic.pkl: %s", _ie)
                    self._isotonic = None
            else:
                self._isotonic = None
            logger.info("GRULSTMPredictor loaded from %s (device=%s)", WEIGHT_FILE, DEVICE)
        except Exception as exc:
            logger.error("GRULSTMPredictor.load failed: %s", exc)
            self._model = None
