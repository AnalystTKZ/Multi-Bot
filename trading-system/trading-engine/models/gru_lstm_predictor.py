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


def _get_device():
    """
    Return best available device.
    Ensures CUDA_VISIBLE_DEVICES is not masking GPUs, then selects cuda:0.
    On Kaggle (KAGGLE_KERNEL_RUN_TYPE set) CUDA is mandatory — raises if absent.
    Locally falls back to CPU so development still works without GPUs.
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
        return torch.device("cuda")

    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        raise RuntimeError(
            "GRULSTMPredictor: CUDA not available on Kaggle — "
            "enable GPU accelerator in notebook settings."
        )
    logger.warning("GRU: CUDA unavailable — using CPU (training will be slow)")
    return torch.device("cpu")


DEVICE = _get_device()


class ModelNotTrainedError(RuntimeError):
    """Raised when a model is used before being trained."""


# ── PyTorch model definition ──────────────────────────────────────────────────

def _build_torch_model():
    import torch
    import torch.nn as nn

    class _MultiHeadGRULSTM(nn.Module):
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
            self.drop2 = nn.Dropout(0.3)
            self.shared = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
            )
            self.head_dir = nn.Linear(64, 1)
            self.head_mag = nn.Linear(64, 1)
            self.head_var = nn.Linear(64, 1)

        def forward(self, x):
            out, _ = self.gru(x)
            out = self.drop1(out)
            out, _ = self.lstm(out)
            out = self.drop2(out[:, -1, :])   # last timestep
            shared = self.shared(out)
            dir_logits = self.head_dir(shared).squeeze(-1)
            mag = self.head_mag(shared).squeeze(-1)
            log_variance = self.head_var(shared).squeeze(-1)
            return dir_logits, mag, log_variance

        def encode(self, x):
            """Return the 64-dim shared embedding without running the output heads."""
            out, _ = self.gru(x)
            out = self.drop1(out)
            out, _ = self.lstm(out)
            out = self.drop2(out[:, -1, :])
            return self.shared(out)  # (batch, 64)

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
                Keys: "5M", "1H", "4H", "1D". Missing keys produce zero features.
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
            x = torch.tensor(seq[np.newaxis, ...], dtype=torch.float32).to(DEVICE)  # (1, 30, 18)

            self._model.eval()
            with torch.no_grad():
                dir_logits, mag_pred, log_variance_pred = self._model(x)
                p_bull = float(torch.sigmoid(dir_logits)[0].item())
                expected_move = float(torch.relu(mag_pred)[0].item())
                expected_variance = float((torch.nn.functional.softplus(log_variance_pred) + 1e-6)[0].item())
                expected_volatility = float(np.sqrt(expected_variance))
                entry_depth = float(np.clip(expected_move * 100.0, 0.0, 1.0))

            return {
                "p_bull": p_bull,
                "p_bear": float(np.clip(1.0 - p_bull, 0.0, 1.0)),
                "entry_depth": entry_depth,
                "expected_move": expected_move,
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

        Used by the retrain pipeline to bulk-index training data into VectorStore.
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
        regime_series: Optional[pd.Series] = None,
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

            feat_df = fe._build_sequence_df(df, df_htf, symbol=symbol, regime_series=regime_series)
            feat_arr = feat_df[SEQUENCE_FEATURES].to_numpy(dtype=np.float32, copy=False)
            feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
            del feat_df

            n_seq = len(feat_arr) - SEQUENCE_LENGTH
            if n_seq <= 0:
                return {"error": "Not enough rows for sequence training"}

            label_cols = labels.iloc[SEQUENCE_LENGTH:]
            y_dir = label_cols.get("direction_up", pd.Series(0.5, index=label_cols.index)).values.astype(np.float32)
            y_mag = label_cols.get("move_magnitude", pd.Series(0.0, index=label_cols.index)).values.astype(np.float32)
            y_vol = label_cols.get("volatility_target", pd.Series(0.0, index=label_cols.index)).values.astype(np.float32)
            targets = np.column_stack([y_dir, y_mag, y_vol]).astype(np.float32)

            if len(targets) != n_seq:
                n_seq = min(n_seq, len(targets))
                targets = targets[:n_seq]

            split = int(n_seq * (1 - validation_split))
            if split <= 0 or split >= n_seq:
                return {"error": "Not enough sequences after validation split"}

            pos_rate = float(np.mean(targets[:, 0]))
            avg_move = float(np.mean(targets[:, 1]))
            avg_vol = float(np.mean(targets[:, 2]))
            logger.info(
                "GRU targets samples=%d pos_rate=%.4f avg_move=%.6f avg_vol=%.6f",
                n_seq, pos_rate, avg_move, avg_vol,
            )

            class _SequenceDataset(torch.utils.data.Dataset):
                def __init__(self, features: np.ndarray, labels_arr: np.ndarray, start: int, end: int):
                    self._features = features
                    self._labels = labels_arr
                    self._start = start
                    self._end = end

                def __len__(self) -> int:
                    return max(0, self._end - self._start)

                def __getitem__(self, idx: int):
                    i = self._start + idx
                    x = self._features[i:i + SEQUENCE_LENGTH]
                    y_item = self._labels[i]
                    return torch.from_numpy(x), torch.from_numpy(y_item)

            train_ds = _SequenceDataset(feat_arr, targets, 0, split)
            val_ds = _SequenceDataset(feat_arr, targets, split, n_seq)
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
            patience, no_improve = 5, 0
            history = {"train_loss": [], "val_loss": []}

            def _compute_loss(dir_logits, mag_pred, log_variance_pred, yb):
                """
                Multi-head loss with dead-zone masking on direction.
                yb[:, 0] = direction (NaN in dead zone → excluded from BCE).
                yb[:, 1] = move_magnitude (always valid after NaN drop).
                yb[:, 2] = volatility_target.
                λ: dir=1.0, mag=0.5, vol=0.3.
                """
                y_dir = yb[:, 0]
                y_mag = yb[:, 1]
                y_vol = yb[:, 2]

                # Direction: mask NaN dead-zone bars (smoothed labels are 0.05/0.95)
                dir_mask = ~torch.isnan(y_dir)
                if dir_mask.sum() > 0:
                    loss_dir = criterion_dir(dir_logits[dir_mask], y_dir[dir_mask])
                else:
                    loss_dir = torch.tensor(0.0, device=dir_logits.device)

                # Magnitude: Huber loss on non-NaN bars
                mag_mask = ~torch.isnan(y_mag)
                if mag_mask.sum() > 0:
                    loss_mag = criterion_mag(torch.relu(mag_pred[mag_mask]), y_mag[mag_mask])
                else:
                    loss_mag = torch.tensor(0.0, device=mag_pred.device)

                # Volatility: Gaussian NLL (heteroscedastic) — clamped to [0, ∞)
                vol_mask = ~torch.isnan(y_vol)
                if vol_mask.sum() > 0:
                    pv = torch.clamp(torch.nn.functional.softplus(log_variance_pred[vol_mask]) + 1e-6,
                                     min=1e-4, max=1.0)
                    se = torch.square(y_vol[vol_mask] - torch.sqrt(pv))
                    loss_vol = torch.clamp(torch.mean((se / pv) + torch.log(pv)), min=0.0)
                else:
                    loss_vol = torch.tensor(0.0, device=log_variance_pred.device)

                return loss_dir + 0.5 * loss_mag + 0.3 * loss_vol

            for epoch in range(epochs):
                self._model.train()
                train_loss = 0.0
                optimiser.zero_grad()
                for step, (xb, yb) in enumerate(train_dl):
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        dir_logits, mag_pred, log_variance_pred = self._model(xb)
                    dir_logits = dir_logits.float()
                    mag_pred = mag_pred.float()
                    log_variance_pred = log_variance_pred.float()
                    loss = _compute_loss(dir_logits, mag_pred, log_variance_pred, yb) / grad_accum_steps
                    scaler.scale(loss).backward()
                    if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_dl):
                        scaler.step(optimiser)
                        scaler.update()
                        optimiser.zero_grad()
                    train_loss += loss.item() * grad_accum_steps * len(xb)
                train_loss /= max(1, len(train_ds))

                self._model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for xb, yb in val_dl:
                        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                        with torch.amp.autocast("cuda", enabled=use_amp):
                            dir_logits, mag_pred, log_variance_pred = self._model(xb)
                        dir_logits = dir_logits.float()
                        mag_pred = mag_pred.float()
                        log_variance_pred = log_variance_pred.float()
                        batch_loss = _compute_loss(dir_logits, mag_pred, log_variance_pred, yb)
                        val_loss += batch_loss.item() * len(xb)
                val_loss /= max(1, len(val_ds))

                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                logger.info("GRU epoch %d/%d — train=%.4f val=%.4f", epoch + 1, epochs, train_loss, val_loss)

                if val_loss < best_val:
                    best_val = val_loss
                    no_improve = 0
                    self.save(WEIGHT_DIR)
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        logger.info("GRU early stop at epoch %d", epoch + 1)
                        break

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
        Train on multiple (df, labels, df_htf, symbol) segments in one combined pass.

        All TFs and symbols train together in a single dataset — avoids the problem of
        per-TF training where the last (smallest) TF overwrites weights learned from
        larger TFs. Sequences are boundary-safe: __getitem__ never crosses symbol/TF
        boundaries. shuffle=True on the DataLoader mixes all symbols/TFs each epoch.

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

                # Build per-segment (feat_arr, tgt_arr) — never concatenate across symbols
                seg_feats: list = []
                seg_tgts: list  = []

                for seg in group:
                    try:
                        feat_df = fe._build_sequence_df(
                            seg["df"], seg.get("df_htf"),
                            symbol=seg.get("symbol"),
                            regime_series=seg.get("regime_series"),
                        )
                        feat_arr = feat_df[SEQUENCE_FEATURES].to_numpy(dtype=np.float32, copy=False)
                        feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)
                        del feat_df

                        lbl = seg["labels"]
                        n_seq = len(feat_arr) - SEQUENCE_LENGTH
                        if n_seq <= 0:
                            continue
                        y_dir = lbl.get("direction_up", pd.Series(0.5, index=lbl.index)).values.astype(np.float32)[SEQUENCE_LENGTH:]
                        y_mag = lbl.get("move_magnitude", pd.Series(0.0, index=lbl.index)).values.astype(np.float32)[SEQUENCE_LENGTH:]
                        y_vol = lbl.get("volatility_target", pd.Series(0.0, index=lbl.index)).values.astype(np.float32)[SEQUENCE_LENGTH:]
                        tgt = np.column_stack([y_dir, y_mag, y_vol]).astype(np.float32)
                        n_seq = min(n_seq, len(tgt))

                        seg_feats.append(feat_arr[:n_seq + SEQUENCE_LENGTH].copy())
                        seg_tgts.append(tgt[:n_seq].copy())
                        del feat_arr, tgt
                    except Exception as exc:
                        logger.warning("train_multi: segment %s/%s failed: %s", seg.get("symbol"), tf, exc)
                        continue

                if not seg_feats:
                    logger.warning("train_multi: no valid segments for TF=%s", tf)
                    continue

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
                Y_train = np.empty((n_train, 3), dtype=np.float32)
                X_val   = np.empty((n_val,   SEQUENCE_LENGTH, n_feat), dtype=np.float32)
                Y_val   = np.empty((n_val,   3), dtype=np.float32)

                tr_off, va_off = 0, 0
                for sf, st in zip(seg_feats, seg_tgts):
                    if tr_off >= n_train and va_off >= n_val:
                        break   # budget exhausted
                    n = len(st)
                    sp = int(n * (1 - validation_split))
                    if sp <= 0 or sp >= n:
                        continue
                    for start_idx, end_idx, X_out, Y_out, is_train in [
                        (0,  sp, X_train, Y_train, True),
                        (sp, n,  X_val,   Y_val,   False),
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
                        if is_train:
                            tr_off += seg_n
                        else:
                            va_off += seg_n
                        del raw

                # Trim pre-alloc arrays to actually filled rows (budget may have been generous)
                X_train = X_train[:tr_off]
                Y_train = Y_train[:tr_off]
                X_val   = X_val[:va_off]
                Y_val   = Y_val[:va_off]
                n_train, n_val = tr_off, va_off

                del seg_feats, seg_tgts
                _gc.collect()

                logger.info("train_multi TF=%s: train=%d val=%d (%.0f MB tensors)",
                            tf, n_train, n_val,
                            (X_train.nbytes + Y_train.nbytes + X_val.nbytes + Y_val.nbytes) / 1e6)

                # Pin to page-locked memory for fast H→D transfers
                X_train_t = torch.from_numpy(X_train).pin_memory()
                Y_train_t = torch.from_numpy(Y_train).pin_memory()
                X_val_t   = torch.from_numpy(X_val).pin_memory()
                Y_val_t   = torch.from_numpy(Y_val).pin_memory()
                del X_train, Y_train, X_val, Y_val
                _gc.collect()

                # Pre-shuffle index array (re-shuffled each epoch) — no DataLoader needed
                train_idx = np.arange(n_train, dtype=np.int64)
                steps_per_epoch = max(1, (n_train + batch_size - 1) // batch_size)

                optimiser = torch.optim.AdamW(self._model.parameters(), lr=3e-4, weight_decay=1e-3)
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimiser, max_lr=3e-4, epochs=epochs,
                    steps_per_epoch=steps_per_epoch, pct_start=0.2,
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

                def _loss_tm(dl, mp, lv, yb):
                    y_dir, y_mag, y_vol = yb[:, 0], yb[:, 1], yb[:, 2]
                    dir_mask = ~torch.isnan(y_dir)
                    loss_dir = criterion_dir(dl[dir_mask], y_dir[dir_mask]).mean() if dir_mask.sum() > 0 else dl.sum() * 0
                    mag_mask = ~torch.isnan(y_mag)
                    loss_mag = criterion_mag(torch.relu(mp[mag_mask]), y_mag[mag_mask]) if mag_mask.sum() > 0 else mp.sum() * 0
                    vol_mask = ~torch.isnan(y_vol)
                    if vol_mask.sum() > 0:
                        # Clamp variance in [1e-4, 1.0] — prevents NLL going negative.
                        # NLL = (se/pv) + log(pv). When pv→0, log(pv)→-∞ which makes the
                        # loss unboundedly negative and causes the optimizer to collapse
                        # the variance head to near-zero, corrupting expected_variance.
                        pv = torch.clamp(torch.nn.functional.softplus(lv[vol_mask]) + 1e-6,
                                         min=1e-4, max=1.0)
                        loss_vol = torch.clamp(
                            torch.mean((torch.square(y_vol[vol_mask] - torch.sqrt(pv)) / pv) + torch.log(pv)),
                            min=0.0)  # floor at 0 — NLL below 0 means overfitted variance, treat as perfect
                    else:
                        loss_vol = lv.sum() * 0
                    return loss_dir + 0.5 * loss_mag + 0.3 * loss_vol

                best_val = float("inf")
                patience, no_improve = 5, 0

                # Move val set to GPU once — it's small enough (~600K × 30 × F ≈ 600MB)
                X_val_gpu = X_val_t.to(DEVICE, non_blocking=True)
                Y_val_gpu = Y_val_t.to(DEVICE, non_blocking=True)

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
                        with torch.amp.autocast("cuda", enabled=use_amp):
                            dl, mp, lv = self._model(xb)
                        dl, mp, lv = dl.float(), mp.float(), lv.float()
                        loss = _loss_tm(dl, mp, lv, yb) / grad_accum_steps
                        scaler.scale(loss).backward()
                        if (step + 1) % grad_accum_steps == 0 or (step + 1) == steps_per_epoch:
                            scaler.unscale_(optimiser)
                            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                            scaler.step(optimiser)
                            scaler.update()
                            optimiser.zero_grad()
                            scheduler.step()
                        train_loss += loss.item() * grad_accum_steps * (b_end - b_start)
                    train_loss /= max(1, n_train)

                    self._model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        val_bs = batch_size * 2   # larger batches during eval (no backward pass)
                        for v_start in range(0, n_val, val_bs):
                            xb = X_val_gpu[v_start: v_start + val_bs]
                            yb = Y_val_gpu[v_start: v_start + val_bs]
                            with torch.amp.autocast("cuda", enabled=use_amp):
                                dl, mp, lv = self._model(xb)
                            dl, mp, lv = dl.float(), mp.float(), lv.float()
                            val_loss += _loss_tm(dl, mp, lv, yb).item() * len(xb)
                    val_loss /= max(1, n_val)

                    combined_history["train_loss"].append(train_loss)
                    combined_history["val_loss"].append(val_loss)
                    logger.info("train_multi TF=%s epoch %d/%d train=%.4f val=%.4f", tf, epoch + 1, epochs, train_loss, val_loss)

                    if val_loss < best_val:
                        best_val = val_loss
                        no_improve = 0
                        self.save(WEIGHT_DIR)
                        logger.info("train_multi TF=%s: new best val=%.4f — saved", tf, best_val)
                    else:
                        no_improve += 1
                        if no_improve >= patience:
                            logger.info("train_multi TF=%s early stop at epoch %d", tf, epoch + 1)
                            break

                combined_history["groups_trained"] += 1
                del X_train_t, Y_train_t, X_val_t, Y_val_t
                del X_val_gpu, Y_val_gpu, train_idx
                import gc; gc.collect()
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

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
        Institutional-grade multi-head labels — predict actionable structure.

        horizon_bars=12 on 15M ≈ 3 hours ahead (enough signal, not 1-bar noise).

        direction_up:
            Dead-zone filtered by ATR threshold — only label bars where the
            k-step move exceeds 0.3× ATR. Labels outside the dead zone are
            smoothed (0.05) to prevent overconfidence. Dead-zone bars = NaN,
            excluded from BCE loss via masking in the training loop.

        move_magnitude:
            log1p(|k-step log return|) — Huber-robust regression target.

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

        # k-step log return — more Gaussian than arithmetic return
        future_close = close.shift(-horizon_bars)
        log_ret_k    = np.log((future_close + 1e-9) / (close + 1e-9))

        # Forward-looking volatility: std of next horizon_bars 1-step log returns
        log_ret_1  = np.log((close + 1e-9) / (close.shift(1) + 1e-9))
        vol_target = (log_ret_1.shift(-1)
                      .rolling(window=horizon_bars, min_periods=horizon_bars)
                      .std()).astype(np.float32)

        # ATR-normalised dead zone — suppress noise below 0.3× ATR
        atr_norm_thresh = (atr * atr_threshold) / (close + 1e-9)

        smoothing    = 0.05
        direction_up = pd.Series(np.nan, index=df.index, dtype=np.float32)
        direction_up[log_ret_k >  atr_norm_thresh] = 1.0 - smoothing
        direction_up[log_ret_k < -atr_norm_thresh] = 0.0 + smoothing
        # Dead zone (|log_ret_k| <= thresh): NaN → masked out in BCE loss

        move_magnitude = np.log1p(np.abs(log_ret_k)).astype(np.float32)

        # Efficiency ratio: directional purity of the move
        abs_moves = np.abs(close.diff()).rolling(horizon_bars, min_periods=horizon_bars).sum().shift(-horizon_bars)
        net_move  = np.abs(close.shift(-horizon_bars) - close)
        eff_ratio = (net_move / (abs_moves + 1e-9)).clip(0, 1).astype(np.float32)

        entry_depth = np.clip(move_magnitude * 100.0, 0.0, 1.0)

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
                    gru_layers=1,
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
            logger.info("GRULSTMPredictor loaded from %s (device=%s)", WEIGHT_FILE, DEVICE)
        except Exception as exc:
            logger.error("GRULSTMPredictor.load failed: %s", exc)
            self._model = None
