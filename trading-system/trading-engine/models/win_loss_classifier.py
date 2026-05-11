"""
win_loss_classifier.py — ANN binary win/loss trade outcome classifier.

Trained on historical trade outcomes (profitable setups labeled 1, losing
labeled 0) from the candidate log / trade journal.  Outputs p_win_ann ∈ [0,1]
as a complementary second opinion alongside the QualityScorer (which regresses
a continuous EV).

Role in the pipeline:
  - Runs at candidate evaluation time alongside QualityScorer.
  - p_win_ann is written to the candidate log for analysis.
  - Optional hard gate: if WIN_LOSS_GATE_ENABLED=1, trades where p_win_ann
    falls below WIN_LOSS_MIN_PROB (default 0.45) are rejected.
  - The gate is intentionally permissive by default — the QualityScorer EV
    gate is the primary filter; this classifier adds a soft second vote.

Architecture:
  Input (N_WL_FEATURES) → 128 → 64 → 32 → 1 (sigmoid)
  Loss: BCEWithLogitsLoss with pos_weight for class imbalance.
  Activation: GELU + BatchNorm + Dropout(0.3).

Features (WL_FEATURES): signal context at execution time — same inputs
as QualityScorer so they remain comparable, but this model classifies
win/loss directly rather than regressing EV.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

WL_FEATURE_VERSION = 1

# ── Feature contract ──────────────────────────────────────────────────────────
# Same columns as QUALITY_FEATURES (20) plus 3 additional execution-context
# features that are known at signal time. Fixed contract — do not reorder.
WL_FEATURES: List[str] = [
    # -- from QUALITY_FEATURES (shared with QualityScorer) ---
    "strategy_id",
    "signal_direction",
    "rr_ratio",
    "p_win_gru",
    "gru_edge",
    "expected_move",
    "gru_uncertainty",
    "trade_regime_code",
    "expected_r_gross",
    "volatility_percentile",
    "chop_score",
    "adx_at_signal",
    "atr_ratio_at_signal",
    "spread_at_signal",
    "session_at_signal",
    "news_in_30min",
    "strategy_win_rate_5",
    "strategy_win_rate_20",
    "strategy_win_rate_50",
    "vol_slope_at_signal",
    # -- additional execution context --
    "p_bull_rf",        # RF blended direction probability at signal time
    "htf_bias_score",   # HTF aligned bias score (0–1)
    "ltf_trend_score",  # LTF trend score (0–1)
]

N_WL_FEATURES = len(WL_FEATURES)

_MODEL_ROOT = Path(__file__).resolve().parent.parent
WEIGHT_DIR  = _MODEL_ROOT / "weights" / "win_loss_classifier"
WEIGHT_FILE = WEIGHT_DIR / "model.pt"
META_FILE   = WEIGHT_DIR / "meta.json"

WIN_LOSS_MIN_PROB_DEFAULT = 0.45


def _get_device():
    import torch
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        del os.environ["CUDA_VISIBLE_DEVICES"]
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_net(n_features: int):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(n_features, 128),
        nn.BatchNorm1d(128),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.GELU(),
        nn.Linear(32, 1),
    )


class WinLossClassifier:
    """
    ANN binary classifier: win (1) vs. loss (0) on historical trade setups.

    Public API:
      - is_trained  → bool
      - train(X, y, val_X, val_y) → metrics dict
      - predict_proba(row_dict) → float p_win_ann ∈ [0,1]
      - predict_proba_batch(X) → np.ndarray
      - save() / load()
    """

    def __init__(self):
        self._model = None
        self._device = _get_device()
        self._meta: dict = {}
        self._load_if_exists()

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_X: Optional[np.ndarray] = None,
        val_y: Optional[np.ndarray] = None,
        epochs: int = 80,
        batch_size: int = 256,
        lr: float = 3e-4,
        patience: int = 12,
    ) -> dict:
        """
        Train ANN on labeled trade outcomes.

        X: (N, N_WL_FEATURES) — feature matrix at signal time
        y: (N,) int — 1=win (TP hit), 0=loss (SL hit or timeout)
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        assert X.shape[1] == N_WL_FEATURES, (
            f"WinLossClassifier: expected {N_WL_FEATURES} features, got {X.shape[1]}"
        )

        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)
        logger.info(
            "WinLoss train: N=%d pos=%d neg=%d ratio=%.3f",
            len(y), n_pos, n_neg, n_pos / max(1, len(y)),
        )

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        ds = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

        net = _build_net(N_WL_FEATURES).to(self._device)

        # Class-weighted loss to handle win/loss imbalance
        pos_weight = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32).to(self._device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)

        best_val_loss = float("inf")
        no_improve = 0
        best_state = None

        for epoch in range(epochs):
            net.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                optimizer.zero_grad()
                logits = net(xb).squeeze(-1)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            scheduler.step()
            epoch_loss /= max(1, len(y))

            if val_X is not None and val_y is not None and len(val_y) > 0:
                net.eval()
                with torch.no_grad():
                    vX = torch.tensor(val_X, dtype=torch.float32).to(self._device)
                    vy = torch.tensor(val_y, dtype=torch.float32).to(self._device)
                    v_logits = net(vX).squeeze(-1)
                    val_loss = criterion(v_logits, vy).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience and epoch >= 20:
                        logger.info("WinLoss early stop at epoch %d", epoch + 1)
                        break
            else:
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}

        if best_state is not None:
            net.load_state_dict({k: v.to(self._device) for k, v in best_state.items()})

        self._model = net
        self._model.eval()

        # Compute metrics
        with torch.no_grad():
            train_logits = net(X_t.to(self._device)).squeeze(-1)
            train_probs  = torch.sigmoid(train_logits).cpu().numpy()

        from sklearn.metrics import accuracy_score, roc_auc_score
        metrics = {
            "n_train": len(y),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "train_acc": float(accuracy_score(y, (train_probs >= 0.5).astype(int))),
            "feature_version": WL_FEATURE_VERSION,
        }
        if n_pos > 0 and n_neg > 0:
            metrics["train_auc"] = float(roc_auc_score(y, train_probs))

        if val_X is not None and val_y is not None and len(val_y) > 0:
            with torch.no_grad():
                vX_t = torch.tensor(val_X, dtype=torch.float32).to(self._device)
                v_logits = net(vX_t).squeeze(-1)
                val_probs = torch.sigmoid(v_logits).cpu().numpy()
            n_vp = int(val_y.sum())
            n_vn = int(len(val_y) - n_vp)
            metrics["val_acc"] = float(accuracy_score(val_y, (val_probs >= 0.5).astype(int)))
            if n_vp > 0 and n_vn > 0:
                metrics["val_auc"] = float(roc_auc_score(val_y, val_probs))
            logger.info(
                "WinLoss val: N=%d acc=%.4f auc=%.4f",
                len(val_y), metrics["val_acc"], metrics.get("val_auc", float("nan")),
            )

        self.save()
        return metrics

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, row: dict) -> float:
        """Single-row inference. Returns p_win_ann ∈ [0,1]."""
        if self._model is None:
            return 0.5
        import torch
        x = np.array(
            [float(row.get(f, 0.0) or 0.0) for f in WL_FEATURES],
            dtype=np.float32,
        ).reshape(1, -1)
        self._model.eval()
        with torch.no_grad():
            xt = torch.tensor(x).to(self._device)
            logit = self._model(xt).squeeze(-1)
            return float(torch.sigmoid(logit).item())

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Batch inference.
        X: (N, N_WL_FEATURES) — returns (N,) float32 p_win_ann
        """
        if self._model is None:
            return np.full(len(X), 0.5, dtype=np.float32)
        import torch
        self._model.eval()
        with torch.no_grad():
            xt = torch.tensor(X, dtype=torch.float32).to(self._device)
            probs = torch.sigmoid(self._model(xt).squeeze(-1)).cpu().numpy()
        return probs.astype(np.float32)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        import torch
        WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
        m = self._model.module if hasattr(self._model, "module") else self._model
        torch.save(m.state_dict(), WEIGHT_FILE)
        meta = {
            "feature_version": WL_FEATURE_VERSION,
            "n_features": N_WL_FEATURES,
            "features": WL_FEATURES,
        }
        with open(META_FILE, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("WinLossClassifier saved to %s", WEIGHT_FILE)

    def load(self) -> bool:
        if not WEIGHT_FILE.exists():
            return False
        try:
            import torch
            net = _build_net(N_WL_FEATURES).to(self._device)
            state = torch.load(WEIGHT_FILE, map_location=self._device, weights_only=True)
            net.load_state_dict(state)
            net.eval()
            self._model = net
            if META_FILE.exists():
                with open(META_FILE) as f:
                    self._meta = json.load(f)
                if self._meta.get("feature_version") != WL_FEATURE_VERSION:
                    logger.warning(
                        "WinLoss: weight feature_version=%s != current=%s — stale weights",
                        self._meta.get("feature_version"), WL_FEATURE_VERSION,
                    )
            logger.info("WinLossClassifier loaded from %s", WEIGHT_FILE)
            return True
        except Exception as exc:
            logger.warning("WinLossClassifier.load failed: %s", exc)
            self._model = None
            return False

    def _load_if_exists(self) -> None:
        self.load()


def build_wl_feature_matrix(
    journal_rows: List[dict],
    extra_cols: Optional[dict] = None,
) -> tuple:
    """
    Build WL feature matrix and labels from journal rows (closed trades).

    journal_rows: list of dicts from trade_journal_detailed.jsonl or candidate_log.csv
    extra_cols: optional dict of {row_index: {col: val}} for features not in journal

    Returns: (X: np.ndarray shape (N, N_WL_FEATURES), y: np.ndarray shape (N,) int)
    """
    rows_X = []
    rows_y = []
    for i, row in enumerate(journal_rows):
        # Determine label: prefer explicit tp_hit/outcome fields, then fall
        # back to pnl and exit_reason (standard journal schema).
        tp_hit = row.get("tp_hit")
        outcome = row.get("outcome", "")
        if tp_hit is not None:
            label = 1 if (str(tp_hit) in ("1", "1.0", "True", "true") or tp_hit == 1) else 0
        elif outcome:
            label = 1 if str(outcome).lower() in ("win", "tp", "tp1", "tp2") else 0
        else:
            pnl = row.get("pnl")
            exit_reason = str(row.get("exit_reason", "")).lower()
            if pnl is None and not exit_reason:
                continue  # skip unclosed / no outcome data
            if pnl is not None:
                try:
                    label = 1 if float(pnl) > 0 else 0
                except (TypeError, ValueError):
                    continue
            else:
                # exit_reason fallback
                if exit_reason in ("tp", "tp1", "tp2", "take_profit", "target"):
                    label = 1
                elif exit_reason in ("sl", "stop_loss", "stop", "loss"):
                    label = 0
                else:
                    continue

        extra = (extra_cols or {}).get(i, {})
        feats = []
        for col in WL_FEATURES:
            val = row.get(col, extra.get(col, 0.0))
            try:
                val = float(val) if val is not None else 0.0
            except (TypeError, ValueError):
                val = 0.0
            if not np.isfinite(val):
                val = 0.0
            feats.append(val)
        rows_X.append(feats)
        rows_y.append(label)

    if not rows_X:
        return np.zeros((0, N_WL_FEATURES), dtype=np.float32), np.zeros(0, dtype=np.int32)

    return (
        np.array(rows_X, dtype=np.float32),
        np.array(rows_y, dtype=np.int32),
    )
