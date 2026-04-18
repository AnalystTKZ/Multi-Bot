"""
weights_manifest.py — Feature contract guard for saved model weights.

Writes a manifest alongside weights at save time recording:
  - feature names and count for each model
  - architecture params (hidden size, layers, etc.)
  - timestamp

At load time, compare the saved manifest against the current code contract.
If they don't match, the caller should delete the stale weights and retrain
from scratch rather than loading incompatible tensors.

Usage:
    from models.weights_manifest import WeightsManifest
    m = WeightsManifest(WEIGHT_DIR)

    # After training — save contract alongside weights
    m.write(gru_features=SEQUENCE_FEATURES, regime_4h_features=REGIME_4H_FEATURES, ...)

    # At startup — check before loading
    compat = m.check(gru_features=SEQUENCE_FEATURES, ...)
    if not compat.ok:
        logger.warning("Weights incompatible: %s — will retrain", compat.reason)
        # delete stale weights, trigger full retrain
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MANIFEST_FILE = "weights_manifest.json"


@dataclass
class CompatResult:
    ok: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ok


def _feature_hash(features: list[str]) -> str:
    """SHA-1 of the ordered feature names. Catches renames and reorderings."""
    blob = json.dumps(features, separators=(",", ":")).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


class WeightsManifest:
    """Read/write the weights_manifest.json in the given directory."""

    def __init__(self, weight_dir: str | Path):
        self._path = Path(weight_dir) / MANIFEST_FILE

    # ── Write ─────────────────────────────────────────────────────────────────

    def write(
        self,
        gru_features: Optional[list[str]] = None,
        regime_4h_features: Optional[list[str]] = None,
        regime_1h_features: Optional[list[str]] = None,
        quality_features: Optional[list[str]] = None,
        gru_hidden: int = 0,
        gru_layers: int = 0,
    ) -> None:
        from datetime import datetime, timezone
        manifest = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "gru": {
                "n_features":    len(gru_features) if gru_features else 0,
                "feature_hash":  _feature_hash(gru_features) if gru_features else "",
                "feature_names": gru_features or [],
                "hidden_size":   gru_hidden,
                "num_layers":    gru_layers,
            },
            "regime_4h": {
                "n_features":    len(regime_4h_features) if regime_4h_features else 0,
                "feature_hash":  _feature_hash(regime_4h_features) if regime_4h_features else "",
                "feature_names": regime_4h_features or [],
            },
            "regime_1h": {
                "n_features":    len(regime_1h_features) if regime_1h_features else 0,
                "feature_hash":  _feature_hash(regime_1h_features) if regime_1h_features else "",
                "feature_names": regime_1h_features or [],
            },
            "quality": {
                "n_features":    len(quality_features) if quality_features else 0,
                "feature_hash":  _feature_hash(quality_features) if quality_features else "",
                "feature_names": quality_features or [],
            },
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info("WeightsManifest written → %s", self._path)

    # ── Read ──────────────────────────────────────────────────────────────────

    def read(self) -> Optional[dict]:
        if not self._path.exists():
            return None
        try:
            with open(self._path) as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("WeightsManifest: failed to read %s: %s", self._path, exc)
            return None

    # ── Check ─────────────────────────────────────────────────────────────────

    def check(
        self,
        gru_features: Optional[list[str]] = None,
        regime_4h_features: Optional[list[str]] = None,
        regime_1h_features: Optional[list[str]] = None,
        quality_features: Optional[list[str]] = None,
    ) -> CompatResult:
        """
        Returns CompatResult(ok=True) if saved weights match current feature contract.
        Returns CompatResult(ok=False, reason=...) if anything is incompatible.
        Returns CompatResult(ok=True) with a warning if no manifest exists yet
        (backwards compat — manifest was added after the weights were saved).
        """
        manifest = self.read()
        if manifest is None:
            logger.warning(
                "WeightsManifest: no manifest at %s — assuming compatible "
                "(run a full retrain to generate one)", self._path
            )
            return CompatResult(ok=True, reason="no manifest — assuming compatible")

        checks = [
            ("gru",        gru_features,        manifest.get("gru", {})),
            ("regime_4h",  regime_4h_features,  manifest.get("regime_4h", {})),
            ("regime_1h",  regime_1h_features,  manifest.get("regime_1h", {})),
            ("quality",    quality_features,     manifest.get("quality", {})),
        ]

        for name, current_features, saved in checks:
            if current_features is None or not saved:
                continue
            saved_hash = saved.get("feature_hash", "")
            current_hash = _feature_hash(current_features)
            if saved_hash and saved_hash != current_hash:
                saved_n = saved.get("n_features", "?")
                cur_n   = len(current_features)
                saved_names = set(saved.get("feature_names", []))
                cur_names   = set(current_features)
                added   = sorted(cur_names - saved_names)
                removed = sorted(saved_names - cur_names)
                detail  = []
                if added:   detail.append(f"added={added}")
                if removed: detail.append(f"removed={removed}")
                if saved_n != cur_n:
                    detail.append(f"count {saved_n}→{cur_n}")
                return CompatResult(
                    ok=False,
                    reason=f"{name} feature contract changed: {'; '.join(detail) or 'order/rename'}"
                )

        return CompatResult(ok=True)

    # ── Convenience: delete stale weights for a given model ───────────────────

    @staticmethod
    def delete_stale(paths: list[str | Path], reason: str) -> None:
        for p in paths:
            p = Path(p)
            if p.exists():
                p.unlink()
                logger.info("Deleted stale weights (%s): %s", reason, p)
