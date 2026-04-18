"""
base_model.py — Hot-reload base class for all ML models.
"""

from __future__ import annotations

import abc
import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

_RELOAD_INTERVAL_SECONDS = 300  # 5 minutes


class BaseModel(abc.ABC):
    """
    Abstract base for all trading ML models.
    Subclasses call super().__init__() to set up hot-reload timer.
    """

    weight_path: str = ""

    def __init__(self):
        self._last_mtime: float = 0.0
        self._last_reload_check: float = 0.0
        self._loaded: bool = False

    @property
    def is_trained(self) -> bool:
        """True if weight file exists and appears loadable."""
        if not self.weight_path:
            return False
        return os.path.exists(self.weight_path) and os.path.getsize(self.weight_path) > 0

    def reload_if_updated(self) -> bool:
        """
        Checks file mtime every 5 min. Reloads on change.
        Returns True if a reload was performed.
        """
        now = time.time()
        if now - self._last_reload_check < _RELOAD_INTERVAL_SECONDS:
            return False
        self._last_reload_check = now

        if not self.is_trained:
            return False

        try:
            current_mtime = os.path.getmtime(self.weight_path)
        except OSError:
            return False

        if current_mtime != self._last_mtime:
            logger.info("%s: weight file changed — reloading", self.__class__.__name__)
            try:
                self.load(self.weight_path)
                self._last_mtime = current_mtime
                self._loaded = True
                logger.info("%s: reload complete", self.__class__.__name__)
                return True
            except Exception as exc:
                logger.error("%s: reload failed: %s", self.__class__.__name__, exc)
        return False

    @abc.abstractmethod
    def save(self, path: str) -> None:
        """Persist model weights to path."""

    @abc.abstractmethod
    def load(self, path: str) -> None:
        """Load model weights from path."""
