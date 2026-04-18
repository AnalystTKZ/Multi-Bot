"""
sentiment_model.py — FinBERT sentiment with VADER fallback.

Primary: ProsusAI/finbert (HuggingFace transformers) — fine-tuned on financial text.
Fallback: vaderSentiment + finance keyword lexicon (see _FINANCE_LEXICON).
Gold: USD sentiment is INVERTED (USD bullish → Gold bearish).
Cache: LRU 50 headlines, TTL 30 minutes.

Every result includes a 'backend' key ('finbert', 'vader', or 'neutral') so
journals and logs always show which engine produced the score.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import Dict

import os

from models.base_model import BaseModel

_MODEL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # trading-engine/

logger = logging.getLogger(__name__)

_CACHE_SIZE = 50
_CACHE_TTL = 1800  # 30 minutes

# Finance-specific keyword adjustments for VADER fallback
_FINANCE_LEXICON = {
    "hawkish": 0.5, "tightening": 0.4, "rate hike": 0.5, "hike": 0.3,
    "dovish": -0.5, "easing": -0.4, "rate cut": -0.5, "cut": -0.2,
    "inflation": -0.3, "recession": -0.7, "default": -0.8,
    "nfp beat": 0.6, "nfp miss": -0.6, "beat": 0.4, "miss": -0.4,
    "better than expected": 0.5, "worse than expected": -0.5,
    "surges": 0.6, "rallies": 0.5, "falls": -0.4, "plunges": -0.7,
    "strong": 0.4, "weak": -0.4, "solid": 0.3, "disappointing": -0.4,
    "crisis": -0.7, "sanctions": -0.4, "geopolitical": -0.3,
}


class SentimentModel(BaseModel):
    """
    FinBERT primary, VADER fallback. LRU cache. Gold inverts USD sentiment.
    All results include 'backend' key for journal/log attribution.
    """

    weight_path = os.path.join(_MODEL_ROOT, "weights", "sentiment_model.pkl")

    def __init__(self):
        super().__init__()
        self._bert_pipeline = None
        self._bert_available = False
        self._vader = None
        self._cache: OrderedDict = OrderedDict()
        self._cache_timestamps: dict = {}
        self._try_load_finbert()
        self._try_load_vader()

    def _try_load_finbert(self) -> None:
        try:
            from transformers import pipeline
            self._bert_pipeline = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                top_k=None,
            )
            self._bert_available = True
            logger.info("SentimentModel: FinBERT loaded (ProsusAI/finbert)")
        except Exception as exc:
            logger.info("SentimentModel: FinBERT unavailable (%s) — will use VADER fallback", exc)
            self._bert_pipeline = None
            self._bert_available = False

    def _try_load_vader(self) -> None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
            self._vader.lexicon.update(_FINANCE_LEXICON)
            logger.info("SentimentModel: VADER loaded with finance lexicon (%d extra terms)", len(_FINANCE_LEXICON))
        except ImportError:
            logger.warning("SentimentModel: vaderSentiment not installed — sentiment scoring disabled")
            self._vader = None

    def analyze(self, text: str, instrument: str = "USD") -> Dict:
        """
        Returns: {'score': float [-1,1], 'label': str, 'confidence': float, 'backend': str}
        'backend' is always included so journals/logs show which engine scored the text.
        Gold inverts USD sentiment (USD bullish = Gold bearish).
        """
        cache_key = f"{text[:100]}:{instrument}"

        # Check cache
        if cache_key in self._cache:
            ts = self._cache_timestamps.get(cache_key, 0)
            if time.time() - ts < _CACHE_TTL:
                return self._cache[cache_key]
            else:
                del self._cache[cache_key]
                del self._cache_timestamps[cache_key]

        result = self._analyze_raw(text)

        # Invert for Gold (XAU)
        if "XAU" in instrument.upper() or instrument.upper() == "GOLD":
            result = {
                "score": -result["score"],
                "label": self._invert_label(result["label"]),
                "confidence": result["confidence"],
                "backend": result["backend"],
            }

        logger.debug(
            "SentimentModel [%s] %s → score=%.3f label=%s conf=%.3f backend=%s",
            instrument, text[:60], result["score"], result["label"],
            result["confidence"], result["backend"],
        )

        # LRU eviction
        if len(self._cache) >= _CACHE_SIZE:
            self._cache.popitem(last=False)
            if self._cache_timestamps:
                oldest = next(iter(self._cache_timestamps))
                del self._cache_timestamps[oldest]

        self._cache[cache_key] = result
        self._cache_timestamps[cache_key] = time.time()
        return result

    def _analyze_raw(self, text: str) -> Dict:
        """FinBERT primary, VADER fallback. Always includes 'backend' key."""
        if self._bert_available and self._bert_pipeline is not None:
            try:
                output = self._bert_pipeline(text[:512])[0]
                label_scores = {item["label"].lower(): item["score"] for item in output}
                pos = label_scores.get("positive", 0.0)
                neg = label_scores.get("negative", 0.0)
                score = pos - neg  # [-1, 1]
                top_label = max(label_scores, key=label_scores.get)
                return {
                    "score": float(score),
                    "label": top_label,
                    "confidence": float(label_scores[top_label]),
                    "backend": "finbert",
                }
            except Exception as exc:
                logger.warning("SentimentModel: FinBERT inference failed (%s) — falling back to VADER", exc)

        if self._vader is not None:
            try:
                scores = self._vader.polarity_scores(text)
                compound = float(scores["compound"])
                if compound >= 0.05:
                    label = "positive"
                elif compound <= -0.05:
                    label = "negative"
                else:
                    label = "neutral"
                return {
                    "score": compound,
                    "label": label,
                    "confidence": abs(compound),
                    "backend": "vader",
                }
            except Exception as exc:
                logger.warning("SentimentModel: VADER failed: %s", exc)

        return {"score": 0.0, "label": "neutral", "confidence": 0.0, "backend": "neutral"}

    @staticmethod
    def _invert_label(label: str) -> str:
        mapping = {"positive": "negative", "negative": "positive", "neutral": "neutral"}
        return mapping.get(label, "neutral")

    def reload_if_updated(self):
        pass  # FinBERT is pre-trained via HuggingFace — no hot-reload needed

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass
