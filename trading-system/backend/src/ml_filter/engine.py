"""ML-based filter for ICT signals with fallback rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ict.models import IctSignal
from .models import FilterRequest, FilterResponse, LabeledTrade, MarketFeatures, TradeFeatures, TrainRequest, TrainResult

FEATURE_NAMES = [
    "ob_strength",
    "fvg_size",
    "liquidity_sweep",
    "htf_alignment",
    "volatility",
    "spread",
    "volume",
    "session_timing",
    "rr_ratio",
    "distance_to_liquidity",
]


@dataclass
class ModelState:
    model: Optional[object] = None
    feature_importance: Dict[str, float] = field(default_factory=dict)
    samples: int = 0


GLOBAL_MODEL_STATE = ModelState()
TRAINING_BUFFER: List[LabeledTrade] = []


def train_filter_model(request: TrainRequest, state: ModelState = GLOBAL_MODEL_STATE) -> TrainResult:
    if request.retrain:
        TRAINING_BUFFER.extend(request.trades)
    else:
        TRAINING_BUFFER[:] = request.trades

    if len(TRAINING_BUFFER) < 20:
        return TrainResult(
            trained=False,
            samples=len(TRAINING_BUFFER),
            feature_importance={},
            message="Not enough samples to train model (need >= 20)",
        )

    features, labels = _build_training_set(TRAINING_BUFFER)
    if not features:
        return TrainResult(
            trained=False,
            samples=len(TRAINING_BUFFER),
            feature_importance={},
            message="No valid feature vectors found",
        )

    model, importances = _fit_model(features, labels)
    if model is None:
        return TrainResult(
            trained=False,
            samples=len(TRAINING_BUFFER),
            feature_importance={},
            message="ML backend unavailable; fallback rules only",
        )

    state.model = model
    state.feature_importance = importances
    state.samples = len(TRAINING_BUFFER)
    return TrainResult(
        trained=True,
        samples=state.samples,
        feature_importance=importances,
        message="Model trained successfully",
    )


def filter_signal(request: FilterRequest, state: ModelState = GLOBAL_MODEL_STATE) -> FilterResponse:
    feature_vector = _extract_features(request.original_signal, request.market_features, request.trade_features)
    probability, used_fallback = _predict_probability(feature_vector, request, state)

    decision = "ACCEPT" if probability >= request.threshold else "REJECT"
    if request.original_signal.direction == "none":
        decision = "REJECT"

    feature_importance = state.feature_importance if state.feature_importance else _fallback_feature_importance()
    return FilterResponse(
        original_signal=request.original_signal,
        probability_score=probability,
        decision=decision,
        feature_importance=feature_importance,
        used_fallback=used_fallback,
    )


def _build_training_set(trades: List[LabeledTrade]) -> Tuple[List[List[float]], List[int]]:
    features: List[List[float]] = []
    labels: List[int] = []
    for trade in trades:
        vector = _extract_features(trade.signal, trade.market_features, trade.trade_features)
        if vector:
            features.append(vector)
            labels.append(trade.label)
    return features, labels


def _fit_model(features: List[List[float]], labels: List[int]) -> Tuple[Optional[object], Dict[str, float]]:
    try:
        from sklearn.ensemble import RandomForestClassifier
    except Exception:
        return None, {}

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=3,
        random_state=42,
    )
    model.fit(features, labels)

    importances = {name: float(val) for name, val in zip(FEATURE_NAMES, model.feature_importances_)}
    return model, importances


def _predict_probability(
    feature_vector: List[float],
    request: FilterRequest,
    state: ModelState,
) -> Tuple[float, bool]:
    if state.model is None or not feature_vector:
        return _fallback_probability(request), True

    try:
        prob = float(state.model.predict_proba([feature_vector])[0][1])
        return prob, False
    except Exception:
        return _fallback_probability(request), True


def _fallback_probability(request: FilterRequest) -> float:
    if request.original_signal.direction == "none":
        return 0.0
    score = request.original_signal.confluence_score / 100.0
    return min(1.0, max(0.0, score))


def _fallback_feature_importance() -> Dict[str, float]:
    base = 1.0 / len(FEATURE_NAMES)
    return {name: base for name in FEATURE_NAMES}


def _extract_features(
    signal: IctSignal,
    market_features: MarketFeatures,
    trade_features: Optional[TradeFeatures],
) -> List[float]:
    payload = signal.feature_payload or {}

    ob_strength = float(payload.get("ob_strength", 0.0))
    fvg_size = float(payload.get("fvg_size", 0.0))
    liquidity_sweep = float(payload.get("liquidity_sweep", 0.0))
    htf_alignment = float(payload.get("htf_alignment", 0.0))

    session_timing = market_features.session_timing
    if session_timing is None:
        session_timing = float(payload.get("session_timing", 0.0))

    rr_ratio = None
    distance_to_liquidity = None
    if trade_features:
        rr_ratio = trade_features.rr_ratio
        distance_to_liquidity = trade_features.distance_to_liquidity

    if rr_ratio is None:
        rr_ratio = float(payload.get("rr_ratio", 0.0))
    if distance_to_liquidity is None:
        distance_to_liquidity = float(payload.get("distance_to_liquidity", 0.0))

    volatility = market_features.volatility if market_features.volatility is not None else 0.0
    spread = market_features.spread if market_features.spread is not None else 0.0
    volume = market_features.volume if market_features.volume is not None else 0.0

    return [
        ob_strength,
        fvg_size,
        liquidity_sweep,
        htf_alignment,
        float(volatility),
        float(spread),
        float(volume),
        float(session_timing),
        float(rr_ratio),
        float(distance_to_liquidity),
    ]
