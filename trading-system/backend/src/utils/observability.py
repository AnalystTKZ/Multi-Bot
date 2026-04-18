import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


# Configurable via DEBUG_LOG_PATH env var. If empty or unset, debug file writing is disabled.
_debug_log_path_str = os.getenv("DEBUG_LOG_PATH", "")
DEBUG_LOG_PATH: Optional[Path] = Path(_debug_log_path_str) if _debug_log_path_str else None


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "service": getattr(record, "service", "backend"),
            "module": getattr(record, "module_name", record.name),
            "correlation_id": getattr(record, "correlation_id", None),
            "strategy_id": getattr(record, "strategy_id", None),
            "event_type": getattr(record, "event_type", None),
            "message": record.getMessage(),
            "data": getattr(record, "data", {}),
        }
        return json.dumps(payload, default=str)


def configure_json_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if any(getattr(handler, "_mb_json_logger", False) for handler in root.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    handler._mb_json_logger = True  # type: ignore[attr-defined]
    root.handlers = [handler]
    root.setLevel(level)


def ensure_correlation_id(value: Optional[str]) -> str:
    return value or str(uuid.uuid4())


def log_event(
    logger: logging.Logger,
    level: str,
    *,
    module: str,
    event_type: str,
    message: str,
    correlation_id: Optional[str],
    strategy_id: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
) -> None:
    getattr(logger, level.lower())(
        message,
        extra={
            "service": "backend",
            "module_name": module,
            "correlation_id": correlation_id,
            "strategy_id": strategy_id,
            "event_type": event_type,
            "data": data or {},
        },
    )


def runtime_debug_log(
    *,
    run_id: str,
    hypothesis_id: str,
    location: str,
    message: str,
    data: Dict[str, Any],
) -> None:
    if not DEBUG_LOG_PATH:
        return
    payload = {
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(datetime.now(tz=timezone.utc).timestamp() * 1000),
    }
    try:
        DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        return
