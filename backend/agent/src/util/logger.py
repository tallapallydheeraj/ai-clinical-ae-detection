import logging
import json
import os
from datetime import datetime, timezone


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)
        for attr in ("request_id", "user_id", "component"):
            if hasattr(record, attr):
                base[attr] = getattr(record, attr)
        return json.dumps(base, ensure_ascii=False)


def _configure_root():
    if getattr(_configure_root, "_configured", False):
        return
    level = os.getenv("AE_LOG_LEVEL", os.getenv("LOG_LEVEL", "INFO")).upper()
    root = logging.getLogger()
    root.handlers.clear()
    h = logging.StreamHandler()
    h.setFormatter(_JsonFormatter())
    root.addHandler(h)
    root.setLevel(getattr(logging, level, logging.INFO))
    _configure_root._configured = True


def get_logger(name: str) -> logging.Logger:
    _configure_root()
    return logging.getLogger(name)


__all__ = ["get_logger"]
