import logging
import json
import os
from datetime import datetime

_CONFIGURED = False


def _configure():
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    json_mode = os.getenv("LOG_JSON", "false").lower() == "true"

    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:  # type: ignore
            from datetime import timezone

            payload = {
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
            }
            env = os.getenv("ENVIRONMENT")
            if env:
                payload["env"] = env
            return json.dumps(payload)

    handler = logging.StreamHandler()
    if json_mode:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
    logging.basicConfig(level=getattr(logging, level, logging.INFO), handlers=[handler])
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    _configure()
    return logging.getLogger(name)
