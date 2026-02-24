"""Audit logging for SHALOM pipeline and DFT executions.

Logs security-relevant events (LLM API calls, DFT executions, pipeline runs)
to a file when ``SHALOM_AUDIT_LOG`` environment variable is set.

Usage::

    export SHALOM_AUDIT_LOG=~/.shalom/audit.log
    python -m shalom pipeline "Find HER catalyst"
    # → audit entries written to ~/.shalom/audit.log

Each log line is a JSON object with timestamp, action, and details.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

_audit_logger: Optional[logging.Logger] = None
_initialized = False


def _get_audit_logger() -> Optional[logging.Logger]:
    """Return the audit file logger, or None if SHALOM_AUDIT_LOG is not set."""
    global _audit_logger, _initialized
    if _initialized:
        return _audit_logger

    _initialized = True
    log_path = os.environ.get("SHALOM_AUDIT_LOG")
    if not log_path:
        return None

    log_path = os.path.expanduser(log_path)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    _audit_logger = logging.getLogger("shalom.audit")
    _audit_logger.setLevel(logging.INFO)
    _audit_logger.propagate = False

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    _audit_logger.addHandler(handler)

    return _audit_logger


def log_event(action: str, details: Optional[Dict[str, Any]] = None) -> None:
    """Log an audit event if ``SHALOM_AUDIT_LOG`` is configured.

    Args:
        action: Event type (e.g. ``"llm_call"``, ``"pipeline_start"``,
            ``"dft_execution"``).
        details: Additional context (provider, model, objective, etc.).
    """
    logger = _get_audit_logger()
    if logger is None:
        return

    entry: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
    }
    if details:
        entry["details"] = details

    logger.info(json.dumps(entry, default=str))
