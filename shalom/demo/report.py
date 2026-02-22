"""JSON report generation for demo runs."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def save_report(
    result: Any,
    tracker: Any,
    filepath: str = "demo_report.json",
) -> Optional[str]:
    """Save a JSON report combining pipeline result and cost summary.

    Args:
        result: PipelineResult from the pipeline run.
        tracker: CostTracker with accumulated usage data.
        filepath: Output path for the JSON report.

    Returns:
        The absolute path of the saved report, or None on failure.
    """
    try:
        report: Dict[str, Any] = {
            "pipeline_result": json.loads(result.model_dump_json(indent=2)),
            "cost_summary": tracker.summary(),
        }
        path = Path(filepath)
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        logger.info("Demo report saved to '%s'.", path.resolve())
        return str(path.resolve())
    except Exception as e:
        logger.warning("Failed to save demo report: %s", e)
        return None
