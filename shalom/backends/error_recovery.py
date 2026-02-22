"""VASP error recovery engine.

Provides pattern-based error detection, progressive correction escalation,
false convergence detection, and ionic sloshing analysis. Inspired by
custodian but tailored for SHALOM's agentic pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from shalom._config_loader import load_config


# ---------------------------------------------------------------------------
# Error severity
# ---------------------------------------------------------------------------

class ErrorSeverity(str, Enum):
    CORRECTABLE = "correctable"
    FATAL = "fatal"


# ---------------------------------------------------------------------------
# VASP Error
# ---------------------------------------------------------------------------

@dataclass
class VASPError:
    """A detected VASP error."""
    error_type: str
    severity: ErrorSeverity
    matched_text: str = ""


# ---------------------------------------------------------------------------
# Correction
# ---------------------------------------------------------------------------

@dataclass
class Correction:
    """A correction to apply to INCAR settings."""
    error_type: str
    step: int
    incar_updates: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


# ---------------------------------------------------------------------------
# Error Pattern Registry (loaded from config/error_patterns.yaml)
# ---------------------------------------------------------------------------

_err_cfg: List[Dict[str, str]] = load_config("error_patterns")  # type: ignore[assignment]
ERROR_PATTERNS: List[tuple] = [
    (p["pattern"], p["type"], ErrorSeverity(p["severity"])) for p in _err_cfg
]


# ---------------------------------------------------------------------------
# Correction Strategies (loaded from config/correction_strategies.yaml)
# ---------------------------------------------------------------------------

CORRECTION_STRATEGIES: Dict[str, List[Dict[str, Any]]] = load_config("correction_strategies")


# ---------------------------------------------------------------------------
# ErrorRecoveryEngine
# ---------------------------------------------------------------------------

class ErrorRecoveryEngine:
    """Detects VASP errors, suggests progressive corrections, and tracks history.

    Usage::

        engine = ErrorRecoveryEngine()
        errors = engine.scan_for_errors(outcar_text)
        if errors:
            correction = engine.get_correction(errors[0])
            if correction:
                # Apply correction.incar_updates to INCAR
                ...
        # Check history
        history = engine.correction_history
    """

    def __init__(self) -> None:
        self._escalation_counters: Dict[str, int] = {}
        self._correction_history: List[Dict[str, Any]] = []

    @property
    def correction_history(self) -> List[Dict[str, Any]]:
        """JSON-serializable correction history."""
        return list(self._correction_history)

    def scan_for_errors(self, output_text: str) -> List[VASPError]:
        """Scan OUTCAR text for known error patterns.

        Args:
            output_text: Full OUTCAR content.

        Returns:
            List of detected VASPErrors, ordered by detection.
        """
        errors: List[VASPError] = []
        seen_types: set = set()
        for pattern, error_type, severity in ERROR_PATTERNS:
            if pattern in output_text and error_type not in seen_types:
                errors.append(VASPError(
                    error_type=error_type,
                    severity=severity,
                    matched_text=pattern,
                ))
                seen_types.add(error_type)
        return errors

    def get_correction(self, error: VASPError) -> Optional[Correction]:
        """Get the next correction for an error, escalating progressively.

        Each call for the same error_type advances to the next strategy step.

        Args:
            error: The VASPError to correct.

        Returns:
            A Correction with INCAR updates, or None if strategies are exhausted.
        """
        if error.severity == ErrorSeverity.FATAL:
            self._correction_history.append({
                "error_type": error.error_type,
                "action": "fatal_no_correction",
            })
            return None

        strategies = CORRECTION_STRATEGIES.get(error.error_type)
        if not strategies:
            return None

        step = self._escalation_counters.get(error.error_type, 0)
        if step >= len(strategies):
            self._correction_history.append({
                "error_type": error.error_type,
                "action": "strategies_exhausted",
                "step": step,
            })
            return None

        incar_updates = dict(strategies[step])
        correction = Correction(
            error_type=error.error_type,
            step=step,
            incar_updates=incar_updates,
            description=f"{error.error_type} correction step {step}",
        )
        self._escalation_counters[error.error_type] = step + 1
        self._correction_history.append({
            "error_type": error.error_type,
            "action": "correction_applied",
            "step": step,
            "incar_updates": incar_updates,
        })
        return correction

    # ------------------------------------------------------------------
    # False convergence detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_false_convergence(
        energies: List[float],
        forces_max_history: Optional[List[float]] = None,
        ediffg: float = -0.02,
        window: int = 10,
        oscillation_threshold: float = 0.1,
    ) -> bool:
        """Detect false convergence via energy + force cross-validation.

        False convergence: energy appears converged but forces oscillate
        near the EDIFFG threshold instead of monotonically decreasing.

        Args:
            energies: Ionic step energies.
            forces_max_history: Max force at each ionic step.
            ediffg: EDIFFG criterion (negative = force-based).
            window: Number of recent steps to analyze.
            oscillation_threshold: Fraction of mean for oscillation detection.

        Returns:
            True if false convergence detected.
        """
        if not energies or len(energies) < 3:
            return False
        if not forces_max_history or len(forces_max_history) < 5:
            return False

        # Check energy convergence (last two steps very close)
        energy_converged = abs(energies[-1] - energies[-2]) < 1e-4

        # Check force oscillation in the last N steps
        recent_forces = forces_max_history[-window:]
        if len(recent_forces) < 5:
            return False

        force_oscillating = _check_force_oscillation(
            recent_forces, threshold=oscillation_threshold,
        )
        return energy_converged and force_oscillating

    # ------------------------------------------------------------------
    # Ionic sloshing detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_ionic_sloshing(
        forces_max_history: List[float],
        window: int = 6,
    ) -> bool:
        """Detect ionic sloshing — forces alternate up/down repeatedly.

        Sloshing = force max alternates increasing/decreasing for >= window steps.

        Args:
            forces_max_history: Max force at each ionic step.
            window: Minimum alternation length to classify as sloshing.

        Returns:
            True if sloshing detected.
        """
        if len(forces_max_history) < window:
            return False

        recent = forces_max_history[-window:]
        alternation_count = 0
        for i in range(2, len(recent)):
            prev_delta = recent[i - 1] - recent[i - 2]
            curr_delta = recent[i] - recent[i - 1]
            if prev_delta * curr_delta < 0:  # Sign change
                alternation_count += 1

        # If most transitions alternate, it's sloshing
        return alternation_count >= (window - 2) * 0.7


# ---------------------------------------------------------------------------
# Helper: force oscillation check
# ---------------------------------------------------------------------------

def _check_force_oscillation(
    forces: List[float], threshold: float = 0.1,
) -> bool:
    """Check if forces oscillate around their mean beyond threshold.

    Args:
        forces: Recent max force values.
        threshold: Relative oscillation threshold (fraction of mean).

    Returns:
        True if oscillation exceeds threshold.
    """
    if len(forces) < 3:
        return False

    mean_f = sum(forces) / len(forces)
    if mean_f <= 0:
        return False

    # Count direction changes (up→down or down→up)
    direction_changes = 0
    for i in range(2, len(forces)):
        delta_prev = forces[i - 1] - forces[i - 2]
        delta_curr = forces[i] - forces[i - 1]
        if delta_prev * delta_curr < 0:
            direction_changes += 1

    # Check if standard deviation is significant relative to mean
    variance = sum((f - mean_f) ** 2 for f in forces) / len(forces)
    std_dev = math.sqrt(variance)
    relative_oscillation = std_dev / mean_f

    # Oscillation: enough direction changes AND significant variance
    min_changes = max(2, len(forces) // 3)
    return direction_changes >= min_changes and relative_oscillation > threshold
