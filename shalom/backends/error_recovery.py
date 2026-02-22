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
# Error Pattern Registry
# ---------------------------------------------------------------------------

# (pattern_substring, error_type, severity)
ERROR_PATTERNS: List[tuple] = [
    ("ZBRENT: fatal internal in bracketing", "ZBRENT", ErrorSeverity.CORRECTABLE),
    ("ZBRENT: fatal error in bracketing", "ZBRENT", ErrorSeverity.CORRECTABLE),
    ("BRMIX: very serious problems", "BRMIX", ErrorSeverity.CORRECTABLE),
    ("EDDDAV: sub-space matrix is not hermitian", "EDDDAV", ErrorSeverity.CORRECTABLE),
    ("NELM reached", "SCF_UNCONVERGED", ErrorSeverity.CORRECTABLE),
    ("WARNING: DENTET", "DENTET", ErrorSeverity.CORRECTABLE),
    ("POSMAP internal error", "POSMAP", ErrorSeverity.CORRECTABLE),
    ("VERY BAD NEWS! internal error in subroutine SGRCON", "SGRCON", ErrorSeverity.FATAL),
    ("PRICEL: found a more primitive cell", "PRICEL", ErrorSeverity.CORRECTABLE),
    ("Error EDDDAV: Call to ZHEGV failed", "ZHEGV", ErrorSeverity.CORRECTABLE),
    ("EDDRMM: call to GR_CGGR failed", "EDDRMM", ErrorSeverity.CORRECTABLE),
]


# ---------------------------------------------------------------------------
# Correction Strategies — Progressive Escalation
# ---------------------------------------------------------------------------

CORRECTION_STRATEGIES: Dict[str, List[Dict[str, Any]]] = {
    "SCF_UNCONVERGED": [
        # Step 1: Increase NELM slightly
        {"NELM": 200},
        # Step 2: Metallization defense — switch smearing
        {"ISMEAR": 1, "SIGMA": 0.1},
        # Step 3: Damped algorithm (more robust than Normal)
        {"ALGO": "Damped", "NELM": 200, "TIME": 0.5},
        # Step 4: Suppress charge sloshing
        {"AMIX": 0.05, "BMIX": 0.001, "NELM": 300},
        # Step 5: Last resort — very aggressive
        {"ALGO": "All", "NELM": 500, "AMIX": 0.02},
    ],
    "BRMIX": [
        # Root causes: (a) distorted geometry, (b) bandgap closing, (c) magnetic
        # Step 1: Metallization defense
        {"ISMEAR": 1, "SIGMA": 0.1},
        # Step 2: Stabilize charge mixing
        {"AMIX": 0.1, "BMIX": 0.01},
        # Step 3: Damped
        {"ALGO": "Damped", "AMIX": 0.05, "BMIX": 0.001},
        # Step 4: Alternative mixing
        {"IMIX": 1, "AMIX": 0.02, "BMIX": 0.001, "AMIX_MAG": 0.1},
        # Step 5: Last resort
        {"ALGO": "All", "AMIX": 0.02, "BMIX": 0.001},
    ],
    "ZBRENT": [
        # EDIFF tightening is NOT effective for line search failure.
        # Go directly to POTIM reduction (root cause).
        {"POTIM": 0.2},
        # Step 2: Switch optimizer CG → DIIS
        {"IBRION": 1, "POTIM": 0.3},
        # Step 3: Damped MD
        {"IBRION": 3, "POTIM": 0.05, "SMASS": 0.5},
    ],
    "IONIC_SLOSHING": [
        # Force sign oscillation detected
        {"POTIM": 0.2},
        {"IBRION": 1},
        {"IBRION": 3, "POTIM": 0.05, "SMASS": 0.5},
    ],
    "EDDDAV": [
        {"ALGO": "Normal"},
        {"ALGO": "Damped", "TIME": 0.5},
        {"ALGO": "All"},
    ],
    "ZHEGV": [
        {"ALGO": "Exact"},
        {"ALGO": "All", "NELM": 200},
    ],
    "EDDRMM": [
        {"ALGO": "Normal"},
        {"ALGO": "Damped", "TIME": 0.5},
    ],
    "DENTET": [
        {"ISMEAR": 0, "SIGMA": 0.05},
    ],
    "PRICEL": [
        {"SYMPREC": 1e-8},
    ],
    "POSMAP": [
        {"SYMPREC": 1e-6},
    ],
}


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
        for i in range(1, len(recent)):
            if i == 1:
                continue
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
