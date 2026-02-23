"""Quantum ESPRESSO error recovery engine.

Provides pattern-based error detection, progressive correction escalation
with physics-aware strategies, S-matrix diagnostic branching, light-atom
dt safety, and quality warning propagation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from ase import Atoms
from ase.data import covalent_radii

from shalom._config_loader import load_config
from shalom.backends.error_recovery import ErrorSeverity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Light atom dt safety (Ry atomic units)
# ---------------------------------------------------------------------------

LIGHT_ATOM_DT: Dict[str, float] = {"H": 5.0, "Li": 10.0, "Be": 12.0}


def compute_safe_dt(atoms: Atoms) -> float:
    """Compute safe timestep for damped dynamics based on lightest element.

    QE default dt=20.0 Ry a.u. (~1 fs). Light atoms (H, Li) need smaller dt
    to prevent overshooting during damped molecular dynamics.

    Args:
        atoms: ASE Atoms object.

    Returns:
        Safe dt in Ry atomic units.
    """
    symbols = set(atoms.get_chemical_symbols())
    candidates = [LIGHT_ATOM_DT.get(s, 20.0) for s in symbols]
    return min(20.0, *candidates) if candidates else 20.0


# ---------------------------------------------------------------------------
# QE Error
# ---------------------------------------------------------------------------

@dataclass
class QEError:
    """A detected Quantum ESPRESSO error."""
    error_type: str
    severity: ErrorSeverity
    matched_text: str = ""


# ---------------------------------------------------------------------------
# QE Correction
# ---------------------------------------------------------------------------

@dataclass
class QECorrection:
    """A correction to apply to QE input configuration."""
    error_type: str
    step: int
    namelist_updates: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    quality_warnings: List[str] = field(default_factory=list)
    electron_maxstep_cap: Optional[int] = None
    nstep_cap: Optional[int] = None
    rollback_geometry: bool = False


# ---------------------------------------------------------------------------
# Error Pattern Registry (loaded from config/qe_error_patterns.yaml)
# ---------------------------------------------------------------------------

_qe_err_cfg: List[Dict[str, str]] = load_config("qe_error_patterns")  # type: ignore[assignment]
QE_ERROR_PATTERNS: List[tuple] = [
    (p["pattern"], p["type"], ErrorSeverity(p["severity"])) for p in _qe_err_cfg
]

# ---------------------------------------------------------------------------
# Correction Strategies (loaded from config/qe_correction_strategies.yaml)
# ---------------------------------------------------------------------------

QE_CORRECTION_STRATEGIES: Dict[str, List[Dict[str, Any]]] = load_config(
    "qe_correction_strategies"
)


# ---------------------------------------------------------------------------
# S-Matrix Diagnostic
# ---------------------------------------------------------------------------

MIN_DISTANCE_THRESHOLD = 0.5  # Angstrom — absolute floor for atomic overlap


def check_atomic_overlap(atoms: Atoms, safety_factor: float = 0.4) -> bool:
    """Check if any atoms are unphysically close (overlap).

    Args:
        atoms: ASE Atoms object.
        safety_factor: Fraction of sum of covalent radii as threshold.

    Returns:
        True if atomic overlap detected.
    """
    if len(atoms) < 2:
        return False
    distances = atoms.get_all_distances(mic=True)
    np.fill_diagonal(distances, np.inf)
    min_dist = float(np.min(distances))

    # Species-dependent threshold
    idx = np.unravel_index(np.argmin(distances), distances.shape)
    z_i = atoms[idx[0]].number
    z_j = atoms[idx[1]].number
    sum_cov = covalent_radii[z_i] + covalent_radii[z_j]
    threshold = max(safety_factor * sum_cov, MIN_DISTANCE_THRESHOLD)

    if min_dist < threshold:
        logger.warning(
            "Atomic overlap detected: min distance %.3f Å < threshold %.3f Å "
            "(atoms %d-%d)", min_dist, threshold, idx[0], idx[1],
        )
        return True
    return False


# ---------------------------------------------------------------------------
# QEErrorRecoveryEngine
# ---------------------------------------------------------------------------

class QEErrorRecoveryEngine:
    """Detects QE errors, suggests progressive corrections, and tracks history.

    Features beyond VASP ErrorRecoveryEngine:
    - S-matrix diagnostic branching (overlap vs basis-set path)
    - Light atom dt safety for damped dynamics
    - Quality warning propagation for physics-compromising corrections
    - Per-step iteration caps (fast-fail for local-TF)

    Usage::

        engine = QEErrorRecoveryEngine()
        errors = engine.scan_for_errors(pw_out_text)
        if errors:
            correction = engine.get_correction(errors[0], atoms=atoms)
            if correction:
                warnings = engine.apply_correction_to_config(config, correction)
    """

    def __init__(self) -> None:
        self._escalation_counters: Dict[str, int] = {}
        self._correction_history: List[Dict[str, Any]] = []
        self._s_matrix_path: Optional[str] = None  # "overlap" or "basis_set"

    @property
    def correction_history(self) -> List[Dict[str, Any]]:
        """JSON-serializable correction history."""
        return list(self._correction_history)

    def scan_for_errors(self, output_text: str) -> List[QEError]:
        """Scan pw.out text for known error patterns.

        Args:
            output_text: Full pw.out content.

        Returns:
            List of detected QEErrors, ordered by pattern priority.
        """
        errors: List[QEError] = []
        seen_types: set = set()
        for pattern, error_type, severity in QE_ERROR_PATTERNS:
            if pattern in output_text and error_type not in seen_types:
                errors.append(QEError(
                    error_type=error_type,
                    severity=severity,
                    matched_text=pattern,
                ))
                seen_types.add(error_type)
        return errors

    def diagnose_s_matrix(self, atoms: Optional[Atoms]) -> str:
        """Diagnose S-matrix error: overlap vs basis-set issue.

        Args:
            atoms: Atoms from the failed geometry (last ionic step).

        Returns:
            "overlap" or "basis_set".
        """
        if atoms is not None and check_atomic_overlap(atoms):
            self._s_matrix_path = "overlap"
            return "overlap"
        self._s_matrix_path = "basis_set"
        return "basis_set"

    def get_correction(
        self, error: QEError, atoms: Optional[Atoms] = None,
    ) -> Optional[QECorrection]:
        """Get the next correction for an error, escalating progressively.

        Args:
            error: The QEError to correct.
            atoms: ASE Atoms (needed for S_MATRIX diagnostic and light atom dt).

        Returns:
            A QECorrection, or None if strategies are exhausted.
        """
        if error.severity == ErrorSeverity.FATAL:
            self._correction_history.append({
                "error_type": error.error_type,
                "action": "fatal_no_correction",
            })
            return None

        # S_MATRIX: diagnostic branching
        if error.error_type == "QE_S_MATRIX":
            return self._get_s_matrix_correction(error, atoms)

        strategies = QE_CORRECTION_STRATEGIES.get(error.error_type)
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

        raw = dict(strategies[step])
        correction = self._build_correction(error.error_type, step, raw, atoms)
        self._escalation_counters[error.error_type] = step + 1
        self._record_history(error.error_type, step, correction)
        return correction

    def _get_s_matrix_correction(
        self, error: QEError, atoms: Optional[Atoms],
    ) -> Optional[QECorrection]:
        """Handle S_MATRIX with diagnostic branching."""
        if self._s_matrix_path is None:
            self.diagnose_s_matrix(atoms)

        strategies = QE_CORRECTION_STRATEGIES.get("QE_S_MATRIX", [])
        path = self._s_matrix_path or "basis_set"

        # Overlap path: steps 0-1 in YAML (rollback + trust_radius, rollback + damp)
        # Basis-set path: steps 2-3 in YAML (diag=cg, diago_david_ndim=4)
        if path == "overlap":
            offset = 0
            path_strategies = strategies[:2]
        else:
            offset = 2
            path_strategies = strategies[2:]

        key = f"QE_S_MATRIX_{path}"
        step = self._escalation_counters.get(key, 0)
        if step >= len(path_strategies):
            self._correction_history.append({
                "error_type": error.error_type,
                "action": "strategies_exhausted",
                "path": path,
                "step": step,
            })
            return None

        raw = dict(path_strategies[step])
        correction = self._build_correction(
            error.error_type, offset + step, raw, atoms,
        )
        self._escalation_counters[key] = step + 1
        self._record_history(error.error_type, offset + step, correction, path=path)
        return correction

    def _build_correction(
        self, error_type: str, step: int, raw: Dict[str, Any],
        atoms: Optional[Atoms],
    ) -> QECorrection:
        """Build a QECorrection from raw strategy dict."""
        # Extract meta-keys
        quality_warning = raw.pop("_quality_warning", None)
        electron_maxstep_cap = raw.pop("_electron_maxstep_cap", None)
        rollback = raw.pop("_rollback_geometry", False)
        needs_atoms = raw.pop("_needs_atoms", False)

        # Light atom dt safety: auto-compute dt for damp corrections
        if needs_atoms and atoms is not None:
            if "ions.ion_dynamics" in raw and raw["ions.ion_dynamics"] == "damp":
                safe_dt = compute_safe_dt(atoms)
                raw["ions.dt"] = safe_dt

        warnings: List[str] = []
        if quality_warning:
            warnings.append(quality_warning)

        return QECorrection(
            error_type=error_type,
            step=step,
            namelist_updates=raw,
            description=f"{error_type} correction step {step}",
            quality_warnings=warnings,
            electron_maxstep_cap=electron_maxstep_cap,
            rollback_geometry=rollback,
        )

    def _record_history(
        self, error_type: str, step: int, correction: QECorrection,
        path: Optional[str] = None,
    ) -> None:
        """Record correction in history."""
        entry: Dict[str, Any] = {
            "error_type": error_type,
            "action": "correction_applied",
            "step": step,
            "namelist_updates": correction.namelist_updates,
        }
        if correction.quality_warnings:
            entry["quality_warnings"] = correction.quality_warnings
        if correction.rollback_geometry:
            entry["rollback_geometry"] = True
        if path:
            entry["path"] = path
        self._correction_history.append(entry)

    @staticmethod
    def apply_correction_to_config(
        config: Any, correction: QECorrection,
    ) -> List[str]:
        """Apply a correction to a QEInputConfig in-place.

        Handles dot-notation keys (e.g., "electrons.mixing_beta" → config.electrons["mixing_beta"]),
        max-semantics for monotonic params, and vc-relax downgrade.

        Args:
            config: QEInputConfig instance (modified in-place).
            correction: The QECorrection to apply.

        Returns:
            List of quality warning strings.
        """
        # Max-semantics parameters: correction should not DECREASE these
        MAX_SEMANTICS = {"ecutwfc", "ecutrho", "electron_maxstep", "nstep"}

        for dotkey, value in correction.namelist_updates.items():
            parts = dotkey.split(".", 1)
            if len(parts) != 2:
                logger.warning("Ignoring non-dotted key: %s", dotkey)
                continue

            namelist, param = parts

            # Special: vc-relax downgrade
            if namelist == "control" and param == "calculation" and value == "relax":
                _downgrade_vc_relax(config)
                continue

            # Get namelist dict from config
            nl_dict = getattr(config, namelist, None)
            if nl_dict is None:
                # Create the namelist dict if it doesn't exist
                setattr(config, namelist, {})
                nl_dict = getattr(config, namelist)

            if not isinstance(nl_dict, dict):
                logger.warning("Config.%s is not a dict, skipping", namelist)
                continue

            # Max-semantics: don't decrease monotonic params
            if param in MAX_SEMANTICS and param in nl_dict:
                current = nl_dict[param]
                if isinstance(current, (int, float)) and isinstance(value, (int, float)):
                    value = max(current, value)

            nl_dict[param] = value

        # Apply electron_maxstep cap (overrides any existing value)
        if correction.electron_maxstep_cap is not None:
            if hasattr(config, "electrons") and isinstance(config.electrons, dict):
                config.electrons["electron_maxstep"] = correction.electron_maxstep_cap

        return list(correction.quality_warnings)


def _downgrade_vc_relax(config: Any) -> None:
    """Downgrade a vc-relax config to relax."""
    from shalom.backends.qe_config import QECalculationType

    logger.warning(
        "Downgrading vc-relax to relax due to cell distortion. "
        "Lattice optimization abandoned."
    )
    if hasattr(config, "control") and isinstance(config.control, dict):
        config.control["calculation"] = "relax"
    if hasattr(config, "calc_type"):
        config.calc_type = QECalculationType.RELAX
    if hasattr(config, "cell"):
        config.cell = {}
