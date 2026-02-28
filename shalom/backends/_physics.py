"""Shared physics constants and utilities for DFT backends.

This module is the **single source of truth** for backend-neutral constants,
conversion factors, and validation thresholds.  Both the VASP and QE
configuration systems import from here, and downstream modules (agents,
workflows, CLI) reference the same values — so changing a constant here
propagates everywhere automatically.

Constants provided
------------------
**Unit conversions** (energy / force):
    RY_TO_EV, HA_TO_EV, EV_TO_RY, RY_PER_BOHR_TO_EV_PER_ANG

**DFT / validation thresholds** (shared across all backends):
    DEFAULT_KPR              — k-point resolution (2π/Å)
    DEFAULT_NSCF_KPR         — NSCF/DOS k-point resolution (2π/Å)
    DEFAULT_BAND_NPOINTS     — k-points per band segment
    DEFAULT_VACUUM_THRESHOLD — 2D detection vacuum (Å)
    MIN_INTERATOMIC_DISTANCE — atomic overlap floor (Å)
    MAX_CELL_VOLUME          — structure validation ceiling (ų)
    FORCE_CONVERGENCE_THRESHOLD  — eV/Å
    ENTROPY_THRESHOLD_PER_ATOM   — eV/atom (1 meV)
    CONVERGENCE_THRESHOLD_PER_ATOM — eV/atom
    DEFAULT_TIMEOUT_SECONDS  — 24 h standard DFT timeout

**Config-loaded data** (from ``shalom/config/*.yaml``):
    MAGNETIC_ELEMENTS, DEFAULT_MAGMOM, HUBBARD_U_VALUES,
    ANION_ELEMENTS, _METALLIC_ELEMENTS

**Backend-specific constants** live in their own modules:
    - ``vasp_config.py``: ENCUT_MULTIPLIER_*, METAL_SIGMA, SLAB_THICKNESS_THRESHOLD
    - ``qe_config.py``:  ECUTWFC_*, ECUTRHO_*, DEGAUSS_*, NBND_*
    - ``runner.py``:     MIN_RECOVERY_TIMEOUT, STDERR_TAIL_CHARS

Physical references
-------------------
- Magnetic moments: high-spin ferromagnetic initialisation defaults.
- Hubbard U: Dudarev scheme, Wang et al. PRB 73, 195107 (2006), PBE-fitted.
- Metallic elements: periodic table classification for ISMEAR selection.
- K-point grid: VASPKIT convention (N_i = max(1, round(|b_i| * kpr / 2pi))).
"""

from __future__ import annotations

import logging
import math
from enum import Enum
from typing import Dict, List, Optional

from ase import Atoms

from shalom._config_loader import load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared Enums
# ---------------------------------------------------------------------------


class AccuracyLevel(str, Enum):
    """Accuracy level controlling convergence parameters (shared across backends)."""

    STANDARD = "standard"
    PRECISE = "precise"


# ---------------------------------------------------------------------------
# Magnetic Elements & MAGMOM Defaults
# ---------------------------------------------------------------------------

_mag_cfg = load_config("magnetic_elements")
DEFAULT_MAGMOM: Dict[str, float] = _mag_cfg["default_magmom"]
MAGNETIC_ELEMENTS: frozenset = frozenset(DEFAULT_MAGMOM.keys())


# ---------------------------------------------------------------------------
# Hubbard U Values — Dudarev Scheme
# Reference: Wang et al. PRB 73, 195107 (2006), PBE-fitted
# ---------------------------------------------------------------------------

_hub_cfg = load_config("hubbard_u")
HUBBARD_U_VALUES: Dict[str, tuple] = {
    k: (v["L"], v["U"], v["J"]) for k, v in _hub_cfg["values"].items()
}
ANION_ELEMENTS: frozenset = frozenset(_hub_cfg["anion_elements"])


# ---------------------------------------------------------------------------
# LMAXMIX Thresholds (VASP-specific but useful for orbital classification)
# ---------------------------------------------------------------------------

_LMAXMIX_D_THRESHOLD = 20   # Z > 20 (Sc onwards): d-electrons
_LMAXMIX_F_THRESHOLD = 56   # Z > 56 (La onwards): f-electrons


# ---------------------------------------------------------------------------
# Pure Metal Detection
# ---------------------------------------------------------------------------

_METALLIC_ELEMENTS: frozenset = frozenset(load_config("metallic_elements")["elements"])


# ---------------------------------------------------------------------------
# Unit conversion constants (single source of truth)
# ---------------------------------------------------------------------------

RY_TO_EV: float = 13.6057                      # Rydberg → eV
HA_TO_EV: float = 27.2114                      # Hartree → eV
EV_TO_RY: float = 1.0 / RY_TO_EV              # eV → Rydberg
RY_PER_BOHR_TO_EV_PER_ANG: float = 25.7112    # force unit: Ry/Bohr → eV/Ang


# ---------------------------------------------------------------------------
# Shared DFT / validation thresholds (single source of truth)
# ---------------------------------------------------------------------------

DEFAULT_KPR: float = 30.0                        # k-point resolution (2π/Å)
DEFAULT_NSCF_KPR: float = 20.0                   # NSCF/DOS k-point resolution (2π/Å)
DEFAULT_BAND_NPOINTS: int = 40                    # k-points per band segment
DEFAULT_VACUUM_THRESHOLD: float = 5.0             # 2D detection vacuum (Å)
MIN_INTERATOMIC_DISTANCE: float = 0.8             # atomic overlap floor (Å)
MAX_CELL_VOLUME: float = 10_000.0                 # structure validation ceiling (ų)
FORCE_CONVERGENCE_THRESHOLD: float = 0.02         # eV/Å
ENTROPY_THRESHOLD_PER_ATOM: float = 0.001         # eV/atom (1 meV)
CONVERGENCE_THRESHOLD_PER_ATOM: float = 1e-3      # eV/atom
DEFAULT_TIMEOUT_SECONDS: int = 86400              # 24 hours


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _get_lmaxmix(atomic_numbers: List[int]) -> Optional[int]:
    """Determine LMAXMIX from atomic numbers in the structure."""
    if not atomic_numbers:
        return None
    max_z = max(atomic_numbers)
    if max_z > _LMAXMIX_F_THRESHOLD:
        return 6
    if max_z > _LMAXMIX_D_THRESHOLD:
        return 4
    return None


def _is_pure_metal(elements: List[str]) -> bool:
    """Check if all elements are metallic (no anions/non-metals)."""
    return all(el in _METALLIC_ELEMENTS for el in elements)


def detect_2d(atoms: Atoms, vacuum_threshold: float = DEFAULT_VACUUM_THRESHOLD) -> bool:
    """Detect whether a structure is 2D based on vacuum in the z-direction.

    Args:
        atoms: ASE Atoms object.
        vacuum_threshold: Minimum vacuum thickness (Angstrom) to classify as 2D.

    Returns:
        True if the structure has z-vacuum exceeding the threshold.
    """
    if len(atoms) == 0:
        return False
    cell = atoms.get_cell()
    c_height = cell[2][2]
    if c_height <= 0:
        return False
    positions = atoms.positions
    z_positions = positions[:, 2]
    slab_thickness = z_positions.max() - z_positions.min()
    vacuum = c_height - slab_thickness
    return bool(vacuum > vacuum_threshold)


def compute_kpoints_grid(
    atoms: Atoms, kpr: float = DEFAULT_KPR, is_2d: bool = False,
) -> List[int]:
    """Compute Gamma-centered k-point grid using VASPKIT convention.

    N_i = max(1, round(|b_i| * kpr / (2*pi)))

    For 2D structures, grid[2] is hardcoded to 1.

    Args:
        atoms: ASE Atoms object.
        kpr: K-point resolution in 2*pi/Angstrom.
        is_2d: Whether the structure is 2D.

    Returns:
        [Nx, Ny, Nz] grid.
    """
    if len(atoms) == 0 or abs(atoms.cell.volume) < 1e-10:
        return [1, 1, 1]
    reciprocal = atoms.cell.reciprocal() * 2 * math.pi
    grid = []
    for i in range(3):
        b_length = float((reciprocal[i] ** 2).sum() ** 0.5)
        n = max(1, round(b_length * kpr / (2 * math.pi)))
        grid.append(n)
    if is_2d:
        grid[2] = 1
    return grid
