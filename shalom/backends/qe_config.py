"""Quantum ESPRESSO input configuration and preset system.

Mirrors the VASP config architecture (``vasp_config.py``): composable presets,
structure-aware auto-detection, dynamic ecutwfc/ecutrho allocation from SSSP
metadata, and K-point grid computation.

Pseudopotential library:
    SSSP Efficiency v1.3.0 (PBE)
    Ref: Prandini et al. npj Comput. Mater. 4, 72 (2018).
    Ref: Bosoni et al. Nat. Rev. Phys. 6, 45 (2024).

Hubbard U:
    Wang et al. PRB 73, 195107 (2006), PBE-fitted.
    NOTE: These U values were derived with PAW projectors in VASP. When used
    with QE's ortho-atomic projectors, quantitative differences are expected.
    For rigorous studies of strongly-correlated systems, re-derive U with hp.x.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from ase import Atoms

from shalom._config_loader import load_config
from shalom.backends._physics import (
    AccuracyLevel,
    MAGNETIC_ELEMENTS,
    DEFAULT_MAGMOM,
    HUBBARD_U_VALUES,
    ANION_ELEMENTS,
    _is_pure_metal,
    detect_2d,
    compute_kpoints_grid,
)

logger = logging.getLogger(__name__)

# Unit conversion: 1 Ry = 13.6057 eV
RY_TO_EV = 13.6057
EV_TO_RY = 1.0 / RY_TO_EV


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class QECalculationType(str, Enum):
    """Supported QE pw.x calculation types."""

    SCF = "scf"
    RELAX = "relax"
    VC_RELAX = "vc-relax"
    BANDS = "bands"
    NSCF = "nscf"


# VASP calc_type string → QE calc_type mapping.
# Key correction: VASP "relaxation" (ISIF=3, cell+atoms) → QE vc-relax, not relax.
VASP_TO_QE_CALC_MAP: Dict[str, QECalculationType] = {
    "relaxation": QECalculationType.VC_RELAX,
    "static": QECalculationType.SCF,
    "band_structure": QECalculationType.BANDS,
    "dos": QECalculationType.NSCF,
}


# ---------------------------------------------------------------------------
# KPoints Config
# ---------------------------------------------------------------------------


@dataclass
class QEKPointsConfig:
    """K-point mesh configuration for QE.

    Attributes:
        mode: K_POINTS card mode — automatic, gamma, or crystal_b.
        grid: Explicit grid [Nx, Ny, Nz].
        shift: Grid offset [sx, sy, sz] (0 or 1).
    """

    mode: Literal["automatic", "gamma", "crystal_b"] = "automatic"
    grid: Optional[List[int]] = None
    shift: List[int] = field(default_factory=lambda: [0, 0, 0])


# ---------------------------------------------------------------------------
# QE Input Config
# ---------------------------------------------------------------------------


@dataclass
class QEInputConfig:
    """Complete QE pw.x input configuration.

    Holds namelist dicts (&CONTROL, &SYSTEM, &ELECTRONS, &IONS, &CELL),
    user overrides, K-point config, and pseudopotential mapping.
    """

    calc_type: QECalculationType = QECalculationType.SCF
    accuracy: AccuracyLevel = AccuracyLevel.STANDARD
    control: Dict[str, Any] = field(default_factory=dict)
    system: Dict[str, Any] = field(default_factory=dict)
    electrons: Dict[str, Any] = field(default_factory=dict)
    ions: Dict[str, Any] = field(default_factory=dict)
    cell: Dict[str, Any] = field(default_factory=dict)
    user_settings: Dict[str, Any] = field(default_factory=dict)
    kpoints: QEKPointsConfig = field(default_factory=QEKPointsConfig)
    pseudo_dir: str = "./"
    pseudo_map: Dict[str, str] = field(default_factory=dict)
    functional: str = "PBE"
    is_2d: bool = False

    def get_merged_namelists(self) -> Dict[str, Dict[str, Any]]:
        """Return namelists with user_settings overrides applied.

        user_settings keys support dot notation (e.g. ``"system.ecutwfc"``).
        Keys without a dot are placed in &SYSTEM by default.
        Only &IONS is included for relax/vc-relax; &CELL only for vc-relax.
        """
        result: Dict[str, Dict[str, Any]] = {
            "CONTROL": dict(self.control),
            "SYSTEM": dict(self.system),
            "ELECTRONS": dict(self.electrons),
        }
        if self.calc_type in (QECalculationType.RELAX, QECalculationType.VC_RELAX):
            result["IONS"] = dict(self.ions)
        if self.calc_type == QECalculationType.VC_RELAX:
            result["CELL"] = dict(self.cell)

        for key, val in self.user_settings.items():
            if "." in key:
                ns, param = key.split(".", 1)
                ns_upper = ns.upper()
                if ns_upper in result:
                    result[ns_upper][param] = val
            else:
                result.setdefault("SYSTEM", {})[key] = val
        return result


# ---------------------------------------------------------------------------
# SSSP Pseudopotential Metadata
# ---------------------------------------------------------------------------

_sssp_cfg = load_config("sssp_metadata")
SSSP_ELEMENTS: Dict[str, Dict[str, Any]] = _sssp_cfg["elements"]


def get_pseudo_filename(element: str) -> str:
    """Return the SSSP pseudopotential filename for an element."""
    entry = SSSP_ELEMENTS.get(element)
    if entry is not None:
        return entry["pseudo"]
    return f"{element}.UPF"


def get_z_valence(element: str) -> int:
    """Return the number of valence electrons from SSSP metadata."""
    entry = SSSP_ELEMENTS.get(element)
    if entry is not None:
        return entry["z_valence"]
    return 1  # conservative fallback


def compute_ecutwfc(elements: List[str], accuracy: AccuracyLevel) -> float:
    """Compute ecutwfc from SSSP per-element hints.

    STANDARD: max(hints), minimum 45 Ry.
    PRECISE:  max(hints) * 1.2, minimum 60 Ry.
    """
    hints = [SSSP_ELEMENTS[el]["ecutwfc"] for el in set(elements) if el in SSSP_ELEMENTS]
    if not hints:
        return 60.0 if accuracy == AccuracyLevel.PRECISE else 45.0
    max_hint = max(hints)
    if accuracy == AccuracyLevel.PRECISE:
        return max(60.0, math.ceil(max_hint * 1.2))
    return max(45.0, float(max_hint))


def compute_ecutrho(elements: List[str], accuracy: AccuracyLevel) -> float:
    """Compute ecutrho from SSSP per-element metadata (NOT blanket 8x).

    Uses max(SSSP ecutrho per element). This respects PP-type dependent ratios:
    NC → ~4x, USPP → ~8-12x, PAW → ~8-12x.
    """
    hints = [SSSP_ELEMENTS[el]["ecutrho"] for el in set(elements) if el in SSSP_ELEMENTS]
    if not hints:
        ecutwfc = compute_ecutwfc(elements, accuracy)
        return ecutwfc * 8.0  # fallback
    max_rho = max(hints)
    if accuracy == AccuracyLevel.PRECISE:
        return max(max_rho, math.ceil(max_rho * 1.2))
    return float(max_rho)


# ---------------------------------------------------------------------------
# Structure-Aware Auto-Detection
# ---------------------------------------------------------------------------


def detect_and_apply_qe_hints(
    atoms: Atoms, config: QEInputConfig,
) -> QEInputConfig:
    """Apply structure-aware auto-detection rules to a QEInputConfig.

    Modifies the config in-place:
    1. Magnetic elements → nspin=2, starting_magnetization = MAGMOM / z_valence
    2. GGA+U (PRECISE only) → lda_plus_u, Hubbard_U(i)
    3. 2D detection → assume_isolated='2D', vdw_corr, cell_dofree='2Dxy'
    4. Pure metal → smearing='methfessel-paxton', degauss=0.00735 Ry
    5. ecutwfc / ecutrho from SSSP
    6. K-points grid (shared compute_kpoints_grid)
    7. ibrav=0, nat, ntyp

    Args:
        atoms: ASE Atoms object.
        config: QEInputConfig to modify.

    Returns:
        The modified config (same object).
    """
    if len(atoms) == 0:
        return config

    symbols = list(atoms.get_chemical_symbols())
    unique_elements = list(set(symbols))
    # Species order preserving first occurrence (for QE type indexing)
    species_order = list(dict.fromkeys(symbols))

    # 1. Magnetic element detection
    has_magnetic = any(el in MAGNETIC_ELEMENTS for el in unique_elements)
    if has_magnetic:
        config.system["nspin"] = 2
        # starting_magnetization(i) = MAGMOM / z_valence
        # Uses z_valence from SSSP metadata, NOT /10.
        for i, el in enumerate(species_order, 1):
            if el in DEFAULT_MAGMOM:
                z_val = get_z_valence(el)
                start_mag = min(DEFAULT_MAGMOM[el] / z_val, 1.0)
                config.system[f"starting_magnetization({i})"] = round(start_mag, 4)

    # 2. GGA+U auto-detection (PRECISE only, same policy as VASP)
    has_anion = any(el in ANION_ELEMENTS for el in unique_elements)
    magnetic_tm = [el for el in unique_elements if el in HUBBARD_U_VALUES]
    if has_magnetic and has_anion and magnetic_tm:
        u_elements = [el for el in magnetic_tm if HUBBARD_U_VALUES[el][1] > 0.0]
        if u_elements and config.accuracy == AccuracyLevel.PRECISE:
            config.system["lda_plus_u"] = True
            for i, el in enumerate(species_order, 1):
                if el in HUBBARD_U_VALUES:
                    _, u_val, _ = HUBBARD_U_VALUES[el]
                    config.system[f"Hubbard_U({i})"] = u_val
            if config.functional != "PBE":
                logger.warning(
                    "GGA+U enabled with functional='%s', but Hubbard U "
                    "values are fitted for PBE only (Wang et al. PRB 73, "
                    "195107). Results may be unreliable.",
                    config.functional,
                )
        elif u_elements and config.accuracy != AccuracyLevel.PRECISE:
            logger.warning(
                "GGA+U disabled at STANDARD accuracy for elements %s. "
                "For transition metal oxides/chalcogenides, use accuracy='precise'.",
                u_elements,
            )

    # 3. 2D detection
    if detect_2d(atoms):
        config.is_2d = True
        config.system["assume_isolated"] = "2D"
        # DFT-D3 with BJ damping to match VASP IVDW=12
        # Ref: Grimme et al. J. Comput. Chem. 32, 1456 (2011)
        config.system["vdw_corr"] = "dft-d3"
        config.system["dftd3_version"] = 4  # BJ damping
        if config.calc_type == QECalculationType.VC_RELAX:
            config.cell["cell_dofree"] = "2Dxy"

    # 4. Occupation / smearing based on material class
    if _is_pure_metal(unique_elements):
        # Methfessel-Paxton order 1, degauss = VASP SIGMA(0.1 eV) / 13.6057
        config.system["occupations"] = "smearing"
        config.system["smearing"] = "methfessel-paxton"
        config.system["degauss"] = 0.00735
        config.electrons["mixing_beta"] = 0.4  # Lower for metallic SCF stability
    elif not has_magnetic:
        # Non-magnetic non-metal (semiconductor/insulator): reduce broadening
        # 0.02 eV / 13.6057 = 0.00147 Ry — negligible for wide-gap materials
        config.system["degauss"] = 0.00147

    # 5. ecutwfc / ecutrho from SSSP metadata
    ecutwfc = compute_ecutwfc(unique_elements, config.accuracy)
    ecutrho = compute_ecutrho(unique_elements, config.accuracy)
    config.system["ecutwfc"] = ecutwfc
    config.system["ecutrho"] = ecutrho

    # 6. K-points grid (reuses shared function)
    config.kpoints.grid = compute_kpoints_grid(
        atoms, kpr=30.0, is_2d=config.is_2d,
    )

    # 7. Pseudo mapping
    config.pseudo_map = {el: get_pseudo_filename(el) for el in species_order}

    # 8. ibrav=0, nat, ntyp
    config.system["ibrav"] = 0
    config.system["nat"] = len(atoms)
    config.system["ntyp"] = len(species_order)

    return config


# ---------------------------------------------------------------------------
# Preset Factory
# ---------------------------------------------------------------------------

_qe_preset_cfg = load_config("qe_presets")
_PRESET_QE: Dict[tuple, Dict[str, Any]] = {}
for _k, _v in _qe_preset_cfg.items():
    _parts = _k.split(":")
    _calc_str, _acc_str = _parts[0], _parts[1]
    _qe_calc = QECalculationType(_calc_str)
    _PRESET_QE[(_qe_calc, AccuracyLevel(_acc_str))] = _v


def get_qe_preset(
    calc_type: QECalculationType = QECalculationType.SCF,
    accuracy: AccuracyLevel = AccuracyLevel.STANDARD,
    atoms: Optional[Atoms] = None,
) -> QEInputConfig:
    """Create a QEInputConfig from preset defaults.

    If ``atoms`` is provided, structure-aware auto-detection is applied.

    Args:
        calc_type: QE calculation type (scf, relax, vc-relax, bands, nscf).
        accuracy: Accuracy level (standard or precise).
        atoms: Optional ASE Atoms for structure-aware configuration.

    Returns:
        Configured QEInputConfig.
    """
    key = (calc_type, accuracy)
    preset = _PRESET_QE.get(key, _PRESET_QE.get(
        (QECalculationType.SCF, accuracy), {}
    ))

    config = QEInputConfig(
        calc_type=calc_type,
        accuracy=accuracy,
        control=dict(preset.get("control", {})),
        system=dict(preset.get("system", {})),
        electrons=dict(preset.get("electrons", {})),
        ions=dict(preset.get("ions", {})),
        cell=dict(preset.get("cell", {})),
    )

    if atoms is not None:
        detect_and_apply_qe_hints(atoms, config)

    return config
