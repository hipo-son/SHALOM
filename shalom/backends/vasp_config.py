"""VASP input configuration and preset system.

Provides composable presets for different calculation types and accuracy levels,
VASP-recommended POTCAR mappings, structure-aware auto-detection (magnetic elements,
2D materials, GGA+U), and dynamic ENCUT allocation.

Inspired by VASPKIT's recipe system and VASP wiki official recommendations.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from ase import Atoms

from shalom._config_loader import load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CalculationType(str, Enum):
    """Supported VASP calculation types."""

    RELAXATION = "relaxation"
    STATIC = "static"
    BAND_STRUCTURE = "band_structure"
    DOS = "dos"
    ELASTIC = "elastic"


class AccuracyLevel(str, Enum):
    """Accuracy level controlling ENCUT, EDIFF, and other convergence parameters."""

    STANDARD = "standard"
    PRECISE = "precise"


# ---------------------------------------------------------------------------
# KPoints Config
# ---------------------------------------------------------------------------


@dataclass
class KPointsConfig:
    """K-point mesh configuration.

    Attributes:
        mode: Generation mode — automatic (Gamma-centered), line, or explicit.
        kpoints_resolution: Resolution parameter kpr in 2pi/Angstrom units.
        grid: Explicit grid override [Nx, Ny, Nz].
        num_kpts_per_segment: Points per segment for band-structure line mode.
    """

    mode: Literal["automatic", "line", "explicit"] = "automatic"
    kpoints_resolution: float = 30.0
    grid: Optional[List[int]] = None
    num_kpts_per_segment: int = 40


# ---------------------------------------------------------------------------
# VASP Input Config
# ---------------------------------------------------------------------------


@dataclass
class VASPInputConfig:
    """Complete VASP input configuration.

    Holds INCAR settings, KPOINTS config, POTCAR variant selection, and
    structure-aware flags (2D, magnetic, GGA+U).

    Attributes:
        calc_type: Type of calculation.
        accuracy: Accuracy level (standard or precise).
        incar_settings: Preset INCAR defaults for the chosen calc_type + accuracy.
        user_incar_settings: User/LLM overrides (highest priority).
        kpoints: K-point mesh configuration.
        potcar_preset: POTCAR variant set to use.
        functional: Exchange-correlation functional.
        vdw_correction: IVDW value (12=D3-BJ preferred, 11=D3).
        ldau_settings: Hubbard U settings dict with LDAUTYPE, LDAUL, LDAUU, LDAUJ.
        double_relaxation: Whether to use two-step relaxation for Pulay stress.
        initial_magmom: Per-element MAGMOM values.
        is_2d: Whether structure is 2D (forces ISIF=4, kz=1).
        enmax_values: Per-element ENMAX from POTCAR (for dynamic ENCUT).
    """

    calc_type: CalculationType = CalculationType.RELAXATION
    accuracy: AccuracyLevel = AccuracyLevel.STANDARD
    incar_settings: Dict[str, Any] = field(default_factory=dict)
    user_incar_settings: Dict[str, Any] = field(default_factory=dict)
    kpoints: KPointsConfig = field(default_factory=KPointsConfig)
    potcar_preset: Literal["vasp_recommended", "mp_default"] = "vasp_recommended"
    functional: str = "PBE"
    vdw_correction: Optional[int] = None
    ldau_settings: Optional[Dict[str, Any]] = None
    double_relaxation: bool = False
    initial_magmom: Optional[Dict[str, float]] = None
    is_2d: bool = False
    enmax_values: Optional[Dict[str, float]] = None

    def get_merged_incar(self) -> Dict[str, Any]:
        """Return INCAR settings with user overrides applied on top of presets."""
        merged = dict(self.incar_settings)
        merged.update(self.user_incar_settings)
        return merged


# ---------------------------------------------------------------------------
# POTCAR Variant Mappings (loaded from config/potcar_mapping.yaml)
# ---------------------------------------------------------------------------

_potcar_cfg = load_config("potcar_mapping")
VASP_RECOMMENDED_POTCARS: Dict[str, str] = _potcar_cfg["vasp_recommended"]
MP_DEFAULT_POTCARS: Dict[str, str] = _potcar_cfg.get("mp_default_overrides", {})


def get_potcar_variant(element: str, preset: str = "vasp_recommended") -> str:
    """Return the POTCAR variant for an element under the given preset.

    Args:
        element: Element symbol (e.g. "Fe", "Ti").
        preset: "vasp_recommended" or "mp_default".

    Returns:
        POTCAR variant string (e.g. "Fe", "Ti_sv").
    """
    if preset == "mp_default":
        if element in MP_DEFAULT_POTCARS:
            return MP_DEFAULT_POTCARS[element]
    return VASP_RECOMMENDED_POTCARS.get(element, element)


# ---------------------------------------------------------------------------
# ENMAX Reference Values (loaded from config/enmax_values.yaml)
# ---------------------------------------------------------------------------

ENMAX_VALUES: Dict[str, float] = load_config("enmax_values")


def compute_encut(
    elements: List[str], accuracy: AccuracyLevel,
    enmax_override: Optional[Dict[str, float]] = None,
) -> int:
    """Compute ENCUT from element ENMAX values.

    STANDARD: max(ENMAX) * 1.3, minimum 520 eV.
    PRECISE: max(ENMAX) * 1.5, minimum 520 eV.

    Args:
        elements: List of element symbols in the structure.
        accuracy: Accuracy level.
        enmax_override: Optional per-element ENMAX overrides.

    Returns:
        ENCUT in eV (integer).
    """
    source = enmax_override or ENMAX_VALUES
    enmax_vals = [source.get(el, 300.0) for el in set(elements)]
    if not enmax_vals:
        return 520
    max_enmax = max(enmax_vals)
    multiplier = 1.3 if accuracy == AccuracyLevel.STANDARD else 1.5
    return max(520, int(math.ceil(max_enmax * multiplier)))


# ---------------------------------------------------------------------------
# Magnetic Elements & MAGMOM Defaults (loaded from config/magnetic_elements.yaml)
# ---------------------------------------------------------------------------

_mag_cfg = load_config("magnetic_elements")
DEFAULT_MAGMOM: Dict[str, float] = _mag_cfg["default_magmom"]
MAGNETIC_ELEMENTS: frozenset = frozenset(DEFAULT_MAGMOM.keys())


# ---------------------------------------------------------------------------
# Hubbard U Values — Dudarev Scheme (loaded from config/hubbard_u.yaml)
# ---------------------------------------------------------------------------

_hub_cfg = load_config("hubbard_u")
HUBBARD_U_VALUES: Dict[str, tuple] = {
    k: (v["L"], v["U"], v["J"]) for k, v in _hub_cfg["values"].items()
}
ANION_ELEMENTS: frozenset = frozenset(_hub_cfg["anion_elements"])


# ---------------------------------------------------------------------------
# LMAXMIX Thresholds
# ---------------------------------------------------------------------------

# Atomic number thresholds for LMAXMIX setting.
# Z > 20 (Sc onwards): d-electrons → LMAXMIX=4
# Z > 56 (La onwards): f-electrons → LMAXMIX=6
_LMAXMIX_D_THRESHOLD = 20
_LMAXMIX_F_THRESHOLD = 56


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


# ---------------------------------------------------------------------------
# Pure Metal Detection (loaded from config/metallic_elements.yaml)
# ---------------------------------------------------------------------------

_METALLIC_ELEMENTS: frozenset = frozenset(load_config("metallic_elements")["elements"])


def _is_pure_metal(elements: List[str]) -> bool:
    """Check if all elements are metallic (no anions/non-metals)."""
    return all(el in _METALLIC_ELEMENTS for el in elements)


# ---------------------------------------------------------------------------
# 2D Structure Detection
# ---------------------------------------------------------------------------


def detect_2d(atoms: Atoms, vacuum_threshold: float = 5.0) -> bool:
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
    if len(cell) < 3 or len(cell[2]) < 3:
        return False
    c_height = cell[2][2]
    if c_height <= 0:
        return False
    positions = atoms.positions
    z_positions = positions[:, 2]
    slab_thickness = z_positions.max() - z_positions.min()
    vacuum = c_height - slab_thickness
    return bool(vacuum > vacuum_threshold)


# ---------------------------------------------------------------------------
# KPOINTS Grid Calculation
# ---------------------------------------------------------------------------


def compute_kpoints_grid(
    atoms: Atoms, kpr: float = 30.0, is_2d: bool = False,
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
    reciprocal = atoms.cell.reciprocal() * 2 * math.pi
    grid = []
    for i in range(3):
        b_length = float((reciprocal[i] ** 2).sum() ** 0.5)
        n = max(1, round(b_length * kpr / (2 * math.pi)))
        grid.append(n)
    if is_2d:
        grid[2] = 1
    return grid


# ---------------------------------------------------------------------------
# Structure-Aware Auto-Detection
# ---------------------------------------------------------------------------


def detect_and_apply_structure_hints(
    atoms: Atoms, config: VASPInputConfig,
) -> VASPInputConfig:
    """Apply structure-aware auto-detection rules to a VASPInputConfig.

    Modifies the config in-place based on the atoms:
    1. Magnetic elements → ISPIN=2, MAGMOM
    2. GGA+U for TMO/chalcogenides (PRECISE only auto-enables; STANDARD warns)
    3. 2D detection → ISIF=4, KPOINTS z=1, IVDW=12
    4. Pure metal → ISMEAR=1, SIGMA=0.1
    5. LMAXMIX for d/f electrons

    Args:
        atoms: ASE Atoms object.
        config: VASPInputConfig to modify.

    Returns:
        The modified config (same object).
    """
    if len(atoms) == 0:
        return config

    symbols = list(atoms.get_chemical_symbols())
    unique_elements = list(set(symbols))
    atomic_numbers = list(atoms.get_atomic_numbers())

    # 1. Magnetic element detection
    has_magnetic = any(el in MAGNETIC_ELEMENTS for el in unique_elements)
    if has_magnetic:
        config.incar_settings["ISPIN"] = 2
        magmom_list = [DEFAULT_MAGMOM.get(s, 0.6) for s in symbols]
        config.incar_settings["MAGMOM"] = magmom_list
        config.initial_magmom = {
            el: DEFAULT_MAGMOM[el] for el in unique_elements if el in DEFAULT_MAGMOM
        }

    # 2. GGA+U auto-detection (TMO / chalcogenides)
    has_anion = any(el in ANION_ELEMENTS for el in unique_elements)
    magnetic_tm = [el for el in unique_elements if el in HUBBARD_U_VALUES]
    if has_magnetic and has_anion and magnetic_tm:
        u_elements = [el for el in magnetic_tm if HUBBARD_U_VALUES[el][1] > 0.0]
        if u_elements:
            if config.accuracy == AccuracyLevel.PRECISE:
                # Auto-enable GGA+U
                all_elements_ordered = list(dict.fromkeys(symbols))
                ldaul = []
                ldauu = []
                ldauj = []
                for el in all_elements_ordered:
                    if el in HUBBARD_U_VALUES:
                        l_val, u_val, j_val = HUBBARD_U_VALUES[el]
                        ldaul.append(l_val)
                        ldauu.append(u_val)
                        ldauj.append(j_val)
                    else:
                        ldaul.append(-1)
                        ldauu.append(0.0)
                        ldauj.append(0.0)
                config.ldau_settings = {
                    "LDAU": True,
                    "LDAUTYPE": 2,
                    "LDAUL": ldaul,
                    "LDAUU": ldauu,
                    "LDAUJ": ldauj,
                    "LDAUPRINT": 1,
                    "elements_order": all_elements_ordered,
                }
                if config.functional != "PBE":
                    logger.warning(
                        "GGA+U enabled with functional='%s', but Hubbard U "
                        "values are fitted for PBE only (Wang et al. PRB 73, "
                        "195107). Results may be unreliable.",
                        config.functional,
                    )

    # 3. 2D structure detection
    if detect_2d(atoms):
        config.is_2d = True
        if config.calc_type == CalculationType.RELAXATION:
            config.incar_settings["ISIF"] = 4  # Prevent vacuum collapse
        config.vdw_correction = 12  # D3(BJ) preferred over D3
        config.incar_settings["IVDW"] = 12

    # 3b. Slab detection (vacuum > 5A but not 2D monolayer)
    if not config.is_2d:
        _cell = atoms.get_cell()
        if len(_cell) >= 3 and len(_cell[2]) >= 3 and _cell[2][2] > 0:
            _z_pos = atoms.positions[:, 2]
            _slab_thickness = _z_pos.max() - _z_pos.min()
            _vacuum = _cell[2][2] - _slab_thickness
            if _vacuum > 5.0 and _slab_thickness > 3.0:
                if config.calc_type == CalculationType.RELAXATION:
                    config.incar_settings["ISIF"] = 2
                config.incar_settings["IDIPOL"] = 3
                config.incar_settings["LDIPOL"] = True

    # 4. Pure metal detection
    if _is_pure_metal(unique_elements):
        config.incar_settings["ISMEAR"] = 1
        config.incar_settings["SIGMA"] = 0.1
        config.incar_settings["ALGO"] = "Fast"

    # 5. LMAXMIX
    lmaxmix = _get_lmaxmix(atomic_numbers)
    if lmaxmix is not None:
        config.incar_settings["LMAXMIX"] = lmaxmix

    # 6. KPOINTS grid with 2D handling
    config.kpoints.grid = compute_kpoints_grid(
        atoms, kpr=config.kpoints.kpoints_resolution, is_2d=config.is_2d,
    )

    # 7. Dynamic ENCUT
    element_enmax = {el: ENMAX_VALUES.get(el, 300.0) for el in unique_elements}
    config.enmax_values = element_enmax
    encut = compute_encut(unique_elements, config.accuracy, element_enmax)
    config.incar_settings["ENCUT"] = encut

    return config


# ---------------------------------------------------------------------------
# Preset Factory
# ---------------------------------------------------------------------------

# Base INCAR presets by (calc_type, accuracy).
# Loaded from config/incar_presets.yaml and converted to tuple keys.
_preset_cfg = load_config("incar_presets")
_PRESET_INCAR: Dict[tuple, Dict[str, Any]] = {
    (CalculationType(k.split(":")[0]), AccuracyLevel(k.split(":")[1])): v
    for k, v in _preset_cfg.items()
}


def get_preset(
    calc_type: CalculationType = CalculationType.RELAXATION,
    accuracy: AccuracyLevel = AccuracyLevel.STANDARD,
    atoms: Optional[Atoms] = None,
) -> VASPInputConfig:
    """Create a VASPInputConfig from preset defaults.

    If ``atoms`` is provided, structure-aware auto-detection is applied:
    magnetic elements, 2D detection, GGA+U, metal smearing, dynamic ENCUT.

    Args:
        calc_type: Calculation type.
        accuracy: Accuracy level.
        atoms: Optional ASE Atoms for structure-aware configuration.

    Returns:
        Configured VASPInputConfig.
    """
    key = (calc_type, accuracy)
    incar = dict(_PRESET_INCAR.get(key, _PRESET_INCAR[(CalculationType.RELAXATION, accuracy)]))

    config = VASPInputConfig(
        calc_type=calc_type,
        accuracy=accuracy,
        incar_settings=incar,
        kpoints=KPointsConfig(),
    )

    if atoms is not None:
        detect_and_apply_structure_hints(atoms, config)

    return config
