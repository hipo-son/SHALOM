"""VASP input configuration and preset system.

Provides composable presets for different calculation types and accuracy levels,
VASP-recommended POTCAR mappings, structure-aware auto-detection (magnetic elements,
2D materials, GGA+U), and dynamic ENCUT allocation.

Inspired by VASPKIT's recipe system and VASP wiki official recommendations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from ase import Atoms


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
# POTCAR Variant Mappings
# ---------------------------------------------------------------------------

# VASP wiki official recommended pseudopotentials (PBE_54).
# Differs from pymatgen/MP defaults for several elements.
VASP_RECOMMENDED_POTCARS: Dict[str, str] = {
    # Main group
    "H": "H", "He": "He",
    "Li": "Li_sv", "Be": "Be", "B": "B", "C": "C", "N": "N", "O": "O",
    "F": "F", "Ne": "Ne",
    "Na": "Na_pv", "Mg": "Mg", "Al": "Al", "Si": "Si", "P": "P", "S": "S",
    "Cl": "Cl", "Ar": "Ar",
    "K": "K_sv", "Ca": "Ca_sv",
    # 3d transition metals
    "Sc": "Sc_sv", "Ti": "Ti_sv", "V": "V_sv", "Cr": "Cr_pv",
    "Mn": "Mn_pv", "Fe": "Fe", "Co": "Co", "Ni": "Ni",
    "Cu": "Cu", "Zn": "Zn",
    # p-block row 4
    "Ga": "Ga_d", "Ge": "Ge_d", "As": "As", "Se": "Se", "Br": "Br", "Kr": "Kr",
    # 4d transition metals
    "Rb": "Rb_sv", "Sr": "Sr_sv",
    "Y": "Y_sv", "Zr": "Zr_sv", "Nb": "Nb_sv", "Mo": "Mo_sv",
    "Tc": "Tc_pv", "Ru": "Ru_pv", "Rh": "Rh_pv", "Pd": "Pd",
    "Ag": "Ag", "Cd": "Cd",
    # p-block row 5
    "In": "In_d", "Sn": "Sn_d", "Sb": "Sb", "Te": "Te", "I": "I", "Xe": "Xe",
    # 5d transition metals
    "Cs": "Cs_sv", "Ba": "Ba_sv",
    "Hf": "Hf_pv", "Ta": "Ta_pv", "W": "W_sv",
    "Re": "Re", "Os": "Os", "Ir": "Ir", "Pt": "Pt",
    "Au": "Au", "Hg": "Hg",
    # p-block row 6
    "Tl": "Tl_d", "Pb": "Pb_d", "Bi": "Bi_d",
    # Lanthanides
    "La": "La", "Ce": "Ce", "Pr": "Pr_3", "Nd": "Nd_3",
    "Sm": "Sm_3", "Eu": "Eu_2", "Gd": "Gd_3",
    "Tb": "Tb_3", "Dy": "Dy_3", "Ho": "Ho_3",
    "Er": "Er_3", "Tm": "Tm_3", "Yb": "Yb_2", "Lu": "Lu_3",
    # Actinides
    "U": "U", "Np": "Np", "Pu": "Pu",
}

# MP default POTCAR variants (subset where they differ from VASP recommended).
MP_DEFAULT_POTCARS: Dict[str, str] = {
    "Fe": "Fe_pv", "Ti": "Ti_pv", "V": "V_pv",
    "Mo": "Mo_pv", "W": "W_pv",
    "Cr": "Cr_pv", "Mn": "Mn_pv",
}


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
# ENMAX Reference Values (PBE POTCAR)
# ---------------------------------------------------------------------------

# ENMAX values from VASP-recommended PBE_54 POTCARs (eV).
# Used for dynamic ENCUT calculation.
ENMAX_VALUES: Dict[str, float] = {
    "H": 250.0, "He": 479.0,
    "Li": 499.0, "Be": 309.0, "B": 318.7, "C": 400.0, "N": 400.0, "O": 400.0,
    "F": 400.0, "Ne": 344.0,
    "Na": 260.0, "Mg": 200.0, "Al": 240.3, "Si": 245.3, "P": 255.0, "S": 280.0,
    "Cl": 262.0, "Ar": 266.0,
    "K": 259.0, "Ca": 267.0,
    "Sc": 223.0, "Ti": 495.0, "V": 476.0, "Cr": 227.0,
    "Mn": 270.0, "Fe": 267.9, "Co": 268.0, "Ni": 270.0,
    "Cu": 295.4, "Zn": 277.0,
    "Ga": 135.0, "Ge": 174.0, "As": 209.0, "Se": 212.0, "Br": 213.0,
    "Rb": 220.0, "Sr": 229.0,
    "Y": 203.0, "Zr": 230.0, "Nb": 209.0, "Mo": 225.0,
    "Ru": 213.0, "Rh": 229.0, "Pd": 251.0, "Ag": 250.0, "Cd": 274.0,
    "In": 96.0, "Sn": 103.0, "Sb": 172.0, "Te": 175.0, "I": 176.0,
    "Cs": 220.0, "Ba": 238.0,
    "Hf": 220.0, "Ta": 224.0, "W": 224.0,
    "Re": 226.0, "Os": 228.0, "Ir": 211.0, "Pt": 230.0,
    "Au": 230.0,
    "La": 219.0, "Ce": 273.0, "Gd": 256.0, "U": 253.0,
}


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
# Magnetic Elements & MAGMOM Defaults
# ---------------------------------------------------------------------------

# Default initial MAGMOM values per element (high-spin initialization).
DEFAULT_MAGMOM: Dict[str, float] = {
    # 3d transition metals
    "Ti": 1.0, "V": 3.0, "Cr": 4.0, "Mn": 5.0,
    "Fe": 5.0, "Co": 3.0, "Ni": 2.0, "Cu": 1.0,
    # 4d/5d (surface/nano structures may show magnetism)
    "Ru": 2.0, "Rh": 1.0, "Os": 2.0, "Ir": 1.0,
    # Lanthanides (4f electron count based)
    "Ce": 1.0, "Pr": 2.0, "Nd": 3.0, "Sm": 5.0,
    "Eu": 7.0, "Gd": 7.0, "Tb": 6.0, "Dy": 5.0,
    "Ho": 4.0, "Er": 3.0, "Tm": 2.0, "Yb": 1.0,
    # Actinides
    "U": 2.0, "Np": 3.0, "Pu": 5.0,
}

MAGNETIC_ELEMENTS: frozenset = frozenset(DEFAULT_MAGMOM.keys())


# ---------------------------------------------------------------------------
# Hubbard U Values — Dudarev Scheme (LDAUTYPE=2)
# ---------------------------------------------------------------------------

# Wang et al. PRB 73, 195107 (2006) — U_eff values.
# Dudarev scheme: J is ignored, only U_eff = U - J matters.
# Tuple: (L, U_eff, J) where L=2 for d-orbitals, L=3 for f-orbitals.
HUBBARD_U_VALUES: Dict[str, tuple] = {
    "Fe": (2, 5.3, 0.0),
    "Co": (2, 3.32, 0.0),
    "Ni": (2, 6.2, 0.0),
    "Mn": (2, 3.9, 0.0),
    "V": (2, 3.25, 0.0),
    "Cr": (2, 3.7, 0.0),
    "Cu": (2, 4.0, 0.0),   # Cu^2+ in oxides; metallic Cu typically U=0
    "Ti": (2, 0.0, 0.0),   # TiO2 etc. usually don't need U
}

# Anion elements that trigger GGA+U when combined with magnetic TM.
ANION_ELEMENTS: frozenset = frozenset({"O", "S", "Se", "Te"})


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
# Pure Metal Detection
# ---------------------------------------------------------------------------

_METALLIC_ELEMENTS: frozenset = frozenset({
    "Li", "Be", "Na", "Mg", "Al", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Cs", "Ba", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
    "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "U", "Np", "Pu",
})


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
                all_elements_ordered = sorted(set(symbols))
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
                config.incar_settings["LDAU"] = True
                config.incar_settings["LDAUTYPE"] = 2
                config.incar_settings["LDAUL"] = " ".join(str(v) for v in ldaul)
                config.incar_settings["LDAUU"] = " ".join(str(v) for v in ldauu)
                config.incar_settings["LDAUJ"] = " ".join(str(v) for v in ldauj)
                config.incar_settings["LDAUPRINT"] = 1

    # 3. 2D structure detection
    if detect_2d(atoms):
        config.is_2d = True
        if config.calc_type == CalculationType.RELAXATION:
            config.incar_settings["ISIF"] = 4  # Prevent vacuum collapse
        config.vdw_correction = 12  # D3(BJ) preferred over D3
        config.incar_settings["IVDW"] = 12

    # 4. Pure metal detection
    if _is_pure_metal(unique_elements):
        config.incar_settings["ISMEAR"] = 1
        config.incar_settings["SIGMA"] = 0.1

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
_PRESET_INCAR: Dict[tuple, Dict[str, Any]] = {
    # STANDARD presets
    (CalculationType.RELAXATION, AccuracyLevel.STANDARD): {
        "ENCUT": 520, "EDIFF": 1e-5, "EDIFFG": -0.02,
        "ISMEAR": 0, "SIGMA": 0.05,
        "NSW": 99, "IBRION": 2, "ISIF": 3,
        "PREC": "Accurate", "LREAL": "Auto", "LORBIT": 11,
        "NELM": 100,
    },
    (CalculationType.STATIC, AccuracyLevel.STANDARD): {
        "ENCUT": 520, "EDIFF": 1e-5,
        "ISMEAR": 0, "SIGMA": 0.05,
        "NSW": 0, "IBRION": -1,
        "PREC": "Accurate", "LREAL": "Auto", "LORBIT": 11,
        "NELM": 100,
    },
    (CalculationType.BAND_STRUCTURE, AccuracyLevel.STANDARD): {
        "ENCUT": 520, "EDIFF": 1e-5,
        "ISMEAR": 0, "SIGMA": 0.05,
        "NSW": 0, "IBRION": -1, "ICHARG": 11,
        "PREC": "Accurate", "LREAL": False, "LORBIT": 11,
        "NELM": 100,
    },
    (CalculationType.DOS, AccuracyLevel.STANDARD): {
        "ENCUT": 520, "EDIFF": 1e-5,
        "ISMEAR": -5, "SIGMA": 0.05,
        "NSW": 0, "IBRION": -1, "ICHARG": 11,
        "NEDOS": 3001,
        "PREC": "Accurate", "LREAL": False, "LORBIT": 11,
        "NELM": 100,
    },
    (CalculationType.ELASTIC, AccuracyLevel.STANDARD): {
        "ENCUT": 520, "EDIFF": 1e-6, "EDIFFG": -0.01,
        "ISMEAR": 0, "SIGMA": 0.05,
        "NSW": 1, "IBRION": 6, "ISIF": 3,
        "PREC": "Accurate", "LREAL": False, "LORBIT": 11,
        "NELM": 100,
    },
    # PRECISE presets (overrides)
    (CalculationType.RELAXATION, AccuracyLevel.PRECISE): {
        "ENCUT": 520, "EDIFF": 1e-6, "EDIFFG": -0.01,
        "ISMEAR": 0, "SIGMA": 0.05,
        "NSW": 99, "IBRION": 2, "ISIF": 3,
        "PREC": "Accurate", "LREAL": False, "LORBIT": 11,
        "NELM": 200,
    },
    (CalculationType.STATIC, AccuracyLevel.PRECISE): {
        "ENCUT": 520, "EDIFF": 1e-6,
        "ISMEAR": 0, "SIGMA": 0.05,
        "NSW": 0, "IBRION": -1,
        "PREC": "Accurate", "LREAL": False, "LORBIT": 11,
        "NELM": 200,
    },
    (CalculationType.BAND_STRUCTURE, AccuracyLevel.PRECISE): {
        "ENCUT": 520, "EDIFF": 1e-6,
        "ISMEAR": 0, "SIGMA": 0.05,
        "NSW": 0, "IBRION": -1, "ICHARG": 11,
        "PREC": "Accurate", "LREAL": False, "LORBIT": 11,
        "NELM": 200,
    },
    (CalculationType.DOS, AccuracyLevel.PRECISE): {
        "ENCUT": 520, "EDIFF": 1e-6,
        "ISMEAR": -5, "SIGMA": 0.05,
        "NSW": 0, "IBRION": -1, "ICHARG": 11,
        "NEDOS": 5001,
        "PREC": "Accurate", "LREAL": False, "LORBIT": 11,
        "NELM": 200,
    },
    (CalculationType.ELASTIC, AccuracyLevel.PRECISE): {
        "ENCUT": 520, "EDIFF": 1e-7, "EDIFFG": -0.005,
        "ISMEAR": 0, "SIGMA": 0.05,
        "NSW": 1, "IBRION": 6, "ISIF": 3,
        "PREC": "Accurate", "LREAL": False, "LORBIT": 11,
        "NELM": 200,
    },
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
