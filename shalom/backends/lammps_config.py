"""LAMMPS input configuration, force field auto-detection, and presets.

Follows the same architecture as ``vasp_config.py`` / ``qe_config.py``:
all force field metadata lives in ``config/lammps_potentials.yaml`` (no hardcoding),
and ``detect_and_apply_lammps_hints()`` auto-configures pair_style, pair_coeff,
timestep, thermostat, and boundary based on the atomic composition.

Usage::

    from shalom.backends.lammps_config import get_lammps_preset, detect_force_field

    config = get_lammps_preset(ensemble="nvt", atoms=atoms)
    # → pair_style, pair_coeff, timestep auto-detected from lammps_potentials.yaml
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ase import Atoms

from shalom._config_loader import load_config
from shalom.backends._physics import detect_2d

logger = logging.getLogger(__name__)

# Light elements that require shorter timesteps
_LIGHT_ELEMENTS = frozenset({"H", "Li", "He"})


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class LAMMPSInputConfig:
    """LAMMPS simulation configuration.

    If ``pair_style`` is left empty, ``detect_and_apply_lammps_hints()``
    auto-selects the force field from ``lammps_potentials.yaml`` based on
    the atomic composition.  If the user specifies ``pair_style`` explicitly,
    auto-detection is skipped entirely (user override wins).

    Attributes:
        ensemble: Thermodynamic ensemble — ``"nve"``, ``"nvt"``, ``"npt"``.
        temperature: Target temperature in K.
        temperature_end: End temperature in K for ramping (None = constant T).
        pressure: Target pressure in bar (NPT only).
        timestep: MD timestep in fs.  Overridden by auto-detection.
        nsteps: Number of MD steps.
        pair_style: LAMMPS pair_style string (empty → auto-detect).
        pair_coeff: LAMMPS pair_coeff commands (empty → auto-generated).
        potential_files: Paths to potential files to copy into calc directory.
        potential_dir: Directory containing potential files.
        detected_ff: Name of auto-detected force field (for logging).
        units: LAMMPS unit system (``"metal"`` → eV, Angstrom, fs).
        atom_style: LAMMPS atom_style (``"atomic"`` / ``"charge"`` / ``"full"``).
        boundary: LAMMPS boundary conditions (``"p p p"`` for fully periodic).
        thermo_interval: Print thermo output every N steps.
        dump_interval: Write dump file every N steps.
        temperature_damp: Thermostat damping parameter in fs.
        pressure_damp: Barostat damping parameter in fs.
        is_2d: True if the structure is 2D (z-direction non-periodic).
        minimize_first: Run energy minimization before MD (default: True).
        accuracy: Accuracy level (``"standard"`` / ``"precise"``).
        extra_commands: Additional LAMMPS commands appended to input script.
    """

    # Ensemble
    ensemble: str = "nvt"
    temperature: float = 300.0
    temperature_end: Optional[float] = None
    pressure: float = 0.0
    timestep: float = 1.0
    nsteps: int = 100000

    # Force field
    pair_style: str = ""
    pair_coeff: List[str] = field(default_factory=list)
    potential_files: List[str] = field(default_factory=list)
    potential_dir: Optional[str] = None
    detected_ff: Optional[str] = None

    # I/O
    units: str = "metal"
    atom_style: str = "atomic"
    boundary: str = "p p p"
    thermo_interval: int = 100
    dump_interval: int = 1000
    temperature_damp: float = 100.0
    pressure_damp: float = 1000.0
    is_2d: bool = False

    # Advanced
    minimize_first: bool = True
    accuracy: str = "standard"
    extra_commands: List[str] = field(default_factory=list)
    _detection_log: List[str] = field(default_factory=list)

    def to_summary_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary of all simulation parameters."""
        d: Dict[str, Any] = {
            "ensemble": self.ensemble,
            "temperature_K": self.temperature,
            "timestep_fs": self.timestep,
            "nsteps": self.nsteps,
            "units": self.units,
            "atom_style": self.atom_style,
            "boundary": self.boundary,
            "accuracy": self.accuracy,
            "is_2d": self.is_2d,
            "minimize_first": self.minimize_first,
        }
        if self.temperature_end is not None:
            d["temperature_end_K"] = self.temperature_end
        if self.ensemble == "npt":
            d["pressure_bar"] = self.pressure
            d["pressure_damp_fs"] = self.pressure_damp
        if self.pair_style:
            d["pair_style"] = self.pair_style
        if self.pair_coeff:
            d["pair_coeff"] = self.pair_coeff
        if self.detected_ff:
            d["detected_ff"] = self.detected_ff
        if self.potential_files:
            d["potential_files"] = self.potential_files
        d["temperature_damp_fs"] = self.temperature_damp
        d["thermo_interval"] = self.thermo_interval
        d["dump_interval"] = self.dump_interval
        return d


# ---------------------------------------------------------------------------
# Potential directory resolution (3-tier, like QE resolve_pseudo_dir)
# ---------------------------------------------------------------------------


def resolve_potential_dir(potential_dir: Optional[str] = None) -> str:
    """Resolve LAMMPS potential file directory with 3-tier fallback.

    Resolution order:
    1. Explicit ``potential_dir`` argument.
    2. ``$SHALOM_LAMMPS_POTENTIALS`` environment variable.
    3. ``~/lammps-potentials`` default.

    Args:
        potential_dir: Explicit path (highest priority).

    Returns:
        Resolved directory path as a string.
    """
    if potential_dir:
        return str(Path(potential_dir).expanduser().resolve())

    env_dir = os.environ.get("SHALOM_LAMMPS_POTENTIALS")
    if env_dir:
        return str(Path(env_dir).expanduser().resolve())

    return str(Path.home() / "lammps-potentials")


# ---------------------------------------------------------------------------
# Force field auto-detection
# ---------------------------------------------------------------------------


def detect_force_field(
    atoms: Atoms, _db: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Auto-detect the best force field for the given structure.

    Searches ``lammps_potentials.yaml`` for force fields whose
    ``supported_elements`` list contains all elements in ``atoms``.
    Returns the match with the highest ``priority``.

    Args:
        atoms: ASE Atoms object.
        _db: Pre-loaded config dict (internal use to avoid redundant loads).

    Returns:
        Force field key (e.g. ``"eam_alloy"``, ``"tersoff"``, ``"lj_cut"``)
        or ``None`` if no matching force field is found.
    """
    if len(atoms) == 0:
        return None

    elements = set(atoms.get_chemical_symbols())
    db = _db if _db is not None else load_config("lammps_potentials")

    candidates: List[tuple] = []
    for ff_name, ff_data in db.items():
        if not isinstance(ff_data, dict) or "pair_style" not in ff_data:
            continue
        supported = set(ff_data.get("supported_elements", []))
        if elements.issubset(supported):
            candidates.append((ff_name, ff_data.get("priority", 50)))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def _generate_pair_coeff(
    ff_name: str, ff_data: Dict[str, Any], species_order: List[str],
) -> tuple:
    """Generate pair_coeff commands and potential file list from YAML metadata.

    Args:
        ff_name: Force field key from YAML.
        ff_data: Force field metadata dict from YAML.
        species_order: Element symbols in species order (first occurrence).

    Returns:
        Tuple of (pair_coeff_list, potential_files_list).
    """
    pair_coeff: List[str] = []
    potential_files: List[str] = []

    pair_style = ff_data.get("pair_style", "")

    if "lj" in pair_style:
        # LJ: pair_coeff i j epsilon sigma — Lorentz-Berthelot mixing
        lj_params = ff_data.get("lj_params", {})
        for i, el_i in enumerate(species_order, 1):
            for j in range(i, len(species_order) + 1):
                el_j = species_order[j - 1]
                p_i = lj_params.get(el_i, {"epsilon": 0.01, "sigma": 3.4})
                p_j = lj_params.get(el_j, {"epsilon": 0.01, "sigma": 3.4})
                eps = math.sqrt(p_i["epsilon"] * p_j["epsilon"])
                sig = (p_i["sigma"] + p_j["sigma"]) / 2.0
                pair_coeff.append(f"{i} {j} {eps:.6f} {sig:.4f}")
    else:
        # EAM/Tersoff style: pair_coeff * * potential_file El1 El2 ...
        potentials = ff_data.get("potentials", {})

        # Collect unique potential files and build element mapping
        pot_files_seen: Dict[str, str] = {}  # file → first element
        element_map: List[str] = []

        for el in species_order:
            el_data = potentials.get(el, {})
            # Use single_file for pure single-element systems
            if len(species_order) == 1 and "single_file" in el_data:
                pot_file = el_data["single_file"]
            else:
                pot_file = el_data.get("file", "")
            map_name = el_data.get("map", el)
            element_map.append(map_name)
            if pot_file and pot_file not in pot_files_seen:
                pot_files_seen[pot_file] = el

        if pot_files_seen:
            # Use the first potential file (most force fields use a single file)
            pot_file = next(iter(pot_files_seen))
            pair_coeff.append(f"* * {pot_file} {' '.join(element_map)}")
            potential_files = list(pot_files_seen.keys())

    return pair_coeff, potential_files


# ---------------------------------------------------------------------------
# Structure-aware auto-detection (same pattern as VASP/QE)
# ---------------------------------------------------------------------------


def detect_and_apply_lammps_hints(
    atoms: Atoms, config: LAMMPSInputConfig,
) -> LAMMPSInputConfig:
    """Apply structure-aware auto-detection rules to a LAMMPSInputConfig.

    Follows the same pattern as ``detect_and_apply_structure_hints()``
    in ``vasp_config.py`` and ``detect_and_apply_qe_hints()`` in
    ``qe_config.py``.

    Auto-detection steps:
    1. If ``pair_style`` is already set → skip (user override).
    2. ``detect_force_field()`` → select FF from YAML metadata.
    3. Generate ``pair_coeff`` and ``potential_files`` from YAML.
    4. Set ``timestep`` based on force field + light element detection.
    5. Set thermostat damping parameters from YAML.
    6. Detect 2D structures → adjust boundary conditions.
    7. Apply simulation_defaults from YAML.

    Args:
        atoms: ASE Atoms object.
        config: LAMMPSInputConfig to modify.

    Returns:
        The modified config (same object).
    """
    if len(atoms) == 0:
        return config

    # 1. User specified pair_style → skip auto-detection entirely
    if config.pair_style:
        config._detection_log.append(
            f"User override: pair_style={config.pair_style} (auto-detect skipped)"
        )
        # Still apply 2D detection
        if detect_2d(atoms):
            config.is_2d = True
            config.boundary = "p p f"
            config._detection_log.append("2D structure → boundary 'p p f'")
        return config

    # 2. Auto-detect force field (load config once, reuse for FF lookup)
    db = load_config("lammps_potentials")
    ff_name = detect_force_field(atoms, _db=db)
    if ff_name is None:
        elements = set(atoms.get_chemical_symbols())
        logger.warning(
            "No matching force field in lammps_potentials.yaml for elements %s. "
            "Specify --pair-style and --pair-coeff manually.",
            sorted(elements),
        )
        return config

    ff_data = db[ff_name]
    config.detected_ff = ff_name

    # 3. Set pair_style, pair_coeff, potential_files
    config.pair_style = ff_data["pair_style"]
    config.units = ff_data.get("units", "metal")
    config.atom_style = ff_data.get("atom_style", "atomic")

    species_order = list(dict.fromkeys(atoms.get_chemical_symbols()))
    pair_coeff, potential_files = _generate_pair_coeff(ff_name, ff_data, species_order)
    config.pair_coeff = pair_coeff
    config.potential_files = potential_files
    config._detection_log.append(
        f"Force field: {ff_name} (pair_style={ff_data['pair_style']})"
    )

    # 4. Timestep (light elements → shorter)
    elements = set(atoms.get_chemical_symbols())
    ts_data = ff_data.get("timestep", {})
    if isinstance(ts_data, dict):
        if any(el in _LIGHT_ELEMENTS for el in elements):
            config.timestep = ts_data.get("light_elements", ts_data.get("default", 1.0))
            config._detection_log.append(
                f"Light elements detected → timestep={config.timestep} fs"
            )
        else:
            config.timestep = ts_data.get("default", 1.0)
            config._detection_log.append(
                f"Timestep={config.timestep} fs (default for {ff_name})"
            )
    else:
        config.timestep = float(ts_data)
        config._detection_log.append(
            f"Timestep={config.timestep} fs (default for {ff_name})"
        )

    # 5. Thermostat damping
    thermo = ff_data.get("thermostat", {})
    config.temperature_damp = thermo.get("nvt_damp", 100.0)
    if config.ensemble == "npt":
        config.pressure_damp = thermo.get("npt_damp", 1000.0)

    # 6. 2D detection
    if detect_2d(atoms):
        config.is_2d = True
        config.boundary = "p p f"
        config._detection_log.append("2D structure → boundary 'p p f'")

    # 7. Apply simulation defaults
    sim_defaults = db.get("simulation_defaults", {})
    if not config.is_2d:
        config.boundary = sim_defaults.get("boundary", "p p p")
    config.thermo_interval = sim_defaults.get("thermo_interval", config.thermo_interval)
    config.dump_interval = sim_defaults.get("dump_interval", config.dump_interval)

    return config


# ---------------------------------------------------------------------------
# Preset factory
# ---------------------------------------------------------------------------


def get_lammps_preset(
    ensemble: str = "nvt",
    atoms: Optional[Atoms] = None,
    accuracy: str = "standard",
) -> LAMMPSInputConfig:
    """Create a LAMMPSInputConfig with auto-detected settings.

    If ``atoms`` is provided, ``detect_and_apply_lammps_hints()`` is called
    to auto-select force field, potential files, timestep, etc.

    Args:
        ensemble: Thermodynamic ensemble (``"nve"`` / ``"nvt"`` / ``"npt"``).
        atoms: ASE Atoms object (triggers auto-detection if provided).
        accuracy: Accuracy level (``"standard"`` / ``"precise"``).

    Returns:
        Configured LAMMPSInputConfig.
    """
    config = LAMMPSInputConfig(
        ensemble=ensemble,
        accuracy=accuracy,
    )

    if atoms is not None:
        detect_and_apply_lammps_hints(atoms, config)

    # Apply accuracy presets from YAML
    db = load_config("lammps_potentials")
    presets = db.get("accuracy_presets", {})
    preset = presets.get(accuracy, {})
    if preset:
        ts_mult = preset.get("timestep_multiplier", 1.0)
        config.timestep *= ts_mult
        ns_mult = preset.get("nsteps_multiplier", 1.0)
        config.nsteps = int(config.nsteps * ns_mult)
        config.dump_interval = preset.get("dump_interval", config.dump_interval)
        config.thermo_interval = preset.get("thermo_interval", config.thermo_interval)

    return config
