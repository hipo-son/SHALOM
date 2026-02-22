"""Direct Material Run — generate DFT input files from a material specification.

Provides a single-function entry point for generating complete DFT input sets
from either a Materials Project ID, chemical formula, or local structure file.
Supports both VASP and QE backends.

Usage::

    from shalom.direct_run import direct_run

    result = direct_run("mp-19717", DirectRunConfig(backend_name="qe"))
    print(result.output_dir)
    print(result.files_generated)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ase import Atoms
from ase.io import read as ase_read

from shalom.agents.simulation_layer import FormFiller
from shalom.backends import get_backend
from shalom.backends._physics import AccuracyLevel

logger = logging.getLogger(__name__)

# Canonical calc_type aliases — accepts both VASP and QE naming conventions.
CALC_TYPE_ALIASES: Dict[str, str] = {
    # VASP names (canonical)
    "relaxation": "relaxation",
    "static": "static",
    "band_structure": "band_structure",
    "dos": "dos",
    "elastic": "elastic",
    # QE / shorthand aliases → VASP canonical
    "scf": "static",
    "relax": "relaxation",
    "vc-relax": "relaxation",
    "bands": "band_structure",
    "nscf": "band_structure",
}


@dataclass
class DirectRunConfig:
    """Configuration for a direct material run."""

    backend_name: str = "vasp"
    calc_type: Optional[str] = None
    accuracy: str = "standard"
    output_dir: Optional[str] = None
    user_settings: Optional[Dict[str, Any]] = None
    functional: str = "PBE"
    potcar_preset: str = "vasp_recommended"
    pseudo_dir: Optional[str] = None
    validate_structure: bool = True
    force_overwrite: bool = False
    structure_file: Optional[str] = None


@dataclass
class DirectRunResult:
    """Result from a direct material run."""

    success: bool
    output_dir: Optional[str] = None
    structure_info: Optional[Dict[str, Any]] = None
    backend_name: str = ""
    files_generated: List[str] = field(default_factory=list)
    error: Optional[str] = None
    auto_detected: Optional[Dict[str, Any]] = None


def _resolve_calc_type(calc_type: Optional[str], backend_name: str) -> str:
    """Resolve calc_type alias to canonical VASP name."""
    if calc_type is None:
        return "relaxation"
    canonical = CALC_TYPE_ALIASES.get(calc_type.lower())
    if canonical is not None:
        return canonical
    logger.warning("Unknown calc_type '%s', defaulting to 'relaxation'.", calc_type)
    return "relaxation"


def _auto_output_dir(
    formula: str, mp_id: Optional[str], backend_name: str, calc_type: str,
) -> str:
    """Generate automatic output directory name."""
    import re as _re
    parts = [formula]
    if mp_id:
        parts.append(mp_id)
    parts.extend([backend_name, calc_type.replace("-", "")])
    name = "_".join(parts)
    return _re.sub(r'[<>:"/\\|?*]', '_', name)


def _create_vasp_config(
    atoms: Atoms, calc_type: str, accuracy: str,
    user_settings: Optional[Dict[str, Any]],
    potcar_preset: str, functional: str,
) -> Any:
    """Create VASP config with structure-aware hints."""
    from shalom.backends.vasp_config import (
        CalculationType, get_preset,
    )
    config = get_preset(
        calc_type=CalculationType(calc_type),
        accuracy=AccuracyLevel(accuracy),
        atoms=atoms,
    )
    config.potcar_preset = potcar_preset
    config.functional = functional
    if user_settings:
        config.user_incar_settings.update(user_settings)
    return config


def _create_qe_config(
    atoms: Atoms, calc_type: str, accuracy: str,
    user_settings: Optional[Dict[str, Any]],
    pseudo_dir: Optional[str], functional: str,
) -> Any:
    """Create QE config with structure-aware hints."""
    from shalom.backends.qe_config import (
        QECalculationType, VASP_TO_QE_CALC_MAP, get_qe_preset,
    )
    qe_calc = VASP_TO_QE_CALC_MAP.get(calc_type)
    if qe_calc is None:
        # Try direct QE calc_type
        try:
            qe_calc = QECalculationType(calc_type)
        except ValueError:
            logger.warning("Unknown QE calc_type '%s', defaulting to SCF.", calc_type)
            qe_calc = QECalculationType.SCF

    config = get_qe_preset(
        calc_type=qe_calc,
        accuracy=AccuracyLevel(accuracy),
        atoms=atoms,
    )
    config.functional = functional
    if pseudo_dir:
        config.pseudo_dir = pseudo_dir
    elif os.environ.get("SHALOM_PSEUDO_DIR"):
        config.pseudo_dir = os.environ["SHALOM_PSEUDO_DIR"]
    if user_settings:
        config.user_settings.update(user_settings)
    return config


def direct_run(
    material_spec: str,
    config: Optional[DirectRunConfig] = None,
) -> DirectRunResult:
    """Generate DFT input files from a material specification.

    Args:
        material_spec: Materials Project ID (e.g. 'mp-19717'), chemical formula
            (e.g. 'Fe2O3'), or path to a local structure file.
        config: Run configuration. Uses defaults if not provided.

    Returns:
        DirectRunResult with output directory and file list.
    """
    if config is None:
        config = DirectRunConfig()

    backend = get_backend(config.backend_name)
    calc_type = _resolve_calc_type(config.calc_type, config.backend_name)

    # 1. Resolve structure
    atoms: Optional[Atoms] = None
    formula = ""
    mp_id: Optional[str] = None
    structure_info: Dict[str, Any] = {}

    if config.structure_file:
        # Local file
        try:
            atoms = ase_read(config.structure_file)
            formula = atoms.get_chemical_formula(mode="reduce")
            structure_info = {"source": "file", "path": config.structure_file}
        except Exception as e:
            return DirectRunResult(
                success=False, backend_name=config.backend_name,
                error=f"Failed to read structure file '{config.structure_file}': {e}",
            )
    elif not material_spec or not material_spec.strip():
        return DirectRunResult(
            success=False, backend_name=config.backend_name,
            error="No material specified. Provide an MP ID, formula, or --structure file.",
        )
    else:
        # Materials Project
        try:
            from shalom.mp_client import fetch_structure
            mp_result = fetch_structure(material_spec)
            atoms = mp_result.atoms
            formula = mp_result.formula
            mp_id = mp_result.mp_id
            structure_info = {
                "source": "materials_project",
                "mp_id": mp_id,
                "formula": formula,
                "energy_above_hull": mp_result.energy_above_hull,
                "space_group": mp_result.space_group,
            }
        except ImportError as e:
            return DirectRunResult(
                success=False, backend_name=config.backend_name,
                error=str(e),
            )
        except (EnvironmentError, ValueError) as e:
            return DirectRunResult(
                success=False, backend_name=config.backend_name,
                error=str(e),
            )

    if atoms is None:
        return DirectRunResult(
            success=False, backend_name=config.backend_name,
            error="No structure resolved.",
        )

    # 2. Validate structure
    if config.validate_structure:
        form = FormFiller.evaluate_atoms(atoms)
        if not form.is_valid:
            return DirectRunResult(
                success=False, backend_name=config.backend_name,
                structure_info=structure_info,
                error=f"Structure validation failed: {form.feedback}",
            )

    # 3. Create backend-specific DFT config
    auto_detected: Dict[str, Any] = {}
    if config.backend_name == "vasp":
        dft_config = _create_vasp_config(
            atoms, calc_type, config.accuracy,
            config.user_settings, config.potcar_preset, config.functional,
        )
        auto_detected = {
            k: v for k, v in {
                "ENCUT": dft_config.incar_settings.get("ENCUT"),
                "ISMEAR": dft_config.incar_settings.get("ISMEAR"),
                "is_2d": dft_config.is_2d,
                "kpoints": dft_config.kpoints.grid,
            }.items() if v is not None
        }
    elif config.backend_name == "qe":
        dft_config = _create_qe_config(
            atoms, calc_type, config.accuracy,
            config.user_settings, config.pseudo_dir, config.functional,
        )
        auto_detected = {
            k: v for k, v in {
                "ecutwfc": dft_config.system.get("ecutwfc"),
                "ecutrho": dft_config.system.get("ecutrho"),
                "is_2d": dft_config.is_2d,
                "kpoints": dft_config.kpoints.grid,
            }.items() if v is not None
        }
    else:
        dft_config = None

    # 4. Output directory
    output_dir = config.output_dir or _auto_output_dir(
        formula, mp_id, config.backend_name, calc_type,
    )

    # 5. Overwrite protection
    if os.path.exists(output_dir) and not config.force_overwrite:
        existing = os.listdir(output_dir)
        if existing:
            return DirectRunResult(
                success=False, backend_name=config.backend_name,
                output_dir=output_dir, structure_info=structure_info,
                error=(
                    f"Output directory '{output_dir}' already exists with "
                    f"{len(existing)} file(s). Use --force to overwrite or "
                    f"-o DIR to specify a different directory."
                ),
            )

    # 6. Write input files
    write_params: Dict[str, Any] = {}
    if dft_config is not None:
        write_params["config"] = dft_config

    try:
        backend.write_input(atoms, output_dir, **write_params)
    except Exception as e:
        return DirectRunResult(
            success=False, backend_name=config.backend_name,
            output_dir=output_dir, structure_info=structure_info,
            error=f"Failed to write input files: {e}",
        )

    # 7. List generated files
    files_generated = sorted(os.listdir(output_dir)) if os.path.isdir(output_dir) else []

    return DirectRunResult(
        success=True,
        output_dir=output_dir,
        structure_info=structure_info,
        backend_name=config.backend_name,
        files_generated=files_generated,
        auto_detected=auto_detected,
    )
