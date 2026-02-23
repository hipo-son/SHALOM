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
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

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
    potcar_preset: Literal["vasp_recommended", "mp_default"] = "vasp_recommended"
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


def _write_output_readme(
    output_dir: str,
    backend_name: str,
    calc_type: str,
    structure_info: Optional[Dict[str, Any]],
    auto_detected: Optional[Dict[str, Any]],
    files_generated: List[str],
) -> None:
    """Write a README.md into the DFT output folder explaining the generated files."""
    import datetime

    try:
        import shalom as _shalom_pkg
        version = getattr(_shalom_pkg, "__version__", "0.1.0")
    except Exception:
        version = "0.1.0"

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info = structure_info or {}
    formula = info.get("formula", "Unknown")
    mp_id = info.get("mp_id", "")
    spacegroup = info.get("space_group", "")
    e_hull = info.get("energy_above_hull")
    source = info.get("source", "")

    # Title line
    title = f"SHALOM DFT Calculation: {formula}"
    if mp_id:
        title += f" ({mp_id})"

    lines: List[str] = [
        f"# {title}",
        "",
        f"Generated: {now}  ",
        f"SHALOM v{version} | Backend: `{backend_name}` | Calc: `{calc_type}`",
        "",
    ]

    # Material information
    lines += ["## Material Information", ""]
    lines.append(f"- **Formula**: {formula}")
    if mp_id:
        lines.append(f"- **MP ID**: [{mp_id}](https://materialsproject.org/materials/{mp_id})")
    if spacegroup:
        lines.append(f"- **Spacegroup**: {spacegroup}")
    if e_hull is not None:
        lines.append(f"- **E above hull**: {e_hull:.4f} eV/atom")
    if source and not mp_id:
        lines.append(f"- **Source**: {source}")
    lines.append("")

    # Auto-detected parameters
    if auto_detected:
        lines += ["## Auto-Detected Parameters", ""]
        for key, val in auto_detected.items():
            lines.append(f"- **{key}**: {val}")
        lines.append("")

    # File descriptions
    _FILE_DESC: Dict[str, str] = {
        "POSCAR": "Crystal structure (fractional coordinates, VASP format)",
        "INCAR": "VASP calculation parameters",
        "KPOINTS": "k-point mesh for Brillouin zone sampling",
        "POTCAR.spec": "List of required pseudopotentials — obtain POTCAR files separately",
        "POTCAR": "Pseudopotential file (PAW, PBE)",
        "OUTCAR": "VASP output (energy, forces, stress, etc.)",
        "pw.in": "Quantum ESPRESSO input file (pw.x)",
        "pw.out": "Quantum ESPRESSO output (after calculation)",
        "atoms.txt": "ASE-format structure file (human-readable)",
    }
    if files_generated:
        lines += ["## Generated Files", ""]
        for fname in files_generated:
            desc = _FILE_DESC.get(fname, "")
            if desc:
                lines.append(f"- `{fname}` — {desc}")
            else:
                lines.append(f"- `{fname}`")
        lines.append("")

    # Next steps
    lines += ["## Next Steps", ""]
    if backend_name == "vasp":
        potcar_spec = f"{output_dir}/POTCAR.spec"
        lines += [
            f"1. Obtain POTCAR files listed in `{potcar_spec}`",
            f"2. Run VASP:",
            f"   ```bash",
            f"   cd {output_dir}",
            f"   mpirun -np 4 vasp_std > vasp.log 2>&1",
            f"   ```",
        ]
    else:
        lines += [
            "1. Ensure pseudopotentials are available:",
            "   ```bash",
            "   python -m shalom setup-qe --elements <ELEMENTS> --download",
            "   ```",
            f"2. Run Quantum ESPRESSO:",
            f"   ```bash",
            f"   cd {output_dir}",
            f"   mpirun -np 4 pw.x < pw.in > pw.out 2>&1",
            f"   ```",
            "   (On Windows, run from inside WSL2)",
        ]
    lines.append("")

    # Reproduce command
    lines += ["## Reproduce This Run", ""]
    cmd_parts = ["python -m shalom run"]
    if mp_id:
        cmd_parts.append(mp_id)
    elif source and source != "local file":
        cmd_parts.append(formula)
    else:
        cmd_parts.extend(["--structure", "<your_structure_file>"])
    cmd_parts.extend(["--backend", backend_name])
    if calc_type not in ("relaxation", "vc-relax"):
        cmd_parts.extend(["--calc", calc_type])
    lines.append("```bash")
    lines.append(" ".join(cmd_parts))
    lines.append("```")
    lines.append("")

    readme_path = os.path.join(output_dir, "README.md")
    try:
        with open(readme_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
    except OSError as exc:
        logger.warning("Could not write output README.md: %s", exc)


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
    potcar_preset: Literal["vasp_recommended", "mp_default"], functional: str,
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
    else:
        resolved = os.environ.get(
            "SHALOM_PSEUDO_DIR",
            str(Path.home() / "pseudopotentials"),
        )
        config.pseudo_dir = resolved
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
            result = ase_read(config.structure_file)
            atoms = result[0] if isinstance(result, list) else result
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

    # 8. Write README.md into the output folder
    _write_output_readme(
        output_dir=output_dir,
        backend_name=config.backend_name,
        calc_type=calc_type,
        structure_info=structure_info,
        auto_detected=auto_detected,
        files_generated=[f for f in files_generated if f != "README.md"],
    )

    return DirectRunResult(
        success=True,
        output_dir=output_dir,
        structure_info=structure_info,
        backend_name=config.backend_name,
        files_generated=files_generated,
        auto_detected=auto_detected,
    )
