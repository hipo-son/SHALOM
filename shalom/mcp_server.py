"""SHALOM MCP Server — Claude Code integration via Model Context Protocol.

Exposes SHALOM's DFT tools as MCP tools that Claude Code can call directly.
Users with a Claude subscription can use natural language to run material
discovery workflows without a separate API key.

Setup::

    # Install
    pip install "shalom[mcp]"

    # Register in Claude Code
    claude mcp add shalom -- python -m shalom.mcp_server

    # Or add to .mcp.json (project-scoped)
    # See project root .mcp.json for example

Usage in Claude Code::

    "Si의 SCF 계산 입력 파일을 만들어줘"
    "mp-1040425 그래핀 밴드 구조 계산해줘"
    "QE 환경이 제대로 설정되어 있는지 확인해줘"
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List, Optional

# CRITICAL: stdio transport uses stdout for JSON-RPC protocol.
# All logging MUST go to stderr. Never use print() in this module.
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("shalom.mcp")

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    _err = (
        "MCP server requires the 'mcp' package.\n"
        "Install with: pip install 'shalom[mcp]'\n"
        "  or: pip install 'mcp[cli]>=1.2.0'"
    )
    print(_err, file=sys.stderr)
    sys.exit(1)


mcp = FastMCP("shalom", json_response=True)


# ---------------------------------------------------------------------------
# Helper: load atoms from material spec or structure file
# ---------------------------------------------------------------------------

def _load_atoms(
    material: str = "",
    structure_file: Optional[str] = None,
) -> tuple:
    """Load ASE Atoms from a material spec or structure file.

    Priority: structure_file > MP ID/formula > ASE bulk fallback.

    Returns:
        ``(atoms, source)`` where *source* is one of ``"structure_file"``,
        ``"materials_project"``, or ``"ase_bulk_fallback"``.
    """
    from ase import Atoms  # noqa: F811

    if structure_file:
        from ase.io import read as ase_read
        return ase_read(structure_file), "structure_file"

    if not material:
        raise ValueError(
            "Provide either 'material' (MP ID or formula) or 'structure_file'."
        )

    # Try Materials Project first
    try:
        from shalom.mp_client import is_mp_available, fetch_structure

        if is_mp_available():
            result = fetch_structure(material)
            return result.atoms, "materials_project"
    except Exception as mp_exc:
        logger.info("MP fetch failed for '%s': %s. Trying ASE bulk.", material, mp_exc)

    # Fallback: ASE bulk builder
    try:
        from ase.build import bulk
        atoms = bulk(material)
        logger.info("Built bulk %s from ASE.", material)
        return atoms, "ase_bulk_fallback"
    except Exception:
        raise ValueError(
            f"Cannot resolve material '{material}'. "
            "Provide an MP ID (mp-19717), chemical formula (Si), "
            "or a structure file path."
        )


# ---------------------------------------------------------------------------
# Tool 1: search_material
# ---------------------------------------------------------------------------

@mcp.tool()
def search_material(query: str, max_results: int = 5) -> dict:
    """Search for a material structure on Materials Project.

    Args:
        query: MP ID (e.g. mp-1040425) or chemical formula (e.g. Fe2O3, Si, C).
        max_results: Maximum number of results when searching by formula.

    Returns:
        Dictionary with search results including formula, space group,
        energy above hull, and structure details.
    """
    try:
        from shalom.mp_client import (
            is_mp_available, is_mp_id, fetch_by_mp_id, search_by_formula,
        )

        if not is_mp_available():
            return {
                "success": False,
                "error": (
                    "mp-api package not installed. "
                    "Install with: pip install 'shalom[mp]'"
                ),
            }

        if is_mp_id(query):
            r = fetch_by_mp_id(query)
            atoms = r.atoms
            return {
                "success": True,
                "results": [{
                    "mp_id": r.mp_id,
                    "formula": r.formula,
                    "space_group": r.space_group,
                    "energy_above_hull_eV": r.energy_above_hull,
                    "n_atoms": len(atoms),
                    "cell_parameters": atoms.cell.cellpar().tolist(),
                    "positions": atoms.get_scaled_positions().tolist(),
                    "symbols": list(atoms.get_chemical_symbols()),
                }],
            }
        else:
            results = search_by_formula(query, max_results=max_results)
            return {
                "success": True,
                "results": [
                    {
                        "mp_id": r.mp_id,
                        "formula": r.formula,
                        "space_group": r.space_group,
                        "energy_above_hull_eV": r.energy_above_hull,
                        "n_atoms": len(r.atoms),
                    }
                    for r in results
                ],
            }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Tool 2: generate_dft_input
# ---------------------------------------------------------------------------

@mcp.tool()
def generate_dft_input(
    material: str = "",
    backend: str = "qe",
    calc_type: str = "scf",
    accuracy: str = "standard",
    output_dir: Optional[str] = None,
    pseudo_dir: Optional[str] = None,
    structure_file: Optional[str] = None,
) -> dict:
    """Generate DFT input files for a material.

    Creates complete input file sets (QE pw.in or VASP INCAR/POSCAR/KPOINTS)
    ready for DFT calculation.

    Args:
        material: Material identifier. MP ID (mp-19717), formula (Fe2O3, Si),
            or empty string when using structure_file.
        backend: DFT backend — 'qe' (Quantum ESPRESSO) or 'vasp'.
        calc_type: Calculation type. QE: scf/relax/vc-relax/bands/nscf.
            VASP: relaxation/static/band_structure/dos.
        accuracy: Accuracy preset — 'standard' or 'precise'.
        output_dir: Output directory path. Auto-generated if not provided.
        pseudo_dir: QE pseudopotential directory (default: $SHALOM_PSEUDO_DIR).
        structure_file: Path to local structure file (POSCAR, CIF, etc.).
    """
    if backend not in ("qe", "vasp"):
        return {
            "success": False,
            "error": f"Invalid backend '{backend}'. Use 'qe' or 'vasp'.",
        }
    try:
        from shalom.direct_run import direct_run, DirectRunConfig

        config = DirectRunConfig(
            backend_name=backend,
            calc_type=calc_type,
            accuracy=accuracy,
            output_dir=output_dir,
            pseudo_dir=pseudo_dir,
            structure_file=structure_file,
        )
        result = direct_run(material, config)
        return {
            "success": result.success,
            "output_dir": result.output_dir,
            "files_generated": result.files_generated,
            "backend": result.backend_name,
            "structure_info": result.structure_info,
            "auto_detected": result.auto_detected,
            "error": result.error,
        }
    except FileNotFoundError as exc:
        return {
            "success": False,
            "error": (
                f"File not found: {exc}. "
                "Check that the structure_file path is correct."
            ),
        }
    except Exception as exc:
        return {
            "success": False,
            "error": (
                f"{exc}. Provide an MP ID (mp-19717), chemical formula (Si), "
                "or a valid structure_file path."
            ),
        }


# ---------------------------------------------------------------------------
# Tool 3: run_workflow
# ---------------------------------------------------------------------------

@mcp.tool()
def run_workflow(
    material: str = "",
    output_dir: str = "./shalom_workflow",
    pseudo_dir: Optional[str] = None,
    nprocs: int = 1,
    mpi_command: str = "mpirun",
    pw_executable: str = "pw.x",
    dos_executable: str = "dos.x",
    accuracy: str = "standard",
    skip_relax: bool = False,
    is_2d: bool = False,
    timeout: int = 7200,
    npoints_kpath: int = 40,
    structure_file: Optional[str] = None,
    dos_emin: float = -20.0,
    dos_emax: float = 10.0,
    dos_deltaE: float = 0.01,
) -> dict:
    """Run the full 5-step QE workflow: vc-relax → scf → bands → nscf → dos.

    Produces band structure and DOS plots. Requires QE (pw.x, dos.x) in PATH
    and pseudopotentials in pseudo_dir.

    Args:
        material: MP ID (mp-1040425) or chemical formula (Si, Fe2O3),
            or empty string when using structure_file.
        output_dir: Output directory for all workflow steps.
        pseudo_dir: QE pseudopotential directory (default: $SHALOM_PSEUDO_DIR).
        nprocs: Number of MPI processes for parallel execution.
        mpi_command: MPI launcher command (mpirun, srun, etc.).
        pw_executable: Path or name of the pw.x binary.
        dos_executable: Path or name of the dos.x binary.
        accuracy: Accuracy preset — 'standard' or 'precise'.
        skip_relax: Skip the vc-relax step (use input structure as-is).
        is_2d: Treat as 2D material (kz=0 k-path, 2D isolation flags).
        timeout: Per-step timeout in seconds.
        npoints_kpath: Number of k-points per segment on the band path.
        structure_file: Path to local structure file (POSCAR, CIF, etc.).
        dos_emin: DOS energy window minimum in eV.
        dos_emax: DOS energy window maximum in eV.
        dos_deltaE: DOS energy step size in eV.
    """
    try:
        atoms, source = _load_atoms(material, structure_file)

        from shalom.workflows.standard import StandardWorkflow

        wf = StandardWorkflow(
            atoms=atoms,
            output_dir=output_dir,
            pseudo_dir=pseudo_dir,
            nprocs=nprocs,
            mpi_command=mpi_command,
            pw_executable=pw_executable,
            dos_executable=dos_executable,
            accuracy=accuracy,
            skip_relax=skip_relax,
            is_2d=is_2d,
            timeout=timeout,
            npoints_kpath=npoints_kpath,
            dos_emin=dos_emin,
            dos_emax=dos_emax,
            dos_deltaE=dos_deltaE,
        )
        result = wf.run()

        response: Dict[str, Any] = {
            "success": True,
            "output_dir": output_dir,
            "fermi_energy_eV": result.get("fermi_energy"),
            "bands_png": result.get("bands_png"),
            "dos_png": result.get("dos_png"),
            "calc_dirs": result.get("calc_dirs", {}),
        }
        if source == "ase_bulk_fallback":
            response["warning"] = (
                f"Structure for '{material}' was built from ASE bulk (idealized). "
                "For accurate results, use an MP ID or provide a structure file."
            )
        return response
    except FileNotFoundError as exc:
        return {
            "success": False,
            "error": (
                f"File not found: {exc}. "
                "Check that pw.x/dos.x are installed and pseudo_dir is set. "
                "Run check_qe_setup to diagnose."
            ),
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Tool 4: execute_dft
# ---------------------------------------------------------------------------

@mcp.tool()
def execute_dft(
    calc_dir: str,
    nprocs: int = 1,
    timeout: int = 7200,
    mpi_command: str = "mpirun",
    max_retries: int = 3,
) -> dict:
    """Execute a QE DFT calculation with automatic error recovery.

    Runs pw.x on an existing input directory. Automatically diagnoses
    and corrects common QE errors (convergence, memory, etc.).

    Args:
        calc_dir: Directory containing pw.in input file.
        nprocs: Number of MPI processes.
        timeout: Execution timeout in seconds.
        mpi_command: MPI launcher command (mpirun, srun, etc.).
        max_retries: Maximum error recovery retries.
    """
    try:
        from shalom.backends.qe import QEBackend
        from shalom.backends.runner import (
            ExecutionConfig, ExecutionRunner, execute_with_recovery,
        )
        from shalom.backends.qe_error_recovery import QEErrorRecoveryEngine

        exec_config = ExecutionConfig(
            nprocs=nprocs,
            mpi_command=mpi_command,
            timeout_seconds=timeout,
        )
        runner = ExecutionRunner(config=exec_config)

        prereq_errors = runner.validate_prerequisites(calc_dir)
        if prereq_errors:
            return {"success": False, "error": "; ".join(prereq_errors)}

        backend = QEBackend()
        recovery = QEErrorRecoveryEngine()

        exec_result, dft_result, history = execute_with_recovery(
            backend, runner, recovery, calc_dir,
            config=None, atoms=None, max_retries=max_retries,
        )

        response: Dict[str, Any] = {
            "success": exec_result.success,
            "wall_time_seconds": exec_result.wall_time_seconds,
            "corrections": len(history) if history else 0,
        }

        if dft_result is not None:
            response.update({
                "converged": dft_result.is_converged,
                "energy_eV": dft_result.energy,
                "forces_max_eV_A": dft_result.forces_max,
                "quality_warnings": dft_result.quality_warnings,
            })

        if history:
            response["correction_history"] = [
                {"error_type": h.get("error_type"), "step": h.get("step")}
                for h in history
            ]

        if not exec_result.success:
            response["error"] = exec_result.error_message

        return response
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Tool 5: parse_dft_output
# ---------------------------------------------------------------------------

@mcp.tool()
def parse_dft_output(calc_dir: str, backend: str = "qe") -> dict:
    """Parse DFT output from a completed calculation directory.

    Extracts energy, forces, convergence status, and other results
    from QE pw.out or VASP OUTCAR/vasprun.xml.

    Args:
        calc_dir: Directory containing DFT output files.
        backend: DFT backend — 'qe' or 'vasp'.
    """
    if backend not in ("qe", "vasp"):
        return {
            "success": False,
            "error": f"Invalid backend '{backend}'. Use 'qe' or 'vasp'.",
        }
    try:
        from shalom.backends import get_backend

        b = get_backend(backend)
        result = b.parse_output(calc_dir)

        response: Dict[str, Any] = {
            "success": True,
            "converged": result.is_converged,
            "energy_eV": result.energy,
        }

        if result.forces_max is not None:
            response["forces_max_eV_A"] = result.forces_max
        if result.stress_tensor is not None and len(result.stress_tensor) >= 3:
            # Pressure = negative trace of stress tensor / 3 (Voigt: xx, yy, zz, ...)
            pressure_kbar = -(
                result.stress_tensor[0] + result.stress_tensor[1] + result.stress_tensor[2]
            ) / 3.0
            response["pressure_kbar"] = pressure_kbar
        if result.quality_warnings:
            response["quality_warnings"] = result.quality_warnings

        return response
    except FileNotFoundError as exc:
        expected = "pw.out" if backend == "qe" else "OUTCAR or vasprun.xml"
        return {
            "success": False,
            "error": (
                f"Output file not found: {exc}. "
                f"Ensure {expected} exists in '{calc_dir}'. "
                "Run the DFT calculation first with execute_dft or run_workflow."
            ),
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Tool 6: plot_bands
# ---------------------------------------------------------------------------

@mcp.tool()
def plot_bands(
    calc_dir: str,
    fermi_from: Optional[str] = None,
    emin: float = -6.0,
    emax: float = 6.0,
    title: Optional[str] = None,
    output: Optional[str] = None,
) -> dict:
    """Plot the electronic band structure from a QE bands calculation.

    Reads data-file-schema.xml from the calculation directory and
    generates a publication-quality band structure plot.

    Args:
        calc_dir: QE bands calculation directory.
        fermi_from: Directory to read Fermi energy from (e.g. NSCF run dir).
            If not provided, reads from calc_dir.
        emin: Energy window minimum in eV relative to Fermi level.
        emax: Energy window maximum in eV relative to Fermi level.
        title: Plot title.
        output: Output file path. Default: calc_dir/bands.png.
    """
    try:
        from shalom.backends.qe_parser import (
            find_xml_path, parse_xml_bands, extract_fermi_energy,
        )
        from shalom.plotting.band_plot import BandStructurePlotter

        # Fermi energy
        fermi: Optional[float] = None
        if fermi_from:
            fermi = extract_fermi_energy(os.path.join(fermi_from, "pw.out"))
        if fermi is None:
            fermi = extract_fermi_energy(os.path.join(calc_dir, "pw.out"))
        if fermi is None:
            fermi = 0.0

        xml_path = find_xml_path(calc_dir)
        if xml_path is None:
            return {"success": False, "error": f"data-file-schema.xml not found in {calc_dir}"}

        bs = parse_xml_bands(xml_path, fermi_energy=fermi)
        out = output or os.path.join(calc_dir, "bands.png")

        plotter = BandStructurePlotter(bs)
        plotter.plot(output_path=out, title=title, energy_window=(emin, emax))

        return {
            "success": True,
            "output_path": out,
            "n_kpoints": bs.eigenvalues.shape[0],
            "n_bands": bs.eigenvalues.shape[1],
            "fermi_energy_eV": fermi,
        }
    except ImportError:
        return {
            "success": False,
            "error": "matplotlib not installed. Run: pip install 'shalom[plotting]'",
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Tool 7: plot_dos
# ---------------------------------------------------------------------------

@mcp.tool()
def plot_dos(
    calc_dir: str,
    fermi_from: Optional[str] = None,
    emin: float = -6.0,
    emax: float = 6.0,
    title: Optional[str] = None,
    output: Optional[str] = None,
) -> dict:
    """Plot the density of states (DOS) from a QE NSCF/dos.x calculation.

    Reads pwscf.dos from the calculation directory and generates
    a publication-quality DOS plot.

    Args:
        calc_dir: QE NSCF calculation directory containing pwscf.dos.
        fermi_from: Directory to read Fermi energy from.
            If not provided, reads from calc_dir.
        emin: Energy window minimum in eV relative to Fermi level.
        emax: Energy window maximum in eV relative to Fermi level.
        title: Plot title.
        output: Output file path. Default: calc_dir/dos.png.
    """
    try:
        from shalom.backends.qe_parser import parse_dos_file, extract_fermi_energy
        from shalom.plotting.dos_plot import DOSPlotter

        dos_path = os.path.join(calc_dir, "pwscf.dos")
        if not os.path.isfile(dos_path):
            return {"success": False, "error": f"pwscf.dos not found in {calc_dir}"}

        # Fermi energy
        fermi: Optional[float] = None
        if fermi_from:
            fermi = extract_fermi_energy(os.path.join(fermi_from, "pw.out"))
        if fermi is None:
            fermi = extract_fermi_energy(os.path.join(calc_dir, "pw.out"))
        if fermi is None:
            fermi = 0.0

        dos_data = parse_dos_file(dos_path)
        dos_data.fermi_energy = fermi

        out = output or os.path.join(calc_dir, "dos.png")
        plotter = DOSPlotter(dos_data)
        plotter.plot(output_path=out, title=title, energy_window=(emin, emax))

        return {
            "success": True,
            "output_path": out,
            "n_points": len(dos_data.energies),
            "energy_range_eV": [float(dos_data.energies.min()), float(dos_data.energies.max())],
            "fermi_energy_eV": fermi,
        }
    except ImportError:
        return {
            "success": False,
            "error": "matplotlib not installed. Run: pip install 'shalom[plotting]'",
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Tool 8: run_convergence
# ---------------------------------------------------------------------------

@mcp.tool()
def run_convergence(
    material: str = "",
    test_type: str = "cutoff",
    output_dir: str = "./convergence_test",
    values: Optional[str] = None,
    ecutwfc: Optional[float] = None,
    kgrid: Optional[str] = None,
    pseudo_dir: Optional[str] = None,
    nprocs: int = 1,
    mpi_command: str = "mpirun",
    timeout: int = 3600,
    accuracy: str = "standard",
    threshold: float = 1e-3,
    structure_file: Optional[str] = None,
) -> dict:
    """Run cutoff or k-point convergence tests.

    Sweeps ecutwfc or k-point resolution to find converged DFT parameters.
    Run cutoff convergence FIRST, then k-point convergence with the
    converged ecutwfc value.

    Args:
        material: MP ID or chemical formula, or empty when using structure_file.
        test_type: Type of convergence test — 'cutoff' or 'kpoints'.
        output_dir: Output directory for convergence runs.
        values: Comma-separated values to test. Cutoff: ecutwfc in Ry (e.g. "30,40,50,60").
            Kpoints: resolutions in 1/Bohr (e.g. "20,30,40,50").
            Uses sensible defaults if not provided.
        ecutwfc: Fixed ecutwfc for k-point test (required for test_type='kpoints').
        kgrid: Fixed Monkhorst-Pack k-grid for cutoff test (e.g. "4,4,4").
            Controls the k-point grid during ecutwfc sweep to isolate the
            cutoff variable. If not set, uses the default from the QE preset.
        pseudo_dir: QE pseudopotential directory.
        nprocs: Number of MPI processes.
        mpi_command: MPI launcher command (mpirun, srun, etc.).
        timeout: Per-step timeout in seconds.
        accuracy: Accuracy preset — 'standard' or 'precise'.
        threshold: Convergence threshold in eV/atom.
        structure_file: Path to local structure file.
    """
    try:
        atoms, source = _load_atoms(material, structure_file)

        # Parse values
        if values:
            value_list = [float(v.strip()) for v in values.split(",")]
        elif test_type == "cutoff":
            value_list = [30.0, 40.0, 50.0, 60.0, 80.0]
        else:
            value_list = [20.0, 30.0, 40.0, 50.0]

        # Parse kgrid
        parsed_kgrid: Optional[List[int]] = None
        if kgrid:
            parsed_kgrid = [int(v.strip()) for v in kgrid.split(",")]

        if test_type == "cutoff":
            from shalom.workflows.convergence import CutoffConvergence

            conv = CutoffConvergence(
                atoms=atoms,
                output_dir=output_dir,
                values=value_list,
                kgrid=parsed_kgrid,
                pseudo_dir=pseudo_dir,
                nprocs=nprocs,
                mpi_command=mpi_command,
                timeout=timeout,
                accuracy=accuracy,
                threshold_per_atom=threshold,
            )
        elif test_type == "kpoints":
            from shalom.workflows.convergence import KpointConvergence

            conv = KpointConvergence(
                atoms=atoms,
                output_dir=output_dir,
                resolutions=value_list,
                ecutwfc=ecutwfc,
                pseudo_dir=pseudo_dir,
                nprocs=nprocs,
                mpi_command=mpi_command,
                timeout=timeout,
                accuracy=accuracy,
                threshold_per_atom=threshold,
            )
        else:
            return {
                "success": False,
                "error": (
                    f"Invalid test_type: '{test_type}'. "
                    "Use 'cutoff' (ecutwfc sweep) or 'kpoints' (k-mesh sweep)."
                ),
            }

        result = conv.run()

        response: Dict[str, Any] = {
            "success": True,
            "converged": result.converged_value is not None,
            "converged_value": result.converged_value,
            "summary": result.summary(),
        }

        try:
            plot_path = conv.plot(result)
            if plot_path:
                response["plot_path"] = plot_path
        except Exception:
            pass

        if source == "ase_bulk_fallback":
            response["warning"] = (
                f"Structure for '{material}' was built from ASE bulk (idealized). "
                "For accurate convergence tests, use an MP ID or structure file."
            )

        return response
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Tool 9: check_qe_setup
# ---------------------------------------------------------------------------

@mcp.tool()
def check_qe_setup(
    pseudo_dir: Optional[str] = None,
    elements: Optional[str] = None,
) -> dict:
    """Check QE (Quantum ESPRESSO) environment and prerequisites.

    Verifies pw.x/dos.x installation, pseudopotential directory,
    and checks for specific element pseudopotential files.

    Args:
        pseudo_dir: Pseudopotential directory to check.
            Default: $SHALOM_PSEUDO_DIR or ~/pseudopotentials.
        elements: Comma-separated list of elements to check (e.g. "Si,Fe,O").
            Default: check all SSSP elements.
    """
    import shutil
    from pathlib import Path

    issues: List[str] = []
    info: Dict[str, Any] = {}

    # Check executables
    pw_path = shutil.which("pw.x")
    dos_path_exec = shutil.which("dos.x")
    info["pw_x"] = pw_path or "NOT FOUND"
    info["dos_x"] = dos_path_exec or "NOT FOUND"
    if not pw_path:
        issues.append("pw.x not found in PATH. Install QE: sudo apt install quantum-espresso")
    if not dos_path_exec:
        issues.append("dos.x not found in PATH.")

    # Check pseudo_dir
    from shalom.backends.qe_config import SSSP_ELEMENTS, get_pseudo_filename

    default_pseudo = str(Path.home() / "pseudopotentials")
    pd = pseudo_dir or os.environ.get("SHALOM_PSEUDO_DIR", default_pseudo)
    pseudo_path = Path(pd).expanduser().resolve()
    info["pseudo_dir"] = str(pseudo_path)

    if not pseudo_path.is_dir():
        issues.append(f"pseudo_dir not found: {pseudo_path}")
    else:
        # Check element pseudopotentials
        if elements:
            check_elements = [e.strip() for e in elements.split(",")]
        else:
            check_elements = sorted(SSSP_ELEMENTS.keys())

        missing = []
        found = 0
        for el in check_elements:
            if el not in SSSP_ELEMENTS:
                continue
            upf = get_pseudo_filename(el)
            upf_path = pseudo_path / upf
            if upf_path.exists():
                found += 1
            else:
                # Case-insensitive check
                exists = False
                try:
                    for entry in pseudo_path.iterdir():
                        if entry.name.lower() == upf.lower():
                            exists = True
                            break
                except OSError:
                    pass
                if exists:
                    found += 1
                else:
                    missing.append({"element": el, "filename": upf})

        info["pseudopotentials_found"] = found
        info["pseudopotentials_total"] = found + len(missing)
        if missing:
            issues.append(f"{len(missing)} pseudopotential(s) missing")
            info["missing_pseudopotentials"] = missing[:10]

    # MP API check
    try:
        from shalom.mp_client import is_mp_available
        info["mp_api"] = "available" if is_mp_available() else "not installed"
    except ImportError:
        info["mp_api"] = "not installed"

    info["mp_api_key_set"] = bool(os.environ.get("MP_API_KEY"))

    return {
        "success": len(issues) == 0,
        "ready": len(issues) == 0,
        "issues": issues,
        "info": info,
    }


# ---------------------------------------------------------------------------
# Tool 10: run_pipeline (LLM-driven multi-agent, requires API key)
# ---------------------------------------------------------------------------

@mcp.tool()
def run_pipeline(
    objective: str,
    backend: str = "qe",
    provider: str = "openai",
    model: Optional[str] = None,
    material: Optional[str] = None,
    steps: Optional[str] = None,
    output_dir: Optional[str] = None,
    calc_type: str = "relaxation",
    accuracy: str = "standard",
    execute: bool = False,
    nprocs: int = 1,
    max_outer_loops: int = 1,
    selector_mode: str = "simple",
    base_url: Optional[str] = None,
) -> dict:
    """Run the full LLM-driven multi-agent material discovery pipeline.

    This is the autonomous pipeline with Design → Simulation → Review agents.
    Each agent uses LLM reasoning with structured output, sandboxed code
    execution, parallel evaluators, and closed-loop feedback.

    REQUIRES an API key OR a local LLM server (via base_url).
    Set OPENAI_API_KEY or ANTHROPIC_API_KEY, or provide base_url for a local
    server (Ollama, vLLM, etc.). This is separate from Claude Code subscription.

    For users WITHOUT an API key or local LLM, use the other tools
    (search_material, generate_dft_input, run_workflow, etc.) and let
    Claude Code handle the reasoning directly.

    Args:
        objective: Natural language research objective.
            E.g. "Find a 2D HER catalyst" or "Stable Li-ion cathode material".
        backend: DFT backend — 'qe' or 'vasp'.
        provider: LLM provider — 'openai' or 'anthropic'.
        model: LLM model name. Default: gpt-4o (openai) or claude-sonnet-4-6 (anthropic).
        material: Skip Design layer and use this material directly (e.g. 'MoS2').
        steps: Comma-separated pipeline steps: 'design,simulation,review'.
            Default: design,simulation.
        output_dir: Output directory for pipeline results.
        calc_type: DFT calculation type (relaxation, scf, etc.).
        accuracy: Accuracy preset — 'standard' or 'precise'.
        execute: Run DFT after input generation (QE only).
        nprocs: Number of MPI processes for DFT execution.
        max_outer_loops: Max Design→Sim→Review iterations for closed-loop retry.
        selector_mode: Fine selector mode — 'simple' or 'multi_agent'
            (multi_agent uses 6 parallel specialist evaluators).
        base_url: Custom LLM API base URL for local/self-hosted models.
            E.g. http://localhost:11434/v1 (Ollama), http://localhost:8000/v1 (vLLM).
            Falls back to $SHALOM_LLM_BASE_URL env var.
    """
    # Resolve base_url
    resolved_base_url = base_url or os.environ.get("SHALOM_LLM_BASE_URL")

    # Check API key (not required when using a local LLM via base_url)
    key_env = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    if not os.environ.get(key_env) and not resolved_base_url:
        return {
            "success": False,
            "error": (
                f"{key_env} not set. This tool requires an LLM API key for "
                f"multi-agent reasoning. Set it with: export {key_env}='your-key'\n"
                "Or use a local LLM server via base_url parameter "
                "(e.g. http://localhost:11434/v1 for Ollama).\n"
                "If you don't have an API key, use the other SHALOM tools "
                "(search_material, generate_dft_input, run_workflow) instead — "
                "Claude Code can handle the reasoning directly."
            ),
        }

    try:
        from shalom.pipeline import Pipeline, PipelineConfig
        from shalom.core.schemas import PipelineStatus

        resolved_model = model
        if resolved_model is None:
            resolved_model = (
                "claude-sonnet-4-6" if provider == "anthropic" else "gpt-4o"
            )

        parsed_steps = None
        if steps:
            parsed_steps = [s.strip() for s in steps.split(",")]

        config = PipelineConfig(
            backend_name=backend,
            provider_type=provider,
            model_name=resolved_model,
            output_dir=output_dir or "",
            calc_type=calc_type,
            accuracy=accuracy,
            execute=execute,
            nprocs=nprocs,
            max_outer_loops=max_outer_loops,
            selector_mode=selector_mode,
            material_name=material,
            steps=parsed_steps,
            base_url=resolved_base_url,
        )

        pipeline = Pipeline(objective=objective, config=config)
        result = pipeline.run()

        response: Dict[str, Any] = {
            "success": result.status in (
                PipelineStatus.COMPLETED,
                PipelineStatus.COMPLETED_DESIGN,
                PipelineStatus.AWAITING_DFT,
            ),
            "status": result.status.value,
            "objective": result.objective,
            "steps_completed": result.steps_completed,
        }

        if result.ranked_material:
            response["material"] = {
                "name": result.ranked_material.candidate.material_name,
                "score": result.ranked_material.score,
                "reasoning": result.ranked_material.ranking_justification,
            }

        if result.candidates:
            response["candidates"] = [
                {
                    "name": c.material_name,
                    "elements": c.elements,
                    "reasoning": c.reasoning,
                }
                for c in result.candidates
            ]

        if result.structure_path:
            response["structure_path"] = result.structure_path
        if result.execution_wall_time is not None:
            response["execution_wall_time_s"] = result.execution_wall_time
        if result.quality_warnings:
            response["quality_warnings"] = result.quality_warnings
        if result.error_message:
            response["error"] = result.error_message
        if result.elapsed_seconds is not None:
            response["elapsed_seconds"] = result.elapsed_seconds
        if result.correction_history:
            response["corrections"] = len(result.correction_history)

        if result.review_result:
            response["review"] = {
                "successful": result.review_result.is_successful,
                "feedback": result.review_result.feedback_for_design,
            }

        return response
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Tool 11: analyze_elastic
# ---------------------------------------------------------------------------

@mcp.tool()
def analyze_elastic(elastic_tensor_json: str) -> dict:
    """Analyze a 6x6 elastic tensor to derive mechanical properties.

    Computes bulk modulus, shear modulus, Young's modulus, Poisson ratio,
    universal anisotropy index, and Born mechanical stability.

    Args:
        elastic_tensor_json: 6x6 Voigt elastic tensor in GPa as a JSON array.
            Example: "[[165.7,63.9,63.9,0,0,0],[63.9,165.7,63.9,0,0,0],
            [63.9,63.9,165.7,0,0,0],[0,0,0,79.6,0,0],[0,0,0,0,79.6,0],
            [0,0,0,0,0,79.6]]"
    """
    try:
        from shalom.analysis.elastic import is_elastic_available, analyze_elastic_tensor

        if not is_elastic_available():
            return {
                "success": False,
                "error": (
                    "pymatgen not installed. "
                    "Install with: pip install 'shalom[analysis]'"
                ),
            }

        import json
        tensor = json.loads(elastic_tensor_json)

        result = analyze_elastic_tensor(tensor)

        return {
            "success": True,
            "bulk_modulus_GPa": result.bulk_modulus_vrh,
            "shear_modulus_GPa": result.shear_modulus_vrh,
            "youngs_modulus_GPa": result.youngs_modulus,
            "poisson_ratio": result.poisson_ratio,
            "universal_anisotropy": result.universal_anisotropy,
            "is_mechanically_stable": result.is_stable,
            "stability_violations": result.stability_violations,
        }
    except json.JSONDecodeError as exc:
        return {
            "success": False,
            "error": f"Invalid JSON: {exc}. Provide a 6x6 nested array.",
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# Tool 12: Phonon analysis  ------------------------------------------------


@mcp.tool()
def analyze_phonon_properties(
    structure_file: str,
    supercell: str = "2,2,2",
    force_sets_file: str = "",
    force_constants_file: str = "",
    mesh: str = "20,20,20",
) -> dict:
    """Analyze phonon properties from a structure and force data.

    Computes phonon band structure, density of states, thermal properties
    (Cv, entropy, free energy), and dynamical stability.

    Requires phonopy: ``pip install 'shalom[phonon]'``

    Args:
        structure_file: Path to unit cell structure (POSCAR, CIF, etc.).
        supercell: Supercell dimensions as "Nx,Ny,Nz" (e.g. "2,2,2").
        force_sets_file: Path to phonopy FORCE_SETS file.
        force_constants_file: Path to FORCE_CONSTANTS or .hdf5 file.
        mesh: Q-point mesh as "Nx,Ny,Nz" for phonon DOS (default: "20,20,20").
    """
    try:
        from shalom.analysis.phonon import is_phonopy_available

        if not is_phonopy_available():
            return {
                "success": False,
                "error": (
                    "phonopy not installed. "
                    "Install with: pip install 'shalom[phonon]'"
                ),
            }

        import numpy as np
        from ase.io import read as ase_read

        atoms = ase_read(structure_file)

        # Parse supercell
        parts = supercell.replace("x", ",").split(",")
        if len(parts) == 3:
            sc = [
                [int(parts[0]), 0, 0],
                [0, int(parts[1]), 0],
                [0, 0, int(parts[2])],
            ]
        else:
            return {"success": False, "error": f"Invalid supercell: '{supercell}'"}

        mesh_list = [int(x) for x in mesh.split(",")]

        if force_constants_file:
            from phonopy.file_IO import parse_FORCE_CONSTANTS
            from shalom.analysis.phonon import analyze_phonon_from_force_constants

            fc = parse_FORCE_CONSTANTS(filename=force_constants_file)
            result = analyze_phonon_from_force_constants(
                atoms, fc, sc, mesh=mesh_list,
            )
        elif force_sets_file:
            from phonopy.file_IO import parse_FORCE_SETS
            from shalom.analysis.phonon import (
                generate_phonon_displacements,
                _run_phonon_analysis,
            )

            force_sets_data = parse_FORCE_SETS(filename=force_sets_file)
            _, ph = generate_phonon_displacements(atoms, sc)
            ph.dataset = force_sets_data
            ph.produce_force_constants()
            result = _run_phonon_analysis(ph, mesh_list, 51, 0.0, 1000.0, 10.0)
        else:
            return {
                "success": False,
                "error": "Provide force_sets_file or force_constants_file.",
            }

        # Thermal at 300K
        cv_300k = None
        entropy_300k = None
        fe_300k = None
        if result.thermal_temperatures is not None:
            idx = int(np.argmin(np.abs(result.thermal_temperatures - 300.0)))
            if result.thermal_cv is not None:
                cv_300k = round(float(result.thermal_cv[idx]), 2)
            if result.thermal_entropy is not None:
                entropy_300k = round(float(result.thermal_entropy[idx]), 2)
            if result.thermal_free_energy is not None:
                fe_300k = round(float(result.thermal_free_energy[idx]), 2)

        return {
            "success": True,
            "is_dynamically_stable": result.is_stable,
            "min_frequency_THz": (
                round(result.min_frequency_THz, 4)
                if result.min_frequency_THz is not None
                else None
            ),
            "n_imaginary_modes": len(result.imaginary_modes),
            "n_branches": result.n_branches,
            "cv_300K_J_per_K_mol": cv_300k,
            "entropy_300K_J_per_K_mol": entropy_300k,
            "free_energy_300K_kJ_per_mol": fe_300k,
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Tool 13: analyze_electronic_structure
# ---------------------------------------------------------------------------


@mcp.tool()
def analyze_electronic_structure(
    bands_xml_path: str = "",
    calc_dir: str = "",
    dos_file_path: str = "",
    fermi_energy: Optional[float] = None,
) -> dict:
    """Analyze electronic band structure from QE calculation results.

    Computes band gap, VBM/CBM, effective masses, metallicity, and DOS
    at Fermi level. No additional dependencies beyond numpy.

    Args:
        bands_xml_path: Path to QE data-file-schema.xml with band eigenvalues.
        calc_dir: QE bands calculation directory (auto-finds XML if bands_xml_path
            is empty).
        dos_file_path: Optional path to pwscf.dos for DOS-at-Fermi analysis.
        fermi_energy: Fermi energy in eV. Auto-detected from XML/pw.out if omitted.
    """
    try:
        from shalom.analysis.electronic import analyze_band_structure
        from shalom.backends.qe_parser import (
            find_xml_path, parse_xml_bands, extract_fermi_energy,
        )

        xml_path = bands_xml_path
        if not xml_path and calc_dir:
            xml_path = find_xml_path(calc_dir)
            if xml_path is None:
                return {
                    "success": False,
                    "error": f"data-file-schema.xml not found in {calc_dir}",
                }

        if not xml_path:
            return {
                "success": False,
                "error": "Provide bands_xml_path or calc_dir.",
            }

        fermi = fermi_energy
        if fermi is None:
            pw_out = os.path.join(os.path.dirname(xml_path), "pw.out")
            fermi = extract_fermi_energy(pw_out)

        band_data = parse_xml_bands(xml_path, fermi_energy=fermi)

        dos_data = None
        if dos_file_path:
            from shalom.backends.qe_parser import parse_dos_file
            dos_data = parse_dos_file(dos_file_path)
            if fermi is not None:
                dos_data.fermi_energy = fermi

        result = analyze_band_structure(band_data, dos_data=dos_data)

        response: Dict[str, Any] = {
            "success": True,
            "is_metal": result.is_metal,
            "bandgap_eV": result.bandgap_eV,
            "is_direct_gap": result.is_direct,
            "n_occupied_bands": result.n_occupied_bands,
        }
        if result.vbm_energy is not None:
            response["vbm_energy_eV"] = result.vbm_energy
        if result.cbm_energy is not None:
            response["cbm_energy_eV"] = result.cbm_energy
        if result.effective_mass_electron is not None:
            response["effective_mass_electron_me"] = result.effective_mass_electron
        if result.effective_mass_hole is not None:
            response["effective_mass_hole_me"] = result.effective_mass_hole
        if result.dos_at_fermi is not None:
            response["dos_at_fermi_states_per_eV"] = result.dos_at_fermi

        return response
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Tool 14: analyze_xrd_pattern
# ---------------------------------------------------------------------------


@mcp.tool()
def analyze_xrd_pattern(
    structure_file: str,
    wavelength: str = "CuKa",
    theta_min: float = 0.0,
    theta_max: float = 90.0,
    output_plot: str = "",
) -> dict:
    """Calculate powder X-ray diffraction pattern for a crystal structure.

    Computes peak positions (2-theta), intensities, Miller indices, and
    d-spacings. Requires pymatgen.

    Args:
        structure_file: Path to structure file (POSCAR, CIF, etc.).
        wavelength: X-ray source label (CuKa, MoKa, CoKa, etc.).
        theta_min: Minimum 2-theta in degrees.
        theta_max: Maximum 2-theta in degrees.
        output_plot: Optional path to save XRD pattern plot (PNG).
    """
    try:
        from shalom.analysis.xrd import is_xrd_available, calculate_xrd

        if not is_xrd_available():
            return {
                "success": False,
                "error": (
                    "pymatgen not installed. "
                    "Install with: pip install 'shalom[analysis]'"
                ),
            }

        from ase.io import read as ase_read
        atoms = ase_read(structure_file)

        result = calculate_xrd(
            atoms,
            wavelength=wavelength,
            two_theta_range=(theta_min, theta_max),
        )

        import numpy as np
        sorted_idx = np.argsort(result.intensities)[::-1]
        n_show = min(10, result.n_peaks)

        top_peaks = []
        for i in range(n_show):
            idx = sorted_idx[i]
            hkl = result.hkl_indices[idx] if idx < len(result.hkl_indices) else (0, 0, 0)
            top_peaks.append({
                "two_theta_deg": round(float(result.two_theta[idx]), 3),
                "intensity": round(float(result.intensities[idx]), 2),
                "d_spacing_A": round(float(result.d_spacings[idx]), 4),
                "hkl": list(hkl),
            })

        response: Dict[str, Any] = {
            "success": True,
            "formula": atoms.get_chemical_formula(),
            "wavelength": wavelength,
            "wavelength_angstrom": result.wavelength_angstrom,
            "n_peaks": result.n_peaks,
            "top_peaks": top_peaks,
        }

        if output_plot:
            try:
                from shalom.plotting.xrd_plot import XRDPlotter
                plotter = XRDPlotter(result)
                plotter.plot(output_path=output_plot)
                response["plot_path"] = output_plot
            except ImportError:
                response["plot_warning"] = "matplotlib not installed for plotting"

        return response
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Tool 15: analyze_symmetry_properties
# ---------------------------------------------------------------------------


@mcp.tool()
def analyze_symmetry_properties(
    structure_file: str,
    symprec: str = "1e-5",
) -> dict:
    """Analyze the crystal symmetry of a structure using spglib.

    Determines space group, point group, crystal system, Wyckoff positions,
    lattice type, and number of symmetry operations.

    Requires spglib: ``pip install 'shalom[symmetry]'``

    Args:
        structure_file: Path to structure file (POSCAR, CIF, etc.).
        symprec: Symmetry tolerance in Angstrom (default: "1e-5").
            Larger values allow more distortion.
    """
    try:
        from shalom.analysis.symmetry import is_spglib_available, analyze_symmetry

        if not is_spglib_available():
            return {
                "success": False,
                "error": (
                    "spglib not installed. "
                    "Install with: pip install 'shalom[symmetry]'"
                ),
            }

        from ase.io import read as ase_read
        atoms = ase_read(structure_file)

        result = analyze_symmetry(atoms, symprec=float(symprec))

        return {
            "success": True,
            "formula": atoms.get_chemical_formula(),
            "space_group_number": result.space_group_number,
            "space_group_symbol": result.space_group_symbol,
            "point_group": result.point_group,
            "crystal_system": result.crystal_system,
            "lattice_type": result.lattice_type,
            "hall_symbol": result.hall_symbol,
            "n_symmetry_operations": result.n_operations,
            "is_primitive": result.is_primitive,
            "wyckoff_letters": result.wyckoff_letters,
            "equivalent_atoms": result.equivalent_atoms,
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Tool 16: analyze_magnetic_properties
# ---------------------------------------------------------------------------


@mcp.tool()
def analyze_magnetic_properties(
    pw_out_path: str,
    structure_file: str = "",
) -> dict:
    """Analyze magnetic properties from a QE pw.out file.

    Extracts site magnetizations, Lowdin charges, and classifies magnetic
    behaviour per element. No additional dependencies required.

    Args:
        pw_out_path: Path to QE pw.out file.
        structure_file: Optional structure file for per-element analysis.
            If provided, identifies which elements carry magnetic moments.
    """
    try:
        from shalom.analysis.magnetic import (
            extract_site_magnetization,
            extract_lowdin_charges,
            analyze_magnetism,
        )

        site_mags = extract_site_magnetization(pw_out_path)
        lowdin = extract_lowdin_charges(pw_out_path)

        response: Dict[str, Any] = {"success": True}

        if site_mags is not None:
            response["site_magnetizations_bohr_mag"] = [
                round(m, 4) for m in site_mags
            ]
            response["n_atoms"] = len(site_mags)
        else:
            response["site_magnetizations_bohr_mag"] = None

        if lowdin is not None:
            response["lowdin_total_charges"] = [
                round(c, 4) for c in lowdin.get("total_charges", [])
            ]
            response["lowdin_spd_charges"] = lowdin.get("spd_charges", [])
        else:
            response["lowdin_total_charges"] = None

        # Full analysis with structure
        if structure_file:
            import re

            from ase.io import read as ase_read
            atoms = ase_read(structure_file)

            # Extract total magnetization
            total_mag = None
            try:
                with open(pw_out_path, "r", encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
                m = re.findall(
                    r"total magnetization\s+=\s+([-\d.]+)\s+Bohr mag/cell", text
                )
                if m:
                    total_mag = float(m[-1])
            except OSError:
                pass

            from shalom.backends.base import DFTResult
            dft_result = DFTResult(magnetization=total_mag, is_converged=True)
            result = analyze_magnetism(dft_result, atoms, pw_out_path=pw_out_path)

            response.update({
                "total_magnetization_bohr_mag": result.total_magnetization,
                "is_magnetic": result.is_magnetic,
                "is_spin_polarized": result.is_spin_polarized,
                "magnetic_elements": result.magnetic_elements,
                "dominant_moment_element": result.dominant_moment_element,
            })

        return response
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
