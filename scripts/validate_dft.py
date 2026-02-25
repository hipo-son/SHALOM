#!/usr/bin/env python3
"""SHALOM DFT validation — run real QE SCF calculations and compare results.

Generates QE inputs via SHALOM, executes pw.x (WSL on Windows, native on
Linux), and validates convergence, energy, and forces against physical
reference ranges.

Materials tested: Si (diamond), Al (FCC), Cu (FCC).

Usage::

    # From the SHALOM project root (Windows with WSL):
    python scripts/validate_dft.py

    # Linux / WSL native:
    python scripts/validate_dft.py --no-wsl

    # Custom pseudo_dir:
    python scripts/validate_dft.py --pseudo-dir /path/to/pseudopotentials
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ase.build import bulk
from ase.io import write as ase_write

from shalom.backends.qe import QEBackend
from shalom.backends.qe_parser import extract_fermi_energy
from shalom.backends.runner import ExecutionConfig, ExecutionRunner
from shalom.direct_run import DirectRunConfig, direct_run


# ---------------------------------------------------------------------------
# Test material definitions
# ---------------------------------------------------------------------------

@dataclass
class MaterialSpec:
    """Specification for a DFT validation material."""
    name: str
    crystal: str
    a: float
    natoms: int
    # Validation ranges (None = skip check)
    energy_per_atom_range: tuple  # (min, max) eV
    forces_max_threshold: float   # eV/A
    is_metal: bool
    fermi_range: Optional[tuple] = None  # (min, max) eV


MATERIALS: Dict[str, MaterialSpec] = {
    "Si": MaterialSpec(
        name="Si", crystal="diamond", a=5.43, natoms=2,
        # SSSP USPP total energy: ~-155 eV/atom (pseudo-dependent)
        energy_per_atom_range=(-200.0, -50.0),
        forces_max_threshold=0.05,
        is_metal=False,
        fermi_range=(4.0, 9.0),
    ),
    "Al": MaterialSpec(
        name="Al", crystal="fcc", a=4.05, natoms=1,
        # SSSP PAW total energy: ~-537 eV/atom (pseudo-dependent)
        energy_per_atom_range=(-700.0, -100.0),
        forces_max_threshold=0.05,
        is_metal=True,
    ),
    "Cu": MaterialSpec(
        name="Cu", crystal="fcc", a=3.615, natoms=1,
        # SSSP PAW total energy: ~-2899 eV/atom (11 valence e-, high Z)
        energy_per_atom_range=(-3500.0, -1000.0),
        forces_max_threshold=0.05,
        is_metal=True,
    ),
}


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Result of a single material validation."""
    material: str
    passed: bool
    energy: Optional[float] = None
    energy_per_atom: Optional[float] = None
    forces_max: Optional[float] = None
    fermi: Optional[float] = None
    is_converged: bool = False
    wall_time: float = 0.0
    errors: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


# ---------------------------------------------------------------------------
# Core validation logic
# ---------------------------------------------------------------------------

def run_scf(
    spec: MaterialSpec,
    work_dir: Path,
    pseudo_dir: str,
    wsl: bool,
    pw_command: str = "pw.x",
) -> ValidationResult:
    """Generate QE input, run SCF via pw.x, parse and validate results."""
    result = ValidationResult(material=spec.name, passed=False)
    calc_dir = work_dir / f"{spec.name}_scf"

    # 1. Build structure via ASE
    atoms = bulk(spec.name, spec.crystal, a=spec.a)
    poscar = work_dir / f"{spec.name}_POSCAR"
    ase_write(str(poscar), atoms, format="vasp")

    # 2. Generate QE input
    config = DirectRunConfig(
        backend_name="qe",
        calc_type="scf",
        output_dir=str(calc_dir),
        structure_file=str(poscar),
        pseudo_dir=pseudo_dir,
        force_overwrite=True,
    )
    gen = direct_run("", config)
    if not gen.success:
        result.errors.append(f"Input generation failed: {gen.error}")
        return result

    # 3. Execute pw.x (heavy elements like Cu need more time on 1 CPU)
    exec_config = ExecutionConfig(
        command=pw_command, nprocs=1, timeout_seconds=900, wsl=wsl,
    )
    runner = ExecutionRunner(exec_config)

    # Prerequisite check (skip pseudo check in WSL mode)
    prereq_errors = runner.validate_prerequisites(str(calc_dir))
    if prereq_errors:
        result.errors.extend(prereq_errors)
        return result

    exec_result = runner.run(str(calc_dir))
    result.wall_time = exec_result.wall_time_seconds

    if not exec_result.success:
        result.errors.append(
            f"pw.x failed (rc={exec_result.return_code}): "
            f"{exec_result.error_message or 'unknown'}"
        )
        return result

    # 4. Parse output
    backend = QEBackend()
    dft = backend.parse_output(str(calc_dir))
    result.is_converged = dft.is_converged
    result.energy = dft.energy
    result.forces_max = dft.forces_max

    pw_out = str(calc_dir / "pw.out")
    result.fermi = extract_fermi_energy(pw_out)

    if dft.energy is not None:
        result.energy_per_atom = dft.energy / spec.natoms

    # 5. Validate against reference ranges
    errors: List[str] = []

    if not dft.is_converged:
        errors.append("SCF did NOT converge")

    if dft.energy is None:
        errors.append("No energy parsed from output")
    elif dft.energy >= 0:
        errors.append(f"Energy is positive: {dft.energy:.4f} eV")

    if result.energy_per_atom is not None:
        lo, hi = spec.energy_per_atom_range
        if not (lo <= result.energy_per_atom <= hi):
            errors.append(
                f"Energy/atom {result.energy_per_atom:.4f} eV "
                f"outside range [{lo}, {hi}]"
            )

    if dft.forces_max is not None and dft.forces_max > spec.forces_max_threshold:
        errors.append(
            f"Forces too large: {dft.forces_max:.4f} eV/A "
            f"(threshold: {spec.forces_max_threshold})"
        )

    if spec.fermi_range and result.fermi is not None:
        lo, hi = spec.fermi_range
        if not (lo <= result.fermi <= hi):
            errors.append(
                f"Fermi energy {result.fermi:.4f} eV outside range [{lo}, {hi}]"
            )

    result.errors = errors
    result.passed = len(errors) == 0
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve_pw_command(wsl: bool) -> str:
    """Auto-detect the best pw.x executable.

    On Windows (WSL mode), tries conda-forge QE first, then system pw.x.
    On Linux, returns ``"pw.x"`` (relies on PATH).
    """
    if not wsl:
        return "pw.x"

    import subprocess as _sp

    # Prefer conda-forge QE 7.x (Ubuntu 24.04's QE 6.7 has buffer overflow)
    conda_pw = "/opt/micromamba/envs/qe/bin/pw.x"
    try:
        r = _sp.run(["wsl", "-e", "test", "-x", conda_pw],
                     capture_output=True, timeout=5)
        if r.returncode == 0:
            return conda_pw
    except (FileNotFoundError, _sp.TimeoutExpired, OSError):
        pass

    return "pw.x"


def main() -> int:
    parser = argparse.ArgumentParser(description="SHALOM DFT validation")
    parser.add_argument(
        "--pseudo-dir", default="/root/pseudopotentials",
        help="Pseudopotential directory (WSL path if --wsl, default: /root/pseudopotentials)",
    )
    parser.add_argument(
        "--no-wsl", action="store_true",
        help="Do NOT use WSL bridge (for native Linux execution).",
    )
    parser.add_argument(
        "--pw-command", default=None,
        help="Full path to pw.x (e.g. /opt/micromamba/envs/qe/bin/pw.x). "
             "Auto-detected if not specified.",
    )
    parser.add_argument(
        "--work-dir",
        help="Working directory (default: temp dir, cleaned up after).",
    )
    parser.add_argument(
        "--materials", nargs="+", default=list(MATERIALS.keys()),
        help="Materials to test (default: Si Al Cu).",
    )
    args = parser.parse_args()

    wsl = sys.platform == "win32" and not args.no_wsl

    # Resolve pw.x command
    pw_command = args.pw_command or _resolve_pw_command(wsl)

    # Determine working directory
    if args.work_dir:
        work = Path(args.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        tmp = tempfile.mkdtemp(prefix="shalom_validate_")
        work = Path(tmp)
        cleanup = True

    print("=" * 60)
    print("SHALOM DFT Validation")
    print("=" * 60)
    print(f"  WSL mode:   {wsl}")
    print(f"  pw.x:       {pw_command}")
    print(f"  pseudo_dir: {args.pseudo_dir}")
    print(f"  work_dir:   {work}")
    print(f"  materials:  {args.materials}")
    print()

    results: List[ValidationResult] = []
    for mat_name in args.materials:
        spec = MATERIALS.get(mat_name)
        if spec is None:
            print(f"  Unknown material: {mat_name} (available: {list(MATERIALS.keys())})")
            continue

        print(f"  [{mat_name}] Running SCF ... ", end="", flush=True)
        vr = run_scf(spec, work, args.pseudo_dir, wsl, pw_command=pw_command)
        results.append(vr)
        status = "PASS" if vr.passed else "FAIL"
        print(f"{status}  ({vr.wall_time:.1f}s)")
        if vr.errors:
            for err in vr.errors:
                print(f"         ! {err}")

    # Summary table
    print()
    print("-" * 60)
    fmt = "{:<6} {:>7} {:>12} {:>10} {:>10}  {}"
    print(fmt.format("Mat", "Status", "E/atom(eV)", "Fmax(eV/A)", "Fermi(eV)", "Time"))
    print("-" * 60)
    for vr in results:
        e_str = f"{vr.energy_per_atom:.4f}" if vr.energy_per_atom is not None else "N/A"
        f_str = f"{vr.forces_max:.4f}" if vr.forces_max is not None else "N/A"
        fermi_str = f"{vr.fermi:.4f}" if vr.fermi is not None else "N/A"
        status = "PASS" if vr.passed else "FAIL"
        print(fmt.format(vr.material, status, e_str, f_str, fermi_str, f"{vr.wall_time:.1f}s"))
    print("-" * 60)

    n_pass = sum(1 for r in results if r.passed)
    n_total = len(results)
    print(f"\nResult: {n_pass}/{n_total} passed")

    if cleanup:
        import shutil
        shutil.rmtree(work, ignore_errors=True)

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
