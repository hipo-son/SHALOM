"""CLI entry point for SHALOM direct material run.

Usage::

    python -m shalom run mp-19717                              # VASP vc-relax
    python -m shalom run Fe2O3 --backend qe --calc scf         # QE SCF
    python -m shalom run mp-19717 --backend qe --accuracy precise
    python -m shalom run mp-19717 --set ENCUT=600 --set NSW=200  # VASP overrides
    python -m shalom run mp-19717 --backend qe --set ecutwfc=80
    python -m shalom run --structure POSCAR --backend vasp      # Local file
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Dict, Any, Optional


def _parse_set_values(set_args: Optional[list]) -> Dict[str, Any]:
    """Parse --set KEY=VALUE arguments into a dict."""
    if not set_args:
        return {}
    result: Dict[str, Any] = {}
    for item in set_args:
        if "=" not in item:
            print(f"Warning: ignoring --set '{item}' (expected KEY=VALUE format)")
            continue
        key, val_str = item.split("=", 1)
        key = key.strip()
        val_str = val_str.strip()
        if not key:
            print(f"Warning: ignoring --set '{item}' (empty key)")
            continue
        if not val_str:
            print(f"Warning: ignoring --set '{item}' (empty value)")
            continue
        # Auto-convert numeric values
        try:
            if "." in val_str or "e" in val_str.lower():
                result[key] = float(val_str)
            else:
                result[key] = int(val_str)
        except ValueError:
            # Bool detection
            if val_str.lower() in ("true", ".true."):
                result[key] = True
            elif val_str.lower() in ("false", ".false."):
                result[key] = False
            else:
                result[key] = val_str
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="shalom",
        description="SHALOM - Generate DFT input files from material specifications.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 'run' subcommand
    run_parser = subparsers.add_parser(
        "run",
        help="Generate DFT input files for a material.",
        description=(
            "Generate complete DFT input files from a Materials Project ID, "
            "chemical formula, or local structure file."
        ),
    )
    run_parser.add_argument(
        "material",
        nargs="?",
        help="Material spec: MP ID (mp-19717), formula (Fe2O3), or omit with --structure.",
    )
    run_parser.add_argument(
        "--backend", "-b",
        default="vasp",
        choices=["vasp", "qe"],
        help="DFT backend (default: vasp).",
    )
    run_parser.add_argument(
        "--calc", "-c",
        default=None,
        help=(
            "Calculation type. VASP: relaxation/static/band_structure/dos. "
            "QE: scf/relax/vc-relax/bands/nscf. Default: relaxation."
        ),
    )
    run_parser.add_argument(
        "--accuracy", "-a",
        default="standard",
        choices=["standard", "precise"],
        help="Accuracy level (default: standard).",
    )
    run_parser.add_argument(
        "--set", "-s",
        action="append",
        dest="set_values",
        metavar="KEY=VALUE",
        help="Override DFT parameter (repeatable). E.g. --set ENCUT=600 --set ecutwfc=80.",
    )
    run_parser.add_argument(
        "-o", "--output",
        default=None,
        help=(
            "Explicit output directory (bypasses workspace logic). "
            "Default: $SHALOM_WORKSPACE/{project}/{auto_name} "
            "or ~/Desktop/shalom-runs/{auto_name}."
        ),
    )
    run_parser.add_argument(
        "-w", "--workspace",
        default=None,
        metavar="DIR",
        help=(
            "Workspace root directory where all runs are stored. "
            "Default: $SHALOM_WORKSPACE, then ~/Desktop/shalom-runs."
        ),
    )
    run_parser.add_argument(
        "-p", "--project",
        default=None,
        metavar="NAME",
        help=(
            "Project sub-folder inside the workspace (optional grouping). "
            "E.g. --project silicon_study → workspace/silicon_study/Si_qe_scf/."
        ),
    )
    run_parser.add_argument(
        "--structure",
        default=None,
        help="Path to local structure file (POSCAR, CIF, etc.).",
    )
    run_parser.add_argument(
        "--pseudo-dir",
        default=None,
        help="QE pseudopotential directory (default: $SHALOM_PSEUDO_DIR or ~/pseudopotentials).",
    )
    run_parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip structure validation.",
    )
    run_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output directory.",
    )
    run_parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress informational output.",
    )
    run_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    # Execution flags
    run_parser.add_argument(
        "--execute", "-x",
        action="store_true",
        help="Run DFT via subprocess after input generation (QE only).",
    )
    run_parser.add_argument(
        "--nprocs", "-np",
        type=int,
        default=1,
        help="MPI process count for --execute (default: 1).",
    )
    run_parser.add_argument(
        "--timeout",
        type=int,
        default=86400,
        help="Execution timeout in seconds (default: 86400).",
    )
    run_parser.add_argument(
        "--mpi-command",
        default="mpirun",
        help="MPI launcher command (default: mpirun).",
    )

    # 'setup-qe' subcommand
    setup_parser = subparsers.add_parser(
        "setup-qe",
        help="Check QE prerequisites and download pseudopotentials.",
        description=(
            "Verify pw.x installation, validate pseudo_dir, "
            "and manage SSSP pseudopotentials."
        ),
    )
    setup_parser.add_argument(
        "--pseudo-dir",
        default=None,
        help="Pseudopotential directory (default: $SHALOM_PSEUDO_DIR or './').",
    )
    setup_parser.add_argument(
        "--elements",
        default=None,
        help="Comma-separated elements to check (default: all SSSP).",
    )
    setup_parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing pseudopotentials from SSSP repository.",
    )

    return parser


def cmd_run(args: argparse.Namespace) -> int:
    """Execute the 'run' subcommand."""
    from shalom.direct_run import direct_run, DirectRunConfig

    if args.material is None and args.structure is None:
        print("Error: Provide one of:")
        print("  1. Materials Project ID:  python -m shalom run mp-19717")
        print("  2. Chemical formula:      python -m shalom run Fe2O3  (requires MP_API_KEY)")
        print("  3. Local structure file:  python -m shalom run --structure POSCAR")
        return 1

    if args.material and not args.structure:
        from shalom.mp_client import is_mp_available
        if not is_mp_available():
            print("Error: mp-api package not installed (required for MP ID / formula lookup).")
            print("Install with: pip install 'shalom[mp]'")
            return 1

    user_settings = _parse_set_values(args.set_values)

    config = DirectRunConfig(
        backend_name=args.backend,
        calc_type=args.calc,
        accuracy=args.accuracy,
        output_dir=args.output,
        workspace_dir=args.workspace,
        project=args.project,
        user_settings=user_settings if user_settings else None,
        pseudo_dir=args.pseudo_dir,
        validate_structure=not args.no_validate,
        force_overwrite=args.force,
        structure_file=args.structure,
    )

    material_spec = args.material or ""
    result = direct_run(material_spec, config)

    if not args.quiet:
        if result.success:
            print(f"Backend:   {result.backend_name}")
            if result.structure_info:
                info = result.structure_info
                if info.get("mp_id"):
                    print(f"Material:  {info.get('formula', '?')} ({info['mp_id']})")
                    if info.get("energy_above_hull") is not None:
                        print(f"E_hull:    {info['energy_above_hull']:.4f} eV/atom")
                    if info.get("space_group"):
                        print(f"Spacegroup: {info['space_group']}")
                else:
                    print(f"Source:    {info.get('source', '?')}")

            if result.auto_detected:
                print(f"Auto:      {result.auto_detected}")
            print(f"Output:    {result.output_dir}/")
            print(f"Files:     {', '.join(result.files_generated)}")

            # Execute DFT if requested
            if getattr(args, "execute", False):
                if args.backend == "vasp":
                    print("\nError: VASP local execution not yet supported.")
                    print("Run manually: cd {result.output_dir} && mpirun vasp_std")
                    return 1

                exec_code = _execute_dft(result.output_dir or ".", args)
                return exec_code

            print()
            print("Next steps:")
            if result.backend_name == "vasp":
                print(f"  1. Obtain POTCARs listed in {result.output_dir}/POTCAR.spec")
                print(f"  2. cd {result.output_dir} && mpirun vasp_std")
            else:
                print("  1. Ensure pseudopotentials are in the pseudo_dir")
                print(f"  2. cd {result.output_dir} && mpirun pw.x < pw.in > pw.out")
        else:
            print(f"Error: {result.error}")

    return 0 if result.success else 1


def cmd_setup_qe(args: argparse.Namespace) -> int:
    """Execute the 'setup-qe' subcommand."""
    import os
    import platform
    import shutil
    from pathlib import Path

    from shalom.backends.qe_config import SSSP_ELEMENTS, get_pseudo_filename

    issues = 0

    # 1. Check pw.x
    pw_path = shutil.which("pw.x")
    if pw_path:
        print(f"pw.x:       {pw_path}")
    else:
        print("pw.x:       NOT FOUND")
        if platform.system() == "Windows":
            print("  Tip:      Run from inside WSL2 (wsl -d Ubuntu-22.04)")
            print("            sudo apt install quantum-espresso")
        else:
            print("  Install:  sudo apt install quantum-espresso  (Ubuntu/Debian)")
            print("            conda install -c conda-forge qe     (conda)")
        issues += 1

    # 2. Resolve pseudo_dir
    _default_pseudo = str(Path.home() / "pseudopotentials")
    pseudo_dir_arg = args.pseudo_dir or os.environ.get("SHALOM_PSEUDO_DIR", _default_pseudo)
    pseudo_path = Path(pseudo_dir_arg).expanduser().resolve()
    if pseudo_path.is_dir():
        print(f"pseudo_dir: {pseudo_path}")
    else:
        print(f"pseudo_dir: {pseudo_dir_arg} (NOT FOUND)")
        print("  Set:      export SHALOM_PSEUDO_DIR=/path/to/pseudos")
        if not args.download:
            issues += 1

    # 3. Parse elements
    if args.elements:
        elements = [e.strip() for e in args.elements.split(",")]
    else:
        elements = sorted(SSSP_ELEMENTS.keys())

    # 4. Check each UPF file
    missing: list = []
    if pseudo_path.is_dir():
        print(f"\nElements ({len(elements)}):")
        for el in elements:
            if el not in SSSP_ELEMENTS:
                print(f"  {el:<4s} (unknown element, not in SSSP)")
                continue
            upf = get_pseudo_filename(el)
            upf_path = pseudo_path / upf
            # Case-insensitive fallback
            found = upf_path.exists()
            if not found:
                try:
                    for entry in pseudo_path.iterdir():
                        if entry.name.lower() == upf.lower():
                            found = True
                            break
                except OSError:
                    pass
            if found:
                print(f"  {el:<4s} {upf}  [OK]")
            else:
                print(f"  {el:<4s} {upf}  [MISSING]")
                missing.append((el, upf))
                issues += 1

    # 5. Download if requested
    if args.download and missing:
        import urllib.request

        pseudo_path.mkdir(parents=True, exist_ok=True)
        print(f"\nDownloading {len(missing)} pseudopotential(s)...")
        base_url = "https://pseudopotentials.quantum-espresso.org/upf_files"
        for el, upf in missing:
            url = f"{base_url}/{upf}"
            dest = pseudo_path / upf
            print(f"  {upf} ... ", end="", flush=True)
            try:
                urllib.request.urlretrieve(url, str(dest))
                print("OK")
                issues -= 1
            except Exception as e:
                print(f"FAILED ({e})")

    # 6. Summary
    print()
    if issues == 0:
        print("Ready for QE execution.")
    else:
        print(f"{issues} issue(s) found.")
        print("Fix and re-run: python -m shalom setup-qe")
    return 0 if issues == 0 else 1


def _execute_dft(output_dir: str, args: argparse.Namespace) -> int:
    """Execute DFT and display results."""
    from shalom.backends.qe import QEBackend
    from shalom.backends.runner import ExecutionConfig, ExecutionRunner, execute_with_recovery
    from shalom.backends.qe_error_recovery import QEErrorRecoveryEngine

    print(f"\nExecuting QE (nprocs={args.nprocs})...")

    exec_config = ExecutionConfig(
        nprocs=args.nprocs,
        mpi_command=args.mpi_command,
        timeout_seconds=args.timeout,
    )
    runner = ExecutionRunner(config=exec_config)

    # Validate prerequisites
    prereq_errors = runner.validate_prerequisites(output_dir)
    if prereq_errors:
        for err in prereq_errors:
            print(f"  Error: {err}")
        return 1

    backend = QEBackend()
    recovery = QEErrorRecoveryEngine()

    exec_result, dft_result, history = execute_with_recovery(
        backend, runner, recovery, output_dir,
        config=None, atoms=None, max_retries=3,
    )

    print(f"Wall time: {exec_result.wall_time_seconds:.1f}s")

    if history:
        print(f"Corrections: {len(history)}")
        for entry in history:
            print(f"  - {entry['error_type']} step {entry['step']}")

    if dft_result is not None and dft_result.is_converged:
        print("Status:    Converged")
        if dft_result.energy is not None:
            print(f"Energy:    {dft_result.energy:.6f} eV")
        if dft_result.forces_max is not None:
            print(f"Max force: {dft_result.forces_max:.6f} eV/A")
        if dft_result.quality_warnings:
            print(f"Warnings:  {', '.join(dft_result.quality_warnings)}")
        return 0
    else:
        print("Status:    FAILED")
        if exec_result.error_message:
            print(f"Error:     {exec_result.error_message}")
        return 1


def main() -> None:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Logging setup
    if hasattr(args, "verbose") and args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")
    elif hasattr(args, "quiet") and not args.quiet:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.command == "run":
        sys.exit(cmd_run(args))
    elif args.command == "setup-qe":
        sys.exit(cmd_setup_qe(args))
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
