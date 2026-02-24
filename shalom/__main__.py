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
import os
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
            "E.g. --project silicon_study ->workspace/silicon_study/Si_qe_scf/."
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

    # 'plot' subcommand
    plot_parser = subparsers.add_parser(
        "plot",
        help="Plot band structure or DOS from an existing QE calculation directory.",
        description=(
            "Generate band structure or DOS plots from completed QE pw.x / dos.x outputs."
        ),
    )
    plot_parser.add_argument(
        "calc_dir",
        help="QE calculation directory (bands or nscf run).",
    )
    plot_parser.add_argument(
        "--bands",
        action="store_true",
        help="Plot band structure (requires data-file-schema.xml).",
    )
    plot_parser.add_argument(
        "--dos",
        action="store_true",
        help="Plot DOS (requires pwscf.dos).",
    )
    plot_parser.add_argument(
        "--fermi-from",
        default=None,
        metavar="DIR",
        help=(
            "Directory to read Fermi energy from (e.g. nscf run dir). "
            "Overrides Fermi energy found in calc_dir."
        ),
    )
    plot_parser.add_argument(
        "--emin",
        type=float,
        default=-6.0,
        help="Energy window minimum in eV relative to Fermi (default: -6).",
    )
    plot_parser.add_argument(
        "--emax",
        type=float,
        default=6.0,
        help="Energy window maximum in eV relative to Fermi (default: 6).",
    )
    plot_parser.add_argument(
        "--title",
        default=None,
        help="Plot title.",
    )
    plot_parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output file path for the plot (default: bands.png or dos.png in calc_dir).",
    )

    # 'workflow' subcommand
    workflow_parser = subparsers.add_parser(
        "workflow",
        help="Run the 5-step sequential QE workflow: vc-relax ->scf ->bands ->nscf ->dos.",
        description=(
            "Full sequential QE workflow for a given material. "
            "Produces bands.png and dos.png in the output directory."
        ),
    )
    workflow_parser.add_argument(
        "material",
        nargs="?",
        help="Material spec: MP ID (mp-19717), formula (Si), or omit with --structure.",
    )
    workflow_parser.add_argument(
        "--structure",
        default=None,
        help="Path to local structure file (POSCAR, CIF, etc.).",
    )
    workflow_parser.add_argument(
        "-o", "--output",
        default=None,
        required=True,
        help="Output directory for all workflow steps.",
    )
    workflow_parser.add_argument(
        "--skip-relax",
        action="store_true",
        help="Skip the vc-relax step.",
    )
    workflow_parser.add_argument(
        "--accuracy", "-a",
        default="standard",
        choices=["standard", "precise"],
        help="QE preset accuracy level (default: standard).",
    )
    workflow_parser.add_argument(
        "--pseudo-dir",
        default=None,
        help="QE pseudopotential directory.",
    )
    workflow_parser.add_argument(
        "--nprocs", "-np",
        type=int,
        default=1,
        help="MPI process count (default: 1).",
    )
    workflow_parser.add_argument(
        "--mpi-command",
        default="mpirun",
        help="MPI launcher (default: mpirun).",
    )
    workflow_parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="Per-step timeout in seconds (default: 7200).",
    )
    workflow_parser.add_argument(
        "--dos-emin",
        type=float,
        default=-20.0,
        help="DOS energy window min in eV (default: -20).",
    )
    workflow_parser.add_argument(
        "--dos-emax",
        type=float,
        default=10.0,
        help="DOS energy window max in eV (default: 10).",
    )
    workflow_parser.add_argument(
        "--is-2d",
        action="store_true",
        help="Treat structure as 2D (kz=0 k-path, 2D isolation flags).",
    )

    # 'converge' subcommand
    converge_parser = subparsers.add_parser(
        "converge",
        help="Run cutoff or k-point convergence tests.",
        description=(
            "Sweep ecutwfc or k-point resolution to find the converged DFT parameter. "
            "Run cutoff convergence FIRST, then k-point convergence."
        ),
    )
    converge_parser.add_argument(
        "material",
        nargs="?",
        help="Material spec: MP ID, formula, or omit with --structure.",
    )
    converge_parser.add_argument(
        "--structure",
        default=None,
        help="Path to local structure file.",
    )
    converge_parser.add_argument(
        "--test",
        choices=["cutoff", "kpoints"],
        required=True,
        help="Type of convergence test.",
    )
    converge_parser.add_argument(
        "--values",
        default=None,
        help=(
            "Comma-separated list of values to test. "
            "Cutoff: ecutwfc in Ry (e.g. 30,40,50,60). "
            "Kpoints: resolutions in Bohr^-1 (e.g. 20,30,40,50)."
        ),
    )
    converge_parser.add_argument(
        "--ecutwfc",
        type=float,
        default=None,
        help="Fixed ecutwfc for k-point test (required for --test kpoints).",
    )
    converge_parser.add_argument(
        "--kgrid",
        default=None,
        help=(
            "Fixed k-grid for cutoff test as NxNyNz (e.g. 4x4x4). "
            "If omitted, estimated from lattice."
        ),
    )
    converge_parser.add_argument(
        "-o", "--output",
        default=None,
        required=True,
        help="Output directory for convergence runs.",
    )
    converge_parser.add_argument(
        "--accuracy", "-a",
        default="standard",
        choices=["standard", "precise"],
    )
    converge_parser.add_argument(
        "--pseudo-dir",
        default=None,
    )
    converge_parser.add_argument(
        "--nprocs", "-np",
        type=int,
        default=1,
    )
    converge_parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
    )
    converge_parser.add_argument(
        "--threshold",
        type=float,
        default=1e-3,
        help="Convergence threshold in eV/atom (default: 1e-3).",
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

    # 'pipeline' subcommand -LLM-driven autonomous pipeline
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run the LLM-driven autonomous material discovery pipeline.",
        description=(
            "Use LLM agents (OpenAI GPT or Anthropic Claude) to autonomously "
            "select materials, generate structures, and produce DFT input files. "
            "Requires an API key: OPENAI_API_KEY or ANTHROPIC_API_KEY."
        ),
    )
    pipeline_parser.add_argument(
        "objective",
        help=(
            "Natural language objective. "
            "E.g. 'Find a 2D HER catalyst', 'Stable Li-ion cathode material'."
        ),
    )
    pipeline_parser.add_argument(
        "--backend", "-b",
        default="qe",
        choices=["vasp", "qe"],
        help="DFT backend (default: qe).",
    )
    pipeline_parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "anthropic"],
        help="LLM provider (default: openai).",
    )
    pipeline_parser.add_argument(
        "--model",
        default=None,
        help="LLM model name (default: gpt-4o for openai, claude-sonnet-4-6 for anthropic).",
    )
    pipeline_parser.add_argument(
        "--material",
        default=None,
        help="Skip Design layer and use this material directly (e.g. 'MoS2', 'Fe2O3').",
    )
    pipeline_parser.add_argument(
        "--steps",
        default=None,
        help=(
            "Comma-separated pipeline steps: design,simulation,review. "
            "Default: design,simulation (generates input files, awaits DFT)."
        ),
    )
    pipeline_parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory for pipeline results.",
    )
    pipeline_parser.add_argument(
        "--calc", "-c",
        default="relaxation",
        help="Calculation type (default: relaxation).",
    )
    pipeline_parser.add_argument(
        "--accuracy", "-a",
        default="standard",
        choices=["standard", "precise"],
    )
    pipeline_parser.add_argument(
        "--execute", "-x",
        action="store_true",
        help="Execute DFT after input generation (QE only).",
    )
    pipeline_parser.add_argument(
        "--nprocs", "-np",
        type=int,
        default=1,
        help="MPI process count for --execute.",
    )
    pipeline_parser.add_argument(
        "--timeout",
        type=int,
        default=86400,
        help="DFT execution timeout in seconds.",
    )
    pipeline_parser.add_argument(
        "--max-loops",
        type=int,
        default=1,
        help="Max Design->Sim->Review outer loop iterations (default: 1).",
    )
    pipeline_parser.add_argument(
        "--selector-mode",
        default="simple",
        choices=["simple", "multi_agent"],
        help="Fine selector mode (default: simple).",
    )
    pipeline_parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "Custom LLM API base URL for local/self-hosted models. "
            "E.g. http://localhost:11434/v1 (Ollama), "
            "http://localhost:8000/v1 (vLLM). "
            "Falls back to $SHALOM_LLM_BASE_URL env var."
        ),
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
                    print(f"Run manually: cd {result.output_dir} && mpirun vasp_std")
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


def _detect_wsl_distros() -> list:
    """Return list of installed WSL distro names (Windows only).

    WSL outputs UTF-16-LE encoded text; decodes and strips null bytes.
    Returns an empty list on any error or non-Windows platform.
    """
    import subprocess
    try:
        result = subprocess.run(
            ["wsl", "--list", "--quiet"],
            capture_output=True,
            timeout=5,
        )
        # No distros installed ->WSL returns non-zero or outputs help text
        if result.returncode != 0:
            return []
        # WSL --list outputs UTF-16-LE with BOM on some builds
        raw = result.stdout
        try:
            text = raw.decode("utf-16-le", errors="ignore")
        except Exception:
            text = raw.decode("utf-8", errors="ignore")
        distros = [ln.strip().strip("\x00") for ln in text.splitlines() if ln.strip().strip("\x00")]
        # Filter out lines that look like help text rather than distro names
        distros = [d for d in distros if not d.startswith("-") and " " not in d]
        return distros
    except Exception:
        return []


def _print_install_guide_windows(distros: list) -> None:
    """Print step-by-step QE installation guide for Windows."""
    SEP = "=" * 60

    print()
    print("+" + "=" * 58 + "+")
    print("|  QE Installation Guide - Windows                       |")
    print("+" + "=" * 58 + "+")
    print()
    print("  QE (pw.x) cannot run natively on Windows.")
    print("  You need WSL2 (Windows Subsystem for Linux).")
    print()

    if not distros:
        # No WSL distro installed yet
        print(f"  {SEP}")
        print("  Step 1 -Install Ubuntu via WSL2")
        print(f"  {SEP}")
        print("  Open PowerShell as Administrator and run:")
        print()
        print("    wsl --install -d Ubuntu-22.04")
        print()
        print("  Restart your computer if prompted.")
        print("  When Ubuntu opens, set a username and password.")
        print()
    else:
        ubuntu = next((d for d in distros if "ubuntu" in d.lower()), distros[0])
        print(f"  WSL2 distro found: {ubuntu}")
        print()

    wsl_cmd = "wsl -d Ubuntu-22.04" if not distros else f"wsl -d {next((d for d in distros if 'ubuntu' in d.lower()), distros[0])}"

    print(f"  {SEP}")
    step = "Step 2" if not distros else "Step 1"
    print(f"  {step} -Install Quantum ESPRESSO inside Ubuntu")
    print(f"  {SEP}")
    print(f"  In the Ubuntu terminal ({wsl_cmd}):")
    print()
    print("    sudo apt update")
    print("    sudo apt install -y quantum-espresso")
    print("    pw.x --version   # verify")
    print()

    print(f"  {SEP}")
    step2 = "Step 3" if not distros else "Step 2"
    print(f"  {step2} -Run SHALOM from inside Ubuntu")
    print(f"  {SEP}")
    print("  All SHALOM commands must be run from within WSL2:")
    print()
    print(f"    {wsl_cmd}")
    print("    cd /mnt/c/<path-to-SHALOM>          # your SHALOM clone location")
    print()
    print("  Set up Python environment inside Ubuntu:")
    print()
    print("    conda env create -f environment.yml   # if conda installed")
    print("    conda activate shalom-env")
    print()
    print("  Download pseudopotentials:")
    print()
    print("    python -m shalom setup-qe --elements Si --download")
    print()
    print("  Run first calculation:")
    print()
    print("    python -m shalom run Si --backend qe --calc scf")
    print()


def _print_install_guide_linux() -> None:
    """Print QE installation options for Linux/macOS."""
    SEP = "=" * 60
    print()
    print(f"  {SEP}")
    print("  QE Installation Options")
    print(f"  {SEP}")
    print()
    print("  Option A -System package manager (Ubuntu/Debian):")
    print()
    print("    sudo apt update && sudo apt install -y quantum-espresso")
    print()
    print("  Option B -Conda (cross-platform):")
    print()
    print("    conda install -c conda-forge qe")
    print()
    print("  Option C -Build from source (latest features):")
    print("    https://www.quantum-espresso.org/Doc/user_guide/")
    print()


def cmd_setup_qe(args: argparse.Namespace) -> int:
    """Execute the 'setup-qe' subcommand."""
    import os
    import platform
    import shutil
    from pathlib import Path

    from shalom.backends.qe_config import SSSP_ELEMENTS, get_pseudo_filename

    SEP = "=" * 60
    issues = 0
    is_windows = platform.system() == "Windows"

    print("=" * 60)
    print("  SHALOM -QE Environment Check")
    print("=" * 60)

    # 1. Check pw.x
    print()
    pw_path = shutil.which("pw.x")
    if pw_path:
        print(f"  pw.x        OK   {pw_path}")
    else:
        print("  pw.x        NOT FOUND")
        issues += 1
        if is_windows:
            distros = _detect_wsl_distros()
            wsl_available = bool(shutil.which("wsl"))
            if wsl_available:
                _print_install_guide_windows(distros)
            else:
                print()
                print(f"  {SEP}")
                print("  WSL2 is not available on this machine.")
                print("  Enable it in Windows Features ->'Windows Subsystem for Linux'")
                print("  then run:  wsl --install -d Ubuntu-22.04")
                print(f"  {SEP}")
        else:
            _print_install_guide_linux()

    # 2. Resolve pseudo_dir
    print()
    _default_pseudo = str(Path.home() / "pseudopotentials")
    pseudo_dir_arg = args.pseudo_dir or os.environ.get("SHALOM_PSEUDO_DIR", _default_pseudo)
    pseudo_path = Path(pseudo_dir_arg).expanduser().resolve()
    if pseudo_path.is_dir():
        print(f"  pseudo_dir  OK   {pseudo_path}")
    else:
        print(f"  pseudo_dir  NOT FOUND   ({pseudo_path})")
        print()
        print(f"  {SEP}")
        print("  Pseudopotential Directory Setup")
        print(f"  {SEP}")
        print()
        print("  Option A -Auto-download (recommended):")
        print()
        print("    python -m shalom setup-qe --elements Si,Fe,O --download")
        print()
        print("  Option B -Set an existing directory:")
        print()
        print(f"    export SHALOM_PSEUDO_DIR=/path/to/pseudos")
        print(f"    python -m shalom setup-qe")
        print()
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
        n_ok = 0
        for el in elements:
            if el not in SSSP_ELEMENTS:
                continue
            upf = get_pseudo_filename(el)
            upf_path = pseudo_path / upf
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
                n_ok += 1
            else:
                missing.append((el, upf))
                issues += 1

        total = len(elements)
        print()
        if missing:
            print(f"  Pseudopotentials  {n_ok}/{total} OK -{len(missing)} missing")
            if len(missing) <= 10:
                print()
                for el, upf in missing:
                    print(f"    {el:<4s}  {upf}  [MISSING]")
            else:
                print(f"    (showing first 5 of {len(missing)})")
                for el, upf in missing[:5]:
                    print(f"    {el:<4s}  {upf}  [MISSING]")
                print(f"    ... and {len(missing) - 5} more")
            print()
            print(f"  Download missing pseudopotentials:")
            if args.elements:
                print(f"    python -m shalom setup-qe --elements {args.elements} --download")
            else:
                print(f"    python -m shalom setup-qe --elements Si,Fe,O --download  # specific")
                print(f"    python -m shalom setup-qe --download                     # all 60")
        else:
            print(f"  Pseudopotentials  {n_ok}/{total} OK")

    # 5. Download if requested
    if args.download and missing:
        import urllib.request

        pseudo_path.mkdir(parents=True, exist_ok=True)
        print()
        print(f"  Downloading {len(missing)} pseudopotential(s) to {pseudo_path} ...")
        print()
        base_url = "https://pseudopotentials.quantum-espresso.org/upf_files"
        for el, upf in missing:
            url = f"{base_url}/{upf}"
            dest = pseudo_path / upf
            print(f"    {upf:<50s} ", end="", flush=True)
            try:
                urllib.request.urlretrieve(url, str(dest))
                print("OK")
                issues -= 1
            except Exception as e:
                print(f"FAILED ({e})")

    # 6. Summary
    print()
    print("=" * 60)
    if issues == 0:
        print("  Status: READY -QE execution is available.")
    else:
        print(f"  Status: {issues} issue(s) remaining.")
        print("  Fix the items above, then re-run:")
        print("    python -m shalom setup-qe")
    print("=" * 60)
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


def cmd_plot(args: argparse.Namespace) -> int:
    """Execute the 'plot' subcommand."""
    from shalom.backends.qe_parser import (
        find_xml_path, parse_xml_bands, parse_dos_file, extract_fermi_energy,
    )

    if not args.bands and not args.dos:
        print("Error: specify at least one of --bands or --dos.")
        return 1

    calc_dir = os.path.abspath(args.calc_dir)
    if not os.path.isdir(calc_dir):
        print(f"Error: directory not found: {calc_dir}")
        return 1

    # Determine Fermi energy
    fermi: Optional[float] = None
    if args.fermi_from:
        fermi = extract_fermi_energy(os.path.join(args.fermi_from, "pw.out"))
    if fermi is None:
        fermi = extract_fermi_energy(os.path.join(calc_dir, "pw.out"))
    if fermi is None:
        fermi = 0.0
        print("Warning: Fermi energy not found; using 0.0 eV.")

    energy_window = (args.emin, args.emax)
    rc = 0

    if args.bands:
        xml_path = find_xml_path(calc_dir)
        if xml_path is None:
            print(f"Error: data-file-schema.xml not found in {calc_dir}")
            rc = 1
        else:
            try:
                from shalom.plotting.band_plot import BandStructurePlotter
                bs = parse_xml_bands(xml_path, fermi_energy=fermi)
                out = args.output or os.path.join(calc_dir, "bands.png")
                plotter = BandStructurePlotter(bs)
                plotter.plot(output_path=out, title=args.title, energy_window=energy_window)
                print(f"Band structure plot saved: {out}")
            except ImportError:
                print("Error: matplotlib not installed. Run: pip install shalom[plotting]")
                rc = 1

    if args.dos:
        dos_path = os.path.join(calc_dir, "pwscf.dos")
        if not os.path.isfile(dos_path):
            print(f"Error: pwscf.dos not found in {calc_dir}")
            rc = 1
        else:
            try:
                from shalom.plotting.dos_plot import DOSPlotter
                dos_data = parse_dos_file(dos_path)
                dos_data.fermi_energy = fermi
                out = args.output or os.path.join(calc_dir, "dos.png")
                plotter = DOSPlotter(dos_data)
                plotter.plot(output_path=out, title=args.title, energy_window=energy_window)
                print(f"DOS plot saved: {out}")
            except ImportError:
                print("Error: matplotlib not installed. Run: pip install shalom[plotting]")
                rc = 1

    return rc


def cmd_workflow(args: argparse.Namespace) -> int:
    """Execute the 'workflow' subcommand."""
    from shalom.workflows.standard import StandardWorkflow

    atoms = _load_atoms(args)
    if atoms is None:
        return 1

    wf = StandardWorkflow(
        atoms=atoms,
        output_dir=args.output,
        pseudo_dir=args.pseudo_dir,
        nprocs=args.nprocs,
        mpi_command=args.mpi_command,
        timeout=args.timeout,
        accuracy=args.accuracy,
        skip_relax=args.skip_relax,
        is_2d=args.is_2d,
        dos_emin=args.dos_emin,
        dos_emax=args.dos_emax,
    )

    try:
        result = wf.run()
        print(f"Workflow complete: {args.output}")
        if result.get("bands_png"):
            print(f"  bands.png: {result['bands_png']}")
        if result.get("dos_png"):
            print(f"  dos.png:   {result['dos_png']}")
        if result.get("fermi_energy") is not None:
            print(f"  Fermi energy: {result['fermi_energy']:.4f} eV")
        return 0
    except Exception as exc:
        print(f"Error: workflow failed: {exc}")
        return 1


def cmd_converge(args: argparse.Namespace) -> int:
    """Execute the 'converge' subcommand."""
    from shalom.workflows.convergence import CutoffConvergence, KpointConvergence

    atoms = _load_atoms(args)
    if atoms is None:
        return 1

    # Parse values
    if args.values:
        try:
            values = [float(v.strip()) for v in args.values.split(",")]
        except ValueError:
            print(f"Error: invalid --values '{args.values}'. Use comma-separated numbers.")
            return 1
    else:
        if args.test == "cutoff":
            values = [30.0, 40.0, 50.0, 60.0, 80.0]
            print(f"Using default ecutwfc values: {values} Ry")
        else:
            values = [20.0, 30.0, 40.0, 50.0]
            print(f"Using default k-point resolutions: {values} Bohr^-1")

    # Parse kgrid for cutoff test
    kgrid = None
    if args.kgrid:
        try:
            kgrid = [int(n) for n in args.kgrid.lower().replace("x", ",").split(",")]
            if len(kgrid) != 3:
                raise ValueError
        except ValueError:
            print(f"Error: invalid --kgrid '{args.kgrid}'. Use format NxNyNz (e.g. 4x4x4).")
            return 1

    try:
        if args.test == "cutoff":
            conv = CutoffConvergence(
                atoms=atoms,
                output_dir=args.output,
                values=values,
                kgrid=kgrid,
                pseudo_dir=args.pseudo_dir,
                nprocs=args.nprocs,
                timeout=args.timeout,
                accuracy=args.accuracy,
                threshold_per_atom=args.threshold,
            )
        else:  # kpoints
            conv = KpointConvergence(
                atoms=atoms,
                output_dir=args.output,
                resolutions=values,
                ecutwfc=args.ecutwfc,
                pseudo_dir=args.pseudo_dir,
                nprocs=args.nprocs,
                timeout=args.timeout,
                accuracy=args.accuracy,
                threshold_per_atom=args.threshold,
            )

        result = conv.run()
        print(result.summary())

        plot_path = conv.plot(result)
        if plot_path:
            print(f"Convergence plot: {plot_path}")

        return 0 if result.converged_value is not None else 1

    except Exception as exc:
        print(f"Error: convergence test failed: {exc}")
        return 1


def cmd_pipeline(args: argparse.Namespace) -> int:
    """Execute the 'pipeline' subcommand -LLM-driven autonomous discovery."""
    from shalom.pipeline import Pipeline, PipelineConfig
    from shalom.core.schemas import PipelineStatus

    # Resolve model name
    model = args.model
    if model is None:
        model = "claude-sonnet-4-6" if args.provider == "anthropic" else "gpt-4o"

    # Resolve base_url (CLI flag > env var)
    base_url = args.base_url or os.environ.get("SHALOM_LLM_BASE_URL")

    # Check API key (not required when using a local LLM via base_url)
    key_env = "ANTHROPIC_API_KEY" if args.provider == "anthropic" else "OPENAI_API_KEY"
    if not os.environ.get(key_env) and not base_url:
        print(f"Error: {key_env} environment variable not set.")
        print(f"  Get your API key and set it:")
        if args.provider == "anthropic":
            print("    export ANTHROPIC_API_KEY='sk-ant-...'")
            print("  Get key at: https://console.anthropic.com/")
        else:
            print("    export OPENAI_API_KEY='sk-...'")
            print("  Get key at: https://platform.openai.com/api-keys")
        print()
        print("  Or use a local LLM server:")
        print("    python -m shalom pipeline 'objective' --base-url http://localhost:11434/v1")
        return 1

    # Parse steps
    steps = None
    if args.steps:
        steps = [s.strip() for s in args.steps.split(",")]

    # Build config
    config = PipelineConfig(
        backend_name=args.backend,
        provider_type=args.provider,
        model_name=model,
        output_dir=args.output or "",
        calc_type=args.calc,
        accuracy=args.accuracy,
        execute=args.execute,
        nprocs=args.nprocs,
        execution_timeout=args.timeout,
        max_outer_loops=args.max_loops,
        selector_mode=args.selector_mode,
        steps=steps,
        material_name=args.material,
        base_url=base_url,
    )

    print(f"SHALOM Pipeline -LLM-driven Material Discovery")
    print(f"  Objective: {args.objective}")
    print(f"  Provider:  {args.provider} ({model})")
    if base_url:
        print(f"  Base URL:  {base_url}")
    print(f"  Backend:   {args.backend}")
    if args.material:
        print(f"  Material:  {args.material} (Design layer skipped)")
    print()

    try:
        pipeline = Pipeline(objective=args.objective, config=config)
        result = pipeline.run()
    except Exception as exc:
        print(f"Error: pipeline failed: {exc}")
        return 1

    # Display results
    print(f"Status:   {result.status.value}")
    if result.ranked_material:
        mat = result.ranked_material.candidate.material_name
        score = result.ranked_material.score
        print(f"Material: {mat} (score={score:.2f})")
    if result.structure_path:
        print(f"Output:   {result.structure_path}")
    if result.execution_wall_time is not None:
        print(f"DFT time: {result.execution_wall_time:.1f}s")
    if result.quality_warnings:
        print(f"Warnings: {', '.join(result.quality_warnings)}")
    if result.error_message:
        print(f"Error:    {result.error_message}")
    if result.elapsed_seconds is not None:
        print(f"Elapsed:  {result.elapsed_seconds:.1f}s")

    print()
    print(f"Steps completed: {' -> '.join(result.steps_completed)}")

    if result.status == PipelineStatus.AWAITING_DFT:
        print()
        print("Next: Run DFT on the generated input files, then use:")
        print(f"  Pipeline.resume_from_dft(result)  # Python API")

    success = result.status in (
        PipelineStatus.COMPLETED,
        PipelineStatus.COMPLETED_DESIGN,
        PipelineStatus.AWAITING_DFT,
    )
    return 0 if success else 1


def _load_atoms(args: argparse.Namespace):
    """Load ASE Atoms from args.material / args.structure. Returns None on error."""
    structure_file = getattr(args, "structure", None)
    material = getattr(args, "material", None)

    if structure_file:
        try:
            from ase.io import read as ase_read
            return ase_read(structure_file)
        except Exception as exc:
            print(f"Error: cannot read structure file '{structure_file}': {exc}")
            return None

    if material:
        # Try Materials Project first, then ASE bulk fallback
        mp_error = None
        try:
            from shalom.mp_client import fetch_structure
            result = fetch_structure(material)
            return result.atoms
        except (ImportError, EnvironmentError, ValueError) as exc:
            mp_error = str(exc)

        # Fallback: ASE bulk builder (simple elements only)
        if mp_error is not None:
            try:
                from ase.build import bulk
                atoms = bulk(material)
                print(f"Note: built bulk {material} from ASE (MP unavailable: {mp_error}).")
                return atoms
            except Exception:
                print(
                    f"Error: cannot resolve '{material}'.\n"
                    f"  MP: {mp_error}\n"
                    f"  ASE bulk also failed (only simple elements like Si, Fe, Cu work).\n"
                    f"  Use --structure FILE or an MP ID (mp-19717)."
                )
                return None

    print("Error: provide a material spec or --structure file.")
    return None


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
    elif args.command == "plot":
        sys.exit(cmd_plot(args))
    elif args.command == "workflow":
        sys.exit(cmd_workflow(args))
    elif args.command == "converge":
        sys.exit(cmd_converge(args))
    elif args.command == "pipeline":
        sys.exit(cmd_pipeline(args))
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
