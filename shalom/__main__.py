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
        help="Output directory (auto-generated if omitted).",
    )
    run_parser.add_argument(
        "--structure",
        default=None,
        help="Path to local structure file (POSCAR, CIF, etc.).",
    )
    run_parser.add_argument(
        "--pseudo-dir",
        default=None,
        help="QE pseudopotential directory (default: $SHALOM_PSEUDO_DIR or './').",
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
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
