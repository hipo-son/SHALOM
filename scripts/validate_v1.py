#!/usr/bin/env python3
"""SHALOM v1 Physics Validation — Si Diamond Benchmark.

Runs a full 5-step QE workflow on Si diamond and validates the results
against known PBE reference values.

Usage (from WSL):
    export SHALOM_PSEUDO_DIR=/root/pseudopotentials
    python scripts/validate_v1.py [--output-dir /tmp/si_validate]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

import numpy as np
from ase.build import bulk

# ── Reference values (PBE, Si diamond, SSSP Efficiency) ──────────────
REF = {
    "lattice_a": (5.40, 5.48),       # Å — relaxed lattice constant range
    "energy_per_atom": (-160, -150),  # eV — rough per-atom energy
    "fermi_energy": (4.0, 9.0),       # eV
    "bandgap_indirect": (0.3, 0.9),   # eV — PBE underestimates (expt: 1.12)
    "n_atoms": 2,                      # primitive cell
}


def check(name: str, value, lo, hi, unit: str = "") -> bool:
    ok = lo <= value <= hi
    status = "PASS" if ok else "FAIL"
    u = f" {unit}" if unit else ""
    print(f"  [{status}] {name}: {value:.4f}{u}  (expected {lo}–{hi}{u})")
    return ok


def main():
    parser = argparse.ArgumentParser(description="SHALOM v1 Si benchmark")
    parser.add_argument(
        "--output-dir", default="/tmp/shalom_v1_validate",
        help="Directory for workflow output",
    )
    parser.add_argument("--nprocs", type=int, default=1)
    parser.add_argument("--skip-relax", action="store_true")
    args = parser.parse_args()

    pseudo_dir = os.environ.get("SHALOM_PSEUDO_DIR", "")
    if not pseudo_dir or not os.path.isdir(pseudo_dir):
        print("ERROR: Set SHALOM_PSEUDO_DIR to a directory with Si UPF file.")
        sys.exit(1)

    if not shutil.which("pw.x"):
        print("ERROR: pw.x not found in PATH.")
        sys.exit(1)

    print("=" * 60)
    print("  SHALOM v1 — Si Diamond Validation")
    print("=" * 60)
    print(f"  pseudo_dir:  {pseudo_dir}")
    print(f"  output_dir:  {args.output_dir}")
    print(f"  nprocs:      {args.nprocs}")
    print(f"  skip_relax:  {args.skip_relax}")
    print()

    # ── Build structure ──────────────────────────────────────────────
    si = bulk("Si", "diamond", a=5.43)

    # ── Run full workflow ────────────────────────────────────────────
    from shalom.workflows.standard import StandardWorkflow

    t0 = time.time()
    wf = StandardWorkflow(
        atoms=si,
        output_dir=args.output_dir,
        pseudo_dir=pseudo_dir,
        nprocs=args.nprocs,
        accuracy="standard",
        skip_relax=args.skip_relax,
        timeout=1800,
    )

    print("Running 5-step workflow ...")
    result = wf.run()
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    print()

    # ── Validate results ─────────────────────────────────────────────
    all_pass = True

    print("─" * 60)
    print("  Structure Validation")
    print("─" * 60)

    atoms = result.get("atoms")
    if atoms is None:
        print("  [FAIL] No atoms in result")
        all_pass = False
    else:
        n = len(atoms)
        ok = n == REF["n_atoms"]
        print(f"  [{'PASS' if ok else 'FAIL'}] Atom count: {n} (expected {REF['n_atoms']})")
        all_pass &= ok

        # Lattice constant from cell
        a = atoms.cell.cellpar()[0]
        lo, hi = REF["lattice_a"]
        # For primitive FCC cell, cell parameter ≠ conventional a.
        # Conventional a = sqrt(2) * primitive cell edge
        a_conv = np.linalg.norm(atoms.cell[0]) * np.sqrt(2)
        all_pass &= check("Lattice constant (conv)", a_conv, lo, hi, "Å")

    print()
    print("─" * 60)
    print("  Electronic Structure Validation")
    print("─" * 60)

    fermi = result.get("fermi_energy")
    if fermi is not None:
        lo, hi = REF["fermi_energy"]
        all_pass &= check("Fermi energy", fermi, lo, hi, "eV")
    else:
        print("  [FAIL] Fermi energy not extracted")
        all_pass = False

    # ── Parse band structure for bandgap ──────────────────────────
    bands_dir = os.path.join(args.output_dir, "03_bands")
    scf_tmp = os.path.join(args.output_dir, "02_scf", "tmp")
    try:
        from shalom.backends.qe_parser import find_xml_path, parse_xml_bands

        xml_path = find_xml_path(bands_dir)
        if xml_path is None:
            xml_path = find_xml_path(scf_tmp)
        if xml_path and fermi is not None:
            bs = parse_xml_bands(xml_path, fermi_energy=fermi)
            eigs = bs.eigenvalues  # (nkpts, nbands), already in eV

            print(f"  Band data: {eigs.shape[0]} k-points, {eigs.shape[1]} bands")

            # Bandgap: difference between CBM and VBM
            vbm = eigs[eigs <= fermi].max()
            cbm = eigs[eigs > fermi].min()
            gap = cbm - vbm

            lo, hi = REF["bandgap_indirect"]
            all_pass &= check("Indirect bandgap", gap, lo, hi, "eV")
            print(f"         VBM = {vbm:.4f} eV, CBM = {cbm:.4f} eV")
        else:
            print("  [SKIP] Band XML or Fermi not available")
    except Exception as e:
        print(f"  [FAIL] Band parsing error: {e}")
        all_pass = False

    # ── Check DOS ─────────────────────────────────────────────────
    nscf_dir = os.path.join(args.output_dir, "04_nscf")
    dos_file = os.path.join(nscf_dir, "pwscf.dos")
    if os.path.isfile(dos_file):
        from shalom.backends.qe_parser import parse_dos_file
        dos = parse_dos_file(dos_file)
        print(f"  DOS data: {len(dos.energies)} points, "
              f"E=[{dos.energies.min():.1f}, {dos.energies.max():.1f}] eV")
        ok = np.all(dos.dos >= -1e-6)
        print(f"  [{'PASS' if ok else 'FAIL'}] DOS non-negative")
        all_pass &= ok
    else:
        print("  [SKIP] DOS file not found")

    # ── Plot files ────────────────────────────────────────────────
    print()
    print("─" * 60)
    print("  Output Files")
    print("─" * 60)

    for key in ("bands_png", "dos_png"):
        path = result.get(key)
        if path and os.path.isfile(path):
            size = os.path.getsize(path) / 1024
            print(f"  [PASS] {key}: {path} ({size:.1f} KB)")
        else:
            print(f"  [WARN] {key}: not generated")

    calc_dirs = result.get("calc_dirs", {})
    for step, d in calc_dirs.items():
        exists = os.path.isdir(d)
        print(f"  {'[OK]' if exists else '[--]'} {step}: {d}")

    # ── Summary ───────────────────────────────────────────────────
    print()
    print("=" * 60)
    if all_pass:
        print("  RESULT: ALL CHECKS PASSED")
    else:
        print("  RESULT: SOME CHECKS FAILED")
    print(f"  Total time: {elapsed:.1f}s")
    print("=" * 60)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
