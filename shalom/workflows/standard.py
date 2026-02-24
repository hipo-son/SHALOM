"""5-step sequential DFT workflow for QE: vc-relax → scf → bands → nscf → dos.

Produces ``bands.png`` and ``dos.png`` in the output directory.

The ``outdir`` for the bands and nscf pw.x runs is **always set to the absolute
path of the scf ``tmp/`` directory** so that the charge density written by scf
is shared by all subsequent non-self-consistent calculations.

Energy units in dos.x namelist
--------------------------------
The ``dos.x`` namelist (``&DOS``) expects **Emin/Emax/DeltaE in Rydberg**,
not in eV.  The conversion ``EV_TO_RY = 1.0 / 13.6057`` is applied
automatically when writing ``dos.in``.

Fermi energy priority
----------------------
NSCF → SCF (NSCF uses a dense k-mesh and is therefore more accurate for metals).

Usage::

    from ase.build import bulk
    from shalom.workflows.standard import StandardWorkflow

    si = bulk("Si", "diamond", a=5.43)
    wf = StandardWorkflow(atoms=si, output_dir="/tmp/si_wf", nprocs=4)
    result = wf.run()
    print("bands:", result["bands_png"])
    print("dos:  ", result["dos_png"])
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import TYPE_CHECKING, Any, Dict, Optional

from ase.io import read as ase_read

from shalom.backends._physics import AccuracyLevel
from shalom.backends.qe import QEBackend
from shalom.backends.qe_config import (
    QECalculationType,
    QEKPointsConfig,
    compute_ecutrho,
    generate_band_kpath,
    get_band_calc_atoms,
    get_qe_preset,
)
from shalom.backends.qe_parser import (
    compute_nbnd,
    extract_fermi_energy,
    find_xml_path,
    parse_dos_file,
    parse_xml_bands,
)
from shalom.backends.runner import ExecutionConfig, ExecutionRunner

if TYPE_CHECKING:
    from ase import Atoms

logger = logging.getLogger(__name__)

EV_TO_RY: float = 1.0 / 13.6057   # dos.x namelist energies must be in Ry


class StandardWorkflow:
    """vc-relax → scf → bands → nscf → dos.x → plots.

    Directory layout::

        output_dir/
          01_vc_relax/  pw.in, pw.out, tmp/
          02_scf/       pw.in, pw.out, tmp/   ← shalom.save/ lives here
          03_bands/     pw.in, pw.out
          04_nscf/      pw.in, pw.out, dos.in, dos.out, pwscf.dos
          bands.png
          dos.png

    Args:
        atoms: Starting structure (ASE ``Atoms``).
        output_dir: Root output directory.  Created if absent.
        pseudo_dir: Override for the pseudopotential directory.
        nprocs: Number of MPI processes for ``pw.x``.
        mpi_command: MPI launcher (e.g. ``"mpirun"``).
        pw_executable: Path/name of the ``pw.x`` binary.
        dos_executable: Path/name of the ``dos.x`` binary.
        timeout: Per-step wall-time limit in seconds.
        accuracy: QE preset accuracy — ``"standard"`` or ``"precise"``.
        skip_relax: If ``True``, skip the vc-relax step and use ``atoms``
            directly as the starting geometry for scf.
        npoints_kpath: Number of k-points per segment on the band path.
        is_2d: If ``True``, enforce ``kz = 0`` on the band k-path and set
            QE 2D isolation flags for all steps.
        dos_emin: DOS energy window minimum in eV (converted to Ry internally).
        dos_emax: DOS energy window maximum in eV (converted to Ry internally).
        dos_deltaE: DOS energy step in eV (converted to Ry internally).
    """

    def __init__(
        self,
        atoms: "Atoms",
        output_dir: str,
        *,
        pseudo_dir: Optional[str] = None,
        nprocs: int = 1,
        mpi_command: str = "mpirun",
        pw_executable: str = "pw.x",
        dos_executable: str = "dos.x",
        timeout: int = 7200,
        accuracy: str = "standard",
        skip_relax: bool = False,
        npoints_kpath: int = 40,
        is_2d: bool = False,
        dos_emin: float = -20.0,
        dos_emax: float = 10.0,
        dos_deltaE: float = 0.01,
    ) -> None:
        self.atoms = atoms
        self.output_dir = os.path.abspath(output_dir)
        self.pseudo_dir = pseudo_dir
        self.nprocs = nprocs
        self.mpi_command = mpi_command
        self.pw_executable = pw_executable
        self.dos_executable = dos_executable
        self.timeout = timeout
        self.accuracy = accuracy
        self.skip_relax = skip_relax
        self.npoints_kpath = npoints_kpath
        self.is_2d = is_2d
        self.dos_emin = dos_emin
        self.dos_emax = dos_emax
        self.dos_deltaE = dos_deltaE
        # Cached after run(): seekpath primitive cell and pre-computed kpath
        self._calc_atoms: Optional["Atoms"] = None
        self._kpath_cfg: Optional[Any] = None

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Execute the full workflow.

        Returns:
            Dict with keys:
            - ``"atoms"`` — seekpath standardized primitive cell used for SCF,
              bands, and NSCF (``get_band_calc_atoms(relaxed)`` when seekpath
              is available, otherwise the post-vc-relax or original structure).
            - ``"fermi_energy"`` — Fermi energy in eV (NSCF > SCF priority).
            - ``"bands_png"`` — absolute path to the band structure plot, or
              ``None`` if plotting failed.
            - ``"dos_png"`` — absolute path to the DOS plot, or ``None``.
            - ``"calc_dirs"`` — dict mapping step names to directories.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        relax_dir = os.path.join(self.output_dir, "01_vc_relax")
        scf_dir   = os.path.join(self.output_dir, "02_scf")
        bands_dir = os.path.join(self.output_dir, "03_bands")
        nscf_dir  = os.path.join(self.output_dir, "04_nscf")

        # Absolute SCF tmp dir — shared by bands & nscf
        scf_tmp_dir = os.path.abspath(os.path.join(scf_dir, "tmp"))

        # ------------------------------------------------------------------
        # Step 1: vc-relax (optional)
        # ------------------------------------------------------------------
        current_atoms = self.atoms
        if not self.skip_relax:
            logger.info("[1/5] vc-relax")
            current_atoms = self._run_vc_relax(relax_dir, current_atoms)
        else:
            logger.info("[1/5] vc-relax SKIPPED")

        # Convert to seekpath standardized primitive cell so that the k-path
        # coordinates returned by seekpath (crystal_b frame) match the cell
        # written to QE CELL_PARAMETERS. SCF, bands and NSCF must all share
        # the same cell for the charge-density transfer to be valid.
        calc_atoms = get_band_calc_atoms(current_atoms, is_2d=self.is_2d) or current_atoms
        self._calc_atoms = calc_atoms
        self._kpath_cfg = generate_band_kpath(
            calc_atoms, npoints=self.npoints_kpath, is_2d=self.is_2d
        )

        # ------------------------------------------------------------------
        # Step 2: scf
        # ------------------------------------------------------------------
        logger.info("[2/5] scf")
        self._run_scf(scf_dir, calc_atoms)

        # ------------------------------------------------------------------
        # Step 3: bands
        # ------------------------------------------------------------------
        logger.info("[3/5] bands")
        self._run_bands(bands_dir, calc_atoms, scf_tmp_dir)

        # ------------------------------------------------------------------
        # Step 4: nscf
        # ------------------------------------------------------------------
        logger.info("[4/5] nscf")
        self._run_nscf(nscf_dir, calc_atoms, scf_tmp_dir)

        # ------------------------------------------------------------------
        # Step 5: dos.x
        # ------------------------------------------------------------------
        logger.info("[5/5] dos.x")
        self._run_dos(nscf_dir, scf_tmp_dir)

        # ------------------------------------------------------------------
        # Fermi energy (NSCF preferred)
        # ------------------------------------------------------------------
        fermi = self._get_best_fermi_energy(scf_dir, nscf_dir)

        # ------------------------------------------------------------------
        # Plotting
        # ------------------------------------------------------------------
        bands_png = self._plot_bands(bands_dir, fermi, scf_tmp_dir)
        dos_png   = self._plot_dos(nscf_dir, fermi)

        return {
            "atoms": calc_atoms,
            "fermi_energy": fermi,
            "bands_png": bands_png,
            "dos_png": dos_png,
            "calc_dirs": {
                "vc_relax": relax_dir,
                "scf": scf_dir,
                "bands": bands_dir,
                "nscf": nscf_dir,
            },
        }

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    def _run_vc_relax(self, calc_dir: str, atoms: "Atoms") -> "Atoms":
        """Run vc-relax and return the relaxed structure."""
        os.makedirs(calc_dir, exist_ok=True)
        acc = AccuracyLevel.PRECISE if self.accuracy == "precise" else AccuracyLevel.STANDARD
        config = get_qe_preset(QECalculationType.VC_RELAX, accuracy=acc, atoms=atoms)
        if self.pseudo_dir:
            config.pseudo_dir = self.pseudo_dir

        backend = QEBackend()
        backend.write_input(atoms, calc_dir, config=config)
        self._pw_run(calc_dir)

        # Try to read relaxed structure
        pw_out = os.path.join(calc_dir, "pw.out")
        try:
            relaxed = ase_read(pw_out, format="espresso-out", index=-1)
            logger.info("vc-relax: read relaxed structure (%d atoms)", len(relaxed))
            return relaxed
        except Exception as exc:
            logger.warning("vc-relax: could not read relaxed structure (%s); using input.", exc)
            return atoms

    def _run_scf(self, calc_dir: str, atoms: "Atoms") -> None:
        """Run scf calculation."""
        os.makedirs(calc_dir, exist_ok=True)
        acc = AccuracyLevel.PRECISE if self.accuracy == "precise" else AccuracyLevel.STANDARD
        config = get_qe_preset(QECalculationType.SCF, accuracy=acc, atoms=atoms)
        if self.pseudo_dir:
            config.pseudo_dir = self.pseudo_dir

        backend = QEBackend()
        backend.write_input(atoms, calc_dir, config=config)
        self._pw_run(calc_dir)

    def _run_bands(self, calc_dir: str, atoms: "Atoms", scf_tmp_dir: str) -> None:
        """Run bands calculation using the scf charge density."""
        os.makedirs(calc_dir, exist_ok=True)
        acc = AccuracyLevel.PRECISE if self.accuracy == "precise" else AccuracyLevel.STANDARD
        config = get_qe_preset(QECalculationType.BANDS, accuracy=acc, atoms=atoms)
        if self.pseudo_dir:
            config.pseudo_dir = self.pseudo_dir

        # Share scf charge density (absolute path required!)
        config.control["outdir"] = scf_tmp_dir

        # Use pre-computed kpath (from run()) or generate on standalone call
        config.kpoints = self._kpath_cfg or generate_band_kpath(
            atoms, npoints=self.npoints_kpath, is_2d=self.is_2d
        )

        # Auto-set nbnd
        nbnd = compute_nbnd(atoms, multiplier=1.3)
        config.system["nbnd"] = max(nbnd, config.system.get("nbnd", 20))

        backend = QEBackend()
        backend.write_input(atoms, calc_dir, config=config)
        self._pw_run(calc_dir)

    def _run_nscf(self, calc_dir: str, atoms: "Atoms", scf_tmp_dir: str) -> None:
        """Run nscf calculation using the scf charge density."""
        os.makedirs(calc_dir, exist_ok=True)
        acc = AccuracyLevel.PRECISE if self.accuracy == "precise" else AccuracyLevel.STANDARD
        config = get_qe_preset(QECalculationType.NSCF, accuracy=acc, atoms=atoms)
        if self.pseudo_dir:
            config.pseudo_dir = self.pseudo_dir

        # Share scf charge density (absolute path required!)
        config.control["outdir"] = scf_tmp_dir

        backend = QEBackend()
        backend.write_input(atoms, calc_dir, config=config)
        self._pw_run(calc_dir)

    def _run_dos(self, calc_dir: str, scf_tmp_dir: str) -> None:
        """Write dos.in and run dos.x (all energies in Ry!)."""
        os.makedirs(calc_dir, exist_ok=True)
        dos_in_path = os.path.join(calc_dir, "dos.in")
        dos_out_path = os.path.join(calc_dir, "dos.out")

        # Convert to Ry — dos.x namelist requires Ry, NOT eV
        emin_ry = self.dos_emin * EV_TO_RY
        emax_ry = self.dos_emax * EV_TO_RY
        delta_ry = self.dos_deltaE * EV_TO_RY

        dos_in_content = (
            "&DOS\n"
            f"  outdir = '{scf_tmp_dir}'\n"
            f"  prefix = 'shalom'\n"
            f"  fildos = 'pwscf.dos'\n"
            f"  Emin = {emin_ry:.6f}\n"
            f"  Emax = {emax_ry:.6f}\n"
            f"  DeltaE = {delta_ry:.8f}\n"
            "/\n"
        )

        with open(dos_in_path, "w", encoding="utf-8") as fh:
            fh.write(dos_in_content)

        self._dos_run(calc_dir)

    # ------------------------------------------------------------------
    # Fermi energy
    # ------------------------------------------------------------------

    def _get_best_fermi_energy(self, scf_dir: str, nscf_dir: str) -> Optional[float]:
        """Return Fermi energy in eV: NSCF preferred, fallback to SCF."""
        nscf_efermi = extract_fermi_energy(os.path.join(nscf_dir, "pw.out"))
        if nscf_efermi is not None:
            logger.info("Fermi energy from NSCF: %.4f eV", nscf_efermi)
            return nscf_efermi
        scf_efermi = extract_fermi_energy(os.path.join(scf_dir, "pw.out"))
        if scf_efermi is not None:
            logger.info("Fermi energy from SCF (fallback): %.4f eV", scf_efermi)
        return scf_efermi

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _plot_bands(
        self,
        bands_dir: str,
        fermi: Optional[float],
        scf_tmp_dir: Optional[str] = None,
    ) -> Optional[str]:
        """Parse bands XML and generate band structure plot."""
        try:
            from shalom.plotting.band_plot import BandStructurePlotter
        except ImportError:
            logger.warning("matplotlib not installed; band plot skipped.")
            return None

        xml_path = find_xml_path(bands_dir)
        if xml_path is None and scf_tmp_dir:
            xml_path = find_xml_path(scf_tmp_dir)
        if xml_path is None:
            logger.warning("bands XML not found in %s; band plot skipped.", bands_dir)
            return None

        bs = parse_xml_bands(xml_path, fermi_energy=fermi or 0.0)

        # Use cached kpath (consistent with calc_atoms used in _run_bands).
        # Fall back to a fresh generation from self.atoms for standalone calls.
        kpath_cfg = self._kpath_cfg or generate_band_kpath(
            self.atoms, npoints=self.npoints_kpath, is_2d=self.is_2d
        )
        if kpath_cfg.kpath_labels:
            # Build cumulative k-point index → label mapping
            cumulative_idx = 0
            label_by_idx: Dict[int, str] = {}
            if kpath_cfg.kpath_points:
                for seg_idx, (_, npts) in enumerate(kpath_cfg.kpath_points):
                    label = kpath_cfg.kpath_labels.get(seg_idx)
                    if label:
                        label_by_idx[cumulative_idx] = label
                    cumulative_idx += npts
            bs.high_sym_labels = label_by_idx

            # Collapse spurious x-axis gaps at path discontinuities.
            # parse_xml_bands() computes cumulative |Δk| distances, so at a
            # break point "X|U" (npts=1), the next k-point U is at
            # dist[b+1] = dist[b] + |X-U|.  Standard band-structure plots
            # reset the distance counter at discontinuities so the sub-paths
            # are plotted side by side with no artificial gap.
            if len(bs.kpath_distances) > 1:
                import numpy as np
                dist = bs.kpath_distances.copy()
                for k_idx in sorted(label_by_idx):
                    if "|" in label_by_idx[k_idx] and k_idx + 1 < len(dist):
                        gap = dist[k_idx + 1] - dist[k_idx]
                        if gap > 0.0:
                            dist[k_idx + 1:] -= gap
                bs.kpath_distances = dist

        output_path = os.path.join(self.output_dir, "bands.png")
        plotter = BandStructurePlotter(bs)
        plotter.plot(output_path=output_path)
        logger.info("Band structure plot saved: %s", output_path)
        return output_path

    def _plot_dos(self, nscf_dir: str, fermi: Optional[float]) -> Optional[str]:
        """Parse pwscf.dos and generate DOS plot."""
        try:
            from shalom.plotting.dos_plot import DOSPlotter
        except ImportError:
            logger.warning("matplotlib not installed; DOS plot skipped.")
            return None

        dos_path = os.path.join(nscf_dir, "pwscf.dos")
        if not os.path.isfile(dos_path):
            logger.warning("pwscf.dos not found in %s; DOS plot skipped.", nscf_dir)
            return None

        dos_data = parse_dos_file(dos_path)
        if fermi is not None:
            dos_data.fermi_energy = fermi

        output_path = os.path.join(self.output_dir, "dos.png")
        plotter = DOSPlotter(dos_data)
        plotter.plot(output_path=output_path)
        logger.info("DOS plot saved: %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Subprocess helpers
    # ------------------------------------------------------------------

    def _pw_run(self, calc_dir: str) -> None:
        """Run pw.x in ``calc_dir`` and raise on failure."""
        exec_config = ExecutionConfig(
            command=self.pw_executable,
            input_file="pw.in",
            output_file="pw.out",
            nprocs=self.nprocs,
            mpi_command=self.mpi_command,
            timeout_seconds=self.timeout,
        )
        runner = ExecutionRunner(exec_config)
        result = runner.run(calc_dir)
        if not result.success:
            raise RuntimeError(
                f"pw.x failed in {calc_dir}: {result.error_message or 'unknown error'}"
            )

    def _dos_run(self, calc_dir: str) -> None:
        """Run dos.x in ``calc_dir`` (serial; no MPI needed)."""
        exec_config = ExecutionConfig(
            command=self.dos_executable,
            input_file="dos.in",
            output_file="dos.out",
            nprocs=1,          # dos.x is serial
            mpi_command=self.mpi_command,
            timeout_seconds=300,
        )
        runner = ExecutionRunner(exec_config)
        result = runner.run(calc_dir)
        if not result.success:
            logger.warning(
                "dos.x failed in %s: %s", calc_dir, result.error_message or "unknown error"
            )
