"""5-step sequential DFT workflow for QE: vc-relax → scf → bands → nscf → dos.

Produces ``bands.png`` and ``dos.png`` in the output directory.

The ``outdir`` for the bands and nscf pw.x runs is **always set to the absolute
path of the scf ``tmp/`` directory** so that the charge density written by scf
is shared by all subsequent non-self-consistent calculations.

Energy units in dos.x namelist
--------------------------------
The ``dos.x`` namelist (``&DOS``) expects **Emin/Emax/DeltaE in eV**.

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
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ase.io import read as ase_read

from shalom.backends._physics import AccuracyLevel, DEFAULT_BAND_NPOINTS, RY_TO_EV
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
from shalom.backends.runner import ExecutionConfig, create_runner

if TYPE_CHECKING:
    from ase import Atoms

logger = logging.getLogger(__name__)

# dos.x Emin/Emax/DeltaE are in eV (QE divides by Ry internally).


@dataclass
class StepStatus:
    """Status of a single workflow step."""

    name: str
    step_number: int
    success: bool
    error_message: Optional[str] = None
    elapsed_seconds: float = 0.0
    summary: str = ""


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
        dos_emin: DOS energy window minimum in eV (dos.x expects eV directly).
        dos_emax: DOS energy window maximum in eV (dos.x expects eV directly).
        dos_deltaE: DOS energy step in eV (dos.x expects eV directly).
        nscf_kmesh: Explicit NSCF k-point grid, e.g. ``[6, 6, 6]``.
            If ``None`` (default), computed from ``DEFAULT_NSCF_KPR``.
        resume: If ``True``, load checkpoint from output_dir and skip
            already-completed steps.
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
        npoints_kpath: int = DEFAULT_BAND_NPOINTS,
        is_2d: bool = False,
        dos_emin: float = -20.0,
        dos_emax: float = 10.0,
        dos_deltaE: float = 0.01,
        nscf_kmesh: Optional[list] = None,
        resume: bool = False,
        wsl: bool = False,
        slurm_config: Optional[Any] = None,
    ) -> None:
        self.atoms = atoms
        self.output_dir = os.path.abspath(output_dir)
        self.pseudo_dir = pseudo_dir
        self.nprocs = nprocs
        self.mpi_command = mpi_command
        self.pw_executable = pw_executable
        self.dos_executable = dos_executable
        self.timeout = timeout
        if nscf_kmesh is not None:
            if len(nscf_kmesh) != 3 or any(n < 1 for n in nscf_kmesh):
                raise ValueError(
                    f"nscf_kmesh must be [Nx, Ny, Nz] with positive ints, got {nscf_kmesh}"
                )
        self.nscf_kmesh = nscf_kmesh
        self.resume = resume
        self.wsl = wsl
        self.slurm_config = slurm_config
        if accuracy not in ("standard", "precise"):
            raise ValueError(
                f"accuracy must be 'standard' or 'precise', got '{accuracy}'"
            )
        self.accuracy = accuracy
        self._accuracy_level = (
            AccuracyLevel.PRECISE if accuracy == "precise"
            else AccuracyLevel.STANDARD
        )
        self.skip_relax = skip_relax
        self.npoints_kpath = npoints_kpath
        self.is_2d = is_2d
        if dos_emin >= dos_emax:
            raise ValueError(
                f"dos_emin ({dos_emin}) must be < dos_emax ({dos_emax})"
            )
        if dos_deltaE <= 0:
            raise ValueError(f"dos_deltaE ({dos_deltaE}) must be > 0")
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
            Dict with keys (backward-compatible):
            - ``"atoms"`` — seekpath standardized primitive cell.
            - ``"fermi_energy"`` — Fermi energy in eV (NSCF > SCF priority).
            - ``"bands_png"`` — path to band structure plot, or ``None``.
            - ``"dos_png"`` — path to DOS plot, or ``None``.
            - ``"calc_dirs"`` — dict mapping step names to directories.

            New keys (v2):
            - ``"step_results"`` — list of :class:`StepStatus` objects.
            - ``"completed_steps"`` — list of step names that succeeded.
            - ``"failed_step"`` — first step that failed, or ``None``.
        """
        self._validate_environment()
        os.makedirs(self.output_dir, exist_ok=True)

        relax_dir = os.path.join(self.output_dir, "01_vc_relax")
        scf_dir   = os.path.join(self.output_dir, "02_scf")
        bands_dir = os.path.join(self.output_dir, "03_bands")
        nscf_dir  = os.path.join(self.output_dir, "04_nscf")

        # Absolute SCF tmp dir — shared by bands & nscf
        scf_tmp_dir = os.path.abspath(os.path.join(scf_dir, "tmp"))

        step_results: List[StepStatus] = []
        failed_step: Optional[str] = None
        scf_ok = False
        bands_ok = False
        nscf_ok = False

        # Resume support: load checkpoint and determine which steps to skip.
        done: set = set()
        if self.resume:
            ckpt = self._load_checkpoint()
            if ckpt:
                done = set(ckpt.get("completed_steps", []))
                logger.info("Resuming workflow — completed steps: %s", sorted(done))

        # ------------------------------------------------------------------
        # Step 1: vc-relax (optional)
        # ------------------------------------------------------------------
        current_atoms = self.atoms
        if "vc_relax" in done:
            step_results.append(StepStatus("vc_relax", 1, True, summary="resumed"))
            logger.info("[1/5] vc-relax — already complete (resumed)")
            # Recover relaxed structure from pw.out if available
            pw_out = os.path.join(relax_dir, "pw.out")
            if os.path.isfile(pw_out):
                try:
                    current_atoms = ase_read(pw_out, format="espresso-out", index=-1)
                except Exception:
                    pass
        elif not self.skip_relax:
            t0 = self._log_step_start(1, 5, "vc-relax")
            try:
                current_atoms = self._run_vc_relax(relax_dir, current_atoms)
                summary = self._extract_pw_summary(os.path.join(relax_dir, "pw.out"))
                elapsed = time.monotonic() - t0
                step_results.append(StepStatus("vc_relax", 1, True,
                                               elapsed_seconds=elapsed, summary=summary))
                self._log_step_end(1, 5, "vc-relax", t0, summary)
                self._save_checkpoint(["vc_relax"])
            except RuntimeError as exc:
                elapsed = time.monotonic() - t0
                step_results.append(StepStatus("vc_relax", 1, False, str(exc), elapsed))
                logger.error("[1/5] vc-relax FAILED (%s) — using input geometry", exc)
                # vc-relax failure is non-fatal: continue with input atoms
        else:
            step_results.append(StepStatus("vc_relax", 1, True, summary="skipped"))
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
        # Step 2: scf (fatal — prerequisite for everything)
        # ------------------------------------------------------------------
        if "scf" in done:
            scf_pw_out = os.path.join(scf_dir, "pw.out")
            if os.path.isfile(scf_pw_out):
                step_results.append(StepStatus("scf", 2, True, summary="resumed"))
                logger.info("[2/5] scf — already complete (resumed)")
                scf_ok = True
            else:
                logger.warning(
                    "Checkpoint says SCF done but pw.out missing; re-running SCF"
                )
                done.discard("scf")
        if "scf" not in done:
            t0 = self._log_step_start(2, 5, "scf")
            try:
                self._run_scf(scf_dir, calc_atoms)
                summary = self._extract_pw_summary(os.path.join(scf_dir, "pw.out"))
                elapsed = time.monotonic() - t0
                step_results.append(StepStatus("scf", 2, True,
                                               elapsed_seconds=elapsed, summary=summary))
                self._log_step_end(2, 5, "scf", t0, summary)
                scf_ok = True
                self._save_checkpoint(
                    [s.name for s in step_results if s.success])
            except RuntimeError as exc:
                elapsed = time.monotonic() - t0
                step_results.append(StepStatus("scf", 2, False, str(exc), elapsed))
                logger.error("[2/5] scf FAILED — %s", exc)
                failed_step = "scf"
                return self._build_result(
                    calc_atoms, None, None, None,
                    relax_dir, scf_dir, bands_dir, nscf_dir,
                    step_results, failed_step,
                )

        # ------------------------------------------------------------------
        # Step 3: bands (depends on SCF)
        # ------------------------------------------------------------------
        if "bands" in done:
            step_results.append(StepStatus("bands", 3, True, summary="resumed"))
            logger.info("[3/5] bands — already complete (resumed)")
            bands_ok = True
        else:
            t0 = self._log_step_start(3, 5, "bands")
            try:
                self._run_bands(bands_dir, calc_atoms, scf_tmp_dir)

                # Preserve bands XML before NSCF overwrites it in scf_tmp_dir.
                bands_xml_src = find_xml_path(scf_tmp_dir)
                if bands_xml_src and os.path.isfile(bands_xml_src):
                    import shutil as _shutil
                    bands_xml_dst = os.path.join(bands_dir, "data-file-schema.xml")
                    try:
                        _shutil.copy2(bands_xml_src, bands_xml_dst)
                        logger.debug("Copied bands XML to %s", bands_xml_dst)
                    except OSError as copy_exc:
                        logger.warning(
                            "Failed to preserve bands XML (%s); "
                            "band plot may show NSCF data.", copy_exc
                        )

                elapsed = time.monotonic() - t0
                step_results.append(StepStatus("bands", 3, True,
                                               elapsed_seconds=elapsed))
                self._log_step_end(3, 5, "bands", t0)
                bands_ok = True
                self._save_checkpoint(
                    [s.name for s in step_results if s.success])
            except RuntimeError as exc:
                elapsed = time.monotonic() - t0
                step_results.append(StepStatus("bands", 3, False, str(exc), elapsed))
                logger.error("[3/5] bands FAILED — %s", exc)
                if not failed_step:
                    failed_step = "bands"

        # ------------------------------------------------------------------
        # Step 4: nscf (depends on SCF, independent of bands)
        # ------------------------------------------------------------------
        if "nscf" in done:
            step_results.append(StepStatus("nscf", 4, True, summary="resumed"))
            logger.info("[4/5] nscf — already complete (resumed)")
            nscf_ok = True
        else:
            t0 = self._log_step_start(4, 5, "nscf")
            try:
                self._run_nscf(nscf_dir, calc_atoms, scf_tmp_dir)
                elapsed = time.monotonic() - t0
                step_results.append(StepStatus("nscf", 4, True,
                                               elapsed_seconds=elapsed))
                self._log_step_end(4, 5, "nscf", t0)
                nscf_ok = True
                self._save_checkpoint(
                    [s.name for s in step_results if s.success])
            except RuntimeError as exc:
                elapsed = time.monotonic() - t0
                step_results.append(StepStatus("nscf", 4, False, str(exc), elapsed))
                logger.error("[4/5] nscf FAILED — %s", exc)
                if not failed_step:
                    failed_step = "nscf"

        # ------------------------------------------------------------------
        # Step 5: dos.x (depends on NSCF)
        # ------------------------------------------------------------------
        if "dos" in done:
            step_results.append(StepStatus("dos", 5, True, summary="resumed"))
            logger.info("[5/5] dos.x — already complete (resumed)")
        elif nscf_ok:
            t0 = self._log_step_start(5, 5, "dos.x")
            try:
                self._run_dos(nscf_dir, scf_tmp_dir)
                elapsed = time.monotonic() - t0
                step_results.append(StepStatus("dos", 5, True,
                                               elapsed_seconds=elapsed))
                self._log_step_end(5, 5, "dos.x", t0)
                self._save_checkpoint(
                    [s.name for s in step_results if s.success])
            except Exception as exc:
                elapsed = time.monotonic() - t0
                step_results.append(StepStatus("dos", 5, False, str(exc), elapsed))
                logger.error("[5/5] dos.x FAILED — %s", exc)
        else:
            step_results.append(StepStatus("dos", 5, False,
                                           "skipped: NSCF failed"))
            logger.info("[5/5] dos.x SKIPPED (NSCF failed)")

        # ------------------------------------------------------------------
        # Fermi energy (NSCF preferred)
        # ------------------------------------------------------------------
        fermi = self._get_best_fermi_energy(scf_dir, nscf_dir) if scf_ok else None
        if scf_ok and fermi is None:
            logger.warning(
                "Fermi energy not found in NSCF or SCF output; "
                "band/DOS plots will use 0 eV."
            )

        # ------------------------------------------------------------------
        # Plotting (best-effort)
        # ------------------------------------------------------------------
        bands_png = self._plot_bands(bands_dir, fermi, scf_tmp_dir) if bands_ok else None
        dos_png   = self._plot_dos(nscf_dir, fermi) if nscf_ok else None

        return self._build_result(
            calc_atoms, fermi, bands_png, dos_png,
            relax_dir, scf_dir, bands_dir, nscf_dir,
            step_results, failed_step,
        )

    # ------------------------------------------------------------------
    # Result builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_result(
        calc_atoms: Any,
        fermi: Optional[float],
        bands_png: Optional[str],
        dos_png: Optional[str],
        relax_dir: str,
        scf_dir: str,
        bands_dir: str,
        nscf_dir: str,
        step_results: List["StepStatus"],
        failed_step: Optional[str],
    ) -> Dict[str, Any]:
        """Build the result dict (backward-compatible + new keys)."""
        completed = [s.name for s in step_results if s.success]
        return {
            # Existing keys (unchanged API)
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
            # New keys
            "step_results": step_results,
            "completed_steps": completed,
            "failed_step": failed_step,
        }

    # ------------------------------------------------------------------
    # Checkpoint (resume support)
    # ------------------------------------------------------------------

    _CHECKPOINT_FILE = "workflow_state.json"

    def _save_checkpoint(self, completed_steps: List[str]) -> None:
        """Write workflow state to output_dir/workflow_state.json."""
        import json
        from datetime import datetime
        state = {
            "version": 1,
            "completed_steps": completed_steps,
            "timestamp": datetime.now().isoformat(),
        }
        path = os.path.join(self.output_dir, self._CHECKPOINT_FILE)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(state, fh, indent=2)
            logger.debug("Checkpoint saved: %s (steps: %s)", path, completed_steps)
        except OSError as exc:
            logger.warning("Could not save checkpoint: %s", exc)

    def _load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint from output_dir.  Returns ``None`` if not found."""
        import json
        path = os.path.join(self.output_dir, self._CHECKPOINT_FILE)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load checkpoint %s: %s", path, exc)
            return None

    # ------------------------------------------------------------------
    # Progress helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_step_start(step_num: int, total: int, name: str) -> float:
        """Log step start and return monotonic timestamp."""
        logger.info("[%d/%d] %s — starting", step_num, total, name)
        return time.monotonic()

    @staticmethod
    def _log_step_end(
        step_num: int, total: int, name: str,
        t0: float, summary: str = "",
    ) -> None:
        """Log step completion with elapsed time and optional summary."""
        elapsed = time.monotonic() - t0
        minutes, seconds = divmod(elapsed, 60)
        if minutes >= 1:
            time_str = f"{int(minutes)}m{seconds:.1f}s"
        else:
            time_str = f"{elapsed:.1f}s"
        msg = f"[{step_num}/{total}] {name} — done ({time_str})"
        if summary:
            msg += f" — {summary}"
        logger.info(msg)

    @staticmethod
    def _extract_pw_summary(pw_out_path: str) -> str:
        """Extract one-line summary from pw.out (best-effort, never raises)."""
        try:
            if not os.path.isfile(pw_out_path):
                return ""
            with open(pw_out_path, encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()[-200:]
            text = "".join(lines)
            parts: List[str] = []
            m = re.search(r"!\s+total energy\s+=\s+([-\d.]+)\s+Ry", text)
            if m:
                parts.append(f"E={float(m.group(1)) * RY_TO_EV:.4f} eV")
            m = re.search(r"the Fermi energy is\s+([-\d.]+)\s+ev", text, re.IGNORECASE)
            if m:
                parts.append(f"Ef={m.group(1)} eV")
            iterations = re.findall(r"iteration #\s*(\d+)", text)
            if iterations:
                parts.append(f"{iterations[-1]} SCF iter")
            if "convergence has been achieved" in text.lower():
                parts.append("converged")
            return ", ".join(parts)
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Pre-flight validation
    # ------------------------------------------------------------------

    def _validate_environment(self) -> None:
        """Check pw.x, dos.x, and pseudo_dir before starting the pipeline."""
        import shutil as _shutil

        warnings = []
        if self.wsl:
            # WSL mode: check inside WSL, not the Windows PATH
            from shalom.backends.runner import detect_wsl_executable
            if not detect_wsl_executable(self.pw_executable):
                warnings.append(f"'{self.pw_executable}' not found in WSL.")
            if not detect_wsl_executable(self.dos_executable):
                warnings.append(f"'{self.dos_executable}' not found in WSL.")
            # Skip pseudo_dir check — Windows os.path.isdir() cannot
            # resolve WSL-converted paths (e.g. /mnt/c/...).
        else:
            if _shutil.which(self.pw_executable) is None:
                warnings.append(f"'{self.pw_executable}' not found in PATH.")
            if _shutil.which(self.dos_executable) is None:
                warnings.append(f"'{self.dos_executable}' not found in PATH.")
            if self.pseudo_dir and not os.path.isdir(self.pseudo_dir):
                warnings.append(f"pseudo_dir does not exist: {self.pseudo_dir}")
        if warnings:
            for w in warnings:
                logger.warning("Pre-flight: %s", w)

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    def _run_vc_relax(self, calc_dir: str, atoms: "Atoms") -> "Atoms":
        """Run vc-relax and return the relaxed structure."""
        os.makedirs(calc_dir, exist_ok=True)
        acc = self._accuracy_level
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
            logger.warning(
                "vc-relax: could not parse relaxed structure (%s); "
                "using UNRELAXED input. Check %s for details.", exc, pw_out
            )
            return atoms

    def _run_scf(self, calc_dir: str, atoms: "Atoms") -> None:
        """Run scf calculation."""
        os.makedirs(calc_dir, exist_ok=True)
        acc = self._accuracy_level
        config = get_qe_preset(QECalculationType.SCF, accuracy=acc, atoms=atoms)
        if self.pseudo_dir:
            config.pseudo_dir = self.pseudo_dir

        backend = QEBackend()
        backend.write_input(atoms, calc_dir, config=config)
        self._pw_run(calc_dir)

    def _run_bands(self, calc_dir: str, atoms: "Atoms", scf_tmp_dir: str) -> None:
        """Run bands calculation using the scf charge density."""
        os.makedirs(calc_dir, exist_ok=True)
        acc = self._accuracy_level
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
        acc = self._accuracy_level
        config = get_qe_preset(QECalculationType.NSCF, accuracy=acc, atoms=atoms)
        if self.pseudo_dir:
            config.pseudo_dir = self.pseudo_dir

        # Share scf charge density (absolute path required!)
        config.control["outdir"] = scf_tmp_dir

        # Override k-mesh: explicit user value or reduced default for NSCF
        if self.nscf_kmesh is not None:
            config.kpoints.grid = list(self.nscf_kmesh)
        else:
            from shalom.backends._physics import DEFAULT_NSCF_KPR, compute_kpoints_grid
            config.kpoints.grid = compute_kpoints_grid(
                atoms, kpr=DEFAULT_NSCF_KPR, is_2d=config.is_2d,
            )

        backend = QEBackend()
        backend.write_input(atoms, calc_dir, config=config)
        self._pw_run(calc_dir)

    def _run_dos(self, calc_dir: str, scf_tmp_dir: str) -> None:
        """Write dos.in and run dos.x (all energies in eV)."""
        os.makedirs(calc_dir, exist_ok=True)
        dos_in_path = os.path.join(calc_dir, "dos.in")
        dos_out_path = os.path.join(calc_dir, "dos.out")

        dos_in_content = (
            "&DOS\n"
            f"  outdir = '{scf_tmp_dir}'\n"
            f"  prefix = 'shalom'\n"
            f"  fildos = 'pwscf.dos'\n"
            f"  Emin = {self.dos_emin:.6f}\n"
            f"  Emax = {self.dos_emax:.6f}\n"
            f"  DeltaE = {self.dos_deltaE:.8f}\n"
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
        try:
            plotter = BandStructurePlotter(bs)
            plotter.plot(output_path=output_path)
        except Exception as exc:
            logger.warning("Band plot failed: %s", exc)
            return None
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
        try:
            plotter = DOSPlotter(dos_data)
            plotter.plot(output_path=output_path)
        except Exception as exc:
            logger.warning("DOS plot failed: %s", exc)
            return None
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
            wsl=self.wsl,
        )
        runner = create_runner(exec_config, self.slurm_config)
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
            wsl=self.wsl,
        )
        runner = create_runner(exec_config, self.slurm_config)
        result = runner.run(calc_dir)
        if not result.success:
            logger.warning(
                "dos.x failed in %s: %s", calc_dir, result.error_message or "unknown error"
            )
