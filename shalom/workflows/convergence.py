"""Cutoff energy and k-point convergence tests for QE calculations.

Run ecutwfc first, then use the converged value to fix the k-point sweep.

Usage::

    from ase.build import bulk
    from shalom.workflows.convergence import CutoffConvergence, KpointConvergence

    si = bulk("Si", "diamond", a=5.43)

    # Step 1: ecutwfc convergence (k-grid fixed)
    conv_ecut = CutoffConvergence(
        atoms=si,
        output_dir="/tmp/si_ecut",
        values=[30, 40, 50, 60, 80],  # Ry
        kgrid=[4, 4, 4],
        nprocs=4,
    )
    result_ecut = conv_ecut.run()
    print(result_ecut.summary())
    conv_ecut.plot(result_ecut, "/tmp/si_ecut/plot.png")

    # Step 2: k-point convergence (ecutwfc fixed from step 1)
    ecutwfc = result_ecut.converged_value or 60.0
    conv_kpt = KpointConvergence(
        atoms=si,
        output_dir="/tmp/si_kpts",
        resolutions=[20, 30, 40, 50],
        ecutwfc=ecutwfc,
        nprocs=4,
    )
    result_kpt = conv_kpt.run()
    print(result_kpt.summary())
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional

from shalom.backends._physics import CONVERGENCE_THRESHOLD_PER_ATOM
from shalom.workflows.base import ConvergenceResult, ConvergenceWorkflow

if TYPE_CHECKING:
    from ase import Atoms

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ecutwfc convergence
# ---------------------------------------------------------------------------


class CutoffConvergence(ConvergenceWorkflow):
    """Sweep ecutwfc values and identify the converged cutoff energy.

    ``ecutrho`` is automatically derived from each ``ecutwfc`` value using
    the per-element SSSP ratio (or a global fallback of 8×).

    .. important::
        Run cutoff convergence **before** k-point convergence.
        Fix the k-grid during this sweep (use ``kgrid``), then pass the
        converged ``ecutwfc`` to :class:`KpointConvergence`.

    Args:
        atoms: ASE ``Atoms`` structure.
        output_dir: Root directory for sweep sub-directories.
        values: List of ``ecutwfc`` values to test (Ry).
        kgrid: Explicit ``[Nx, Ny, Nz]`` Monkhorst-Pack grid to use during
            the sweep.  If ``None``, a default grid is estimated from the
            lattice using ``compute_kpoints_grid``.
        pseudo_dir: Pseudopotential directory override.
        nprocs: MPI processes for ``pw.x``.
        timeout: Per-run timeout (seconds).
        mpi_command: MPI launcher.
        accuracy: QE preset accuracy level.
        parallel: Run all values concurrently (multiprocessing).
        threshold_per_atom: Convergence threshold in eV/atom.
    """

    _parameter_name = "ecutwfc"

    def __init__(
        self,
        atoms: "Atoms",
        output_dir: str,
        values: List[float],
        *,
        kgrid: Optional[List[int]] = None,
        pseudo_dir: Optional[str] = None,
        nprocs: int = 1,
        timeout: int = 3600,
        mpi_command: str = "mpirun",
        accuracy: str = "standard",
        parallel: bool = False,
        threshold_per_atom: float = CONVERGENCE_THRESHOLD_PER_ATOM,
        wsl: bool = False,
        slurm_config: Optional[Any] = None,
    ) -> None:
        super().__init__(
            atoms=atoms,
            output_dir=output_dir,
            pseudo_dir=pseudo_dir,
            nprocs=nprocs,
            timeout=timeout,
            mpi_command=mpi_command,
            accuracy=accuracy,
            parallel=parallel,
            threshold_per_atom=threshold_per_atom,
            wsl=wsl,
            slurm_config=slurm_config,
        )
        self._values_list = sorted(values)
        self.kgrid = kgrid

    @property
    def _values(self) -> List[float]:
        return self._values_list

    def _run_single(self, ecut: float, calc_dir: str) -> ConvergenceResult:
        """Run a single SCF calculation with ``ecutwfc=ecut``."""
        from shalom.backends.qe import QEBackend
        from shalom.backends.qe_config import (
            QECalculationType, QEKPointsConfig, get_qe_preset, compute_ecutrho,
        )
        from shalom.backends._physics import AccuracyLevel, compute_kpoints_grid
        from shalom.backends.runner import ExecutionConfig, create_runner

        try:
            acc = AccuracyLevel.PRECISE if self.accuracy == "precise" else AccuracyLevel.STANDARD
            elements = self.atoms.get_chemical_symbols()
            config = get_qe_preset(QECalculationType.SCF, accuracy=acc, atoms=self.atoms)

            # Set cutoff energies (override what the preset computed)
            config.system["ecutwfc"] = ecut
            config.system["ecutrho"] = compute_ecutrho(elements, acc)

            # Apply pseudo_dir override
            if self.pseudo_dir:
                config.pseudo_dir = self.pseudo_dir

            # Apply k-grid
            if self.kgrid:
                config.kpoints = QEKPointsConfig(mode="automatic", grid=self.kgrid)
            else:
                default_grid = compute_kpoints_grid(self.atoms)
                config.kpoints = QEKPointsConfig(mode="automatic", grid=default_grid)

            backend = QEBackend()
            backend.write_input(self.atoms, calc_dir, config=config)

            exec_config = ExecutionConfig(
                command="pw.x",
                input_file="pw.in",
                output_file="pw.out",
                nprocs=self.nprocs,
                mpi_command=self.mpi_command,
                timeout_seconds=self.timeout,
                wsl=self.wsl,
            )
            runner = create_runner(exec_config, self.slurm_config)
            exec_result = runner.run(calc_dir)

            result = backend.parse_output(calc_dir)

            return ConvergenceResult(
                parameter_value=ecut,
                energy=result.energy,
                is_converged=result.is_converged and exec_result.success,
                calc_dir=calc_dir,
            )

        except Exception as exc:  # noqa: BLE001
            logger.error("ecutwfc=%.4g failed: %s", ecut, exc)
            return ConvergenceResult(
                parameter_value=ecut,
                energy=None,
                is_converged=False,
                calc_dir=calc_dir,
                error_message=str(exc),
            )


# ---------------------------------------------------------------------------
# K-point convergence
# ---------------------------------------------------------------------------


class KpointConvergence(ConvergenceWorkflow):
    """Sweep k-point density and identify the converged k-point sampling.

    Each test value is a k-point **resolution** (reciprocal-space density in
    Bohr^-1), converted to a Monkhorst-Pack grid via ``compute_kpoints_grid``.

    .. important::
        Run :class:`CutoffConvergence` first.  Pass the converged ``ecutwfc``
        via the ``ecutwfc`` argument so the cutoff is fixed during this sweep.

    Args:
        atoms: ASE ``Atoms`` structure.
        output_dir: Root directory for sweep sub-directories.
        resolutions: List of k-point resolution values (Bohr^-1) to test.
        ecutwfc: Fixed ecutwfc (Ry) from prior cutoff convergence.  If ``None``,
            uses the value from the QE preset (may be inaccurate).
        pseudo_dir: Pseudopotential directory override.
        nprocs: MPI processes for ``pw.x``.
        timeout: Per-run timeout (seconds).
        mpi_command: MPI launcher.
        accuracy: QE preset accuracy level.
        parallel: Run all values concurrently.
        threshold_per_atom: Convergence threshold in eV/atom.
    """

    _parameter_name = "kpoint_resolution"

    def __init__(
        self,
        atoms: "Atoms",
        output_dir: str,
        resolutions: List[float],
        *,
        ecutwfc: Optional[float] = None,
        pseudo_dir: Optional[str] = None,
        nprocs: int = 1,
        timeout: int = 3600,
        mpi_command: str = "mpirun",
        accuracy: str = "standard",
        parallel: bool = False,
        threshold_per_atom: float = CONVERGENCE_THRESHOLD_PER_ATOM,
        wsl: bool = False,
        slurm_config: Optional[Any] = None,
    ) -> None:
        super().__init__(
            atoms=atoms,
            output_dir=output_dir,
            pseudo_dir=pseudo_dir,
            nprocs=nprocs,
            timeout=timeout,
            mpi_command=mpi_command,
            accuracy=accuracy,
            parallel=parallel,
            threshold_per_atom=threshold_per_atom,
            wsl=wsl,
            slurm_config=slurm_config,
        )
        self._values_list = sorted(resolutions)
        self.ecutwfc = ecutwfc

    @property
    def _values(self) -> List[float]:
        return self._values_list

    def _run_single(self, resolution: float, calc_dir: str) -> ConvergenceResult:
        """Run a single SCF calculation at the given k-point resolution."""
        from shalom.backends.qe import QEBackend
        from shalom.backends.qe_config import (
            QECalculationType, QEKPointsConfig, get_qe_preset, compute_ecutrho,
        )
        from shalom.backends._physics import AccuracyLevel, compute_kpoints_grid
        from shalom.backends.runner import ExecutionConfig, create_runner

        try:
            acc = AccuracyLevel.PRECISE if self.accuracy == "precise" else AccuracyLevel.STANDARD
            elements = self.atoms.get_chemical_symbols()
            config = get_qe_preset(QECalculationType.SCF, accuracy=acc, atoms=self.atoms)

            # Fix cutoff energy from prior convergence
            if self.ecutwfc is not None:
                config.system["ecutwfc"] = self.ecutwfc
                config.system["ecutrho"] = compute_ecutrho(elements, acc)

            # Apply pseudo_dir override
            if self.pseudo_dir:
                config.pseudo_dir = self.pseudo_dir

            # Convert resolution to grid
            grid = compute_kpoints_grid(self.atoms, kpr=resolution)
            config.kpoints = QEKPointsConfig(mode="automatic", grid=grid)

            backend = QEBackend()
            backend.write_input(self.atoms, calc_dir, config=config)

            exec_config = ExecutionConfig(
                command="pw.x",
                input_file="pw.in",
                output_file="pw.out",
                nprocs=self.nprocs,
                mpi_command=self.mpi_command,
                timeout_seconds=self.timeout,
                wsl=self.wsl,
            )
            runner = create_runner(exec_config, self.slurm_config)
            exec_result = runner.run(calc_dir)

            result = backend.parse_output(calc_dir)

            return ConvergenceResult(
                parameter_value=resolution,
                energy=result.energy,
                is_converged=result.is_converged and exec_result.success,
                calc_dir=calc_dir,
            )

        except Exception as exc:  # noqa: BLE001
            logger.error("kpoint_resolution=%.4g failed: %s", resolution, exc)
            return ConvergenceResult(
                parameter_value=resolution,
                energy=None,
                is_converged=False,
                calc_dir=calc_dir,
                error_message=str(exc),
            )
