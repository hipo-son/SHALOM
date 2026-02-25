"""Slurm HPC job submission and monitoring.

Provides :class:`SlurmConfig` and :class:`SlurmRunner` for submitting DFT
jobs via ``sbatch`` and monitoring them via ``squeue``/``sacct``.  Implements
the same ``run(directory) → ExecutionResult`` interface as
:class:`ExecutionRunner` so callers (``execute_with_recovery``, workflows,
etc.) can use either runner transparently.

Usage::

    from shalom.backends.slurm import SlurmConfig, SlurmRunner
    from shalom.backends.runner import ExecutionConfig

    slurm_cfg = SlurmConfig(
        partition="compute",
        account="mat_sci",
        nodes=2,
        ntasks_per_node=32,
        walltime="24:00:00",
    )
    runner = SlurmRunner(
        exec_config=ExecutionConfig(command="pw.x", nprocs=64),
        slurm_config=slurm_cfg,
    )
    result = runner.run("/scratch/user/Si_scf")
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from shalom.backends.runner import ExecutionConfig, ExecutionResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SlurmConfig:
    """Slurm scheduler configuration.

    All fields map directly to ``#SBATCH`` directives in the generated job
    script.  Polling parameters control how :class:`SlurmRunner` monitors job
    status after submission.
    """

    partition: str = "compute"
    account: Optional[str] = None
    nodes: int = 1
    ntasks_per_node: int = 1
    walltime: str = "24:00:00"
    qos: Optional[str] = None
    mem: Optional[str] = None
    job_name: str = "shalom-dft"
    output_pattern: str = "slurm-%j.out"
    error_pattern: str = "slurm-%j.err"
    extra_directives: List[str] = field(default_factory=list)
    module_loads: List[str] = field(default_factory=list)
    pre_commands: List[str] = field(default_factory=list)

    # Polling tuning
    poll_interval_initial: float = 10.0
    poll_interval_max: float = 60.0
    poll_interval_growth: float = 5.0
    client_timeout: float = 0.0  # 0 = rely on Slurm walltime only

    @property
    def total_tasks(self) -> int:
        """Total MPI tasks (nodes × ntasks_per_node)."""
        return self.nodes * self.ntasks_per_node


# ---------------------------------------------------------------------------
# Job state enum
# ---------------------------------------------------------------------------


class SlurmJobState(Enum):
    """Slurm job states relevant to monitoring."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETING = "COMPLETING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"
    NODE_FAIL = "NODE_FAIL"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    UNKNOWN = "UNKNOWN"


_TERMINAL_STATES = frozenset({
    SlurmJobState.COMPLETED,
    SlurmJobState.FAILED,
    SlurmJobState.TIMEOUT,
    SlurmJobState.CANCELLED,
    SlurmJobState.NODE_FAIL,
    SlurmJobState.OUT_OF_MEMORY,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_elapsed(elapsed_str: str) -> Optional[float]:
    """Parse sacct elapsed-time string to seconds.

    Accepted formats: ``"HH:MM:SS"``, ``"D-HH:MM:SS"``, ``"MM:SS"``.
    """
    try:
        parts = elapsed_str.split("-")
        if len(parts) == 2:
            days = int(parts[0])
            time_part = parts[1]
        else:
            days = 0
            time_part = parts[0]

        segments = time_part.split(":")
        if len(segments) == 3:
            h, m, s = int(segments[0]), int(segments[1]), float(segments[2])
        elif len(segments) == 2:
            h, m, s = 0, int(segments[0]), float(segments[1])
        else:
            return None

        return float(days * 86400 + h * 3600 + m * 60 + s)
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# SlurmRunner
# ---------------------------------------------------------------------------


class SlurmRunner:
    """Submit and monitor DFT jobs via Slurm.

    Implements the same ``run()`` / ``cancel()`` interface as
    :class:`~shalom.backends.runner.ExecutionRunner` so that
    ``execute_with_recovery()`` and all workflow code can use either runner
    interchangeably.
    """

    def __init__(
        self,
        exec_config: Optional[ExecutionConfig] = None,
        slurm_config: Optional[SlurmConfig] = None,
    ) -> None:
        self.config = exec_config or ExecutionConfig()
        self.slurm = slurm_config or SlurmConfig()
        self._job_id: Optional[str] = None

    # -- Job script generation -----------------------------------------------

    def generate_job_script(self, directory: str) -> str:
        """Generate a complete ``sbatch`` job script.

        Args:
            directory: Calculation directory (absolute path).

        Returns:
            Complete bash script content as a string.
        """
        lines: List[str] = ["#!/bin/bash"]

        # SBATCH directives
        lines.append(f"#SBATCH --job-name={self.slurm.job_name}")
        lines.append(f"#SBATCH --partition={self.slurm.partition}")
        if self.slurm.account:
            lines.append(f"#SBATCH --account={self.slurm.account}")
        lines.append(f"#SBATCH --nodes={self.slurm.nodes}")
        lines.append(
            f"#SBATCH --ntasks-per-node={self.slurm.ntasks_per_node}"
        )
        lines.append(f"#SBATCH --time={self.slurm.walltime}")
        if self.slurm.qos:
            lines.append(f"#SBATCH --qos={self.slurm.qos}")
        if self.slurm.mem:
            lines.append(f"#SBATCH --mem={self.slurm.mem}")
        lines.append(f"#SBATCH --output={self.slurm.output_pattern}")
        lines.append(f"#SBATCH --error={self.slurm.error_pattern}")
        lines.append(f"#SBATCH --chdir={directory}")

        for extra in self.slurm.extra_directives:
            lines.append(f"#SBATCH {extra}")

        lines.append("")

        # Module loads
        for mod in self.slurm.module_loads:
            lines.append(f"module load {mod}")

        # Pre-commands (e.g. export OMP_NUM_THREADS=1)
        for cmd in self.slurm.pre_commands:
            lines.append(cmd)

        if self.slurm.module_loads or self.slurm.pre_commands:
            lines.append("")

        # Environment overrides from ExecutionConfig
        for key, val in self.config.env_overrides.items():
            lines.append(f"export {key}={val}")

        if self.config.env_overrides:
            lines.append("")

        # DFT command
        nprocs = self.slurm.total_tasks
        dft_cmd = (
            f"srun -n {nprocs} {self.config.command}"
            f" < {self.config.input_file}"
            f" > {self.config.output_file}"
        )
        lines.append(dft_cmd)
        lines.append("")

        return "\n".join(lines)

    # -- Submission ----------------------------------------------------------

    def _submit(self, directory: str) -> str:
        """Submit job via ``sbatch`` and return the Slurm job ID.

        Raises:
            RuntimeError: If ``sbatch`` fails or returns unexpected output.
        """
        script_content = self.generate_job_script(directory)
        script_path = os.path.join(directory, "job.sh")

        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)

        result = subprocess.run(
            ["sbatch", script_path],
            capture_output=True,
            text=True,
            cwd=directory,
            timeout=30,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"sbatch failed (rc={result.returncode}): "
                f"{result.stderr.strip()}"
            )

        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if not match:
            raise RuntimeError(
                f"Could not parse job ID from sbatch output: "
                f"{result.stdout.strip()}"
            )

        job_id = match.group(1)
        logger.info("Submitted Slurm job %s in %s", job_id, directory)
        return job_id

    # -- Polling -------------------------------------------------------------

    def _query_squeue(self, job_id: str) -> Optional[SlurmJobState]:
        """Query job state via ``squeue``.

        Returns ``None`` if the job is no longer in the queue.
        """
        try:
            result = subprocess.run(
                ["squeue", "--job", job_id, "--noheader", "--format=%T"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            state_str = result.stdout.strip()
            if not state_str:
                return None
            try:
                return SlurmJobState(state_str)
            except ValueError:
                logger.debug(
                    "Unknown squeue state '%s' for job %s",
                    state_str, job_id,
                )
                return SlurmJobState.UNKNOWN
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning("squeue query failed: %s", e)
            return SlurmJobState.UNKNOWN

    def _query_sacct(
        self, job_id: str,
    ) -> Tuple[SlurmJobState, Optional[float]]:
        """Query completed job via ``sacct``.

        Returns:
            Tuple of (final state, elapsed seconds or None).
        """
        try:
            result = subprocess.run(
                [
                    "sacct", "--job", job_id,
                    "--format=JobID,State,Elapsed",
                    "--noheader", "--parsable2",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            for line in result.stdout.strip().split("\n"):
                parts = line.split("|")
                if len(parts) >= 3 and parts[0] == job_id:
                    state_str = parts[1].split()[0]
                    elapsed_sec = _parse_elapsed(parts[2])
                    try:
                        state = SlurmJobState(state_str)
                    except ValueError:
                        state = SlurmJobState.UNKNOWN
                    return state, elapsed_sec

            return SlurmJobState.UNKNOWN, None
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning("sacct query failed: %s", e)
            return SlurmJobState.UNKNOWN, None

    def _poll_until_done(
        self, job_id: str,
    ) -> Tuple[SlurmJobState, float]:
        """Poll job until a terminal state is reached.

        Returns:
            Tuple of (final state, wall-clock time in seconds).
        """
        interval = self.slurm.poll_interval_initial
        start_time = time.monotonic()
        unknown_streak = 0
        _MAX_UNKNOWN_STREAK = 30  # ~30 min at max interval

        while True:
            elapsed = time.monotonic() - start_time

            # Client-side timeout
            if self.slurm.client_timeout > 0 and elapsed > self.slurm.client_timeout:
                logger.warning(
                    "Client-side timeout (%.0fs) for job %s",
                    self.slurm.client_timeout, job_id,
                )
                self._scancel(job_id)
                return SlurmJobState.TIMEOUT, elapsed

            # Check squeue
            state = self._query_squeue(job_id)

            if state is None:
                # Job left the queue — check sacct for final state
                final_state, sacct_elapsed = self._query_sacct(job_id)
                wall_time = (
                    sacct_elapsed
                    if sacct_elapsed is not None
                    else time.monotonic() - start_time
                )
                return final_state, wall_time

            if state in _TERMINAL_STATES:
                wall_time = time.monotonic() - start_time
                return state, wall_time

            # Guard against squeue returning UNKNOWN indefinitely
            if state == SlurmJobState.UNKNOWN:
                unknown_streak += 1
                if unknown_streak >= _MAX_UNKNOWN_STREAK:
                    logger.error(
                        "squeue returned UNKNOWN %d times for job %s; "
                        "falling back to sacct",
                        unknown_streak, job_id,
                    )
                    final_state, sacct_elapsed = self._query_sacct(job_id)
                    wall_time = (
                        sacct_elapsed
                        if sacct_elapsed is not None
                        else elapsed
                    )
                    return final_state, wall_time
            else:
                unknown_streak = 0

            # Still active — wait then re-poll
            logger.debug(
                "Job %s: %s (next poll in %.0fs)",
                job_id, state.value, interval,
            )
            time.sleep(interval)
            interval = min(
                interval + self.slurm.poll_interval_growth,
                self.slurm.poll_interval_max,
            )

    # -- Main interface (matches ExecutionRunner) ----------------------------

    def run(self, directory: str) -> ExecutionResult:
        """Submit and monitor a Slurm job.

        Compatible with :meth:`ExecutionRunner.run`.
        """
        t0 = time.monotonic()

        try:
            job_id = self._submit(directory)
            self._job_id = job_id
        except (RuntimeError, subprocess.TimeoutExpired, OSError) as e:
            return ExecutionResult(
                success=False,
                wall_time_seconds=time.monotonic() - t0,
                error_message=str(e),
            )

        final_state, wall_time = self._poll_until_done(job_id)
        self._job_id = None

        output_path = os.path.join(directory, self.config.output_file)

        if final_state == SlurmJobState.COMPLETED:
            return ExecutionResult(
                success=True,
                return_code=0,
                wall_time_seconds=wall_time,
                output_file=output_path if os.path.exists(output_path) else None,
            )

        timed_out = final_state in (
            SlurmJobState.TIMEOUT,
            SlurmJobState.CANCELLED,
        )

        stderr_tail = self._read_slurm_stderr(directory, job_id)

        return ExecutionResult(
            success=False,
            return_code=1,
            wall_time_seconds=wall_time,
            timed_out=timed_out,
            output_file=output_path if os.path.exists(output_path) else None,
            error_message=(
                f"Slurm job {job_id} ended with state {final_state.value}"
            ),
            stderr_tail=stderr_tail,
        )

    def cancel(self) -> None:
        """Cancel the currently tracked Slurm job."""
        if self._job_id is not None:
            self._scancel(self._job_id)

    def validate_prerequisites(self, directory: str) -> List[str]:
        """Check Slurm-specific prerequisites.

        Returns:
            List of error messages (empty if all OK).
        """
        import shutil as _shutil

        errors: List[str] = []

        if _shutil.which("sbatch") is None:
            errors.append(
                "sbatch not found on PATH. "
                "Slurm mode requires a working Slurm installation."
            )
        if _shutil.which("squeue") is None:
            errors.append("squeue not found on PATH.")

        input_path = os.path.join(directory, self.config.input_file)
        if not os.path.exists(input_path):
            errors.append(f"Input file not found: {input_path}")

        return errors

    # -- Helpers -------------------------------------------------------------

    def _scancel(self, job_id: str) -> None:
        """Cancel a Slurm job via ``scancel``."""
        try:
            subprocess.run(
                ["scancel", job_id],
                capture_output=True,
                timeout=10,
            )
            logger.info("Cancelled Slurm job %s", job_id)
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning("scancel failed for job %s: %s", job_id, e)

    def _read_slurm_stderr(
        self, directory: str, job_id: str,
    ) -> Optional[str]:
        """Read the last 2000 chars of the Slurm stderr file."""
        err_filename = self.slurm.error_pattern.replace("%j", job_id)
        err_path = os.path.join(directory, err_filename)
        try:
            with open(err_path, encoding="utf-8", errors="replace") as f:
                content = f.read()
            return content[-2000:] if content else None
        except OSError:
            return None
