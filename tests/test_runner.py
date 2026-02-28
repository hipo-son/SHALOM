"""Tests for shalom.backends.runner — DFT execution and error recovery loop."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from shalom.backends.base import DFTResult
from shalom.backends.qe_error_recovery import QEErrorRecoveryEngine
from shalom.backends.runner import (
    ExecutionConfig,
    ExecutionResult,
    ExecutionRunner,
    _patch_input_paths_for_wsl,
    _windows_to_wsl_path,
    execute_with_recovery,
)


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def default_config():
    return ExecutionConfig()


@pytest.fixture
def parallel_config():
    return ExecutionConfig(nprocs=4, mpi_command="mpirun")


@pytest.fixture
def srun_config():
    return ExecutionConfig(nprocs=8, mpi_command="srun")


@pytest.fixture
def tmp_calc_dir(tmp_path):
    """Create a temporary calculation directory with pw.in."""
    pw_in = tmp_path / "pw.in"
    pw_in.write_text(
        "&CONTROL\n  pseudo_dir = './pseudo'\n/\n"
        "&SYSTEM\n  ecutwfc = 60\n/\n"
        "&ELECTRONS\n/\n"
        "ATOMIC_SPECIES\n  Si 28.086 Si.pbe-n-rrkjus_psl.1.0.0.UPF\n"
    )
    pseudo_dir = tmp_path / "pseudo"
    pseudo_dir.mkdir()
    (pseudo_dir / "Si.pbe-n-rrkjus_psl.1.0.0.UPF").write_text("pseudo")
    return str(tmp_path)


# =========================================================================
# TestBuildCommand
# =========================================================================

class TestBuildCommand:
    """Test command construction for serial and parallel execution."""

    def test_serial_command(self, default_config):
        cmd = ExecutionRunner.build_command(default_config)
        assert cmd == ["pw.x"]

    def test_parallel_command(self, parallel_config):
        cmd = ExecutionRunner.build_command(parallel_config)
        assert cmd == ["mpirun", "-np", "4", "pw.x"]

    def test_single_proc_is_serial(self):
        config = ExecutionConfig(nprocs=1)
        cmd = ExecutionRunner.build_command(config)
        assert cmd == ["pw.x"]

    def test_custom_mpi_command(self, srun_config):
        cmd = ExecutionRunner.build_command(srun_config)
        assert cmd == ["srun", "-np", "8", "pw.x"]

    def test_custom_executable(self):
        config = ExecutionConfig(command="/opt/qe/bin/pw.x", nprocs=2)
        cmd = ExecutionRunner.build_command(config)
        assert cmd == ["mpirun", "-np", "2", "/opt/qe/bin/pw.x"]

    def test_root_adds_allow_run_as_root(self):
        """Open MPI ≥5 root execution → --allow-run-as-root flag added."""
        import shalom.backends.runner as runner_mod

        config = ExecutionConfig(nprocs=4, mpi_command="mpirun")
        with patch.object(runner_mod.os, "getuid", create=True, return_value=0):
            cmd = ExecutionRunner.build_command(config)
        assert "--allow-run-as-root" in cmd
        assert cmd == ["mpirun", "--allow-run-as-root", "-np", "4", "pw.x"]

    def test_non_root_no_allow_flag(self):
        """Non-root user → no --allow-run-as-root flag."""
        import shalom.backends.runner as runner_mod

        config = ExecutionConfig(nprocs=4, mpi_command="mpirun")
        with patch.object(runner_mod.os, "getuid", create=True, return_value=1000):
            cmd = ExecutionRunner.build_command(config)
        assert "--allow-run-as-root" not in cmd

    def test_no_getuid_skips_root_check(self):
        """Windows (no getuid) → no crash, no root flag."""
        config = ExecutionConfig(nprocs=4, mpi_command="mpirun")
        # On Windows, os.getuid doesn't exist; the code uses hasattr check
        # Just verify that calling build_command doesn't crash
        cmd = ExecutionRunner.build_command(config)
        assert "mpirun" in cmd
        assert "-np" in cmd


# =========================================================================
# TestPrerequisiteValidation
# =========================================================================

class TestPrerequisiteValidation:
    """Test prerequisite validation before execution."""

    def test_valid_prerequisites(self, tmp_calc_dir):
        runner = ExecutionRunner()
        with patch("shutil.which", return_value="/usr/bin/pw.x"):
            errors = runner.validate_prerequisites(tmp_calc_dir)
        assert errors == []

    def test_missing_executable(self, tmp_calc_dir):
        runner = ExecutionRunner()
        with patch("shutil.which", return_value=None):
            errors = runner.validate_prerequisites(tmp_calc_dir)
        assert any("pw.x" in e and "not found" in e for e in errors)

    def test_missing_input_file(self, tmp_path):
        runner = ExecutionRunner()
        with patch("shutil.which", return_value="/usr/bin/pw.x"):
            errors = runner.validate_prerequisites(str(tmp_path))
        assert any("Input file not found" in e for e in errors)

    def test_missing_mpi_for_parallel(self, tmp_calc_dir):
        runner = ExecutionRunner(ExecutionConfig(nprocs=4))
        def which_side_effect(name):
            if name == "pw.x":
                return "/usr/bin/pw.x"
            return None  # mpirun not found
        with patch("shutil.which", side_effect=which_side_effect):
            errors = runner.validate_prerequisites(tmp_calc_dir)
        assert any("mpirun" in e and "not found" in e for e in errors)

    def test_serial_no_mpi_check(self, tmp_calc_dir):
        runner = ExecutionRunner(ExecutionConfig(nprocs=1))
        with patch("shutil.which", return_value="/usr/bin/pw.x"):
            errors = runner.validate_prerequisites(tmp_calc_dir)
        # No MPI error for serial
        assert not any("mpirun" in e for e in errors)

    def test_missing_pseudo_dir(self, tmp_path):
        pw_in = tmp_path / "pw.in"
        pw_in.write_text("pseudo_dir = '/nonexistent/pseudo'\n")
        runner = ExecutionRunner()
        with patch("shutil.which", return_value="/usr/bin/pw.x"):
            errors = runner.validate_prerequisites(str(tmp_path))
        assert any("pseudo_dir not found" in e for e in errors)


# =========================================================================
# TestExecutionRunner
# =========================================================================

class TestExecutionRunner:
    """Test the ExecutionRunner.run() method with mocked subprocess."""

    @patch("subprocess.Popen")
    def test_successful_execution(self, mock_popen, tmp_calc_dir):
        proc = MagicMock()
        proc.communicate.return_value = (None, b"")
        proc.returncode = 0
        mock_popen.return_value = proc

        runner = ExecutionRunner()
        result = runner.run(tmp_calc_dir)

        assert result.success is True
        assert result.return_code == 0
        assert result.timed_out is False
        assert result.wall_time_seconds >= 0
        assert result.error_message is None

    @patch("subprocess.Popen")
    def test_failed_execution_nonzero_exit(self, mock_popen, tmp_calc_dir):
        proc = MagicMock()
        proc.communicate.return_value = (None, b"CRASH in pw.x")
        proc.returncode = 1
        mock_popen.return_value = proc

        runner = ExecutionRunner()
        result = runner.run(tmp_calc_dir)

        assert result.success is False
        assert result.return_code == 1
        assert "exited with code 1" in result.error_message
        assert result.stderr_tail is not None
        assert "CRASH" in result.stderr_tail

    @patch("subprocess.Popen")
    def test_timeout_execution(self, mock_popen, tmp_calc_dir):
        proc = MagicMock()
        proc.communicate.return_value = (None, b"")
        proc.returncode = -9  # killed by timer
        mock_popen.return_value = proc

        runner = ExecutionRunner(ExecutionConfig(timeout_seconds=1))
        result = runner.run(tmp_calc_dir)

        assert result.timed_out is True
        assert result.success is False
        assert "timed out" in result.error_message.lower()

    @patch("subprocess.Popen")
    def test_stderr_capture(self, mock_popen, tmp_calc_dir):
        stderr_msg = b"Warning: some MPI message\nAnother warning"
        proc = MagicMock()
        proc.communicate.return_value = (None, stderr_msg)
        proc.returncode = 0
        mock_popen.return_value = proc

        runner = ExecutionRunner()
        result = runner.run(tmp_calc_dir)

        assert result.success is True
        assert result.stderr_tail is not None
        assert "MPI message" in result.stderr_tail

    @patch("subprocess.Popen")
    def test_cancel_running_process(self, mock_popen, tmp_calc_dir):
        proc = MagicMock()
        proc.communicate.return_value = (None, b"")
        proc.returncode = 0
        mock_popen.return_value = proc

        runner = ExecutionRunner()
        # Simulate setting _process during run
        runner._process = proc
        runner.cancel()

        proc.kill.assert_called_once()
        proc.wait.assert_called_once()

    @patch("subprocess.Popen")
    def test_keyboard_interrupt_cleanup(self, mock_popen, tmp_calc_dir):
        proc = MagicMock()
        proc.communicate.side_effect = KeyboardInterrupt()
        proc.returncode = None
        mock_popen.return_value = proc

        runner = ExecutionRunner()
        with pytest.raises(KeyboardInterrupt):
            runner.run(tmp_calc_dir)

        proc.kill.assert_called_once()
        proc.wait.assert_called_once()

    @patch("subprocess.Popen")
    def test_command_not_found(self, mock_popen, tmp_calc_dir):
        mock_popen.side_effect = FileNotFoundError("pw.x not found")

        runner = ExecutionRunner()
        result = runner.run(tmp_calc_dir)

        assert result.success is False
        assert "not found" in result.error_message.lower()

    @patch("subprocess.Popen")
    def test_env_overrides_passed(self, mock_popen, tmp_calc_dir):
        proc = MagicMock()
        proc.communicate.return_value = (None, b"")
        proc.returncode = 0
        mock_popen.return_value = proc

        config = ExecutionConfig(env_overrides={"OMP_NUM_THREADS": "1"})
        runner = ExecutionRunner(config)
        runner.run(tmp_calc_dir)

        call_kwargs = mock_popen.call_args[1]
        assert call_kwargs["env"]["OMP_NUM_THREADS"] == "1"

    @patch("subprocess.Popen")
    def test_empty_stderr(self, mock_popen, tmp_calc_dir):
        proc = MagicMock()
        proc.communicate.return_value = (None, b"")
        proc.returncode = 0
        mock_popen.return_value = proc

        runner = ExecutionRunner()
        result = runner.run(tmp_calc_dir)

        assert result.stderr_tail is None

    @patch("subprocess.Popen")
    def test_output_file_path(self, mock_popen, tmp_calc_dir):
        proc = MagicMock()
        proc.communicate.return_value = (None, b"")
        proc.returncode = 0
        mock_popen.return_value = proc

        runner = ExecutionRunner()
        result = runner.run(tmp_calc_dir)

        expected = os.path.join(tmp_calc_dir, "pw.out")
        assert result.output_file == expected


# =========================================================================
# TestExecutionConfig
# =========================================================================

class TestExecutionConfig:
    """Test ExecutionConfig defaults and customization."""

    def test_defaults(self):
        config = ExecutionConfig()
        assert config.command == "pw.x"
        assert config.input_file == "pw.in"
        assert config.output_file == "pw.out"
        assert config.nprocs == 1
        assert config.mpi_command == "mpirun"
        assert config.timeout_seconds == 86400
        assert config.env_overrides == {}

    def test_custom_values(self):
        config = ExecutionConfig(
            command="/opt/qe/bin/pw.x",
            nprocs=16,
            mpi_command="srun",
            timeout_seconds=7200,
        )
        assert config.command == "/opt/qe/bin/pw.x"
        assert config.nprocs == 16
        assert config.timeout_seconds == 7200


# =========================================================================
# TestExecutionResult
# =========================================================================

class TestExecutionResult:
    """Test ExecutionResult dataclass."""

    def test_success_result(self):
        result = ExecutionResult(success=True, return_code=0, wall_time_seconds=120.5)
        assert result.success is True
        assert result.return_code == 0
        assert result.timed_out is False

    def test_failure_result(self):
        result = ExecutionResult(
            success=False, return_code=1,
            error_message="Segfault",
            stderr_tail="Signal 11",
        )
        assert result.success is False
        assert result.error_message == "Segfault"
        assert result.stderr_tail == "Signal 11"


# =========================================================================
# Helpers for execute_with_recovery tests
# =========================================================================

@dataclass
class _MockQEConfig:
    """Minimal mock of QEInputConfig for testing."""
    electrons: Dict[str, Any] = field(default_factory=dict)
    control: Dict[str, Any] = field(default_factory=dict)
    ions: Dict[str, Any] = field(default_factory=dict)
    cell: Dict[str, Any] = field(default_factory=dict)


def _make_backend(parse_results: list) -> MagicMock:
    """Create a mock backend with sequential parse_output results."""
    backend = MagicMock()
    backend.name = "qe"
    backend.parse_output.side_effect = parse_results
    backend.write_input.return_value = "/calc"
    return backend


def _make_runner(run_results: list) -> MagicMock:
    """Create a mock runner with sequential run() results."""
    runner = MagicMock(spec=ExecutionRunner)
    runner.run.side_effect = run_results
    runner.config = ExecutionConfig()
    return runner


def _converged_result(**kwargs) -> DFTResult:
    return DFTResult(energy=-100.0, is_converged=True, **kwargs)


def _unconverged_result(**kwargs) -> DFTResult:
    return DFTResult(energy=-50.0, is_converged=False, **kwargs)


def _success_exec() -> ExecutionResult:
    return ExecutionResult(success=True, return_code=0, wall_time_seconds=60.0)


def _failed_exec(code=1) -> ExecutionResult:
    return ExecutionResult(
        success=False, return_code=code,
        error_message=f"Process exited with code {code}",
    )


def _timeout_exec() -> ExecutionResult:
    return ExecutionResult(
        success=False, timed_out=True, return_code=-9,
        error_message="Timed out",
    )


# =========================================================================
# TestExecuteWithRecovery
# =========================================================================

class TestExecuteWithRecovery:
    """Test the execute_with_recovery loop."""

    def test_immediate_success_no_recovery(self, tmp_path):
        """First run converges — no error recovery needed."""
        backend = _make_backend([_converged_result()])
        runner = _make_runner([_success_exec()])
        engine = QEErrorRecoveryEngine()

        exec_r, dft_r, history = execute_with_recovery(
            backend, runner, engine, str(tmp_path), _MockQEConfig(),
        )

        assert dft_r.is_converged is True
        assert history == []
        runner.run.assert_called_once()

    def test_timeout_returns_immediately(self, tmp_path):
        """Timeout should return immediately without parsing."""
        backend = _make_backend([])  # Should NOT be called
        runner = _make_runner([_timeout_exec()])
        engine = QEErrorRecoveryEngine()

        exec_r, dft_r, history = execute_with_recovery(
            backend, runner, engine, str(tmp_path), _MockQEConfig(),
        )

        assert exec_r.timed_out is True
        assert dft_r is None
        backend.parse_output.assert_not_called()

    def test_fatal_error_stops_immediately(self, tmp_path):
        """Fatal errors should stop without attempting correction."""
        pw_out = tmp_path / "pw.out"
        pw_out.write_text("Error in routine readpp\nPseudo not found")

        backend = _make_backend([_unconverged_result()])
        runner = _make_runner([_success_exec()])
        runner.config.output_file = "pw.out"
        engine = QEErrorRecoveryEngine()

        exec_r, dft_r, history = execute_with_recovery(
            backend, runner, engine, str(tmp_path), _MockQEConfig(),
        )

        # Only 1 run attempt (fatal stops retry)
        assert runner.run.call_count == 1
        assert history == []

    def test_correctable_retry_succeeds(self, tmp_path):
        """Correctable error → fix → second run succeeds."""
        pw_out = tmp_path / "pw.out"
        pw_out.write_text("convergence NOT achieved after 100 iterations\n")

        backend = _make_backend([_unconverged_result(), _converged_result()])
        runner = _make_runner([_success_exec(), _success_exec()])
        runner.config.output_file = "pw.out"
        engine = QEErrorRecoveryEngine()
        config = _MockQEConfig()

        exec_r, dft_r, history = execute_with_recovery(
            backend, runner, engine, str(tmp_path), config,
        )

        assert dft_r.is_converged is True
        assert len(history) == 1
        assert history[0]["error_type"] == "QE_SCF_UNCONVERGED"
        assert runner.run.call_count == 2
        backend.write_input.assert_called_once()

    def test_max_retries_exhausted(self, tmp_path):
        """All retries fail — returns last result."""
        pw_out = tmp_path / "pw.out"
        pw_out.write_text("convergence NOT achieved after 100 iterations\n")

        # 4 attempts (1 initial + 3 retries), all unconverged
        backend = _make_backend([_unconverged_result()] * 4)
        runner = _make_runner([_success_exec()] * 4)
        runner.config.output_file = "pw.out"
        engine = QEErrorRecoveryEngine()

        exec_r, dft_r, history = execute_with_recovery(
            backend, runner, engine, str(tmp_path), _MockQEConfig(),
            max_retries=3,
        )

        assert dft_r is not None
        assert dft_r.is_converged is False
        # 3 corrections recorded (attempts 1-3)
        assert len(history) == 3

    def test_pw_out_backup_created(self, tmp_path):
        """pw.out should be backed up before rewrite."""
        pw_out = tmp_path / "pw.out"
        pw_out.write_text("convergence NOT achieved\n")

        backend = _make_backend([_unconverged_result(), _converged_result()])
        runner = _make_runner([_success_exec(), _success_exec()])
        runner.config.output_file = "pw.out"
        engine = QEErrorRecoveryEngine()

        execute_with_recovery(
            backend, runner, engine, str(tmp_path), _MockQEConfig(),
        )

        backup = tmp_path / "pw.out.retry_0"
        assert backup.exists()

    def test_input_rewritten_after_correction(self, tmp_path):
        """write_input called after correction applied."""
        pw_out = tmp_path / "pw.out"
        pw_out.write_text("convergence NOT achieved\n")

        backend = _make_backend([_unconverged_result(), _converged_result()])
        runner = _make_runner([_success_exec(), _success_exec()])
        runner.config.output_file = "pw.out"
        engine = QEErrorRecoveryEngine()
        atoms = Atoms("Si2", positions=[[0, 0, 0], [1.3, 1.3, 1.3]])

        execute_with_recovery(
            backend, runner, engine, str(tmp_path), _MockQEConfig(),
            atoms=atoms,
        )

        backend.write_input.assert_called_once()

    def test_quality_warnings_propagated(self, tmp_path):
        """Quality warnings from corrections propagated to DFTResult."""
        pw_out = tmp_path / "pw.out"
        # Use BFGS failed which has quality warnings at step 2
        pw_out.write_text("bfgs failed after 100 iterations\n")

        # Need to exhaust steps 0 and 1 to get to step 2 (quality_warning)
        backend = _make_backend([
            _unconverged_result(),
            _unconverged_result(),
            _unconverged_result(),
            _converged_result(),
        ])
        runner = _make_runner([_success_exec()] * 4)
        runner.config.output_file = "pw.out"
        engine = QEErrorRecoveryEngine()
        atoms = Atoms("Fe2", positions=[[0, 0, 0], [2.0, 0, 0]], cell=[4, 4, 4])

        exec_r, dft_r, history = execute_with_recovery(
            backend, runner, engine, str(tmp_path), _MockQEConfig(),
            atoms=atoms, max_retries=3,
        )

        assert dft_r.is_converged is True
        assert "loosely_relaxed" in dft_r.quality_warnings

    def test_no_errors_detected_stops(self, tmp_path):
        """Unknown failure (no patterns match) stops loop."""
        pw_out = tmp_path / "pw.out"
        pw_out.write_text("Some unknown error occurred\n")

        backend = _make_backend([_unconverged_result()])
        runner = _make_runner([_success_exec()])
        runner.config.output_file = "pw.out"
        engine = QEErrorRecoveryEngine()

        exec_r, dft_r, history = execute_with_recovery(
            backend, runner, engine, str(tmp_path), _MockQEConfig(),
        )

        assert dft_r.is_converged is False
        assert history == []
        assert runner.run.call_count == 1

    def test_parse_failure_handled(self, tmp_path):
        """parse_output raising exception doesn't crash loop."""
        pw_out = tmp_path / "pw.out"
        pw_out.write_text("convergence NOT achieved\n")

        backend = MagicMock()
        backend.parse_output.side_effect = [Exception("parse error"), _converged_result()]
        backend.write_input.return_value = str(tmp_path)
        runner = _make_runner([_success_exec(), _success_exec()])
        runner.config.output_file = "pw.out"
        engine = QEErrorRecoveryEngine()

        exec_r, dft_r, history = execute_with_recovery(
            backend, runner, engine, str(tmp_path), _MockQEConfig(),
        )

        assert dft_r.is_converged is True
        assert len(history) == 1

    def test_wall_clock_decay(self, tmp_path):
        """Timeout should decay on subsequent attempts."""
        pw_out = tmp_path / "pw.out"
        pw_out.write_text("convergence NOT achieved\n")

        backend = _make_backend([_unconverged_result(), _converged_result()])
        runner = _make_runner([_success_exec(), _success_exec()])
        runner.config.output_file = "pw.out"
        runner.config.timeout_seconds = 86400
        engine = QEErrorRecoveryEngine()

        execute_with_recovery(
            backend, runner, engine, str(tmp_path), _MockQEConfig(),
        )

        # After first retry: timeout = max(7200, 86400 // 2) = 43200
        assert runner.config.timeout_seconds == 43200

    def test_correction_history_structure(self, tmp_path):
        """Correction history entries have expected keys."""
        pw_out = tmp_path / "pw.out"
        pw_out.write_text("convergence NOT achieved\n")

        backend = _make_backend([_unconverged_result(), _converged_result()])
        runner = _make_runner([_success_exec(), _success_exec()])
        runner.config.output_file = "pw.out"
        engine = QEErrorRecoveryEngine()

        _, _, history = execute_with_recovery(
            backend, runner, engine, str(tmp_path), _MockQEConfig(),
        )

        assert len(history) == 1
        entry = history[0]
        assert "attempt" in entry
        assert "error_type" in entry
        assert "step" in entry
        assert "namelist_updates" in entry
        assert "quality_warnings" in entry

    def test_fatal_first_priority(self, tmp_path):
        """Fatal error takes priority even with correctable errors present."""
        pw_out = tmp_path / "pw.out"
        # Both correctable AND fatal error in output
        pw_out.write_text(
            "convergence NOT achieved\n"
            "Error in routine readpp\n"
        )

        backend = _make_backend([_unconverged_result()])
        runner = _make_runner([_success_exec()])
        runner.config.output_file = "pw.out"
        engine = QEErrorRecoveryEngine()

        exec_r, dft_r, history = execute_with_recovery(
            backend, runner, engine, str(tmp_path), _MockQEConfig(),
        )

        # Fatal should prevent any correction
        assert runner.run.call_count == 1
        assert history == []

    def test_strategies_exhausted_stops(self, tmp_path):
        """When correction strategies run out, loop stops."""
        pw_out = tmp_path / "pw.out"
        pw_out.write_text("eigenvalues not converged\n")

        # QE_EIGVAL_NOT_CONVERGED has only 2 steps
        backend = _make_backend([_unconverged_result()] * 4)
        runner = _make_runner([_success_exec()] * 4)
        runner.config.output_file = "pw.out"
        engine = QEErrorRecoveryEngine()

        exec_r, dft_r, history = execute_with_recovery(
            backend, runner, engine, str(tmp_path), _MockQEConfig(),
            max_retries=5,
        )

        # 2 corrections max (steps 0 and 1), then exhausted
        assert len(history) == 2
        assert runner.run.call_count == 3  # initial + 2 retries

    def test_zero_retries(self, tmp_path):
        """max_retries=0 means only initial execution, no recovery."""
        backend = _make_backend([_unconverged_result()])
        runner = _make_runner([_success_exec()])
        engine = QEErrorRecoveryEngine()

        exec_r, dft_r, history = execute_with_recovery(
            backend, runner, engine, str(tmp_path), _MockQEConfig(),
            max_retries=0,
        )

        assert runner.run.call_count == 1
        assert history == []

    def test_correction_applied_to_config(self, tmp_path):
        """Config object is modified in-place by corrections."""
        pw_out = tmp_path / "pw.out"
        pw_out.write_text("convergence NOT achieved\n")

        backend = _make_backend([_unconverged_result(), _converged_result()])
        runner = _make_runner([_success_exec(), _success_exec()])
        runner.config.output_file = "pw.out"
        engine = QEErrorRecoveryEngine()
        config = _MockQEConfig()

        execute_with_recovery(
            backend, runner, engine, str(tmp_path), config,
        )

        # SCF step 0 sets mixing_beta=0.3, electron_maxstep=150
        assert config.electrons.get("mixing_beta") == 0.3
        assert config.electrons.get("electron_maxstep") == 150

    def test_atoms_none_still_works(self, tmp_path):
        """Recovery works even without atoms (non-S_MATRIX errors)."""
        pw_out = tmp_path / "pw.out"
        pw_out.write_text("convergence NOT achieved\n")

        backend = _make_backend([_unconverged_result(), _converged_result()])
        runner = _make_runner([_success_exec(), _success_exec()])
        runner.config.output_file = "pw.out"
        engine = QEErrorRecoveryEngine()

        exec_r, dft_r, history = execute_with_recovery(
            backend, runner, engine, str(tmp_path), _MockQEConfig(),
            atoms=None,
        )

        assert dft_r.is_converged is True

    def test_correction_history_in_dft_result(self, tmp_path):
        """DFTResult.correction_history populated on success."""
        pw_out = tmp_path / "pw.out"
        pw_out.write_text("convergence NOT achieved\n")

        backend = _make_backend([_unconverged_result(), _converged_result()])
        runner = _make_runner([_success_exec(), _success_exec()])
        runner.config.output_file = "pw.out"
        engine = QEErrorRecoveryEngine()

        _, dft_r, history = execute_with_recovery(
            backend, runner, engine, str(tmp_path), _MockQEConfig(),
        )

        assert dft_r.correction_history == history
        assert len(dft_r.correction_history) == 1


# =========================================================================
# TestExecutionRunnerEdgeCases
# =========================================================================

class TestExecutionRunnerEdgeCases:
    """Edge cases for ExecutionRunner."""

    def test_default_config(self):
        runner = ExecutionRunner()
        assert runner.config.command == "pw.x"

    def test_custom_config(self):
        config = ExecutionConfig(command="ph.x", timeout_seconds=3600)
        runner = ExecutionRunner(config)
        assert runner.config.command == "ph.x"
        assert runner.config.timeout_seconds == 3600

    def test_cancel_no_process(self):
        """cancel() when no process is running should not raise."""
        runner = ExecutionRunner()
        runner.cancel()  # Should not raise

    @patch("subprocess.Popen")
    def test_process_cleared_after_run(self, mock_popen, tmp_calc_dir):
        proc = MagicMock()
        proc.communicate.return_value = (None, b"")
        proc.returncode = 0
        mock_popen.return_value = proc

        runner = ExecutionRunner()
        runner.run(tmp_calc_dir)

        assert runner._process is None

    @patch("subprocess.Popen")
    def test_long_stderr_truncated(self, mock_popen, tmp_calc_dir):
        # Stderr longer than 2000 chars
        long_stderr = b"X" * 5000
        proc = MagicMock()
        proc.communicate.return_value = (None, long_stderr)
        proc.returncode = 1
        mock_popen.return_value = proc

        runner = ExecutionRunner()
        result = runner.run(tmp_calc_dir)

        assert len(result.stderr_tail) <= 2000

    @patch("subprocess.Popen")
    def test_os_error_handling(self, mock_popen, tmp_calc_dir):
        mock_popen.side_effect = OSError("Permission denied")

        runner = ExecutionRunner()
        result = runner.run(tmp_calc_dir)

        assert result.success is False
        assert "Permission denied" in result.error_message


# =========================================================================
# TestUPFValidation
# =========================================================================

class TestUPFValidation:
    """Test individual UPF pseudopotential file validation."""

    def test_missing_upf_file_detected(self, tmp_path):
        """UPF file in ATOMIC_SPECIES but not in pseudo_dir."""
        pw_in = tmp_path / "pw.in"
        pw_in.write_text(
            "&CONTROL\n  pseudo_dir = './pseudo'\n/\n"
            "ATOMIC_SPECIES\n  Fe 55.845 Fe.pbe-spn-kjpaw_psl.1.0.0.UPF\n"
        )
        (tmp_path / "pseudo").mkdir()
        runner = ExecutionRunner()
        with patch("shutil.which", return_value="/usr/bin/pw.x"):
            errors = runner.validate_prerequisites(str(tmp_path))
        assert any("Fe.pbe-spn-kjpaw_psl" in e for e in errors)

    def test_present_upf_passes(self, tmp_path):
        """UPF file exists — no error."""
        pw_in = tmp_path / "pw.in"
        pw_in.write_text(
            "&CONTROL\n  pseudo_dir = './pseudo'\n/\n"
            "ATOMIC_SPECIES\n  Si 28.086 Si.pbe-n-rrkjus_psl.1.0.0.UPF\n"
        )
        pseudo_dir = tmp_path / "pseudo"
        pseudo_dir.mkdir()
        (pseudo_dir / "Si.pbe-n-rrkjus_psl.1.0.0.UPF").write_text("pseudo")
        runner = ExecutionRunner()
        with patch("shutil.which", return_value="/usr/bin/pw.x"):
            errors = runner.validate_prerequisites(str(tmp_path))
        assert errors == []

    def test_multi_element_upf_check(self, tmp_path):
        """Multi-element: report all missing UPFs."""
        pw_in = tmp_path / "pw.in"
        pw_in.write_text(
            "&CONTROL\n  pseudo_dir = './pseudo'\n/\n"
            "ATOMIC_SPECIES\n"
            "  Fe 55.845 Fe.pbe-spn-kjpaw_psl.1.0.0.UPF\n"
            "  O  15.999 O.pbe-n-kjpaw_psl.1.0.0.UPF\n"
        )
        (tmp_path / "pseudo").mkdir()
        runner = ExecutionRunner()
        with patch("shutil.which", return_value="/usr/bin/pw.x"):
            errors = runner.validate_prerequisites(str(tmp_path))
        upf_errors = [e for e in errors if "Pseudopotential not found" in e]
        assert len(upf_errors) == 2

    def test_case_insensitive_upf(self, tmp_path):
        """Lowercase .upf file found via case-insensitive fallback."""
        pw_in = tmp_path / "pw.in"
        pw_in.write_text(
            "&CONTROL\n  pseudo_dir = './pseudo'\n/\n"
            "ATOMIC_SPECIES\n  Mo 95.94 Mo_ONCV_PBE-1.2.upf\n"
        )
        pseudo_dir = tmp_path / "pseudo"
        pseudo_dir.mkdir()
        (pseudo_dir / "Mo_ONCV_PBE-1.2.upf").write_text("pseudo")
        runner = ExecutionRunner()
        with patch("shutil.which", return_value="/usr/bin/pw.x"):
            errors = runner.validate_prerequisites(str(tmp_path))
        assert errors == []

    def test_improved_error_message_for_missing_executable(self, tmp_path):
        """Missing pw.x shows install instructions."""
        pw_in = tmp_path / "pw.in"
        pw_in.write_text("&CONTROL\n/\n")
        runner = ExecutionRunner()
        with patch("shutil.which", return_value=None):
            errors = runner.validate_prerequisites(str(tmp_path))
        msg = errors[0]
        assert "sudo apt install" in msg
        assert "setup-qe" in msg


# =========================================================================
# TestWindowsToWSLPath
# =========================================================================

class TestWindowsToWSLPath:
    """Test Windows-to-WSL path conversion utility."""

    def test_c_drive_forward_slash(self):
        assert _windows_to_wsl_path("C:/Users/Foo") == "/mnt/c/Users/Foo"

    def test_c_drive_backslash(self):
        assert _windows_to_wsl_path("C:\\Users\\Foo") == "/mnt/c/Users/Foo"

    def test_d_drive(self):
        assert _windows_to_wsl_path("D:/data/pseudo") == "/mnt/d/data/pseudo"

    def test_lowercase_drive(self):
        assert _windows_to_wsl_path("c:/Users") == "/mnt/c/Users"

    def test_unix_path_unchanged(self):
        assert _windows_to_wsl_path("/mnt/c/Users") == "/mnt/c/Users"

    def test_relative_path_unchanged(self):
        assert _windows_to_wsl_path("./tmp") == "./tmp"

    def test_empty_string(self):
        assert _windows_to_wsl_path("") == ""


# =========================================================================
# TestPatchInputPathsForWSL
# =========================================================================

class TestPatchInputPathsForWSL:
    """Test pw.in / dos.in path patching for WSL execution."""

    def test_patches_pseudo_dir(self, tmp_path):
        pw_in = tmp_path / "pw.in"
        pw_in.write_text(
            "&CONTROL\n"
            "  pseudo_dir = 'C:/Users/Sejong/pseudopotentials'\n"
            "  outdir = './tmp'\n"
            "/\n"
        )
        _patch_input_paths_for_wsl(str(tmp_path), "pw.in")
        content = pw_in.read_text()
        assert "/mnt/c/Users/Sejong/pseudopotentials" in content
        # outdir should also have been converted (Bug 5)
        assert "'./tmp'" not in content

    def test_patches_outdir_absolute(self, tmp_path):
        pw_in = tmp_path / "pw.in"
        pw_in.write_text(
            "&CONTROL\n"
            "  outdir = 'D:\\data\\calc\\tmp'\n"
            "/\n"
        )
        _patch_input_paths_for_wsl(str(tmp_path), "pw.in")
        content = pw_in.read_text()
        assert "/mnt/d/data/calc/tmp" in content

    def test_relative_outdir_converted_to_absolute_wsl(self, tmp_path):
        """Relative ./tmp should be converted to absolute WSL path."""
        pw_in = tmp_path / "pw.in"
        pw_in.write_text(
            "&CONTROL\n"
            "  outdir = './tmp'\n"
            "/\n"
        )
        _patch_input_paths_for_wsl(str(tmp_path), "pw.in")
        content = pw_in.read_text()
        # Compute expected WSL-converted path
        expected_wsl_dir = _windows_to_wsl_path(os.path.abspath(str(tmp_path)))
        expected_outdir = f"{expected_wsl_dir}/tmp"
        assert f"outdir = '{expected_outdir}'" in content
        assert "'./tmp'" not in content

    def test_relative_outdir_nested_path(self, tmp_path):
        """Nested relative path ./foo/bar is converted correctly."""
        pw_in = tmp_path / "pw.in"
        pw_in.write_text(
            "&CONTROL\n"
            "  outdir = './subdir/tmp'\n"
            "/\n"
        )
        _patch_input_paths_for_wsl(str(tmp_path), "pw.in")
        content = pw_in.read_text()
        expected_wsl_dir = _windows_to_wsl_path(os.path.abspath(str(tmp_path)))
        assert f"outdir = '{expected_wsl_dir}/subdir/tmp'" in content

    def test_relative_pseudo_dir_converted(self, tmp_path):
        """Relative pseudo_dir = './pseudo' is also converted."""
        pw_in = tmp_path / "pw.in"
        pw_in.write_text(
            "&CONTROL\n"
            "  pseudo_dir = './pseudo'\n"
            "/\n"
        )
        _patch_input_paths_for_wsl(str(tmp_path), "pw.in")
        content = pw_in.read_text()
        expected_wsl_dir = _windows_to_wsl_path(os.path.abspath(str(tmp_path)))
        assert f"pseudo_dir = '{expected_wsl_dir}/pseudo'" in content

    def test_parent_relative_path_unchanged(self, tmp_path):
        """Parent-relative paths (../tmp) are NOT converted (unsupported)."""
        pw_in = tmp_path / "pw.in"
        pw_in.write_text(
            "&CONTROL\n"
            "  outdir = '../tmp'\n"
            "/\n"
        )
        _patch_input_paths_for_wsl(str(tmp_path), "pw.in")
        content = pw_in.read_text()
        # ../tmp doesn't start with "./" so it passes through unchanged
        assert "'../tmp'" in content

    def test_dos_in_outdir_patched(self, tmp_path):
        dos_in = tmp_path / "dos.in"
        dos_in.write_text(
            "&DOS\n"
            "  outdir = 'C:/Users/Sejong/calc/02_scf/tmp'\n"
            "  prefix = 'shalom'\n"
            "/\n"
        )
        _patch_input_paths_for_wsl(str(tmp_path), "dos.in")
        content = dos_in.read_text()
        assert "/mnt/c/Users/Sejong/calc/02_scf/tmp" in content

    def test_no_file_no_error(self, tmp_path):
        """Missing input file should not raise."""
        _patch_input_paths_for_wsl(str(tmp_path), "nonexistent.in")  # no error

    def test_no_windows_paths_unchanged(self, tmp_path):
        pw_in = tmp_path / "pw.in"
        original = (
            "&CONTROL\n"
            "  pseudo_dir = '/home/user/pseudo'\n"
            "  outdir = '/tmp/calc'\n"
            "/\n"
        )
        pw_in.write_text(original)
        _patch_input_paths_for_wsl(str(tmp_path), "pw.in")
        assert pw_in.read_text() == original


# =========================================================================
# TestBuildCommandWSL
# =========================================================================

class TestBuildCommandWSL:
    """Test WSL-specific command construction."""

    def test_wsl_serial(self):
        config = ExecutionConfig(command="/opt/qe/bin/pw.x", wsl=True)
        cmd = ExecutionRunner.build_command(config)
        assert cmd == ["wsl", "-e", "bash", "-c", "/opt/qe/bin/pw.x"]

    def test_wsl_parallel(self):
        config = ExecutionConfig(
            command="/opt/qe/bin/pw.x",
            nprocs=4,
            mpi_command="/opt/conda/bin/mpirun",
            wsl=True,
        )
        cmd = ExecutionRunner.build_command(config)
        assert cmd == [
            "wsl", "-e", "bash", "-c",
            "/opt/conda/bin/mpirun --allow-run-as-root -np 4 /opt/qe/bin/pw.x",
        ]

    def test_wsl_single_proc_serial(self):
        config = ExecutionConfig(command="pw.x", nprocs=1, wsl=True)
        cmd = ExecutionRunner.build_command(config)
        assert cmd == ["wsl", "-e", "bash", "-c", "pw.x"]
