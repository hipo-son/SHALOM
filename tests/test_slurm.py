"""Tests for shalom.backends.slurm — Slurm job submission and monitoring."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, call, patch

import pytest

from shalom.backends.runner import ExecutionConfig, ExecutionResult
from shalom.backends.slurm import (
    SlurmConfig,
    SlurmJobState,
    SlurmRunner,
    _parse_elapsed,
)


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def default_slurm_config():
    return SlurmConfig()


@pytest.fixture
def custom_slurm_config():
    return SlurmConfig(
        partition="gpu",
        account="mat_sci",
        nodes=2,
        ntasks_per_node=32,
        walltime="48:00:00",
        qos="high",
        mem="128G",
        job_name="test-job",
        module_loads=["quantum-espresso/7.2", "intel-mpi/2021"],
        pre_commands=["export OMP_NUM_THREADS=1"],
        extra_directives=["--gres=gpu:4", "--exclusive"],
    )


@pytest.fixture
def exec_config():
    return ExecutionConfig(command="pw.x", nprocs=64)


@pytest.fixture
def vasp_exec_config():
    return ExecutionConfig(
        command="vasp_std",
        input_file="/dev/null",
        output_file="vasp.out",
        nprocs=4,
    )


@pytest.fixture
def runner(exec_config, default_slurm_config):
    return SlurmRunner(
        exec_config=exec_config,
        slurm_config=default_slurm_config,
    )


@pytest.fixture
def tmp_calc_dir(tmp_path):
    """Calculation directory with a minimal pw.in."""
    pw_in = tmp_path / "pw.in"
    pw_in.write_text("&CONTROL\n/\n&SYSTEM\n/\n&ELECTRONS\n/\n")
    return str(tmp_path)


# =========================================================================
# TestSlurmConfig
# =========================================================================


class TestSlurmConfig:
    """SlurmConfig defaults and computed properties."""

    def test_defaults(self, default_slurm_config):
        cfg = default_slurm_config
        assert cfg.partition == "compute"
        assert cfg.account is None
        assert cfg.nodes == 1
        assert cfg.ntasks_per_node == 1
        assert cfg.walltime == "24:00:00"
        assert cfg.qos is None
        assert cfg.mem is None
        assert cfg.job_name == "shalom-dft"
        assert cfg.output_pattern == "slurm-%j.out"
        assert cfg.error_pattern == "slurm-%j.err"
        assert cfg.extra_directives == []
        assert cfg.module_loads == []
        assert cfg.pre_commands == []
        assert cfg.poll_interval_initial == 10.0
        assert cfg.poll_interval_max == 60.0
        assert cfg.client_timeout == 0.0

    def test_total_tasks_single_node(self):
        cfg = SlurmConfig(nodes=1, ntasks_per_node=16)
        assert cfg.total_tasks == 16

    def test_total_tasks_multi_node(self):
        cfg = SlurmConfig(nodes=4, ntasks_per_node=32)
        assert cfg.total_tasks == 128

    def test_custom_fields(self, custom_slurm_config):
        cfg = custom_slurm_config
        assert cfg.partition == "gpu"
        assert cfg.account == "mat_sci"
        assert cfg.nodes == 2
        assert cfg.total_tasks == 64
        assert cfg.qos == "high"
        assert cfg.mem == "128G"
        assert len(cfg.module_loads) == 2
        assert len(cfg.extra_directives) == 2


# =========================================================================
# TestJobScriptGeneration
# =========================================================================


class TestJobScriptGeneration:
    """SlurmRunner.generate_job_script() produces valid sbatch scripts."""

    def test_basic_qe_script(self, tmp_calc_dir):
        runner = SlurmRunner(
            exec_config=ExecutionConfig(command="pw.x", nprocs=32),
            slurm_config=SlurmConfig(partition="compute", nodes=1, ntasks_per_node=32),
        )
        script = runner.generate_job_script(tmp_calc_dir)

        assert script.startswith("#!/bin/bash")
        assert "#SBATCH --job-name=shalom-dft" in script
        assert "#SBATCH --partition=compute" in script
        assert "#SBATCH --nodes=1" in script
        assert "#SBATCH --ntasks-per-node=32" in script
        assert "#SBATCH --time=24:00:00" in script
        assert f"#SBATCH --chdir={tmp_calc_dir}" in script
        assert "srun -n 32 pw.x < pw.in > pw.out" in script

    def test_vasp_script(self, tmp_calc_dir, vasp_exec_config):
        runner = SlurmRunner(
            exec_config=vasp_exec_config,
            slurm_config=SlurmConfig(
                partition="gpu",
                nodes=1,
                ntasks_per_node=4,
            ),
        )
        script = runner.generate_job_script(tmp_calc_dir)

        assert "srun -n 4 vasp_std < /dev/null > vasp.out" in script
        assert "#SBATCH --partition=gpu" in script

    def test_account_included(self, tmp_calc_dir):
        runner = SlurmRunner(
            slurm_config=SlurmConfig(account="my_alloc"),
        )
        script = runner.generate_job_script(tmp_calc_dir)
        assert "#SBATCH --account=my_alloc" in script

    def test_account_omitted_when_none(self, tmp_calc_dir):
        runner = SlurmRunner(slurm_config=SlurmConfig(account=None))
        script = runner.generate_job_script(tmp_calc_dir)
        assert "--account" not in script

    def test_qos_included(self, tmp_calc_dir):
        runner = SlurmRunner(slurm_config=SlurmConfig(qos="high"))
        script = runner.generate_job_script(tmp_calc_dir)
        assert "#SBATCH --qos=high" in script

    def test_mem_included(self, tmp_calc_dir):
        runner = SlurmRunner(slurm_config=SlurmConfig(mem="64G"))
        script = runner.generate_job_script(tmp_calc_dir)
        assert "#SBATCH --mem=64G" in script

    def test_module_loads(self, tmp_calc_dir, custom_slurm_config):
        runner = SlurmRunner(slurm_config=custom_slurm_config)
        script = runner.generate_job_script(tmp_calc_dir)
        assert "module load quantum-espresso/7.2" in script
        assert "module load intel-mpi/2021" in script

    def test_pre_commands(self, tmp_calc_dir, custom_slurm_config):
        runner = SlurmRunner(slurm_config=custom_slurm_config)
        script = runner.generate_job_script(tmp_calc_dir)
        assert "export OMP_NUM_THREADS=1" in script

    def test_extra_directives(self, tmp_calc_dir, custom_slurm_config):
        runner = SlurmRunner(slurm_config=custom_slurm_config)
        script = runner.generate_job_script(tmp_calc_dir)
        assert "#SBATCH --gres=gpu:4" in script
        assert "#SBATCH --exclusive" in script

    def test_env_overrides_exported(self, tmp_calc_dir):
        runner = SlurmRunner(
            exec_config=ExecutionConfig(
                env_overrides={"ASE_VASP_COMMAND": "vasp_std"},
            ),
        )
        script = runner.generate_job_script(tmp_calc_dir)
        assert "export ASE_VASP_COMMAND=vasp_std" in script

    def test_output_error_patterns(self, tmp_calc_dir):
        runner = SlurmRunner(
            slurm_config=SlurmConfig(
                output_pattern="job-%j.log",
                error_pattern="job-%j.err",
            ),
        )
        script = runner.generate_job_script(tmp_calc_dir)
        assert "#SBATCH --output=job-%j.log" in script
        assert "#SBATCH --error=job-%j.err" in script

    def test_multi_node_task_count(self, tmp_calc_dir):
        runner = SlurmRunner(
            slurm_config=SlurmConfig(nodes=4, ntasks_per_node=32),
        )
        script = runner.generate_job_script(tmp_calc_dir)
        assert "srun -n 128" in script


# =========================================================================
# TestParseElapsed
# =========================================================================


class TestParseElapsed:
    """_parse_elapsed() converts sacct time strings to seconds."""

    def test_hhmmss(self):
        assert _parse_elapsed("01:23:45") == 5025.0

    def test_dhhmmss(self):
        assert _parse_elapsed("2-03:00:00") == 183600.0

    def test_mmss(self):
        assert _parse_elapsed("05:30") == 330.0

    def test_zero(self):
        assert _parse_elapsed("00:00:00") == 0.0

    def test_large_days(self):
        assert _parse_elapsed("10-00:00:00") == 864000.0

    def test_fractional_seconds(self):
        result = _parse_elapsed("01:23:45.678")
        assert result is not None
        assert abs(result - 5025.678) < 0.001

    def test_fractional_mmss(self):
        result = _parse_elapsed("05:30.5")
        assert result is not None
        assert abs(result - 330.5) < 0.001

    def test_invalid_returns_none(self):
        assert _parse_elapsed("invalid") is None

    def test_empty_returns_none(self):
        assert _parse_elapsed("") is None


# =========================================================================
# TestSubmission
# =========================================================================


class TestSubmission:
    """SlurmRunner._submit() parses sbatch output for job IDs."""

    @patch("shalom.backends.slurm.subprocess.run")
    def test_submit_parses_job_id(self, mock_run, runner, tmp_calc_dir):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Submitted batch job 12345\n",
            stderr="",
        )
        job_id = runner._submit(tmp_calc_dir)
        assert job_id == "12345"

    @patch("shalom.backends.slurm.subprocess.run")
    def test_submit_parses_large_job_id(self, mock_run, runner, tmp_calc_dir):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Submitted batch job 9876543\n",
            stderr="",
        )
        job_id = runner._submit(tmp_calc_dir)
        assert job_id == "9876543"

    @patch("shalom.backends.slurm.subprocess.run")
    def test_submit_failure_raises(self, mock_run, runner, tmp_calc_dir):
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="sbatch: error: Batch job submission failed\n",
        )
        with pytest.raises(RuntimeError, match="sbatch failed"):
            runner._submit(tmp_calc_dir)

    @patch("shalom.backends.slurm.subprocess.run")
    def test_submit_unparseable_output_raises(self, mock_run, runner, tmp_calc_dir):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Something unexpected\n",
            stderr="",
        )
        with pytest.raises(RuntimeError, match="Could not parse job ID"):
            runner._submit(tmp_calc_dir)

    @patch("shalom.backends.slurm.subprocess.run")
    def test_submit_writes_job_script(self, mock_run, runner, tmp_calc_dir):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Submitted batch job 111\n",
            stderr="",
        )
        runner._submit(tmp_calc_dir)
        script_path = os.path.join(tmp_calc_dir, "job.sh")
        assert os.path.exists(script_path)
        with open(script_path) as f:
            content = f.read()
        assert "#!/bin/bash" in content
        assert "#SBATCH" in content

    @patch("shalom.backends.slurm.subprocess.run")
    def test_submit_calls_sbatch_with_correct_args(self, mock_run, runner, tmp_calc_dir):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Submitted batch job 111\n",
            stderr="",
        )
        runner._submit(tmp_calc_dir)
        expected_script = os.path.join(tmp_calc_dir, "job.sh")
        mock_run.assert_called_once_with(
            ["sbatch", expected_script],
            capture_output=True,
            text=True,
            cwd=tmp_calc_dir,
            timeout=30,
        )


# =========================================================================
# TestPolling
# =========================================================================


class TestPolling:
    """SlurmRunner polling methods (squeue, sacct, poll_until_done)."""

    @patch("shalom.backends.slurm.subprocess.run")
    def test_query_squeue_running(self, mock_run, runner):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="RUNNING\n",
        )
        state = runner._query_squeue("12345")
        assert state == SlurmJobState.RUNNING

    @patch("shalom.backends.slurm.subprocess.run")
    def test_query_squeue_pending(self, mock_run, runner):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="PENDING\n",
        )
        state = runner._query_squeue("12345")
        assert state == SlurmJobState.PENDING

    @patch("shalom.backends.slurm.subprocess.run")
    def test_query_squeue_empty_means_gone(self, mock_run, runner):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="",
        )
        state = runner._query_squeue("12345")
        assert state is None

    @patch("shalom.backends.slurm.subprocess.run")
    def test_query_squeue_unknown_state(self, mock_run, runner):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="CONFIGURING\n",
        )
        state = runner._query_squeue("12345")
        assert state == SlurmJobState.UNKNOWN

    @patch("shalom.backends.slurm.subprocess.run")
    def test_query_squeue_timeout_returns_unknown(self, mock_run, runner):
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="squeue", timeout=15)
        state = runner._query_squeue("12345")
        assert state == SlurmJobState.UNKNOWN

    @patch("shalom.backends.slurm.subprocess.run")
    def test_query_sacct_completed(self, mock_run, runner):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="12345|COMPLETED|01:30:00\n12345.batch|COMPLETED|01:30:00\n",
        )
        state, elapsed = runner._query_sacct("12345")
        assert state == SlurmJobState.COMPLETED
        assert elapsed == 5400.0

    @patch("shalom.backends.slurm.subprocess.run")
    def test_query_sacct_failed(self, mock_run, runner):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="12345|FAILED|00:05:30\n",
        )
        state, elapsed = runner._query_sacct("12345")
        assert state == SlurmJobState.FAILED
        assert elapsed == 330.0

    @patch("shalom.backends.slurm.subprocess.run")
    def test_query_sacct_no_match(self, mock_run, runner):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="12345.batch|COMPLETED|01:00:00\n",
        )
        state, elapsed = runner._query_sacct("12345")
        assert state == SlurmJobState.UNKNOWN
        assert elapsed is None

    @patch("shalom.backends.slurm.time.sleep")
    @patch("shalom.backends.slurm.subprocess.run")
    def test_poll_pending_then_completed(self, mock_run, mock_sleep, runner):
        """Simulate PENDING → RUNNING → gone → sacct COMPLETED."""
        mock_run.side_effect = [
            # squeue: PENDING
            MagicMock(returncode=0, stdout="PENDING\n"),
            # squeue: RUNNING
            MagicMock(returncode=0, stdout="RUNNING\n"),
            # squeue: gone (empty)
            MagicMock(returncode=0, stdout=""),
            # sacct: COMPLETED
            MagicMock(returncode=0, stdout="12345|COMPLETED|00:10:00\n"),
        ]
        state, wall_time = runner._poll_until_done("12345")
        assert state == SlurmJobState.COMPLETED
        assert wall_time == 600.0
        assert mock_sleep.call_count == 2

    @patch("shalom.backends.slurm.time.sleep")
    @patch("shalom.backends.slurm.subprocess.run")
    def test_poll_failed_from_squeue(self, mock_run, mock_sleep, runner):
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="FAILED\n"),
        ]
        state, _ = runner._poll_until_done("12345")
        assert state == SlurmJobState.FAILED
        assert mock_sleep.call_count == 0

    @patch("shalom.backends.slurm.time.monotonic")
    @patch("shalom.backends.slurm.time.sleep")
    @patch("shalom.backends.slurm.subprocess.run")
    def test_poll_client_timeout_triggers_scancel(
        self, mock_run, mock_sleep, mock_monotonic, tmp_calc_dir,
    ):
        slurm_cfg = SlurmConfig(client_timeout=30.0)
        runner = SlurmRunner(slurm_config=slurm_cfg)

        # monotonic: start=0, then 35 (beyond timeout)
        mock_monotonic.side_effect = [0.0, 35.0]

        # squeue would return RUNNING, but timeout fires first
        mock_run.side_effect = [
            # scancel call
            MagicMock(returncode=0),
        ]

        state, wall_time = runner._poll_until_done("12345")
        assert state == SlurmJobState.TIMEOUT
        # scancel was called
        mock_run.assert_called_once_with(
            ["scancel", "12345"],
            capture_output=True,
            timeout=10,
        )

    @patch("shalom.backends.slurm.time.sleep")
    @patch("shalom.backends.slurm.subprocess.run")
    def test_poll_unknown_streak_falls_back_to_sacct(self, mock_run, mock_sleep):
        """After 30 consecutive UNKNOWN responses, fall back to sacct."""
        import subprocess as _sp

        slurm_cfg = SlurmConfig()
        runner = SlurmRunner(slurm_config=slurm_cfg)

        # 30 squeue failures (TimeoutExpired → UNKNOWN) then sacct COMPLETED
        squeue_timeout = _sp.TimeoutExpired(cmd="squeue", timeout=15)
        side_effects = [squeue_timeout] * 30 + [
            MagicMock(returncode=0, stdout="12345|COMPLETED|01:00:00\n"),
        ]
        mock_run.side_effect = side_effects

        state, wall_time = runner._poll_until_done("12345")
        assert state == SlurmJobState.COMPLETED
        assert wall_time == 3600.0


# =========================================================================
# TestSlurmRunnerRun
# =========================================================================


class TestSlurmRunnerRun:
    """Full run() lifecycle: submit → poll → ExecutionResult."""

    @patch("shalom.backends.slurm.time.sleep")
    @patch("shalom.backends.slurm.subprocess.run")
    def test_successful_job(self, mock_run, mock_sleep, tmp_calc_dir):
        runner = SlurmRunner(
            exec_config=ExecutionConfig(command="pw.x"),
            slurm_config=SlurmConfig(),
        )
        # Create expected output file
        pw_out = os.path.join(tmp_calc_dir, "pw.out")
        open(pw_out, "w").close()

        mock_run.side_effect = [
            # sbatch
            MagicMock(returncode=0, stdout="Submitted batch job 42\n", stderr=""),
            # squeue: gone
            MagicMock(returncode=0, stdout=""),
            # sacct: COMPLETED
            MagicMock(returncode=0, stdout="42|COMPLETED|00:05:00\n"),
        ]

        result = runner.run(tmp_calc_dir)
        assert result.success is True
        assert result.return_code == 0
        assert result.wall_time_seconds == 300.0
        assert result.output_file == pw_out

    @patch("shalom.backends.slurm.time.sleep")
    @patch("shalom.backends.slurm.subprocess.run")
    def test_failed_job(self, mock_run, mock_sleep, tmp_calc_dir):
        runner = SlurmRunner(
            exec_config=ExecutionConfig(command="pw.x"),
            slurm_config=SlurmConfig(),
        )

        mock_run.side_effect = [
            # sbatch
            MagicMock(returncode=0, stdout="Submitted batch job 99\n", stderr=""),
            # squeue: FAILED
            MagicMock(returncode=0, stdout="FAILED\n"),
        ]

        result = runner.run(tmp_calc_dir)
        assert result.success is False
        assert result.timed_out is False
        assert "FAILED" in result.error_message

    @patch("shalom.backends.slurm.time.sleep")
    @patch("shalom.backends.slurm.subprocess.run")
    def test_timeout_job(self, mock_run, mock_sleep, tmp_calc_dir):
        runner = SlurmRunner(
            exec_config=ExecutionConfig(command="pw.x"),
            slurm_config=SlurmConfig(),
        )

        mock_run.side_effect = [
            # sbatch
            MagicMock(returncode=0, stdout="Submitted batch job 77\n", stderr=""),
            # squeue: TIMEOUT
            MagicMock(returncode=0, stdout="TIMEOUT\n"),
        ]

        result = runner.run(tmp_calc_dir)
        assert result.success is False
        assert result.timed_out is True
        assert "TIMEOUT" in result.error_message

    @patch("shalom.backends.slurm.subprocess.run")
    def test_submit_failure_returns_error(self, mock_run, tmp_calc_dir):
        runner = SlurmRunner(
            exec_config=ExecutionConfig(command="pw.x"),
            slurm_config=SlurmConfig(),
        )

        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="sbatch: error: invalid partition\n",
        )

        result = runner.run(tmp_calc_dir)
        assert result.success is False
        assert "sbatch failed" in result.error_message

    @patch("shalom.backends.slurm.time.sleep")
    @patch("shalom.backends.slurm.subprocess.run")
    def test_cancelled_job_is_timed_out(self, mock_run, mock_sleep, tmp_calc_dir):
        runner = SlurmRunner(
            exec_config=ExecutionConfig(command="pw.x"),
            slurm_config=SlurmConfig(),
        )

        mock_run.side_effect = [
            # sbatch
            MagicMock(returncode=0, stdout="Submitted batch job 55\n", stderr=""),
            # squeue: CANCELLED
            MagicMock(returncode=0, stdout="CANCELLED\n"),
        ]

        result = runner.run(tmp_calc_dir)
        assert result.success is False
        assert result.timed_out is True

    @patch("shalom.backends.slurm.time.sleep")
    @patch("shalom.backends.slurm.subprocess.run")
    def test_node_fail(self, mock_run, mock_sleep, tmp_calc_dir):
        runner = SlurmRunner(
            exec_config=ExecutionConfig(command="pw.x"),
            slurm_config=SlurmConfig(),
        )

        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="Submitted batch job 88\n", stderr=""),
            MagicMock(returncode=0, stdout="NODE_FAIL\n"),
        ]

        result = runner.run(tmp_calc_dir)
        assert result.success is False
        assert "NODE_FAIL" in result.error_message

    @patch("shalom.backends.slurm.time.sleep")
    @patch("shalom.backends.slurm.subprocess.run")
    def test_out_of_memory(self, mock_run, mock_sleep, tmp_calc_dir):
        runner = SlurmRunner(
            exec_config=ExecutionConfig(command="pw.x"),
            slurm_config=SlurmConfig(),
        )

        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="Submitted batch job 66\n", stderr=""),
            MagicMock(returncode=0, stdout="OUT_OF_MEMORY\n"),
        ]

        result = runner.run(tmp_calc_dir)
        assert result.success is False
        assert "OUT_OF_MEMORY" in result.error_message

    @patch("shalom.backends.slurm.time.sleep")
    @patch("shalom.backends.slurm.subprocess.run")
    def test_output_file_missing(self, mock_run, mock_sleep, tmp_calc_dir):
        """When output file doesn't exist, output_file should be None."""
        runner = SlurmRunner(
            exec_config=ExecutionConfig(command="pw.x"),
            slurm_config=SlurmConfig(),
        )

        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="Submitted batch job 42\n", stderr=""),
            MagicMock(returncode=0, stdout=""),
            MagicMock(returncode=0, stdout="42|COMPLETED|00:01:00\n"),
        ]

        result = runner.run(tmp_calc_dir)
        assert result.success is True
        assert result.output_file is None


# =========================================================================
# TestCancel
# =========================================================================


class TestCancel:
    """SlurmRunner.cancel() calls scancel."""

    @patch("shalom.backends.slurm.subprocess.run")
    def test_cancel_with_active_job(self, mock_run, runner):
        runner._job_id = "12345"
        runner.cancel()
        mock_run.assert_called_once_with(
            ["scancel", "12345"],
            capture_output=True,
            timeout=10,
        )

    @patch("shalom.backends.slurm.subprocess.run")
    def test_cancel_without_job_does_nothing(self, mock_run, runner):
        runner._job_id = None
        runner.cancel()
        mock_run.assert_not_called()


# =========================================================================
# TestPrerequisites
# =========================================================================


class TestPrerequisites:
    """SlurmRunner.validate_prerequisites() checks."""

    @patch("shutil.which")
    def test_all_present(self, mock_which, runner, tmp_calc_dir):
        mock_which.return_value = "/usr/bin/sbatch"
        errors = runner.validate_prerequisites(tmp_calc_dir)
        assert errors == []

    @patch("shutil.which")
    def test_missing_sbatch(self, mock_which, runner, tmp_calc_dir):
        mock_which.side_effect = lambda cmd: None if cmd == "sbatch" else "/usr/bin/squeue"
        errors = runner.validate_prerequisites(tmp_calc_dir)
        assert any("sbatch" in e for e in errors)

    @patch("shutil.which")
    def test_missing_squeue(self, mock_which, runner, tmp_calc_dir):
        mock_which.side_effect = lambda cmd: "/usr/bin/sbatch" if cmd == "sbatch" else None
        errors = runner.validate_prerequisites(tmp_calc_dir)
        assert any("squeue" in e for e in errors)

    @patch("shutil.which")
    def test_missing_input_file(self, mock_which, runner, tmp_path):
        mock_which.return_value = "/usr/bin/sbatch"
        empty_dir = str(tmp_path / "empty")
        os.makedirs(empty_dir)
        errors = runner.validate_prerequisites(empty_dir)
        assert any("Input file not found" in e for e in errors)


# =========================================================================
# TestSlurmJobState
# =========================================================================


class TestSlurmJobState:
    """SlurmJobState enum covers all relevant states."""

    def test_all_states_exist(self):
        expected = {
            "PENDING", "RUNNING", "COMPLETING", "COMPLETED",
            "FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL",
            "OUT_OF_MEMORY", "UNKNOWN",
        }
        actual = {s.value for s in SlurmJobState}
        assert actual == expected

    def test_value_roundtrip(self):
        for state in SlurmJobState:
            assert SlurmJobState(state.value) is state


# =========================================================================
# TestSlurmStderrRead
# =========================================================================


class TestSlurmStderrRead:
    """SlurmRunner._read_slurm_stderr() reads error output."""

    def test_reads_stderr_file(self, tmp_calc_dir, runner):
        err_file = os.path.join(tmp_calc_dir, "slurm-12345.err")
        with open(err_file, "w") as f:
            f.write("Some error output\n")
        content = runner._read_slurm_stderr(tmp_calc_dir, "12345")
        assert content == "Some error output\n"

    def test_returns_none_for_missing_file(self, tmp_calc_dir, runner):
        content = runner._read_slurm_stderr(tmp_calc_dir, "99999")
        assert content is None

    def test_truncates_long_content(self, tmp_calc_dir, runner):
        err_file = os.path.join(tmp_calc_dir, "slurm-12345.err")
        with open(err_file, "w") as f:
            f.write("x" * 5000)
        content = runner._read_slurm_stderr(tmp_calc_dir, "12345")
        assert len(content) == 2000


# =========================================================================
# TestDefaultRunner
# =========================================================================


class TestDefaultRunner:
    """SlurmRunner with default configs."""

    def test_default_exec_config(self):
        runner = SlurmRunner()
        assert runner.config.command == "pw.x"
        assert runner.slurm.partition == "compute"

    def test_default_job_id_is_none(self):
        runner = SlurmRunner()
        assert runner._job_id is None
