"""Tests for shalom.__main__ CLI module."""

import pytest

from shalom.__main__ import build_parser, _parse_set_values


# ---------------------------------------------------------------------------
# Argument parser tests
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_run_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["run", "mp-19717"])
        assert args.command == "run"
        assert args.material == "mp-19717"

    @pytest.mark.parametrize("args_list,attr_name,expected", [
        (["run", "mp-19717", "--backend", "qe"], "backend", "qe"),
        (["run", "mp-19717", "-b", "qe"], "backend", "qe"),
        (["run", "mp-19717", "--calc", "scf"], "calc", "scf"),
        (["run", "mp-19717", "--accuracy", "precise"], "accuracy", "precise"),
        (["run", "mp-19717", "--force"], "force", True),
        (["run", "mp-19717", "--no-validate"], "no_validate", True),
        (["run", "mp-19717", "-q"], "quiet", True),
        (["run", "mp-19717", "-v"], "verbose", True),
    ])
    def test_parser_single_flag(self, args_list, attr_name, expected):
        parser = build_parser()
        args = parser.parse_args(args_list)
        assert getattr(args, attr_name) == expected

    def test_set_values(self):
        parser = build_parser()
        args = parser.parse_args(["run", "mp-19717", "--set", "ENCUT=600", "--set", "NSW=200"])
        assert args.set_values == ["ENCUT=600", "NSW=200"]

    def test_output_flag(self):
        parser = build_parser()
        args = parser.parse_args(["run", "mp-19717", "-o", "/tmp/out"])
        assert args.output == "/tmp/out"

    def test_structure_flag(self):
        parser = build_parser()
        args = parser.parse_args(["run", "--structure", "POSCAR"])
        assert args.structure == "POSCAR"
        assert args.material is None

    def test_default_backend_is_vasp(self):
        parser = build_parser()
        args = parser.parse_args(["run", "mp-19717"])
        assert args.backend == "vasp"

    def test_no_command(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.command is None


# ---------------------------------------------------------------------------
# _parse_set_values tests
# ---------------------------------------------------------------------------


class TestParseSetValues:
    @pytest.mark.parametrize("input_str,key,expected_val,expected_type", [
        ("ENCUT=600", "ENCUT", 600, int),
        ("ecutwfc=80.5", "ecutwfc", 80.5, float),
        ("conv_thr=1e-8", "conv_thr", 1e-8, float),
        ("tprnfor=true", "tprnfor", True, bool),
        ("LWAVE=false", "LWAVE", False, bool),
        ("smearing=cold", "smearing", "cold", str),
        ("LREAL=.TRUE.", "LREAL", True, bool),
    ])
    def test_type_parsing(self, input_str, key, expected_val, expected_type):
        result = _parse_set_values([input_str])
        assert result[key] == expected_val
        assert isinstance(result[key], expected_type)

    def test_multiple(self):
        result = _parse_set_values(["ENCUT=600", "EDIFF=1e-6", "ALGO=Fast"])
        assert result == {"ENCUT": 600, "EDIFF": 1e-6, "ALGO": "Fast"}

    def test_none_input(self):
        result = _parse_set_values(None)
        assert result == {}

    def test_empty_list(self):
        result = _parse_set_values([])
        assert result == {}

    def test_missing_equals(self):
        """Items without = are skipped with a warning."""
        result = _parse_set_values(["ENCUT600"])
        assert result == {}

    def test_empty_key(self):
        """'=value' → skipped (empty key)."""
        result = _parse_set_values(["=600"])
        assert result == {}

    def test_empty_value(self):
        """'KEY=' → skipped (empty value)."""
        result = _parse_set_values(["ENCUT="])
        assert result == {}

    def test_multiple_equals(self):
        """'KEY=a=b' → value is 'a=b'."""
        result = _parse_set_values(["PATH=a=b"])
        assert result["PATH"] == "a=b"


# ---------------------------------------------------------------------------
# cmd_run tests
# ---------------------------------------------------------------------------


class TestCmdRun:
    def _make_args(self, **overrides):
        """Build a minimal argparse.Namespace for cmd_run."""
        import argparse
        defaults = dict(
            material=None, structure=None, backend="vasp", calc=None,
            accuracy="standard", set_values=None, output=None,
            workspace=None, project=None,
            pseudo_dir=None, no_validate=False, force=False,
            quiet=True, verbose=False, command="run",
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_no_material_no_structure(self, capsys):
        """Neither material nor structure → returns 1."""
        from shalom.__main__ import cmd_run
        args = self._make_args(quiet=False)
        rc = cmd_run(args)
        assert rc == 1
        captured = capsys.readouterr()
        assert "Error" in captured.out or "Provide" in captured.out

    def test_mp_not_installed(self):
        """Material given but mp-api not available → returns 1."""
        from unittest.mock import patch
        from shalom.__main__ import cmd_run
        args = self._make_args(material="mp-19717", quiet=True)
        with patch("shalom.mp_client.is_mp_available", return_value=False):
            rc = cmd_run(args)
        assert rc == 1

    def test_vasp_success(self, tmp_path):
        """Successful VASP run with local structure file."""
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import cmd_run

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.backend_name = "vasp"
        mock_result.structure_info = None
        mock_result.auto_detected = {"ENCUT": 520}
        mock_result.output_dir = str(tmp_path)
        mock_result.files_generated = ["POSCAR", "INCAR"]

        args = self._make_args(structure="POSCAR", quiet=False)
        with patch("shalom.direct_run.direct_run", return_value=mock_result), \
             patch("shalom.direct_run.DirectRunConfig"):
            rc = cmd_run(args)
        assert rc == 0

    def test_qe_success(self, tmp_path):
        """Successful QE run prints QE-specific instructions."""
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import cmd_run

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.backend_name = "qe"
        mock_result.structure_info = None
        mock_result.auto_detected = {"ecutwfc": 60}
        mock_result.output_dir = str(tmp_path)
        mock_result.files_generated = ["pw.in"]

        args = self._make_args(structure="POSCAR", backend="qe", quiet=False)
        with patch("shalom.direct_run.direct_run", return_value=mock_result), \
             patch("shalom.direct_run.DirectRunConfig"):
            rc = cmd_run(args)
        assert rc == 0

    def test_failure_returns_1(self, capsys):
        """Failed run returns 1 and prints error."""
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import cmd_run

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "Something went wrong"

        args = self._make_args(structure="POSCAR", quiet=False)
        with patch("shalom.direct_run.direct_run", return_value=mock_result), \
             patch("shalom.direct_run.DirectRunConfig"):
            rc = cmd_run(args)
        assert rc == 1
        captured = capsys.readouterr()
        assert "Something went wrong" in captured.out

    def test_set_values_passed(self, tmp_path):
        """--set KEY=VALUE is parsed and passed to DirectRunConfig."""
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import cmd_run

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.backend_name = "vasp"
        mock_result.structure_info = None
        mock_result.auto_detected = {}
        mock_result.output_dir = str(tmp_path)
        mock_result.files_generated = []

        mock_config_cls = MagicMock()

        args = self._make_args(
            structure="POSCAR", set_values=["ENCUT=600"], quiet=True
        )
        with patch("shalom.direct_run.direct_run", return_value=mock_result), \
             patch("shalom.direct_run.DirectRunConfig", mock_config_cls):
            rc = cmd_run(args)
        assert rc == 0
        # Verify user_settings was passed with ENCUT=600
        config_call = mock_config_cls.call_args
        assert config_call.kwargs.get("user_settings") == {"ENCUT": 600}

    def test_mp_info_printed(self, tmp_path, capsys):
        """MP structure info is printed when available."""
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import cmd_run

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.backend_name = "vasp"
        mock_result.structure_info = {
            "mp_id": "mp-19717",
            "formula": "Cu",
            "energy_above_hull": 0.0,
            "space_group": "Fm-3m",
        }
        mock_result.auto_detected = {}
        mock_result.output_dir = str(tmp_path)
        mock_result.files_generated = ["POSCAR"]

        args = self._make_args(structure="POSCAR", quiet=False)
        with patch("shalom.direct_run.direct_run", return_value=mock_result), \
             patch("shalom.direct_run.DirectRunConfig"):
            rc = cmd_run(args)
        assert rc == 0
        captured = capsys.readouterr()
        assert "mp-19717" in captured.out
        assert "Fm-3m" in captured.out


# ---------------------------------------------------------------------------
# main() tests
# ---------------------------------------------------------------------------


class TestMain:
    def test_no_command_prints_help(self, capsys):
        """No subcommand → prints help and exits 0."""
        import sys
        from unittest.mock import patch
        from shalom.__main__ import main
        with patch.object(sys, "argv", ["shalom"]), \
             pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

    def test_verbose_sets_debug(self):
        """--verbose flag sets logging to DEBUG."""
        import sys
        import logging
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import main

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.backend_name = "vasp"
        mock_result.structure_info = None
        mock_result.auto_detected = {}
        mock_result.output_dir = "/tmp"
        mock_result.files_generated = []

        with patch.object(sys, "argv", ["shalom", "run", "--structure", "POSCAR", "-v", "-q"]), \
             patch("shalom.direct_run.direct_run", return_value=mock_result), \
             patch("shalom.direct_run.DirectRunConfig"), \
             patch("logging.basicConfig") as mock_basic, \
             pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        mock_basic.assert_called_once()
        assert mock_basic.call_args.kwargs["level"] == logging.DEBUG


# ---------------------------------------------------------------------------
# Execution CLI flag tests
# ---------------------------------------------------------------------------


class TestExecutionFlags:
    def test_execute_flag_parsed(self):
        parser = build_parser()
        args = parser.parse_args(["run", "mp-19717", "--execute"])
        assert args.execute is True

    def test_execute_short_flag(self):
        parser = build_parser()
        args = parser.parse_args(["run", "mp-19717", "-x"])
        assert args.execute is True

    def test_nprocs_flag(self):
        parser = build_parser()
        args = parser.parse_args(["run", "mp-19717", "--nprocs", "4"])
        assert args.nprocs == 4

    def test_nprocs_short_flag(self):
        parser = build_parser()
        args = parser.parse_args(["run", "mp-19717", "-np", "8"])
        assert args.nprocs == 8

    def test_timeout_flag(self):
        parser = build_parser()
        args = parser.parse_args(["run", "mp-19717", "--timeout", "7200"])
        assert args.timeout == 7200

    def test_mpi_command_flag(self):
        parser = build_parser()
        args = parser.parse_args(["run", "mp-19717", "--mpi-command", "srun"])
        assert args.mpi_command == "srun"

    def test_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["run", "mp-19717"])
        assert args.execute is False
        assert args.nprocs == 1
        assert args.timeout == 86400
        assert args.mpi_command == "mpirun"

    def test_setup_qe_parser(self):
        """setup-qe subcommand parses correctly."""
        parser = build_parser()
        args = parser.parse_args(["setup-qe", "--elements", "Si,Fe"])
        assert args.command == "setup-qe"
        assert args.elements == "Si,Fe"

    def test_setup_qe_download_flag(self):
        parser = build_parser()
        args = parser.parse_args(["setup-qe", "--download"])
        assert args.download is True

    def test_vasp_execute_error(self, capsys):
        """VASP + --execute → error message."""
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import cmd_run
        import argparse

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.backend_name = "vasp"
        mock_result.structure_info = None
        mock_result.auto_detected = {}
        mock_result.output_dir = "/tmp"
        mock_result.files_generated = ["POSCAR"]

        args = argparse.Namespace(
            material=None, structure="POSCAR", backend="vasp", calc=None,
            accuracy="standard", set_values=None, output=None,
            workspace=None, project=None,
            pseudo_dir=None, no_validate=False, force=False,
            quiet=False, verbose=False, command="run",
            execute=True, nprocs=1, timeout=86400, mpi_command="mpirun",
        )
        with patch("shalom.direct_run.direct_run", return_value=mock_result), \
             patch("shalom.direct_run.DirectRunConfig"):
            rc = cmd_run(args)
        assert rc == 1
        captured = capsys.readouterr()
        assert "VASP" in captured.out and "not yet supported" in captured.out


# ---------------------------------------------------------------------------
# setup-qe tests
# ---------------------------------------------------------------------------


class TestCmdSetupQE:
    def _make_args(self, **overrides):
        import argparse
        defaults = dict(
            pseudo_dir=None, elements=None, download=False, command="setup-qe",
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_pw_x_found_and_pseudo_ok(self, capsys, tmp_path):
        """pw.x found + pseudo exists → return 0."""
        from unittest.mock import patch
        from shalom.__main__ import cmd_setup_qe

        pseudo_dir = tmp_path / "pseudo"
        pseudo_dir.mkdir()
        (pseudo_dir / "Si.pbe-n-rrkjus_psl.1.0.0.UPF").write_text("data")

        args = self._make_args(pseudo_dir=str(pseudo_dir), elements="Si")
        with patch("shutil.which", return_value="/usr/bin/pw.x"):
            rc = cmd_setup_qe(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "pw.x" in out
        assert "OK" in out

    def test_pw_x_not_found(self, capsys):
        """pw.x missing → return 1 with install instructions."""
        from unittest.mock import patch
        from shalom.__main__ import cmd_setup_qe

        args = self._make_args(elements="Si")
        with patch("shutil.which", return_value=None):
            rc = cmd_setup_qe(args)
        assert rc == 1
        out = capsys.readouterr().out
        assert "NOT FOUND" in out

    def test_missing_upf_reported(self, capsys, tmp_path):
        """UPF file missing → MISSING reported."""
        from unittest.mock import patch
        from shalom.__main__ import cmd_setup_qe

        pseudo_dir = tmp_path / "pseudo"
        pseudo_dir.mkdir()
        args = self._make_args(pseudo_dir=str(pseudo_dir), elements="Si")
        with patch("shutil.which", return_value="/usr/bin/pw.x"):
            rc = cmd_setup_qe(args)
        assert rc == 1
        out = capsys.readouterr().out
        assert "[MISSING]" in out

    def test_download_triggers_urllib(self, capsys, tmp_path):
        """--download triggers urllib download."""
        from unittest.mock import patch
        from shalom.__main__ import cmd_setup_qe

        pseudo_dir = tmp_path / "pseudo"
        pseudo_dir.mkdir()
        args = self._make_args(
            pseudo_dir=str(pseudo_dir), elements="Si", download=True,
        )
        with patch("shutil.which", return_value="/usr/bin/pw.x"), \
             patch("urllib.request.urlretrieve") as mock_dl:
            cmd_setup_qe(args)
        mock_dl.assert_called_once()
        out = capsys.readouterr().out
        assert "Downloading" in out

    def test_unknown_element_handled(self, capsys, tmp_path):
        """Unknown element gracefully reported."""
        from unittest.mock import patch
        from shalom.__main__ import cmd_setup_qe

        pseudo_dir = tmp_path / "pseudo"
        pseudo_dir.mkdir()
        args = self._make_args(pseudo_dir=str(pseudo_dir), elements="Xx")
        with patch("shutil.which", return_value="/usr/bin/pw.x"):
            cmd_setup_qe(args)
        out = capsys.readouterr().out
        assert "unknown" in out.lower()


# ---------------------------------------------------------------------------
# _detect_wsl_distros tests
# ---------------------------------------------------------------------------


class TestDetectWslDistros:
    def test_successful_utf16le_output(self):
        """Successful wsl --list --quiet → list of distro names."""
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import _detect_wsl_distros

        mock_result = MagicMock()
        mock_result.returncode = 0
        # UTF-16-LE encoding of "Ubuntu-22.04\nDebian\n"
        mock_result.stdout = "Ubuntu-22.04\nDebian\n".encode("utf-16-le")
        with patch("subprocess.run", return_value=mock_result):
            distros = _detect_wsl_distros()
        assert "Ubuntu-22.04" in distros
        assert "Debian" in distros

    def test_nonzero_returncode_returns_empty(self):
        """Non-zero returncode from wsl → empty list."""
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import _detect_wsl_distros

        mock_result = MagicMock()
        mock_result.returncode = 1
        with patch("subprocess.run", return_value=mock_result):
            distros = _detect_wsl_distros()
        assert distros == []

    def test_subprocess_exception_returns_empty(self):
        """subprocess.run raises Exception → empty list."""
        from unittest.mock import patch
        from shalom.__main__ import _detect_wsl_distros

        with patch("subprocess.run", side_effect=FileNotFoundError("wsl not found")):
            distros = _detect_wsl_distros()
        assert distros == []

    def test_utf8_fallback_on_decode_error(self):
        """Falls back to utf-8 when utf-16-le decode fails."""
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import _detect_wsl_distros

        mock_result = MagicMock()
        mock_result.returncode = 0
        # Plain utf-8 bytes — should be handled by fallback
        mock_result.stdout = b"Ubuntu-22.04\n"
        with patch("subprocess.run", return_value=mock_result):
            distros = _detect_wsl_distros()
        # Either returned a distro or empty (help text filtered)
        assert isinstance(distros, list)


# ---------------------------------------------------------------------------
# _print_install_guide_windows / _print_install_guide_linux tests
# ---------------------------------------------------------------------------


class TestPrintInstallGuideWindows:
    def test_no_distros_shows_step1(self, capsys):
        """No distros → Step 1 install instructions shown."""
        from shalom.__main__ import _print_install_guide_windows
        _print_install_guide_windows([])
        out = capsys.readouterr().out
        assert "Step 1" in out
        assert "Ubuntu" in out

    def test_with_ubuntu_distro(self, capsys):
        """With Ubuntu distro → 'WSL2 distro found' shown."""
        from shalom.__main__ import _print_install_guide_windows
        _print_install_guide_windows(["Ubuntu-22.04"])
        out = capsys.readouterr().out
        assert "WSL2 distro found" in out
        assert "Ubuntu-22.04" in out

    def test_non_ubuntu_distro_used(self, capsys):
        """Non-Ubuntu distro still used when no Ubuntu found."""
        from shalom.__main__ import _print_install_guide_windows
        _print_install_guide_windows(["Debian"])
        out = capsys.readouterr().out
        assert "Debian" in out


class TestPrintInstallGuideLinux:
    def test_prints_apt_option(self, capsys):
        """Linux guide shows apt install option."""
        from shalom.__main__ import _print_install_guide_linux
        _print_install_guide_linux()
        out = capsys.readouterr().out
        assert "apt" in out
        assert "quantum-espresso" in out

    def test_prints_conda_option(self, capsys):
        """Linux guide shows conda option."""
        from shalom.__main__ import _print_install_guide_linux
        _print_install_guide_linux()
        out = capsys.readouterr().out
        assert "conda" in out


# ---------------------------------------------------------------------------
# cmd_setup_qe — additional coverage tests
# ---------------------------------------------------------------------------


class TestCmdSetupQeAdditional:
    def _make_args(self, **overrides):
        import argparse
        defaults = dict(pseudo_dir=None, elements=None, download=False, command="setup-qe")
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_pw_x_not_found_linux_prints_guide(self, capsys):
        """Non-Windows + pw.x missing → linux guide printed."""
        from unittest.mock import patch
        from shalom.__main__ import cmd_setup_qe

        args = self._make_args(elements="Si")
        with patch("shutil.which", return_value=None), \
             patch("sys.platform", "linux"):
            cmd_setup_qe(args)
        out = capsys.readouterr().out
        assert "NOT FOUND" in out

    def test_pw_x_not_found_windows_with_wsl(self, capsys):
        """Windows + pw.x missing + wsl available → windows guide printed."""
        from unittest.mock import patch
        from shalom.__main__ import cmd_setup_qe

        args = self._make_args(elements="Si")
        with patch("shutil.which", side_effect=lambda x: "/wsl" if x == "wsl" else None), \
             patch("sys.platform", "win32"), \
             patch("shalom.__main__._detect_wsl_distros", return_value=[]):
            cmd_setup_qe(args)
        out = capsys.readouterr().out
        assert "NOT FOUND" in out

    def test_case_insensitive_upf_found(self, capsys, tmp_path):
        """UPF present with different case → found via iterdir fallback."""
        from unittest.mock import patch
        from shalom.__main__ import cmd_setup_qe
        from shalom.backends.qe_config import get_pseudo_filename

        pseudo_dir = tmp_path / "pseudo"
        pseudo_dir.mkdir()
        # Create file with different case (lowercase)
        upf = get_pseudo_filename("Si")
        (pseudo_dir / upf.lower()).write_text("data")

        args = self._make_args(pseudo_dir=str(pseudo_dir), elements="Si")
        with patch("shutil.which", return_value="/usr/bin/pw.x"):
            cmd_setup_qe(args)
        out = capsys.readouterr().out
        # The file was found via case-insensitive match → it's not in [MISSING]
        assert "[MISSING]" not in out

    def test_many_missing_upfs_shows_first_5(self, capsys, tmp_path):
        """More than 10 missing UPFs → shows 'first 5 of N' summary."""
        from unittest.mock import patch
        from shalom.__main__ import cmd_setup_qe

        pseudo_dir = tmp_path / "pseudo"
        pseudo_dir.mkdir()
        # Request 12 elements, none present → all missing
        elements = "Si,Fe,O,Cu,Al,Ca,Mg,Na,K,Ti,N,C"
        args = self._make_args(pseudo_dir=str(pseudo_dir), elements=elements)
        with patch("shutil.which", return_value="/usr/bin/pw.x"):
            cmd_setup_qe(args)
        out = capsys.readouterr().out
        assert "showing first 5" in out

    def test_download_failure_prints_failed(self, capsys, tmp_path):
        """urllib raises exception during download → FAILED printed."""
        from unittest.mock import patch
        from shalom.__main__ import cmd_setup_qe

        pseudo_dir = tmp_path / "pseudo"
        pseudo_dir.mkdir()
        args = self._make_args(pseudo_dir=str(pseudo_dir), elements="Si", download=True)
        with patch("shutil.which", return_value="/usr/bin/pw.x"), \
             patch("urllib.request.urlretrieve", side_effect=Exception("network error")):
            cmd_setup_qe(args)
        out = capsys.readouterr().out
        assert "FAILED" in out


# ---------------------------------------------------------------------------
# _execute_dft tests
# ---------------------------------------------------------------------------


class TestExecuteDft:
    def _make_args(self, **overrides):
        import argparse
        defaults = dict(nprocs=1, mpi_command="mpirun", timeout=3600)
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_prereq_errors_returns_1(self, tmp_path, capsys):
        """validate_prerequisites failure → return 1."""
        from unittest.mock import patch
        from shalom.__main__ import _execute_dft

        args = self._make_args()
        with patch("shalom.backends.runner.ExecutionRunner.validate_prerequisites",
                   return_value=["pw.x not found"]):
            rc = _execute_dft(str(tmp_path), args)
        assert rc == 1
        out = capsys.readouterr().out
        assert "pw.x not found" in out

    def test_converged_returns_0(self, tmp_path, capsys):
        """Converged DFT result → return 0."""
        from unittest.mock import patch
        from shalom.__main__ import _execute_dft
        from shalom.backends.base import DFTResult
        from shalom.backends.runner import ExecutionResult

        args = self._make_args()
        dft = DFTResult(energy=-10.0, forces_max=0.001, is_converged=True)
        exec_res = ExecutionResult(success=True, return_code=0, wall_time_seconds=5.0)

        with patch("shalom.backends.runner.ExecutionRunner.validate_prerequisites",
                   return_value=[]), \
             patch("shalom.backends.runner.execute_with_recovery",
                   return_value=(exec_res, dft, [])):
            rc = _execute_dft(str(tmp_path), args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "Converged" in out

    def test_failed_returns_1(self, tmp_path, capsys):
        """Non-converged result → return 1."""
        from unittest.mock import patch
        from shalom.__main__ import _execute_dft
        from shalom.backends.base import DFTResult
        from shalom.backends.runner import ExecutionResult

        args = self._make_args()
        dft = DFTResult(is_converged=False)
        exec_res = ExecutionResult(
            success=False, return_code=1, wall_time_seconds=2.0,
            error_message="SCF not converged",
        )

        with patch("shalom.backends.runner.ExecutionRunner.validate_prerequisites",
                   return_value=[]), \
             patch("shalom.backends.runner.execute_with_recovery",
                   return_value=(exec_res, dft, [])):
            rc = _execute_dft(str(tmp_path), args)
        assert rc == 1
        out = capsys.readouterr().out
        assert "FAILED" in out

    def test_corrections_printed(self, tmp_path, capsys):
        """History entries are printed when corrections applied."""
        from unittest.mock import patch
        from shalom.__main__ import _execute_dft
        from shalom.backends.base import DFTResult
        from shalom.backends.runner import ExecutionResult

        args = self._make_args()
        dft = DFTResult(energy=-5.0, is_converged=True)
        exec_res = ExecutionResult(success=True, return_code=0, wall_time_seconds=10.0)
        history = [{"error_type": "no_restart", "step": 1}]

        with patch("shalom.backends.runner.ExecutionRunner.validate_prerequisites",
                   return_value=[]), \
             patch("shalom.backends.runner.execute_with_recovery",
                   return_value=(exec_res, dft, history)):
            rc = _execute_dft(str(tmp_path), args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "Corrections" in out


# ---------------------------------------------------------------------------
# cmd_plot tests
# ---------------------------------------------------------------------------


class TestCmdPlot:
    def _make_args(self, calc_dir, **overrides):
        import argparse
        defaults = dict(
            calc_dir=calc_dir, bands=False, dos=False,
            fermi_from=None, output=None, title=None,
            emin=None, emax=None, command="plot",
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_no_bands_or_dos_returns_1(self, tmp_path, capsys):
        """Neither --bands nor --dos → return 1."""
        from shalom.__main__ import cmd_plot
        args = self._make_args(str(tmp_path))
        rc = cmd_plot(args)
        assert rc == 1
        assert "specify at least one" in capsys.readouterr().out

    def test_nonexistent_dir_returns_1(self, tmp_path, capsys):
        """Non-existent calc_dir → return 1."""
        from shalom.__main__ import cmd_plot
        args = self._make_args(str(tmp_path / "nonexistent"), bands=True)
        rc = cmd_plot(args)
        assert rc == 1
        assert "not found" in capsys.readouterr().out

    def test_bands_xml_not_found(self, tmp_path, capsys):
        """--bands but no XML → error message, return 1."""
        from unittest.mock import patch
        from shalom.__main__ import cmd_plot

        args = self._make_args(str(tmp_path), bands=True)
        with patch("shalom.backends.qe_parser.find_xml_path", return_value=None):
            rc = cmd_plot(args)
        assert rc == 1
        out = capsys.readouterr().out
        assert "not found" in out

    def test_bands_success(self, tmp_path, capsys):
        """--bands with XML → plot saved."""
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import cmd_plot
        from shalom.backends.base import BandStructureData
        import numpy as np

        fake_bs = BandStructureData(
            eigenvalues=np.zeros((3, 5)),
            kpoint_coords=np.zeros((3, 3)),
            kpath_distances=np.linspace(0, 1, 3),
        )

        mock_plotter = MagicMock()
        args = self._make_args(str(tmp_path), bands=True)
        with patch("shalom.backends.qe_parser.find_xml_path", return_value="/x.xml"), \
             patch("shalom.backends.qe_parser.parse_xml_bands", return_value=fake_bs), \
             patch("shalom.backends.qe_parser.extract_fermi_energy", return_value=5.0), \
             patch("shalom.plotting.band_plot.BandStructurePlotter", return_value=mock_plotter):
            rc = cmd_plot(args)
        assert rc == 0
        assert "saved" in capsys.readouterr().out

    def test_bands_import_error(self, tmp_path, capsys):
        """matplotlib not installed for bands → error + return 1."""
        from unittest.mock import patch
        from shalom.__main__ import cmd_plot
        from shalom.backends.base import BandStructureData
        import numpy as np

        fake_bs = BandStructureData(
            eigenvalues=np.zeros((3, 5)),
            kpoint_coords=np.zeros((3, 3)),
            kpath_distances=np.linspace(0, 1, 3),
        )

        args = self._make_args(str(tmp_path), bands=True)
        with patch("shalom.backends.qe_parser.find_xml_path", return_value="/x.xml"), \
             patch("shalom.backends.qe_parser.parse_xml_bands", return_value=fake_bs), \
             patch("shalom.backends.qe_parser.extract_fermi_energy", return_value=None), \
             patch("shalom.plotting.band_plot.BandStructurePlotter",
                   side_effect=ImportError("no matplotlib")):
            rc = cmd_plot(args)
        assert rc == 1
        out = capsys.readouterr().out
        assert "matplotlib" in out

    def test_dos_file_not_found(self, tmp_path, capsys):
        """--dos but pwscf.dos missing → error + return 1."""
        from unittest.mock import patch
        from shalom.__main__ import cmd_plot

        args = self._make_args(str(tmp_path), dos=True)
        with patch("shalom.backends.qe_parser.extract_fermi_energy", return_value=None):
            rc = cmd_plot(args)
        assert rc == 1
        out = capsys.readouterr().out
        assert "pwscf.dos not found" in out

    def test_dos_success(self, tmp_path, capsys):
        """--dos with valid DOS file → plot saved."""
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import cmd_plot
        from shalom.backends.base import DOSData
        import numpy as np

        (tmp_path / "pwscf.dos").write_text("# DOS\n0.0 1.0 0.5\n")
        fake_dos = DOSData(
            energies=np.array([0.0]),
            dos=np.array([1.0]),
            integrated_dos=np.array([0.5]),
        )
        mock_plotter = MagicMock()

        args = self._make_args(str(tmp_path), dos=True)
        with patch("shalom.backends.qe_parser.extract_fermi_energy", return_value=5.0), \
             patch("shalom.backends.qe_parser.parse_dos_file", return_value=fake_dos), \
             patch("shalom.plotting.dos_plot.DOSPlotter", return_value=mock_plotter):
            rc = cmd_plot(args)
        assert rc == 0
        assert "saved" in capsys.readouterr().out

    def test_fermi_from_flag(self, tmp_path, capsys):
        """--fermi-from dir extracts Fermi energy from that dir."""
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import cmd_plot
        from shalom.backends.base import BandStructureData
        import numpy as np

        fermi_dir = tmp_path / "nscf"
        fermi_dir.mkdir()
        fake_bs = BandStructureData(
            eigenvalues=np.zeros((3, 5)),
            kpoint_coords=np.zeros((3, 3)),
            kpath_distances=np.linspace(0, 1, 3),
        )
        mock_plotter = MagicMock()

        args = self._make_args(str(tmp_path), bands=True, fermi_from=str(fermi_dir))
        extract_calls = []
        def fake_extract(path):
            extract_calls.append(path)
            return 5.5

        with patch("shalom.backends.qe_parser.find_xml_path", return_value="/x.xml"), \
             patch("shalom.backends.qe_parser.parse_xml_bands", return_value=fake_bs), \
             patch("shalom.backends.qe_parser.extract_fermi_energy", side_effect=fake_extract), \
             patch("shalom.plotting.band_plot.BandStructurePlotter", return_value=mock_plotter):
            rc = cmd_plot(args)
        assert rc == 0
        # Should have extracted Fermi from the fermi_from dir
        assert any(str(fermi_dir) in p for p in extract_calls)


# ---------------------------------------------------------------------------
# cmd_workflow tests
# ---------------------------------------------------------------------------


class TestCmdWorkflow:
    def _make_args(self, output_dir, **overrides):
        import argparse
        defaults = dict(
            material="Si", structure=None, output=output_dir,
            pseudo_dir=None, nprocs=1, mpi_command="mpirun",
            pw_command="pw.x", dos_command="dos.x",
            timeout=3600, accuracy="standard", skip_relax=True,
            is_2d=False, dos_emin=None, dos_emax=None,
            nscf_kgrid=None, resume=False, wsl=False,
            command="workflow", slurm=False,
            partition="compute", account=None, nodes=1,
            ntasks_per_node=None, walltime="24:00:00",
            qos=None, mem=None, module_loads=[], slurm_extras=[],
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_success_returns_0(self, tmp_path, capsys):
        """cmd_workflow success → return 0."""
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import cmd_workflow
        from ase.build import bulk

        si = bulk("Si", "diamond", a=5.43)
        mock_wf = MagicMock()
        mock_wf.run.return_value = {
            "bands_png": str(tmp_path / "bands.png"),
            "dos_png": str(tmp_path / "dos.png"),
            "fermi_energy": 5.1234,
        }

        args = self._make_args(str(tmp_path))
        with patch("shalom.__main__._load_atoms", return_value=si), \
             patch("shalom.workflows.standard.StandardWorkflow", return_value=mock_wf):
            rc = cmd_workflow(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "Workflow complete" in out
        assert "5.1234" in out

    def test_load_atoms_failure_returns_1(self, tmp_path):
        """_load_atoms returns None → return 1."""
        from unittest.mock import patch
        from shalom.__main__ import cmd_workflow

        args = self._make_args(str(tmp_path))
        with patch("shalom.__main__._load_atoms", return_value=None):
            rc = cmd_workflow(args)
        assert rc == 1

    def test_exception_returns_1(self, tmp_path, capsys):
        """StandardWorkflow.run raises → return 1 with error message."""
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import cmd_workflow
        from ase.build import bulk

        si = bulk("Si", "diamond", a=5.43)
        mock_wf = MagicMock()
        mock_wf.run.side_effect = RuntimeError("pw.x failed")

        args = self._make_args(str(tmp_path))
        with patch("shalom.__main__._load_atoms", return_value=si), \
             patch("shalom.workflows.standard.StandardWorkflow", return_value=mock_wf):
            rc = cmd_workflow(args)
        assert rc == 1
        assert "pw.x failed" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# cmd_converge tests
# ---------------------------------------------------------------------------


class TestCmdConverge:
    def _make_args(self, output_dir, **overrides):
        import argparse
        defaults = dict(
            material="Si", structure=None, output=output_dir,
            pseudo_dir=None, nprocs=1, timeout=3600,
            accuracy="standard", test="cutoff",
            values=None, kgrid=None, ecutwfc=None, threshold=0.01,
            command="converge",
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_cutoff_success_converged(self, tmp_path, capsys):
        """cmd_converge cutoff with converged result → return 0."""
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import cmd_converge
        from ase.build import bulk

        si = bulk("Si", "diamond", a=5.43)
        mock_conv = MagicMock()
        mock_result = MagicMock()
        mock_result.converged_value = 60.0
        mock_result.summary.return_value = "Converged at 60 Ry"
        mock_conv.run.return_value = mock_result
        mock_conv.plot.return_value = str(tmp_path / "conv.png")

        args = self._make_args(str(tmp_path), values="40,60,80")
        with patch("shalom.__main__._load_atoms", return_value=si), \
             patch("shalom.workflows.convergence.CutoffConvergence", return_value=mock_conv):
            rc = cmd_converge(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "Converged at 60 Ry" in out

    def test_kpoints_test(self, tmp_path, capsys):
        """cmd_converge kpoints → KpointConvergence used."""
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import cmd_converge
        from ase.build import bulk

        si = bulk("Si", "diamond", a=5.43)
        mock_conv = MagicMock()
        mock_result = MagicMock()
        mock_result.converged_value = 30.0
        mock_result.summary.return_value = "Converged"
        mock_conv.run.return_value = mock_result
        mock_conv.plot.return_value = None

        args = self._make_args(str(tmp_path), test="kpoints", values="20,30,40")
        with patch("shalom.__main__._load_atoms", return_value=si), \
             patch("shalom.workflows.convergence.KpointConvergence", return_value=mock_conv):
            rc = cmd_converge(args)
        assert rc == 0

    def test_invalid_values_returns_1(self, tmp_path, capsys):
        """Non-numeric --values → return 1 with error."""
        from unittest.mock import patch
        from shalom.__main__ import cmd_converge
        from ase.build import bulk

        si = bulk("Si", "diamond", a=5.43)
        args = self._make_args(str(tmp_path), values="abc,xyz")
        with patch("shalom.__main__._load_atoms", return_value=si):
            rc = cmd_converge(args)
        assert rc == 1
        assert "invalid --values" in capsys.readouterr().out

    def test_invalid_kgrid_returns_1(self, tmp_path, capsys):
        """Invalid --kgrid format → return 1."""
        from unittest.mock import patch
        from shalom.__main__ import cmd_converge
        from ase.build import bulk

        si = bulk("Si", "diamond", a=5.43)
        args = self._make_args(str(tmp_path), values="40,60", kgrid="4x4")
        with patch("shalom.__main__._load_atoms", return_value=si):
            rc = cmd_converge(args)
        assert rc == 1
        assert "invalid --kgrid" in capsys.readouterr().out

    def test_default_cutoff_values(self, tmp_path, capsys):
        """No --values for cutoff → default 30,40,50,60,80 printed."""
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import cmd_converge
        from ase.build import bulk

        si = bulk("Si", "diamond", a=5.43)
        mock_conv = MagicMock()
        mock_result = MagicMock()
        mock_result.converged_value = None
        mock_result.summary.return_value = "not converged"
        mock_conv.run.return_value = mock_result
        mock_conv.plot.return_value = None

        args = self._make_args(str(tmp_path), values=None)
        with patch("shalom.__main__._load_atoms", return_value=si), \
             patch("shalom.workflows.convergence.CutoffConvergence", return_value=mock_conv):
            rc = cmd_converge(args)
        out = capsys.readouterr().out
        assert "default ecutwfc values" in out
        # converged_value is None → return 1
        assert rc == 1

    def test_default_kpoints_values(self, tmp_path, capsys):
        """No --values for kpoints → default 20,30,40,50 printed."""
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import cmd_converge
        from ase.build import bulk

        si = bulk("Si", "diamond", a=5.43)
        mock_conv = MagicMock()
        mock_result = MagicMock()
        mock_result.converged_value = 30.0
        mock_result.summary.return_value = "converged"
        mock_conv.run.return_value = mock_result
        mock_conv.plot.return_value = None

        args = self._make_args(str(tmp_path), test="kpoints", values=None)
        with patch("shalom.__main__._load_atoms", return_value=si), \
             patch("shalom.workflows.convergence.KpointConvergence", return_value=mock_conv):
            cmd_converge(args)
        out = capsys.readouterr().out
        assert "default k-point" in out

    def test_load_atoms_failure(self, tmp_path):
        """_load_atoms returns None → return 1."""
        from unittest.mock import patch
        from shalom.__main__ import cmd_converge

        args = self._make_args(str(tmp_path))
        with patch("shalom.__main__._load_atoms", return_value=None):
            rc = cmd_converge(args)
        assert rc == 1

    def test_exception_returns_1(self, tmp_path, capsys):
        """Convergence run raises → return 1."""
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import cmd_converge
        from ase.build import bulk

        si = bulk("Si", "diamond", a=5.43)
        mock_conv = MagicMock()
        mock_conv.run.side_effect = RuntimeError("DFT failed")

        args = self._make_args(str(tmp_path), values="40,60")
        with patch("shalom.__main__._load_atoms", return_value=si), \
             patch("shalom.workflows.convergence.CutoffConvergence", return_value=mock_conv):
            rc = cmd_converge(args)
        assert rc == 1
        assert "DFT failed" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _load_atoms tests
# ---------------------------------------------------------------------------


class TestLoadAtoms:
    def _make_args(self, **overrides):
        import argparse
        defaults = dict(structure=None, material=None)
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_from_structure_file(self, tmp_path):
        """args.structure → ase_read called."""
        from unittest.mock import patch
        from shalom.__main__ import _load_atoms
        from ase.build import bulk

        si = bulk("Si", "diamond", a=5.43)
        args = self._make_args(structure="Si.cif")
        with patch("ase.io.read", return_value=si) as mock_read:
            result = _load_atoms(args)
        assert result is si
        mock_read.assert_called_once_with("Si.cif")

    def test_structure_file_error_returns_none(self, tmp_path, capsys):
        """ase_read raises → returns None with error message."""
        from unittest.mock import patch
        from shalom.__main__ import _load_atoms

        args = self._make_args(structure="bad.cif")
        with patch("ase.io.read", side_effect=Exception("cannot read")):
            result = _load_atoms(args)
        assert result is None
        assert "cannot read structure" in capsys.readouterr().out

    def test_from_mp_material(self):
        """args.material -> fetch_structure returns atoms."""
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import _load_atoms
        from ase.build import bulk

        si = bulk("Si", "diamond", a=5.43)
        mock_result = MagicMock()
        mock_result.atoms = si
        args = self._make_args(material="mp-149")
        with patch("shalom.mp_client.fetch_structure", return_value=mock_result):
            result = _load_atoms(args)
        assert result is si

    def test_mp_error_fallback_to_ase_bulk(self, capsys):
        """MP fetch_structure raises, ASE bulk fallback succeeds."""
        from unittest.mock import patch
        from shalom.__main__ import _load_atoms
        from ase.build import bulk

        si = bulk("Si", "diamond", a=5.43)
        args = self._make_args(material="Si")
        with patch("shalom.mp_client.fetch_structure",
                   side_effect=EnvironmentError("MP_API_KEY not set")), \
             patch("ase.build.bulk", return_value=si):
            result = _load_atoms(args)
        assert result is si
        out = capsys.readouterr().out
        assert "MP unavailable" in out

    def test_mp_and_bulk_both_fail_returns_none(self, capsys):
        """Both MP and bulk fail -> None with error."""
        from unittest.mock import patch
        from shalom.__main__ import _load_atoms

        args = self._make_args(material="BadFormula")
        with patch("shalom.mp_client.fetch_structure",
                   side_effect=ValueError("Not found")), \
             patch("ase.build.bulk", side_effect=Exception("bulk error")):
            result = _load_atoms(args)
        assert result is None
        assert "cannot resolve" in capsys.readouterr().out

    def test_no_material_or_structure_returns_none(self, capsys):
        """Neither material nor structure → None + error."""
        from shalom.__main__ import _load_atoms
        args = self._make_args()
        result = _load_atoms(args)
        assert result is None
        assert "provide a material" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# main() dispatcher tests
# ---------------------------------------------------------------------------


class TestMainDispatcher:
    def test_run_subcommand_dispatched(self):
        """main() with 'run' subcommand calls cmd_run."""
        import sys
        from unittest.mock import patch
        from shalom.__main__ import main

        with patch.object(sys, "argv", ["shalom", "run", "--structure", "POSCAR"]), \
             patch("shalom.__main__.cmd_run", return_value=0) as mock_run, \
             pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        mock_run.assert_called_once()

    def test_plot_subcommand_dispatched(self, tmp_path):
        """main() with 'plot' subcommand calls cmd_plot."""
        import sys
        from unittest.mock import patch
        from shalom.__main__ import main

        with patch.object(sys, "argv", ["shalom", "plot", str(tmp_path), "--bands"]), \
             patch("shalom.__main__.cmd_plot", return_value=0) as mock_plot, \
             pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        mock_plot.assert_called_once()

    def test_workflow_subcommand_dispatched(self, tmp_path):
        """main() with 'workflow' subcommand calls cmd_workflow."""
        import sys
        from unittest.mock import patch
        from shalom.__main__ import main

        with patch.object(sys, "argv", ["shalom", "workflow", "Si", "-o", str(tmp_path)]), \
             patch("shalom.__main__.cmd_workflow", return_value=0) as mock_wf, \
             pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        mock_wf.assert_called_once()

    def test_converge_subcommand_dispatched(self, tmp_path):
        """main() with 'converge' subcommand calls cmd_converge."""
        import sys
        from unittest.mock import patch
        from shalom.__main__ import main

        with patch.object(sys, "argv", ["shalom", "converge", "Si",
                                         "--test", "cutoff", "-o", str(tmp_path)]), \
             patch("shalom.__main__.cmd_converge", return_value=0) as mock_conv, \
             pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        mock_conv.assert_called_once()

    def test_setup_qe_subcommand_dispatched(self):
        """main() with 'setup-qe' subcommand calls cmd_setup_qe."""
        import sys
        from unittest.mock import patch
        from shalom.__main__ import main

        with patch.object(sys, "argv", ["shalom", "setup-qe"]), \
             patch("shalom.__main__.cmd_setup_qe", return_value=0) as mock_setup, \
             pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        mock_setup.assert_called_once()

    def test_quiet_flag_sets_info_logging(self):
        """main() without quiet flag sets INFO logging."""
        import sys
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import main

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "test"

        with patch.object(sys, "argv", ["shalom", "run", "--structure", "x.cif"]), \
             patch("shalom.direct_run.direct_run", return_value=mock_result), \
             patch("shalom.direct_run.DirectRunConfig"), \
             patch("logging.basicConfig") as mock_lc, \
             pytest.raises(SystemExit):
            main()
        # Should have called basicConfig (for INFO level since no --quiet)
        assert mock_lc.called


# ---------------------------------------------------------------------------
# cmd_run additional coverage (non-MP source)
# ---------------------------------------------------------------------------


class TestCmdRunNonMPSource:
    def test_non_mp_source_printed(self, tmp_path, capsys):
        """cmd_run with non-MP source_info shows 'Source:' line."""
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import cmd_run
        import argparse

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.backend_name = "qe"
        mock_result.structure_info = {"source": "ase_builder"}  # no mp_id key
        mock_result.auto_detected = {}
        mock_result.output_dir = str(tmp_path)
        mock_result.files_generated = ["pw.in"]

        args = argparse.Namespace(
            material="Si", structure=None, backend="qe", calc=None,
            accuracy="standard", set_values=None, output=None,
            workspace=None, project=None,
            pseudo_dir=None, no_validate=False, force=False,
            quiet=False, verbose=False, command="run",
            execute=False, nprocs=1, timeout=86400, mpi_command="mpirun",
        )
        with patch("shalom.direct_run.direct_run", return_value=mock_result), \
             patch("shalom.direct_run.DirectRunConfig"):
            rc = cmd_run(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "Source:" in out
        assert "ase_builder" in out


# ---------------------------------------------------------------------------
# CLI new args tests (--pw-command, --dos-command, --nscf-kgrid, --resume,
# --combined)
# ---------------------------------------------------------------------------


class TestCLINewArgs:
    """Test new CLI argument parsing for recent feature additions."""

    # -- run parser: --pw-command, --dos-command --------------------------

    def test_run_pw_command_default(self):
        """run --pw-command defaults to 'pw.x'."""
        parser = build_parser()
        args = parser.parse_args(["run", "mp-19717"])
        assert args.pw_command == "pw.x"

    def test_run_pw_command_custom(self):
        """run --pw-command accepts custom executable."""
        parser = build_parser()
        args = parser.parse_args(["run", "mp-19717", "--pw-command", "/opt/qe/bin/pw.x"])
        assert args.pw_command == "/opt/qe/bin/pw.x"

    def test_run_dos_command_default(self):
        """run --dos-command defaults to 'dos.x'."""
        parser = build_parser()
        args = parser.parse_args(["run", "mp-19717"])
        assert args.dos_command == "dos.x"

    def test_run_dos_command_custom(self):
        """run --dos-command accepts custom executable."""
        parser = build_parser()
        args = parser.parse_args(["run", "mp-19717", "--dos-command", "/opt/qe/bin/dos.x"])
        assert args.dos_command == "/opt/qe/bin/dos.x"

    # -- workflow parser: --pw-command, --dos-command ----------------------

    def test_workflow_pw_command_default(self):
        """workflow --pw-command defaults to 'pw.x'."""
        parser = build_parser()
        args = parser.parse_args(["workflow", "Si", "-o", "/tmp/wf"])
        assert args.pw_command == "pw.x"

    def test_workflow_pw_command_custom(self):
        """workflow --pw-command accepts custom executable."""
        parser = build_parser()
        args = parser.parse_args([
            "workflow", "Si", "-o", "/tmp/wf",
            "--pw-command", "/usr/local/bin/pw.x",
        ])
        assert args.pw_command == "/usr/local/bin/pw.x"

    def test_workflow_dos_command_default(self):
        """workflow --dos-command defaults to 'dos.x'."""
        parser = build_parser()
        args = parser.parse_args(["workflow", "Si", "-o", "/tmp/wf"])
        assert args.dos_command == "dos.x"

    def test_workflow_dos_command_custom(self):
        """workflow --dos-command accepts custom executable."""
        parser = build_parser()
        args = parser.parse_args([
            "workflow", "Si", "-o", "/tmp/wf",
            "--dos-command", "/usr/local/bin/dos.x",
        ])
        assert args.dos_command == "/usr/local/bin/dos.x"

    # -- workflow parser: --nscf-kgrid ------------------------------------

    def test_workflow_nscf_kgrid_default(self):
        """workflow --nscf-kgrid defaults to None."""
        parser = build_parser()
        args = parser.parse_args(["workflow", "Si", "-o", "/tmp/wf"])
        assert args.nscf_kgrid is None

    def test_workflow_nscf_kgrid_parsed(self):
        """workflow --nscf-kgrid accepts NxNyNz format."""
        parser = build_parser()
        args = parser.parse_args([
            "workflow", "Si", "-o", "/tmp/wf", "--nscf-kgrid", "6x6x6",
        ])
        assert args.nscf_kgrid == "6x6x6"

    def test_workflow_nscf_kgrid_asymmetric(self):
        """workflow --nscf-kgrid accepts asymmetric grid like 8x8x4."""
        parser = build_parser()
        args = parser.parse_args([
            "workflow", "Si", "-o", "/tmp/wf", "--nscf-kgrid", "8x8x4",
        ])
        assert args.nscf_kgrid == "8x8x4"

    # -- workflow parser: --resume ----------------------------------------

    def test_workflow_resume_default_false(self):
        """workflow --resume defaults to False."""
        parser = build_parser()
        args = parser.parse_args(["workflow", "Si", "-o", "/tmp/wf"])
        assert args.resume is False

    def test_workflow_resume_flag(self):
        """workflow --resume sets True."""
        parser = build_parser()
        args = parser.parse_args([
            "workflow", "Si", "-o", "/tmp/wf", "--resume",
        ])
        assert args.resume is True

    # -- plot parser: --combined ------------------------------------------

    def test_plot_combined_default_false(self):
        """plot --combined defaults to False."""
        parser = build_parser()
        args = parser.parse_args(["plot", "/tmp/calc", "--bands"])
        assert args.combined is False

    def test_plot_combined_flag(self):
        """plot --combined sets True."""
        parser = build_parser()
        args = parser.parse_args(["plot", "/tmp/calc", "--combined"])
        assert args.combined is True

    def test_plot_bands_dir_flag(self):
        """plot --bands-dir accepts a directory path."""
        parser = build_parser()
        args = parser.parse_args([
            "plot", "/tmp/nscf", "--combined", "--bands-dir", "/tmp/bands",
        ])
        assert args.bands_dir == "/tmp/bands"

    def test_plot_bands_dir_default_none(self):
        """plot --bands-dir defaults to None."""
        parser = build_parser()
        args = parser.parse_args(["plot", "/tmp/calc", "--bands"])
        assert args.bands_dir is None

    # -- cmd_workflow passes pw_command/dos_command to StandardWorkflow ----

    def test_cmd_workflow_passes_pw_command(self, tmp_path):
        """cmd_workflow forwards --pw-command to StandardWorkflow."""
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import cmd_workflow
        from ase.build import bulk
        import argparse

        si = bulk("Si", "diamond", a=5.43)
        mock_wf_cls = MagicMock()
        mock_wf_cls.return_value.run.return_value = {
            "bands_png": None, "dos_png": None,
            "fermi_energy": None, "failed_step": None,
            "step_results": [], "completed_steps": [],
        }

        args = argparse.Namespace(
            material="Si", structure=None, output=str(tmp_path),
            pseudo_dir=None, nprocs=1, mpi_command="mpirun",
            pw_command="/custom/pw.x", dos_command="/custom/dos.x",
            timeout=3600, accuracy="standard", skip_relax=True,
            is_2d=False, dos_emin=-20.0, dos_emax=20.0,
            nscf_kgrid=None, resume=False, wsl=False,
            command="workflow", slurm=False,
            partition="compute", account=None, nodes=1,
            ntasks_per_node=None, walltime="24:00:00",
            qos=None, mem=None, module_loads=[], slurm_extras=[],
        )
        with patch("shalom.__main__._load_atoms", return_value=si), \
             patch("shalom.workflows.standard.StandardWorkflow", mock_wf_cls):
            cmd_workflow(args)

        call_kwargs = mock_wf_cls.call_args.kwargs
        assert call_kwargs["pw_executable"] == "/custom/pw.x"
        assert call_kwargs["dos_executable"] == "/custom/dos.x"

    # -- cmd_workflow passes resume flag -----------------------------------

    def test_cmd_workflow_passes_resume(self, tmp_path):
        """cmd_workflow forwards --resume to StandardWorkflow."""
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import cmd_workflow
        from ase.build import bulk
        import argparse

        si = bulk("Si", "diamond", a=5.43)
        mock_wf_cls = MagicMock()
        mock_wf_cls.return_value.run.return_value = {
            "bands_png": None, "dos_png": None,
            "fermi_energy": None, "failed_step": None,
            "step_results": [], "completed_steps": [],
        }

        args = argparse.Namespace(
            material="Si", structure=None, output=str(tmp_path),
            pseudo_dir=None, nprocs=1, mpi_command="mpirun",
            pw_command="pw.x", dos_command="dos.x",
            timeout=3600, accuracy="standard", skip_relax=True,
            is_2d=False, dos_emin=-20.0, dos_emax=20.0,
            nscf_kgrid=None, resume=True, wsl=False,
            command="workflow", slurm=False,
            partition="compute", account=None, nodes=1,
            ntasks_per_node=None, walltime="24:00:00",
            qos=None, mem=None, module_loads=[], slurm_extras=[],
        )
        with patch("shalom.__main__._load_atoms", return_value=si), \
             patch("shalom.workflows.standard.StandardWorkflow", mock_wf_cls):
            cmd_workflow(args)

        call_kwargs = mock_wf_cls.call_args.kwargs
        assert call_kwargs["resume"] is True

    # -- cmd_workflow passes nscf_kgrid -----------------------------------

    def test_cmd_workflow_passes_nscf_kgrid(self, tmp_path):
        """cmd_workflow parses --nscf-kgrid and passes nscf_kmesh."""
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import cmd_workflow
        from ase.build import bulk
        import argparse

        si = bulk("Si", "diamond", a=5.43)
        mock_wf_cls = MagicMock()
        mock_wf_cls.return_value.run.return_value = {
            "bands_png": None, "dos_png": None,
            "fermi_energy": None, "failed_step": None,
            "step_results": [], "completed_steps": [],
        }

        args = argparse.Namespace(
            material="Si", structure=None, output=str(tmp_path),
            pseudo_dir=None, nprocs=1, mpi_command="mpirun",
            pw_command="pw.x", dos_command="dos.x",
            timeout=3600, accuracy="standard", skip_relax=True,
            is_2d=False, dos_emin=-20.0, dos_emax=20.0,
            nscf_kgrid="6x6x6", resume=False, wsl=False,
            command="workflow", slurm=False,
            partition="compute", account=None, nodes=1,
            ntasks_per_node=None, walltime="24:00:00",
            qos=None, mem=None, module_loads=[], slurm_extras=[],
        )
        with patch("shalom.__main__._load_atoms", return_value=si), \
             patch("shalom.workflows.standard.StandardWorkflow", mock_wf_cls):
            cmd_workflow(args)

        call_kwargs = mock_wf_cls.call_args.kwargs
        assert call_kwargs["nscf_kmesh"] == [6, 6, 6]

    # -- cmd_workflow invalid nscf-kgrid returns 1 -------------------------

    def test_cmd_workflow_invalid_nscf_kgrid(self, tmp_path, capsys):
        """Invalid --nscf-kgrid format returns 1 with error."""
        from unittest.mock import patch
        from shalom.__main__ import cmd_workflow
        from ase.build import bulk
        import argparse

        si = bulk("Si", "diamond", a=5.43)

        args = argparse.Namespace(
            material="Si", structure=None, output=str(tmp_path),
            pseudo_dir=None, nprocs=1, mpi_command="mpirun",
            pw_command="pw.x", dos_command="dos.x",
            timeout=3600, accuracy="standard", skip_relax=True,
            is_2d=False, dos_emin=-20.0, dos_emax=20.0,
            nscf_kgrid="bad_value", resume=False, wsl=False,
            command="workflow", slurm=False,
            partition="compute", account=None, nodes=1,
            ntasks_per_node=None, walltime="24:00:00",
            qos=None, mem=None, module_loads=[], slurm_extras=[],
        )
        with patch("shalom.__main__._load_atoms", return_value=si):
            rc = cmd_workflow(args)
        assert rc == 1
        assert "invalid --nscf-kgrid" in capsys.readouterr().out

    # -- cmd_plot --combined dispatched ------------------------------------

    def test_cmd_plot_combined_dispatched(self, tmp_path, capsys):
        """cmd_plot with --combined calls CombinedPlotter."""
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import cmd_plot
        from shalom.backends.base import BandStructureData, DOSData
        import numpy as np
        import argparse

        fake_bs = BandStructureData(
            eigenvalues=np.zeros((5, 3)),
            kpoint_coords=np.zeros((5, 3)),
            kpath_distances=np.linspace(0, 1, 5),
        )
        fake_dos = DOSData(
            energies=np.array([0.0]),
            dos=np.array([1.0]),
            integrated_dos=np.array([0.5]),
        )

        (tmp_path / "pwscf.dos").write_text("# DOS\n0.0 1.0 0.5\n")
        mock_plotter = MagicMock()

        args = argparse.Namespace(
            calc_dir=str(tmp_path), bands=False, dos=False, combined=True,
            bands_dir=None, fermi_from=None, output=None,
            title=None, emin=-6.0, emax=6.0, command="plot",
        )
        with patch("shalom.backends.qe_parser.find_xml_path", return_value="/x.xml"), \
             patch("shalom.backends.qe_parser.parse_xml_bands", return_value=fake_bs), \
             patch("shalom.backends.qe_parser.parse_dos_file", return_value=fake_dos), \
             patch("shalom.backends.qe_parser.extract_fermi_energy", return_value=5.0), \
             patch("shalom.plotting.combined_plot.CombinedPlotter", return_value=mock_plotter):
            rc = cmd_plot(args)
        assert rc == 0
        mock_plotter.plot.assert_called_once()
        out = capsys.readouterr().out
        assert "Combined plot saved" in out


# ---------------------------------------------------------------------------
# setup-qe pseudo_dir creation on --download
# ---------------------------------------------------------------------------


class TestCmdSetupQeDownloadCreatesPseudoDir:
    """Test that setup-qe --download creates pseudo_dir when missing."""

    def _make_args(self, **overrides):
        import argparse
        defaults = dict(pseudo_dir=None, elements=None, download=False, command="setup-qe")
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_download_creates_missing_pseudo_dir(self, tmp_path, capsys):
        """--download with non-existent pseudo_dir creates the directory."""
        from unittest.mock import patch
        from shalom.__main__ import cmd_setup_qe

        pseudo_dir = tmp_path / "new_pseudo_dir"
        assert not pseudo_dir.exists()

        args = self._make_args(
            pseudo_dir=str(pseudo_dir), elements="Si", download=True,
        )
        with patch("shutil.which", return_value="/usr/bin/pw.x"), \
             patch("urllib.request.urlretrieve"):
            cmd_setup_qe(args)

        # Directory should now exist
        assert pseudo_dir.is_dir()

    def test_no_download_does_not_create_dir(self, tmp_path, capsys):
        """Without --download, missing pseudo_dir is NOT created."""
        from unittest.mock import patch
        from shalom.__main__ import cmd_setup_qe

        pseudo_dir = tmp_path / "nonexistent_pseudo"
        args = self._make_args(pseudo_dir=str(pseudo_dir), elements="Si")
        with patch("shutil.which", return_value="/usr/bin/pw.x"):
            cmd_setup_qe(args)

        assert not pseudo_dir.exists()

    def test_download_with_missing_dir_lists_all_as_missing(self, tmp_path, capsys):
        """When pseudo_dir doesn't exist, all requested elements are listed as missing."""
        from unittest.mock import patch
        from shalom.__main__ import cmd_setup_qe

        pseudo_dir = tmp_path / "no_such_dir"
        args = self._make_args(
            pseudo_dir=str(pseudo_dir), elements="Si,Fe", download=True,
        )
        with patch("shutil.which", return_value="/usr/bin/pw.x"), \
             patch("urllib.request.urlretrieve") as mock_urlretrieve:
            cmd_setup_qe(args)

        out = capsys.readouterr().out
        assert "Downloading" in out
        # urlretrieve should have been called exactly twice (Si + Fe)
        assert mock_urlretrieve.call_count == 2


# ---------------------------------------------------------------------------
# Additional CLI edge case tests
# ---------------------------------------------------------------------------


class TestCLIEdgeCases:
    """Test invalid inputs and error paths for new CLI arguments."""

    def _make_workflow_args(self, tmp_path, **overrides):
        import argparse
        defaults = dict(
            material="Si", structure=None, output=str(tmp_path),
            pseudo_dir=None, nprocs=1, mpi_command="mpirun",
            pw_command="pw.x", dos_command="dos.x",
            timeout=3600, accuracy="standard", skip_relax=True,
            is_2d=False, dos_emin=-20.0, dos_emax=20.0,
            nscf_kgrid=None, resume=False, wsl=False,
            command="workflow", slurm=False,
            partition="compute", account=None, nodes=1,
            ntasks_per_node=None, walltime="24:00:00",
            qos=None, mem=None, module_loads=[], slurm_extras=[],
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    @pytest.mark.parametrize("kgrid", ["6x6", "6x6x6x6", "0x6x6", "-1x6x6"])
    def test_nscf_kgrid_invalid_returns_1(self, tmp_path, capsys, kgrid):
        """Invalid --nscf-kgrid values return error code 1."""
        from unittest.mock import patch
        from shalom.__main__ import cmd_workflow
        from ase.build import bulk
        si = bulk("Si", "diamond", a=5.43)
        args = self._make_workflow_args(tmp_path, nscf_kgrid=kgrid)
        with patch("shalom.__main__._load_atoms", return_value=si):
            rc = cmd_workflow(args)
        assert rc == 1
        assert "invalid --nscf-kgrid" in capsys.readouterr().out

    def test_nscf_kgrid_comma_separated_format(self, tmp_path):
        """--nscf-kgrid with comma-separated format (6,6,6) is parsed."""
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import cmd_workflow
        from ase.build import bulk
        si = bulk("Si", "diamond", a=5.43)
        mock_wf_cls = MagicMock()
        mock_wf_cls.return_value.run.return_value = {
            "bands_png": None, "dos_png": None,
            "fermi_energy": None, "failed_step": None,
            "step_results": [], "completed_steps": [],
        }
        args = self._make_workflow_args(tmp_path, nscf_kgrid="8,8,4")
        with patch("shalom.__main__._load_atoms", return_value=si), \
             patch("shalom.workflows.standard.StandardWorkflow", mock_wf_cls):
            rc = cmd_workflow(args)
        assert rc == 0
        assert mock_wf_cls.call_args.kwargs["nscf_kmesh"] == [8, 8, 4]

    def test_combined_plot_missing_xml_returns_1(self, tmp_path, capsys):
        """--combined when no band XML found returns error."""
        from unittest.mock import patch
        from shalom.__main__ import cmd_plot
        import argparse

        (tmp_path / "pwscf.dos").write_text("# DOS\n0.0 1.0 0.5\n")
        args = argparse.Namespace(
            calc_dir=str(tmp_path), bands=False, dos=False, combined=True,
            bands_dir=None, fermi_from=None, output=None,
            title=None, emin=-6.0, emax=6.0, command="plot",
        )
        with patch("shalom.backends.qe_parser.find_xml_path", return_value=None):
            rc = cmd_plot(args)
        assert rc == 1
        out = capsys.readouterr().out
        assert "xml" in out.lower() or "band" in out.lower()

    def test_combined_plot_missing_dos_returns_1(self, tmp_path, capsys):
        """--combined when pwscf.dos is missing returns error."""
        from unittest.mock import patch
        from shalom.__main__ import cmd_plot
        from shalom.backends.base import BandStructureData
        import numpy as np
        import argparse

        fake_bs = BandStructureData(
            eigenvalues=np.zeros((5, 3)),
            kpoint_coords=np.zeros((5, 3)),
            kpath_distances=np.linspace(0, 1, 5),
        )
        # No pwscf.dos file in tmp_path
        args = argparse.Namespace(
            calc_dir=str(tmp_path), bands=False, dos=False, combined=True,
            bands_dir=None, fermi_from=None, output=None,
            title=None, emin=-6.0, emax=6.0, command="plot",
        )
        with patch("shalom.backends.qe_parser.find_xml_path", return_value="/x.xml"), \
             patch("shalom.backends.qe_parser.parse_xml_bands", return_value=fake_bs), \
             patch("shalom.backends.qe_parser.extract_fermi_energy", return_value=5.0):
            rc = cmd_plot(args)
        assert rc == 1
        out = capsys.readouterr().out
        assert "dos" in out.lower()

    def test_combined_plot_with_bands_dir(self, tmp_path, capsys):
        """--combined --bands-dir uses bands_dir for XML lookup."""
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import cmd_plot
        from shalom.backends.base import BandStructureData, DOSData
        import numpy as np
        import argparse

        bands_dir = tmp_path / "03_bands"
        bands_dir.mkdir()
        nscf_dir = tmp_path / "04_nscf"
        nscf_dir.mkdir()
        (nscf_dir / "pwscf.dos").write_text("# DOS\n0.0 1.0 0.5\n")

        fake_bs = BandStructureData(
            eigenvalues=np.zeros((5, 3)),
            kpoint_coords=np.zeros((5, 3)),
            kpath_distances=np.linspace(0, 1, 5),
        )
        fake_dos = DOSData(
            energies=np.array([0.0]),
            dos=np.array([1.0]),
            integrated_dos=np.array([0.5]),
        )
        mock_plotter = MagicMock()

        xml_calls = []
        def tracking_find_xml(path):
            xml_calls.append(path)
            return "/x.xml"

        args = argparse.Namespace(
            calc_dir=str(nscf_dir), bands=False, dos=False, combined=True,
            bands_dir=str(bands_dir), fermi_from=None, output=None,
            title=None, emin=-6.0, emax=6.0, command="plot",
        )
        with patch("shalom.backends.qe_parser.find_xml_path", side_effect=tracking_find_xml), \
             patch("shalom.backends.qe_parser.parse_xml_bands", return_value=fake_bs), \
             patch("shalom.backends.qe_parser.parse_dos_file", return_value=fake_dos), \
             patch("shalom.backends.qe_parser.extract_fermi_energy", return_value=5.0), \
             patch("shalom.plotting.combined_plot.CombinedPlotter", return_value=mock_plotter):
            rc = cmd_plot(args)

        assert rc == 0
        # find_xml_path should have been called with bands_dir, not nscf_dir
        assert any(str(bands_dir) in c for c in xml_calls)


# ---------------------------------------------------------------------------
# _parse_supercell tests
# ---------------------------------------------------------------------------


class TestParseSupercell:
    def test_nxnxn_format(self):
        from shalom.__main__ import _parse_supercell
        result = _parse_supercell("2x2x2")
        assert result == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]

    def test_json_format(self):
        from shalom.__main__ import _parse_supercell
        result = _parse_supercell("[[3,0,0],[0,3,0],[0,0,3]]")
        assert result == [[3, 0, 0], [0, 3, 0], [0, 0, 3]]

    def test_invalid_returns_none(self):
        from shalom.__main__ import _parse_supercell
        assert _parse_supercell("bad") is None

    def test_comma_format(self):
        from shalom.__main__ import _parse_supercell
        result = _parse_supercell("2,3,4")
        assert result == [[2, 0, 0], [0, 3, 0], [0, 0, 4]]


# ---------------------------------------------------------------------------
# _save_analysis_json tests
# ---------------------------------------------------------------------------


class TestSaveAnalysisJson:
    def test_save_to_output_dir(self, tmp_path, capsys):
        from shalom.__main__ import _save_analysis_json
        _save_analysis_json({"key": "val"}, output_dir=str(tmp_path), auto_name="test.json")
        out = capsys.readouterr().out
        assert "Saved" in out

    def test_save_to_explicit_path(self, tmp_path, capsys):
        from shalom.__main__ import _save_analysis_json
        path = str(tmp_path / "result.json")
        _save_analysis_json({"key": "val"}, save_json_path=path)
        out = capsys.readouterr().out
        assert "Results saved" in out

    def test_no_paths_no_output(self, capsys):
        from shalom.__main__ import _save_analysis_json
        _save_analysis_json({"key": "val"})
        out = capsys.readouterr().out
        assert out == ""


# ---------------------------------------------------------------------------
# _build_slurm_config tests
# ---------------------------------------------------------------------------


class TestBuildSlurmConfig:
    def test_no_slurm_returns_none(self):
        import argparse
        from shalom.__main__ import _build_slurm_config
        args = argparse.Namespace(slurm=False)
        assert _build_slurm_config(args) is None

    def test_slurm_returns_config(self):
        import argparse
        from shalom.__main__ import _build_slurm_config
        args = argparse.Namespace(
            slurm=True, partition="gpu", account="sci",
            nodes=2, ntasks_per_node=32, walltime="12:00:00",
            qos="high", mem="64G", module_loads=["qe/7.0"],
            slurm_extras=["--gres=gpu:4"], nprocs=4,
        )
        config = _build_slurm_config(args)
        assert config.partition == "gpu"
        assert config.nodes == 2
        assert config.ntasks_per_node == 32

    def test_slurm_ntasks_fallback_to_nprocs(self):
        import argparse
        from shalom.__main__ import _build_slurm_config
        args = argparse.Namespace(
            slurm=True, partition="compute", account=None,
            nodes=1, ntasks_per_node=None, walltime="24:00:00",
            qos=None, mem=None, module_loads=[], slurm_extras=[],
            nprocs=8,
        )
        config = _build_slurm_config(args)
        assert config.ntasks_per_node == 8


# ---------------------------------------------------------------------------
# cmd_run LAMMPS args forwarding
# ---------------------------------------------------------------------------


class TestCmdRunLAMMPS:
    def _make_lammps_args(self, **overrides):
        import argparse
        defaults = dict(
            material=None, structure="POSCAR", backend="lammps", calc="md",
            accuracy="standard", set_values=None, output=None,
            workspace=None, project=None,
            pseudo_dir=None, no_validate=False, force=False,
            quiet=True, verbose=False, command="run",
            execute=False, nprocs=1, timeout=86400, mpi_command="mpirun",
            pair_style=None, pair_coeff=[], md_ensemble="nvt",
            temperature=300.0, md_steps=None, timestep=None,
            potential_dir=None,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_lammps_args_forwarded(self, tmp_path):
        """LAMMPS CLI args forwarded into user_settings."""
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import cmd_run

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.backend_name = "lammps"
        mock_result.structure_info = None
        mock_result.auto_detected = {}
        mock_result.output_dir = str(tmp_path)
        mock_result.files_generated = ["in.lammps"]

        mock_config_cls = MagicMock()
        args = self._make_lammps_args(
            pair_style="eam/alloy",
            pair_coeff=["* * Fe.eam.alloy Fe"],
            temperature=500.0,
            md_steps=50000,
            timestep=0.5,
            potential_dir="/opt/lammps",
        )
        with patch("shalom.direct_run.direct_run", return_value=mock_result), \
             patch("shalom.direct_run.DirectRunConfig", mock_config_cls):
            rc = cmd_run(args)
        assert rc == 0
        settings = mock_config_cls.call_args.kwargs["user_settings"]
        assert settings["pair_style"] == "eam/alloy"
        assert settings["temperature"] == 500.0
        assert settings["nsteps"] == 50000
        assert settings["timestep"] == 0.5
        assert settings["potential_dir"] == "/opt/lammps"

    def test_lammps_success_prints_next_steps(self, tmp_path, capsys):
        """LAMMPS success prints LAMMPS-specific next steps."""
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import cmd_run

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.backend_name = "lammps"
        mock_result.structure_info = None
        mock_result.auto_detected = {}
        mock_result.output_dir = str(tmp_path)
        mock_result.files_generated = ["in.lammps"]

        args = self._make_lammps_args(quiet=False)
        with patch("shalom.direct_run.direct_run", return_value=mock_result), \
             patch("shalom.direct_run.DirectRunConfig"):
            cmd_run(args)
        out = capsys.readouterr().out
        assert "in.lammps" in out


# ---------------------------------------------------------------------------
# cmd_analyze dispatcher tests
# ---------------------------------------------------------------------------


class TestCmdAnalyze:
    def _make_args(self, analyze_type=None, **overrides):
        import argparse
        defaults = dict(analyze_type=analyze_type, save_json=None, command="analyze")
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_no_type_returns_1(self, capsys):
        from shalom.__main__ import cmd_analyze
        args = self._make_args(analyze_type=None)
        rc = cmd_analyze(args)
        assert rc == 1
        assert "specify an analysis type" in capsys.readouterr().out

    def test_unknown_type_returns_1(self, capsys):
        from shalom.__main__ import cmd_analyze
        args = self._make_args(analyze_type="unknown_type")
        rc = cmd_analyze(args)
        assert rc == 1
        assert "unknown analysis type" in capsys.readouterr().out

    def test_dispatches_to_elastic(self):
        from unittest.mock import patch
        from shalom.__main__ import cmd_analyze
        args = self._make_args(analyze_type="elastic")
        with patch("shalom.__main__._cmd_analyze_elastic", return_value=0) as mock_el:
            rc = cmd_analyze(args)
        assert rc == 0
        mock_el.assert_called_once()

    def test_dispatches_to_md(self):
        from unittest.mock import patch
        from shalom.__main__ import cmd_analyze
        args = self._make_args(analyze_type="md")
        with patch("shalom.__main__._cmd_analyze_md", return_value=0) as mock_md:
            rc = cmd_analyze(args)
        assert rc == 0
        mock_md.assert_called_once()


# ---------------------------------------------------------------------------
# _cmd_analyze_elastic tests
# ---------------------------------------------------------------------------


class TestCmdAnalyzeElastic:
    def _make_args(self, **overrides):
        import argparse
        defaults = dict(tensor=None, file=None, save_json=None)
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_not_available_returns_1(self, capsys):
        from unittest.mock import patch
        from shalom.__main__ import _cmd_analyze_elastic
        args = self._make_args()
        with patch("shalom.analysis.elastic.is_elastic_available", return_value=False):
            rc = _cmd_analyze_elastic(args)
        assert rc == 1
        assert "pymatgen" in capsys.readouterr().out

    def test_no_tensor_or_file_returns_1(self, capsys):
        from unittest.mock import patch
        from shalom.__main__ import _cmd_analyze_elastic
        args = self._make_args()
        with patch("shalom.analysis.elastic.is_elastic_available", return_value=True):
            rc = _cmd_analyze_elastic(args)
        assert rc == 1
        assert "provide --tensor" in capsys.readouterr().out

    def test_invalid_json_tensor_returns_1(self, capsys):
        from unittest.mock import patch
        from shalom.__main__ import _cmd_analyze_elastic
        args = self._make_args(tensor="{bad json")
        with patch("shalom.analysis.elastic.is_elastic_available", return_value=True):
            rc = _cmd_analyze_elastic(args)
        assert rc == 1
        assert "invalid JSON" in capsys.readouterr().out

    def test_file_read_error_returns_1(self, capsys, tmp_path):
        from unittest.mock import patch
        from shalom.__main__ import _cmd_analyze_elastic
        args = self._make_args(file=str(tmp_path / "nonexistent.json"))
        with patch("shalom.analysis.elastic.is_elastic_available", return_value=True):
            rc = _cmd_analyze_elastic(args)
        assert rc == 1
        assert "cannot read" in capsys.readouterr().out

    def test_success_from_tensor(self, capsys):
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import _cmd_analyze_elastic
        import json

        tensor = [[165.7, 63.9, 63.9, 0, 0, 0]] * 6
        mock_result = MagicMock()
        mock_result.bulk_modulus_vrh = 100.0
        mock_result.shear_modulus_vrh = 50.0
        mock_result.youngs_modulus = 130.0
        mock_result.poisson_ratio = 0.3
        mock_result.universal_anisotropy = 0.5
        mock_result.is_stable = True
        mock_result.stability_violations = []
        mock_result.to_dict.return_value = {}

        args = self._make_args(tensor=json.dumps(tensor))
        with patch("shalom.analysis.elastic.is_elastic_available", return_value=True), \
             patch("shalom.analysis.elastic.analyze_elastic_tensor",
                   return_value=mock_result):
            rc = _cmd_analyze_elastic(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "100.00" in out
        assert "STABLE" in out

    def test_unstable_result(self, capsys):
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import _cmd_analyze_elastic
        import json

        mock_result = MagicMock()
        mock_result.bulk_modulus_vrh = None
        mock_result.shear_modulus_vrh = None
        mock_result.youngs_modulus = None
        mock_result.poisson_ratio = None
        mock_result.universal_anisotropy = None
        mock_result.is_stable = False
        mock_result.stability_violations = ["C11 < 0"]
        mock_result.to_dict.return_value = {}

        args = self._make_args(tensor="[[1,2,3,4,5,6]]*6")
        # Use a valid tensor for the mock
        args.tensor = json.dumps([[1]*6]*6)
        with patch("shalom.analysis.elastic.is_elastic_available", return_value=True), \
             patch("shalom.analysis.elastic.analyze_elastic_tensor",
                   return_value=mock_result):
            rc = _cmd_analyze_elastic(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "UNSTABLE" in out
        assert "C11 < 0" in out

    def test_from_file(self, tmp_path, capsys):
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import _cmd_analyze_elastic
        import json

        tensor = [[165]*6]*6
        tensor_file = tmp_path / "tensor.json"
        tensor_file.write_text(json.dumps(tensor))

        mock_result = MagicMock()
        mock_result.bulk_modulus_vrh = 80.0
        mock_result.shear_modulus_vrh = 40.0
        mock_result.youngs_modulus = 100.0
        mock_result.poisson_ratio = 0.25
        mock_result.universal_anisotropy = 0.1
        mock_result.is_stable = True
        mock_result.stability_violations = []
        mock_result.to_dict.return_value = {}

        args = self._make_args(file=str(tensor_file))
        with patch("shalom.analysis.elastic.is_elastic_available", return_value=True), \
             patch("shalom.analysis.elastic.analyze_elastic_tensor",
                   return_value=mock_result):
            rc = _cmd_analyze_elastic(args)
        assert rc == 0

    def test_analysis_exception_returns_1(self, capsys):
        from unittest.mock import patch
        from shalom.__main__ import _cmd_analyze_elastic
        import json

        args = self._make_args(tensor=json.dumps([[1]*6]*6))
        with patch("shalom.analysis.elastic.is_elastic_available", return_value=True), \
             patch("shalom.analysis.elastic.analyze_elastic_tensor",
                   side_effect=ValueError("bad tensor")):
            rc = _cmd_analyze_elastic(args)
        assert rc == 1
        assert "bad tensor" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _cmd_analyze_electronic tests
# ---------------------------------------------------------------------------


class TestCmdAnalyzeElectronic:
    def _make_args(self, **overrides):
        import argparse
        defaults = dict(
            bands_xml=None, calc_dir=None, dos_file=None,
            fermi_energy=None, save_json=None,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_not_available_returns_1(self, capsys):
        from unittest.mock import patch
        from shalom.__main__ import _cmd_analyze_electronic
        args = self._make_args()
        with patch("shalom.analysis.electronic.is_electronic_available", return_value=False):
            rc = _cmd_analyze_electronic(args)
        assert rc == 1

    def test_no_xml_or_dir_returns_1(self, capsys):
        from unittest.mock import patch
        from shalom.__main__ import _cmd_analyze_electronic
        args = self._make_args()
        with patch("shalom.analysis.electronic.is_electronic_available", return_value=True):
            rc = _cmd_analyze_electronic(args)
        assert rc == 1
        assert "provide --bands-xml" in capsys.readouterr().out

    def test_calc_dir_no_xml_returns_1(self, tmp_path, capsys):
        from unittest.mock import patch
        from shalom.__main__ import _cmd_analyze_electronic
        args = self._make_args(calc_dir=str(tmp_path))
        with patch("shalom.analysis.electronic.is_electronic_available", return_value=True), \
             patch("shalom.backends.qe_parser.find_xml_path", return_value=None):
            rc = _cmd_analyze_electronic(args)
        assert rc == 1
        assert "not found" in capsys.readouterr().out

    def test_success_semiconductor(self, capsys):
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import _cmd_analyze_electronic

        mock_result = MagicMock()
        mock_result.is_metal = False
        mock_result.is_direct = True
        mock_result.bandgap_eV = 1.12
        mock_result.vbm_energy = -0.5
        mock_result.cbm_energy = 0.62
        mock_result.dos_at_fermi = None
        mock_result.effective_mass_electron = 0.26
        mock_result.effective_mass_hole = 0.49
        mock_result.n_occupied_bands = 4
        mock_result.to_dict.return_value = {}

        args = self._make_args(bands_xml="/path/to/xml")
        with patch("shalom.analysis.electronic.is_electronic_available", return_value=True), \
             patch("shalom.backends.qe_parser.parse_xml_bands"), \
             patch("shalom.backends.qe_parser.extract_fermi_energy", return_value=5.0), \
             patch("shalom.analysis.electronic.analyze_band_structure",
                   return_value=mock_result):
            rc = _cmd_analyze_electronic(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "1.12" in out
        assert "direct" in out

    def test_success_metal(self, capsys):
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import _cmd_analyze_electronic

        mock_result = MagicMock()
        mock_result.is_metal = True
        mock_result.dos_at_fermi = 2.5
        mock_result.effective_mass_electron = None
        mock_result.effective_mass_hole = None
        mock_result.n_occupied_bands = 6
        mock_result.to_dict.return_value = {}

        args = self._make_args(bands_xml="/path/to/xml")
        with patch("shalom.analysis.electronic.is_electronic_available", return_value=True), \
             patch("shalom.backends.qe_parser.parse_xml_bands"), \
             patch("shalom.backends.qe_parser.extract_fermi_energy", return_value=None), \
             patch("shalom.analysis.electronic.analyze_band_structure",
                   return_value=mock_result):
            rc = _cmd_analyze_electronic(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "Metal" in out
        assert "2.5" in out

    def test_exception_returns_1(self, capsys):
        from unittest.mock import patch
        from shalom.__main__ import _cmd_analyze_electronic
        args = self._make_args(bands_xml="/bad/xml")
        with patch("shalom.analysis.electronic.is_electronic_available", return_value=True), \
             patch("shalom.backends.qe_parser.parse_xml_bands",
                   side_effect=Exception("parse error")):
            rc = _cmd_analyze_electronic(args)
        assert rc == 1


# ---------------------------------------------------------------------------
# _cmd_analyze_xrd tests
# ---------------------------------------------------------------------------


class TestCmdAnalyzeXrd:
    def _make_args(self, **overrides):
        import argparse
        defaults = dict(
            structure="POSCAR", wavelength="CuKa",
            theta_min=0.0, theta_max=90.0,
            output=None, save_json=None,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_not_available_returns_1(self, capsys):
        from unittest.mock import patch
        from shalom.__main__ import _cmd_analyze_xrd
        args = self._make_args()
        with patch("shalom.analysis.xrd.is_xrd_available", return_value=False):
            rc = _cmd_analyze_xrd(args)
        assert rc == 1
        assert "pymatgen" in capsys.readouterr().out

    def test_structure_read_error_returns_1(self, capsys):
        from unittest.mock import patch
        from shalom.__main__ import _cmd_analyze_xrd
        args = self._make_args(structure="bad.cif")
        with patch("shalom.analysis.xrd.is_xrd_available", return_value=True), \
             patch("ase.io.read", side_effect=Exception("bad file")):
            rc = _cmd_analyze_xrd(args)
        assert rc == 1

    def test_success(self, capsys):
        import numpy as np
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import _cmd_analyze_xrd

        mock_result = MagicMock()
        mock_result.wavelength = "CuKa"
        mock_result.wavelength_angstrom = 1.5406
        mock_result.n_peaks = 5
        mock_result.two_theta = np.array([28.4, 47.3, 56.1, 69.1, 76.4])
        mock_result.intensities = np.array([100, 55, 30, 6, 11])
        mock_result.d_spacings = np.array([3.14, 1.92, 1.64, 1.36, 1.25])
        mock_result.hkl_indices = [(1, 1, 1), (2, 2, 0), (3, 1, 1), (4, 0, 0), (3, 3, 1)]
        mock_result.to_dict.return_value = {}

        mock_atoms = MagicMock()
        mock_atoms.get_chemical_formula.return_value = "Si2"

        args = self._make_args()
        with patch("shalom.analysis.xrd.is_xrd_available", return_value=True), \
             patch("ase.io.read", return_value=mock_atoms), \
             patch("shalom.analysis.xrd.calculate_xrd", return_value=mock_result):
            rc = _cmd_analyze_xrd(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "5" in out
        assert "CuKa" in out

    def test_with_output_plot(self, tmp_path, capsys):
        import numpy as np
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import _cmd_analyze_xrd

        mock_result = MagicMock()
        mock_result.wavelength = "CuKa"
        mock_result.wavelength_angstrom = 1.5406
        mock_result.n_peaks = 1
        mock_result.two_theta = np.array([28.4])
        mock_result.intensities = np.array([100])
        mock_result.d_spacings = np.array([3.14])
        mock_result.hkl_indices = [(1, 1, 1)]
        mock_result.to_dict.return_value = {}

        mock_atoms = MagicMock()
        mock_atoms.get_chemical_formula.return_value = "Si"
        mock_plotter = MagicMock()

        output = str(tmp_path / "xrd.png")
        args = self._make_args(output=output)
        with patch("shalom.analysis.xrd.is_xrd_available", return_value=True), \
             patch("ase.io.read", return_value=mock_atoms), \
             patch("shalom.analysis.xrd.calculate_xrd", return_value=mock_result), \
             patch("shalom.plotting.xrd_plot.XRDPlotter", return_value=mock_plotter):
            rc = _cmd_analyze_xrd(args)
        assert rc == 0
        assert "Plot saved" in capsys.readouterr().out

    def test_exception_returns_1(self, capsys):
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import _cmd_analyze_xrd
        mock_atoms = MagicMock()
        args = self._make_args()
        with patch("shalom.analysis.xrd.is_xrd_available", return_value=True), \
             patch("ase.io.read", return_value=mock_atoms), \
             patch("shalom.analysis.xrd.calculate_xrd",
                   side_effect=ValueError("xrd error")):
            rc = _cmd_analyze_xrd(args)
        assert rc == 1


# ---------------------------------------------------------------------------
# _cmd_analyze_symmetry tests
# ---------------------------------------------------------------------------


class TestCmdAnalyzeSymmetry:
    def _make_args(self, **overrides):
        import argparse
        defaults = dict(structure="POSCAR", symprec=1e-5, save_json=None)
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_not_available_returns_1(self, capsys):
        from unittest.mock import patch
        from shalom.__main__ import _cmd_analyze_symmetry
        args = self._make_args()
        with patch("shalom.analysis.symmetry.is_spglib_available", return_value=False):
            rc = _cmd_analyze_symmetry(args)
        assert rc == 1

    def test_structure_error_returns_1(self, capsys):
        from unittest.mock import patch
        from shalom.__main__ import _cmd_analyze_symmetry
        args = self._make_args()
        with patch("shalom.analysis.symmetry.is_spglib_available", return_value=True), \
             patch("ase.io.read", side_effect=Exception("no file")):
            rc = _cmd_analyze_symmetry(args)
        assert rc == 1

    def test_success(self, capsys):
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import _cmd_analyze_symmetry

        mock_result = MagicMock()
        mock_result.space_group_symbol = "Fd-3m"
        mock_result.space_group_number = 227
        mock_result.point_group = "m-3m"
        mock_result.crystal_system = "cubic"
        mock_result.lattice_type = "face-centered"
        mock_result.hall_symbol = "-F 4vw 2vw 3"
        mock_result.n_operations = 192
        mock_result.is_primitive = False
        mock_result.wyckoff_letters = ["a", "c"]
        mock_result.to_dict.return_value = {}

        mock_atoms = MagicMock()
        mock_atoms.get_chemical_formula.return_value = "Si2"

        args = self._make_args()
        with patch("shalom.analysis.symmetry.is_spglib_available", return_value=True), \
             patch("ase.io.read", return_value=mock_atoms), \
             patch("shalom.analysis.symmetry.analyze_symmetry", return_value=mock_result):
            rc = _cmd_analyze_symmetry(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "Fd-3m" in out
        assert "227" in out
        assert "a, c" in out

    def test_exception_returns_1(self, capsys):
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import _cmd_analyze_symmetry
        args = self._make_args()
        with patch("shalom.analysis.symmetry.is_spglib_available", return_value=True), \
             patch("ase.io.read", return_value=MagicMock()), \
             patch("shalom.analysis.symmetry.analyze_symmetry",
                   side_effect=RuntimeError("spglib error")):
            rc = _cmd_analyze_symmetry(args)
        assert rc == 1


# ---------------------------------------------------------------------------
# _cmd_analyze_magnetic tests
# ---------------------------------------------------------------------------


class TestCmdAnalyzeMagnetic:
    def _make_args(self, **overrides):
        import argparse
        defaults = dict(pw_out="pw.out", structure=None, save_json=None)
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_missing_pw_out_returns_1(self, tmp_path, capsys):
        from shalom.__main__ import _cmd_analyze_magnetic
        args = self._make_args(pw_out=str(tmp_path / "no_such_pw.out"))
        rc = _cmd_analyze_magnetic(args)
        assert rc == 1
        assert "not found" in capsys.readouterr().out

    def test_success_without_structure(self, tmp_path, capsys):
        from unittest.mock import patch
        from shalom.__main__ import _cmd_analyze_magnetic

        pw_out = tmp_path / "pw.out"
        pw_out.write_text("dummy")

        args = self._make_args(pw_out=str(pw_out))
        with patch("shalom.analysis.magnetic.extract_site_magnetization",
                   return_value=[0.5, -0.5, 0.3]), \
             patch("shalom.analysis.magnetic.extract_lowdin_charges",
                   return_value={"total_charges": [3.5, 3.5], "spd_charges": [{"s": 0.5, "d": 3.0}, {}]}):
            rc = _cmd_analyze_magnetic(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "Atom" in out
        assert "Bohr mag" in out

    def test_success_with_structure(self, tmp_path, capsys):
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import _cmd_analyze_magnetic

        pw_out = tmp_path / "pw.out"
        pw_out.write_text("dummy")

        mock_mag_result = MagicMock()
        mock_mag_result.total_magnetization = 4.0
        mock_mag_result.is_magnetic = True
        mock_mag_result.is_spin_polarized = True
        mock_mag_result.magnetic_elements = ["Fe"]
        mock_mag_result.dominant_moment_element = "Fe"
        mock_mag_result.to_dict.return_value = {"total": 4.0}

        args = self._make_args(pw_out=str(pw_out), structure="POSCAR", save_json=str(tmp_path / "mag.json"))
        with patch("shalom.analysis.magnetic.extract_site_magnetization", return_value=[3.5]), \
             patch("shalom.analysis.magnetic.extract_lowdin_charges", return_value=None), \
             patch("shalom.analysis.magnetic.extract_total_magnetization", return_value=4.0), \
             patch("shalom.analysis.magnetic.analyze_magnetism", return_value=mock_mag_result), \
             patch("ase.io.read", return_value=MagicMock()):
            rc = _cmd_analyze_magnetic(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "4.0000" in out
        assert "Fe" in out

    def test_no_mag_data_found(self, tmp_path, capsys):
        from unittest.mock import patch
        from shalom.__main__ import _cmd_analyze_magnetic

        pw_out = tmp_path / "pw.out"
        pw_out.write_text("dummy")

        args = self._make_args(pw_out=str(pw_out))
        with patch("shalom.analysis.magnetic.extract_site_magnetization", return_value=None), \
             patch("shalom.analysis.magnetic.extract_lowdin_charges", return_value=None):
            rc = _cmd_analyze_magnetic(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "No site magnetization" in out
        assert "No Lowdin" in out


# ---------------------------------------------------------------------------
# _cmd_analyze_md tests
# ---------------------------------------------------------------------------


class TestCmdAnalyzeMd:
    def _make_args(self, **overrides):
        import argparse
        defaults = dict(
            calc_dir="/tmp/calc", backend="lammps",
            r_max=10.0, output=None, save_json=None,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_nonexistent_dir_returns_1(self, capsys):
        from shalom.__main__ import _cmd_analyze_md
        args = self._make_args(calc_dir="/nonexistent/path")
        rc = _cmd_analyze_md(args)
        assert rc == 1
        assert "not found" in capsys.readouterr().out

    def test_file_not_found_returns_1(self, tmp_path, capsys):
        from unittest.mock import patch
        from shalom.__main__ import _cmd_analyze_md
        args = self._make_args(calc_dir=str(tmp_path))
        with patch("shalom.backends.get_backend") as mock_be:
            mock_be.return_value.parse_trajectory.side_effect = FileNotFoundError("no dump")
            rc = _cmd_analyze_md(args)
        assert rc == 1

    def test_success(self, tmp_path, capsys):
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import _cmd_analyze_md

        mock_traj = MagicMock()
        mock_traj.n_frames = 100
        mock_traj.n_atoms = 32
        mock_traj.source = "lammps"
        mock_traj.ensemble = "nvt"
        mock_traj.timestep_fs = 1.0

        mock_result = MagicMock()
        mock_result.avg_temperature = 300.0
        mock_result.temperature_std = 15.0
        mock_result.avg_energy = -5.0
        mock_result.avg_pressure = 1.0
        mock_result.diffusion_coefficient = 1e-5
        mock_result.energy_drift_per_atom = 1e-6
        mock_result.is_equilibrated = True
        mock_result.equilibration_step = 20
        mock_result.rdf_r = None
        mock_result.msd_t = None
        mock_result.to_dict.return_value = {}

        args = self._make_args(calc_dir=str(tmp_path))
        with patch("shalom.backends.get_backend") as mock_be, \
             patch("shalom.analysis.md.analyze_md_trajectory", return_value=mock_result):
            mock_be.return_value.parse_trajectory.return_value = mock_traj
            rc = _cmd_analyze_md(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "300.0" in out
        assert "Equilibrated" in out

    def test_with_output_plots(self, tmp_path, capsys):
        import numpy as np
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import _cmd_analyze_md

        mock_traj = MagicMock()
        mock_traj.n_frames = 50
        mock_traj.n_atoms = 8
        mock_traj.source = "vasp"
        mock_traj.ensemble = "nvt"
        mock_traj.timestep_fs = 1.0

        mock_result = MagicMock()
        mock_result.avg_temperature = None
        mock_result.temperature_std = None
        mock_result.avg_energy = None
        mock_result.avg_pressure = None
        mock_result.diffusion_coefficient = None
        mock_result.energy_drift_per_atom = None
        mock_result.is_equilibrated = False
        mock_result.equilibration_step = None
        mock_result.rdf_r = np.array([1.0, 2.0])
        mock_result.msd_t = np.array([0.0, 1.0])
        mock_result.to_dict.return_value = {}

        out_dir = str(tmp_path / "plots")
        args = self._make_args(calc_dir=str(tmp_path), output=out_dir, backend="vasp")
        with patch("shalom.backends.get_backend") as mock_be, \
             patch("shalom.analysis.md.analyze_md_trajectory", return_value=mock_result), \
             patch("shalom.plotting.md_plot.MDEnergyPlotter") as mock_ep, \
             patch("shalom.plotting.md_plot.MDTemperaturePlotter") as mock_tp, \
             patch("shalom.plotting.md_plot.RDFPlotter") as mock_rp, \
             patch("shalom.plotting.md_plot.MSDPlotter") as mock_mp, \
             patch("matplotlib.pyplot.close"):
            mock_be.return_value.parse_trajectory.return_value = mock_traj
            rc = _cmd_analyze_md(args)
        assert rc == 0
        mock_ep.assert_called_once()
        mock_rp.assert_called_once()
        mock_mp.assert_called_once()

    def test_exception_returns_1(self, tmp_path, capsys):
        from unittest.mock import patch
        from shalom.__main__ import _cmd_analyze_md
        args = self._make_args(calc_dir=str(tmp_path))
        with patch("shalom.backends.get_backend") as mock_be:
            mock_be.return_value.parse_trajectory.side_effect = RuntimeError("parse error")
            rc = _cmd_analyze_md(args)
        assert rc == 1


# ---------------------------------------------------------------------------
# _cmd_analyze_phonon tests
# ---------------------------------------------------------------------------


class TestCmdAnalyzePhonon:
    def _make_args(self, **overrides):
        import argparse
        defaults = dict(
            structure="POSCAR", supercell="2x2x2",
            force_sets=None, force_constants=None,
            generate_displacements=False,
            mesh="20,20,20", output=None, save_json=None,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_not_available_returns_1(self, capsys):
        from unittest.mock import patch
        from shalom.__main__ import _cmd_analyze_phonon
        args = self._make_args()
        with patch("shalom.analysis.phonon.is_phonopy_available", return_value=False):
            rc = _cmd_analyze_phonon(args)
        assert rc == 1
        assert "phonopy" in capsys.readouterr().out

    def test_invalid_supercell_returns_1(self, capsys):
        from unittest.mock import patch
        from shalom.__main__ import _cmd_analyze_phonon
        args = self._make_args(supercell="bad")
        with patch("shalom.analysis.phonon.is_phonopy_available", return_value=True):
            rc = _cmd_analyze_phonon(args)
        assert rc == 1
        assert "invalid supercell" in capsys.readouterr().out

    def test_structure_read_error_returns_1(self, capsys):
        from unittest.mock import patch
        from shalom.__main__ import _cmd_analyze_phonon
        args = self._make_args()
        with patch("shalom.analysis.phonon.is_phonopy_available", return_value=True), \
             patch("ase.io.read", side_effect=Exception("no file")):
            rc = _cmd_analyze_phonon(args)
        assert rc == 1

    def test_no_force_data_returns_1(self, capsys):
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import _cmd_analyze_phonon
        args = self._make_args()  # no force_sets or force_constants
        with patch("shalom.analysis.phonon.is_phonopy_available", return_value=True), \
             patch("ase.io.read", return_value=MagicMock()):
            rc = _cmd_analyze_phonon(args)
        assert rc == 1
        assert "provide --force-sets" in capsys.readouterr().out

    def test_generate_displacements(self, tmp_path, capsys):
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import _cmd_analyze_phonon

        mock_disp = [MagicMock(), MagicMock()]
        args = self._make_args(generate_displacements=True, output=str(tmp_path / "disps"))
        with patch("shalom.analysis.phonon.is_phonopy_available", return_value=True), \
             patch("ase.io.read", return_value=MagicMock()), \
             patch("shalom.analysis.phonon.generate_phonon_displacements",
                   return_value=(mock_disp, MagicMock())), \
             patch("ase.io.write"):
            rc = _cmd_analyze_phonon(args)
        assert rc == 0
        assert "Generated 2" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# cmd_pipeline tests
# ---------------------------------------------------------------------------


class TestCmdPipeline:
    def _make_args(self, **overrides):
        import argparse
        defaults = dict(
            objective="Find stable alloy", backend="qe",
            provider="openai", model=None, material=None,
            steps=None, output=None, calc="relaxation",
            accuracy="standard", execute=False, nprocs=1,
            timeout=86400, max_loops=1, selector_mode="simple",
            base_url=None, command="pipeline",
            slurm=False, partition="compute", account=None,
            nodes=1, ntasks_per_node=None, walltime="24:00:00",
            qos=None, mem=None, module_loads=[], slurm_extras=[],
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_no_api_key_returns_1(self, capsys):
        from unittest.mock import patch
        from shalom.__main__ import cmd_pipeline
        args = self._make_args()
        with patch.dict("os.environ", {}, clear=True):
            rc = cmd_pipeline(args)
        assert rc == 1
        out = capsys.readouterr().out
        assert "OPENAI_API_KEY" in out

    def test_anthropic_no_key_returns_1(self, capsys):
        from unittest.mock import patch
        from shalom.__main__ import cmd_pipeline
        args = self._make_args(provider="anthropic")
        with patch.dict("os.environ", {}, clear=True):
            rc = cmd_pipeline(args)
        assert rc == 1
        out = capsys.readouterr().out
        assert "ANTHROPIC_API_KEY" in out

    def test_base_url_skips_key_check(self, capsys):
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import cmd_pipeline

        mock_pipeline_cls = MagicMock()
        mock_result = MagicMock()
        mock_result.status.value = "completed"
        mock_result.ranked_material.candidate.material_name = "Fe2O3"
        mock_result.ranked_material.score = 0.85
        mock_result.structure_path = "/out/POSCAR"
        mock_result.execution_wall_time = 10.0
        mock_result.quality_warnings = ["high force"]
        mock_result.error_message = None
        mock_result.elapsed_seconds = 20.0
        mock_result.steps_completed = ["design", "simulation"]
        # Make status comparison work
        from shalom.core.schemas import PipelineStatus
        mock_result.status = PipelineStatus.COMPLETED
        mock_pipeline_cls.return_value.run.return_value = mock_result

        args = self._make_args(base_url="http://localhost:11434/v1")
        with patch.dict("os.environ", {}, clear=True), \
             patch("shalom.pipeline.Pipeline", mock_pipeline_cls), \
             patch("shalom.pipeline.PipelineConfig"):
            rc = cmd_pipeline(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "Fe2O3" in out

    def test_default_model_openai(self, capsys):
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import cmd_pipeline

        mock_pipeline_cls = MagicMock()
        from shalom.core.schemas import PipelineStatus
        mock_result = MagicMock()
        mock_result.status = PipelineStatus.AWAITING_DFT
        mock_result.ranked_material = None
        mock_result.structure_path = None
        mock_result.execution_wall_time = None
        mock_result.quality_warnings = []
        mock_result.error_message = None
        mock_result.elapsed_seconds = None
        mock_result.steps_completed = ["design"]
        mock_pipeline_cls.return_value.run.return_value = mock_result
        mock_config_cls = MagicMock()

        args = self._make_args(model=None, provider="openai")
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=False), \
             patch("shalom.pipeline.Pipeline", mock_pipeline_cls), \
             patch("shalom.pipeline.PipelineConfig", mock_config_cls):
            rc = cmd_pipeline(args)
        assert rc == 0
        # Check PipelineConfig called with model_name="gpt-4o"
        call_kwargs = mock_config_cls.call_args.kwargs
        assert call_kwargs["model_name"] == "gpt-4o"
        out = capsys.readouterr().out
        assert "AWAITING_DFT" in out.upper() or "awaiting_dft" in out

    def test_anthropic_default_model(self, capsys):
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import cmd_pipeline

        mock_config_cls = MagicMock()
        mock_pipeline_cls = MagicMock()
        from shalom.core.schemas import PipelineStatus
        mock_result = MagicMock()
        mock_result.status = PipelineStatus.COMPLETED
        mock_result.ranked_material = None
        mock_result.structure_path = None
        mock_result.execution_wall_time = None
        mock_result.quality_warnings = []
        mock_result.error_message = None
        mock_result.elapsed_seconds = None
        mock_result.steps_completed = []
        mock_pipeline_cls.return_value.run.return_value = mock_result

        args = self._make_args(provider="anthropic", model=None)
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}, clear=False), \
             patch("shalom.pipeline.Pipeline", mock_pipeline_cls), \
             patch("shalom.pipeline.PipelineConfig", mock_config_cls):
            cmd_pipeline(args)
        assert mock_config_cls.call_args.kwargs["model_name"] == "claude-sonnet-4-6"

    def test_pipeline_exception_returns_1(self, capsys):
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import cmd_pipeline

        mock_pipeline_cls = MagicMock()
        mock_pipeline_cls.return_value.run.side_effect = RuntimeError("LLM error")
        mock_config_cls = MagicMock()

        args = self._make_args()
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=False), \
             patch("shalom.pipeline.Pipeline", mock_pipeline_cls), \
             patch("shalom.pipeline.PipelineConfig", mock_config_cls):
            rc = cmd_pipeline(args)
        assert rc == 1
        assert "LLM error" in capsys.readouterr().out

    def test_failed_status_returns_1(self, capsys):
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import cmd_pipeline
        from shalom.core.schemas import PipelineStatus

        mock_result = MagicMock()
        mock_result.status = PipelineStatus.FAILED_EXECUTION
        mock_result.ranked_material = None
        mock_result.structure_path = None
        mock_result.execution_wall_time = None
        mock_result.quality_warnings = []
        mock_result.error_message = "DFT crashed"
        mock_result.elapsed_seconds = 5.0
        mock_result.steps_completed = ["design", "simulation"]
        mock_pipeline_cls = MagicMock()
        mock_pipeline_cls.return_value.run.return_value = mock_result

        args = self._make_args()
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=False), \
             patch("shalom.pipeline.Pipeline", mock_pipeline_cls), \
             patch("shalom.pipeline.PipelineConfig"):
            rc = cmd_pipeline(args)
        assert rc == 1
        out = capsys.readouterr().out
        assert "DFT crashed" in out

    def test_steps_parsing(self, capsys):
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import cmd_pipeline
        from shalom.core.schemas import PipelineStatus

        mock_config_cls = MagicMock()
        mock_pipeline_cls = MagicMock()
        mock_result = MagicMock()
        mock_result.status = PipelineStatus.COMPLETED
        mock_result.ranked_material = None
        mock_result.structure_path = None
        mock_result.execution_wall_time = None
        mock_result.quality_warnings = []
        mock_result.error_message = None
        mock_result.elapsed_seconds = None
        mock_result.steps_completed = []
        mock_pipeline_cls.return_value.run.return_value = mock_result

        args = self._make_args(steps="design,simulation,review")
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=False), \
             patch("shalom.pipeline.Pipeline", mock_pipeline_cls), \
             patch("shalom.pipeline.PipelineConfig", mock_config_cls):
            cmd_pipeline(args)
        assert mock_config_cls.call_args.kwargs["steps"] == ["design", "simulation", "review"]

    def test_material_skip_design(self, capsys):
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import cmd_pipeline
        from shalom.core.schemas import PipelineStatus

        mock_config_cls = MagicMock()
        mock_pipeline_cls = MagicMock()
        mock_result = MagicMock()
        mock_result.status = PipelineStatus.COMPLETED_DESIGN
        mock_result.ranked_material = None
        mock_result.structure_path = "/out"
        mock_result.execution_wall_time = None
        mock_result.quality_warnings = []
        mock_result.error_message = None
        mock_result.elapsed_seconds = None
        mock_result.steps_completed = ["simulation"]
        mock_pipeline_cls.return_value.run.return_value = mock_result

        args = self._make_args(material="MoS2")
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=False), \
             patch("shalom.pipeline.Pipeline", mock_pipeline_cls), \
             patch("shalom.pipeline.PipelineConfig", mock_config_cls):
            rc = cmd_pipeline(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "MoS2" in out
        assert "Design layer skipped" in out

    def test_slurm_kwargs_forwarded(self, capsys):
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import cmd_pipeline
        from shalom.core.schemas import PipelineStatus

        mock_config_cls = MagicMock()
        mock_pipeline_cls = MagicMock()
        mock_result = MagicMock()
        mock_result.status = PipelineStatus.COMPLETED
        mock_result.ranked_material = None
        mock_result.structure_path = None
        mock_result.execution_wall_time = None
        mock_result.quality_warnings = []
        mock_result.error_message = None
        mock_result.elapsed_seconds = None
        mock_result.steps_completed = []
        mock_pipeline_cls.return_value.run.return_value = mock_result

        args = self._make_args(slurm=True, partition="gpu", account="sci")
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=False), \
             patch("shalom.pipeline.Pipeline", mock_pipeline_cls), \
             patch("shalom.pipeline.PipelineConfig", mock_config_cls):
            cmd_pipeline(args)
        call_kwargs = mock_config_cls.call_args.kwargs
        assert call_kwargs["slurm_partition"] == "gpu"
        assert call_kwargs["slurm_account"] == "sci"

    def test_env_var_base_url(self, capsys):
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import cmd_pipeline
        from shalom.core.schemas import PipelineStatus

        mock_config_cls = MagicMock()
        mock_pipeline_cls = MagicMock()
        mock_result = MagicMock()
        mock_result.status = PipelineStatus.COMPLETED
        mock_result.ranked_material = None
        mock_result.structure_path = None
        mock_result.execution_wall_time = None
        mock_result.quality_warnings = []
        mock_result.error_message = None
        mock_result.elapsed_seconds = None
        mock_result.steps_completed = []
        mock_pipeline_cls.return_value.run.return_value = mock_result

        args = self._make_args(base_url=None)
        with patch.dict("os.environ", {"SHALOM_LLM_BASE_URL": "http://localhost:8000/v1"}, clear=False), \
             patch("shalom.pipeline.Pipeline", mock_pipeline_cls), \
             patch("shalom.pipeline.PipelineConfig", mock_config_cls):
            cmd_pipeline(args)
        assert mock_config_cls.call_args.kwargs["base_url"] == "http://localhost:8000/v1"


# ---------------------------------------------------------------------------
# cmd_workflow step_results printing
# ---------------------------------------------------------------------------


class TestCmdWorkflowStepResults:
    def test_step_results_printed(self, tmp_path, capsys):
        """cmd_workflow prints step results with timing."""
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import cmd_workflow
        from ase.build import bulk
        import argparse

        si = bulk("Si", "diamond", a=5.43)
        mock_wf = MagicMock()

        # Create mock StepStatus objects
        step1 = MagicMock()
        step1.success = True
        step1.error_message = ""
        step1.elapsed_seconds = 120.0
        step1.summary = "converged"
        step1.step_number = 1
        step1.name = "vc-relax"

        step2 = MagicMock()
        step2.success = False
        step2.error_message = "SCF diverged"
        step2.elapsed_seconds = 5.0
        step2.summary = ""
        step2.step_number = 2
        step2.name = "scf"

        mock_wf.run.return_value = {
            "bands_png": None,
            "dos_png": None,
            "fermi_energy": None,
            "failed_step": "scf",
            "step_results": [step1, step2],
        }

        args = argparse.Namespace(
            material="Si", structure=None, output=str(tmp_path),
            pseudo_dir=None, nprocs=1, mpi_command="mpirun",
            pw_command="pw.x", dos_command="dos.x",
            timeout=3600, accuracy="standard", skip_relax=False,
            is_2d=False, dos_emin=-20.0, dos_emax=20.0,
            nscf_kgrid=None, resume=False, wsl=False,
            command="workflow", slurm=False,
            partition="compute", account=None, nodes=1,
            ntasks_per_node=None, walltime="24:00:00",
            qos=None, mem=None, module_loads=[], slurm_extras=[],
        )
        with patch("shalom.__main__._load_atoms", return_value=si), \
             patch("shalom.workflows.standard.StandardWorkflow", return_value=mock_wf):
            rc = cmd_workflow(args)
        assert rc == 1  # failed_step is set
        out = capsys.readouterr().out
        assert "PARTIALLY complete" in out
        assert "vc-relax" in out
        assert "2m0s" in out
        assert "converged" in out
        assert "SCF diverged" in out


# ---------------------------------------------------------------------------
# cmd_run --execute QE success path
# ---------------------------------------------------------------------------


class TestCmdRunExecuteQE:
    def test_execute_qe_success(self, tmp_path, capsys):
        """cmd_run --execute with QE backend calls _execute_dft."""
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import cmd_run
        import argparse

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.backend_name = "qe"
        mock_result.structure_info = None
        mock_result.auto_detected = {}
        mock_result.output_dir = str(tmp_path)
        mock_result.files_generated = ["pw.in"]

        args = argparse.Namespace(
            material=None, structure="POSCAR", backend="qe", calc=None,
            accuracy="standard", set_values=None, output=None,
            workspace=None, project=None,
            pseudo_dir=None, no_validate=False, force=False,
            quiet=False, verbose=False, command="run",
            execute=True, nprocs=1, timeout=3600, mpi_command="mpirun",
            pw_command="pw.x", dos_command="dos.x", wsl=False,
            slurm=False, partition="compute", account=None, nodes=1,
            ntasks_per_node=None, walltime="24:00:00",
            qos=None, mem=None, module_loads=[], slurm_extras=[],
            pair_style=None, pair_coeff=[], md_ensemble="nvt",
            temperature=300.0, md_steps=None, timestep=None,
            potential_dir=None,
        )
        with patch("shalom.direct_run.direct_run", return_value=mock_result), \
             patch("shalom.direct_run.DirectRunConfig"), \
             patch("shalom.__main__._execute_dft", return_value=0) as mock_exec:
            rc = cmd_run(args)
        assert rc == 0
        mock_exec.assert_called_once()


# ---------------------------------------------------------------------------
# main() dispatches pipeline and analyze
# ---------------------------------------------------------------------------


class TestMainDispatcherAdditional:
    def test_pipeline_subcommand_dispatched(self):
        import sys
        from unittest.mock import patch
        from shalom.__main__ import main

        with patch.object(sys, "argv", ["shalom", "pipeline", "Find catalyst"]), \
             patch("shalom.__main__.cmd_pipeline", return_value=0) as mock_pipe, \
             pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        mock_pipe.assert_called_once()

    def test_analyze_subcommand_dispatched(self):
        import sys
        from unittest.mock import patch
        from shalom.__main__ import main

        with patch.object(sys, "argv", ["shalom", "analyze"]), \
             patch("shalom.__main__.cmd_analyze", return_value=1) as mock_an, \
             pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
        mock_an.assert_called_once()


# ---------------------------------------------------------------------------
# _cmd_analyze_phonon — force_constants path + display + plots
# ---------------------------------------------------------------------------


class TestCmdAnalyzePhononForceConstants:
    """Test phonon analysis from force_constants (covers lines 1919-2004)."""

    def _make_args(self, **overrides):
        import argparse
        defaults = dict(
            structure="POSCAR", supercell="2x2x2",
            force_sets=None, force_constants=None,
            generate_displacements=False,
            mesh="20,20,20", output=None, save_json=None,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_force_constants_success_stable(self, capsys):
        import numpy as np
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import _cmd_analyze_phonon

        mock_result = MagicMock()
        mock_result.n_branches = 6
        mock_result.min_frequency_THz = 0.5
        mock_result.is_stable = True
        mock_result.imaginary_modes = []
        mock_result.thermal_temperatures = np.array([100.0, 200.0, 300.0, 400.0])
        mock_result.thermal_cv = np.array([10.0, 20.0, 25.0, 27.0])
        mock_result.thermal_entropy = np.array([5.0, 12.0, 18.0, 22.0])
        mock_result.thermal_free_energy = np.array([-1.0, -5.0, -10.0, -16.0])
        mock_result.to_dict.return_value = {}

        mock_atoms = MagicMock()
        mock_atoms.get_chemical_formula.return_value = "Si2"

        args = self._make_args(force_constants="/path/to/FC")
        with patch("shalom.analysis.phonon.is_phonopy_available", return_value=True), \
             patch("ase.io.read", return_value=mock_atoms), \
             patch("phonopy.file_IO.parse_FORCE_CONSTANTS", return_value=MagicMock()), \
             patch("shalom.analysis.phonon.analyze_phonon_from_force_constants",
                   return_value=mock_result):
            rc = _cmd_analyze_phonon(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "STABLE" in out
        assert "25.00" in out  # Cv at 300K
        assert "18.00" in out  # Entropy at 300K
        assert "-10.00" in out  # Free energy at 300K

    def test_force_constants_unstable(self, capsys):
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import _cmd_analyze_phonon

        mock_result = MagicMock()
        mock_result.n_branches = 6
        mock_result.min_frequency_THz = -1.5
        mock_result.is_stable = False
        mock_result.imaginary_modes = [(-1.5, 0), (-0.8, 1)]
        mock_result.thermal_temperatures = None
        mock_result.to_dict.return_value = {}

        mock_atoms = MagicMock()
        mock_atoms.get_chemical_formula.return_value = "Si2"

        args = self._make_args(force_constants="/path/to/FC")
        with patch("shalom.analysis.phonon.is_phonopy_available", return_value=True), \
             patch("ase.io.read", return_value=mock_atoms), \
             patch("phonopy.file_IO.parse_FORCE_CONSTANTS", return_value=MagicMock()), \
             patch("shalom.analysis.phonon.analyze_phonon_from_force_constants",
                   return_value=mock_result):
            rc = _cmd_analyze_phonon(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "UNSTABLE" in out
        assert "Imaginary modes: 2" in out

    def test_force_sets_success(self, capsys):
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import _cmd_analyze_phonon

        mock_result = MagicMock()
        mock_result.n_branches = 3
        mock_result.min_frequency_THz = 0.1
        mock_result.is_stable = True
        mock_result.thermal_temperatures = None
        mock_result.to_dict.return_value = {}

        mock_atoms = MagicMock()
        mock_atoms.get_chemical_formula.return_value = "Si"
        mock_ph = MagicMock()

        args = self._make_args(force_sets="/path/to/FORCE_SETS")
        with patch("shalom.analysis.phonon.is_phonopy_available", return_value=True), \
             patch("ase.io.read", return_value=mock_atoms), \
             patch("phonopy.file_IO.parse_FORCE_SETS", return_value=MagicMock()), \
             patch("shalom.analysis.phonon.generate_phonon_displacements",
                   return_value=([], mock_ph)), \
             patch("shalom.analysis.phonon._run_phonon_analysis",
                   return_value=mock_result):
            rc = _cmd_analyze_phonon(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "STABLE" in out

    def test_with_output_plots(self, tmp_path, capsys):
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import _cmd_analyze_phonon

        mock_result = MagicMock()
        mock_result.n_branches = 6
        mock_result.min_frequency_THz = 0.5
        mock_result.is_stable = True
        mock_result.thermal_temperatures = None
        mock_result.to_dict.return_value = {}

        mock_atoms = MagicMock()
        mock_atoms.get_chemical_formula.return_value = "Si2"

        out_dir = str(tmp_path / "phonon_plots")
        args = self._make_args(force_constants="/path/to/FC", output=out_dir)
        with patch("shalom.analysis.phonon.is_phonopy_available", return_value=True), \
             patch("ase.io.read", return_value=mock_atoms), \
             patch("phonopy.file_IO.parse_FORCE_CONSTANTS", return_value=MagicMock()), \
             patch("shalom.analysis.phonon.analyze_phonon_from_force_constants",
                   return_value=mock_result), \
             patch("shalom.plotting.PhononBandPlotter") as mock_bp, \
             patch("shalom.plotting.PhononDOSPlotter") as mock_dp:
            rc = _cmd_analyze_phonon(args)
        assert rc == 0
        mock_bp.assert_called_once()
        mock_dp.assert_called_once()
        out = capsys.readouterr().out
        assert "Saved" in out

    def test_plot_import_error(self, tmp_path, capsys):
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import _cmd_analyze_phonon

        mock_result = MagicMock()
        mock_result.n_branches = 6
        mock_result.min_frequency_THz = 0.5
        mock_result.is_stable = True
        mock_result.thermal_temperatures = None
        mock_result.to_dict.return_value = {}

        mock_atoms = MagicMock()
        mock_atoms.get_chemical_formula.return_value = "Si2"

        out_dir = str(tmp_path / "phonon_plots")
        args = self._make_args(force_constants="/path/to/FC", output=out_dir)
        with patch("shalom.analysis.phonon.is_phonopy_available", return_value=True), \
             patch("ase.io.read", return_value=mock_atoms), \
             patch("phonopy.file_IO.parse_FORCE_CONSTANTS", return_value=MagicMock()), \
             patch("shalom.analysis.phonon.analyze_phonon_from_force_constants",
                   return_value=mock_result), \
             patch("shalom.plotting.PhononBandPlotter",
                   side_effect=ImportError("no matplotlib")):
            rc = _cmd_analyze_phonon(args)
        assert rc == 0
        assert "matplotlib not installed" in capsys.readouterr().out

    def test_analysis_exception_returns_1(self, capsys):
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import _cmd_analyze_phonon

        mock_atoms = MagicMock()
        args = self._make_args(force_constants="/path/to/FC")
        with patch("shalom.analysis.phonon.is_phonopy_available", return_value=True), \
             patch("ase.io.read", return_value=mock_atoms), \
             patch("phonopy.file_IO.parse_FORCE_CONSTANTS",
                   side_effect=Exception("bad FC file")):
            rc = _cmd_analyze_phonon(args)
        assert rc == 1
        assert "bad FC file" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Electronic with DOS file test
# ---------------------------------------------------------------------------


class TestCmdAnalyzeElectronicDOS:
    def test_with_dos_file(self, capsys):
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import _cmd_analyze_electronic
        import argparse

        mock_result = MagicMock()
        mock_result.is_metal = False
        mock_result.is_direct = False
        mock_result.bandgap_eV = 0.65
        mock_result.vbm_energy = -0.3
        mock_result.cbm_energy = 0.35
        mock_result.dos_at_fermi = None
        mock_result.effective_mass_electron = None
        mock_result.effective_mass_hole = None
        mock_result.n_occupied_bands = 4
        mock_result.to_dict.return_value = {}

        args = argparse.Namespace(
            bands_xml="/path/xml", calc_dir=None,
            dos_file="/path/dos", fermi_energy=5.0,
            save_json=None,
        )
        with patch("shalom.analysis.electronic.is_electronic_available", return_value=True), \
             patch("shalom.backends.qe_parser.parse_xml_bands"), \
             patch("shalom.backends.qe_parser.extract_fermi_energy", return_value=5.0), \
             patch("shalom.backends.qe_parser.parse_dos_file") as mock_parse_dos, \
             patch("shalom.analysis.electronic.analyze_band_structure",
                   return_value=mock_result):
            mock_dos = MagicMock()
            mock_parse_dos.return_value = mock_dos
            rc = _cmd_analyze_electronic(args)
        assert rc == 0
        # Verify fermi_energy was set on dos_data
        assert mock_dos.fermi_energy == 5.0
        out = capsys.readouterr().out
        assert "indirect" in out


# ---------------------------------------------------------------------------
# Combined plot ImportError
# ---------------------------------------------------------------------------


class TestCmdPlotCombinedImportError:
    def test_combined_matplotlib_import_error(self, tmp_path, capsys):
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import cmd_plot
        from shalom.backends.base import BandStructureData, DOSData
        import numpy as np
        import argparse

        fake_bs = BandStructureData(
            eigenvalues=np.zeros((5, 3)),
            kpoint_coords=np.zeros((5, 3)),
            kpath_distances=np.linspace(0, 1, 5),
        )
        fake_dos = DOSData(
            energies=np.array([0.0]),
            dos=np.array([1.0]),
            integrated_dos=np.array([0.5]),
        )
        (tmp_path / "pwscf.dos").write_text("# DOS\n0.0 1.0 0.5\n")

        args = argparse.Namespace(
            calc_dir=str(tmp_path), bands=False, dos=False, combined=True,
            bands_dir=None, fermi_from=None, output=None,
            title=None, emin=-6.0, emax=6.0, command="plot",
        )
        with patch("shalom.backends.qe_parser.find_xml_path", return_value="/x.xml"), \
             patch("shalom.backends.qe_parser.parse_xml_bands", return_value=fake_bs), \
             patch("shalom.backends.qe_parser.parse_dos_file", return_value=fake_dos), \
             patch("shalom.backends.qe_parser.extract_fermi_energy", return_value=5.0), \
             patch("shalom.plotting.combined_plot.CombinedPlotter",
                   side_effect=ImportError("no matplotlib")):
            rc = cmd_plot(args)
        assert rc == 1
        assert "matplotlib" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# DOS ImportError in cmd_plot
# ---------------------------------------------------------------------------


class TestCmdPlotDOSImportError:
    def test_dos_matplotlib_import_error(self, tmp_path, capsys):
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import cmd_plot
        from shalom.backends.base import DOSData
        import numpy as np
        import argparse

        (tmp_path / "pwscf.dos").write_text("# DOS\n0.0 1.0 0.5\n")
        fake_dos = DOSData(
            energies=np.array([0.0]),
            dos=np.array([1.0]),
            integrated_dos=np.array([0.5]),
        )

        args = argparse.Namespace(
            calc_dir=str(tmp_path), bands=False, dos=True, combined=False,
            bands_dir=None, fermi_from=None, output=None,
            title=None, emin=-6.0, emax=6.0, command="plot",
        )
        with patch("shalom.backends.qe_parser.extract_fermi_energy", return_value=5.0), \
             patch("shalom.backends.qe_parser.parse_dos_file", return_value=fake_dos), \
             patch("shalom.plotting.dos_plot.DOSPlotter",
                   side_effect=ImportError("no mpl")):
            rc = cmd_plot(args)
        assert rc == 1
        assert "matplotlib" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Pipeline ImportError (no openai/anthropic)
# ---------------------------------------------------------------------------


class TestCmdPipelineImportError:
    def test_pipeline_import_missing(self, capsys):
        from unittest.mock import patch
        from shalom.__main__ import cmd_pipeline
        import argparse

        args = argparse.Namespace(
            objective="test", backend="qe", provider="openai",
            model=None, material=None, steps=None, output=None,
            calc="relaxation", accuracy="standard", execute=False,
            nprocs=1, timeout=86400, max_loops=1,
            selector_mode="simple", base_url=None, command="pipeline",
            slurm=False, partition="compute", account=None,
            nodes=1, ntasks_per_node=None, walltime="24:00:00",
            qos=None, mem=None, module_loads=[], slurm_extras=[],
        )
        with patch.dict("sys.modules", {"shalom.pipeline": None}):
            rc = cmd_pipeline(args)
        assert rc == 1
        out = capsys.readouterr().out
        assert "LLM dependencies" in out or "not installed" in out


# ---------------------------------------------------------------------------
# XRD no wavelength_angstrom test + plot import error
# ---------------------------------------------------------------------------


class TestCmdAnalyzeXrdEdgeCases:
    def _make_args(self, **overrides):
        import argparse
        defaults = dict(
            structure="POSCAR", wavelength="Custom",
            theta_min=0.0, theta_max=90.0,
            output=None, save_json=None,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_no_wavelength_angstrom(self, capsys):
        import numpy as np
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import _cmd_analyze_xrd

        mock_result = MagicMock()
        mock_result.wavelength = "Custom"
        mock_result.wavelength_angstrom = None  # No angstrom value
        mock_result.n_peaks = 0
        mock_result.two_theta = np.array([])
        mock_result.intensities = np.array([])
        mock_result.d_spacings = np.array([])
        mock_result.hkl_indices = []
        mock_result.to_dict.return_value = {}

        mock_atoms = MagicMock()
        mock_atoms.get_chemical_formula.return_value = "Si"

        args = self._make_args()
        with patch("shalom.analysis.xrd.is_xrd_available", return_value=True), \
             patch("ase.io.read", return_value=mock_atoms), \
             patch("shalom.analysis.xrd.calculate_xrd", return_value=mock_result):
            rc = _cmd_analyze_xrd(args)
        assert rc == 0

    def test_plot_import_error(self, tmp_path, capsys):
        import numpy as np
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import _cmd_analyze_xrd

        mock_result = MagicMock()
        mock_result.wavelength = "CuKa"
        mock_result.wavelength_angstrom = 1.5406
        mock_result.n_peaks = 1
        mock_result.two_theta = np.array([28.4])
        mock_result.intensities = np.array([100])
        mock_result.d_spacings = np.array([3.14])
        mock_result.hkl_indices = [(1, 1, 1)]
        mock_result.to_dict.return_value = {}

        mock_atoms = MagicMock()
        mock_atoms.get_chemical_formula.return_value = "Si"

        output = str(tmp_path / "xrd.png")
        args = self._make_args(output=output)
        with patch("shalom.analysis.xrd.is_xrd_available", return_value=True), \
             patch("ase.io.read", return_value=mock_atoms), \
             patch("shalom.analysis.xrd.calculate_xrd", return_value=mock_result), \
             patch("shalom.plotting.xrd_plot.XRDPlotter",
                   side_effect=ImportError("no mpl")):
            rc = _cmd_analyze_xrd(args)
        assert rc == 0
        assert "matplotlib not installed" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Magnetic analysis with structure exception path
# ---------------------------------------------------------------------------


class TestCmdAnalyzeMagneticException:
    def test_full_analysis_exception_handled(self, tmp_path, capsys):
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import _cmd_analyze_magnetic
        import argparse

        pw_out = tmp_path / "pw.out"
        pw_out.write_text("dummy")

        args = argparse.Namespace(
            pw_out=str(pw_out), structure="POSCAR", save_json=None,
        )
        with patch("shalom.analysis.magnetic.extract_site_magnetization", return_value=[0.5]), \
             patch("shalom.analysis.magnetic.extract_lowdin_charges", return_value=None), \
             patch("ase.io.read", side_effect=Exception("cannot read structure")):
            rc = _cmd_analyze_magnetic(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "full analysis failed" in out


# ---------------------------------------------------------------------------
# MD plot matplotlib ImportError
# ---------------------------------------------------------------------------


class TestCmdAnalyzeMdPlotImportError:
    def _make_args(self, **overrides):
        import argparse
        defaults = dict(
            calc_dir="/tmp", backend="lammps",
            r_max=10.0, output=None, save_json=None,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_md_plot_import_error(self, tmp_path, capsys):
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import _cmd_analyze_md

        mock_traj = MagicMock()
        mock_traj.n_frames = 10
        mock_traj.n_atoms = 4
        mock_traj.source = "lammps"
        mock_traj.ensemble = "nvt"
        mock_traj.timestep_fs = 1.0

        mock_result = MagicMock()
        mock_result.avg_temperature = None
        mock_result.temperature_std = None
        mock_result.avg_energy = None
        mock_result.avg_pressure = None
        mock_result.diffusion_coefficient = None
        mock_result.energy_drift_per_atom = None
        mock_result.is_equilibrated = False
        mock_result.equilibration_step = None
        mock_result.rdf_r = None
        mock_result.msd_t = None
        mock_result.to_dict.return_value = {}

        out_dir = str(tmp_path / "md_plots")
        args = self._make_args(calc_dir=str(tmp_path), output=out_dir)
        with patch("shalom.backends.get_backend") as mock_be, \
             patch("shalom.analysis.md.analyze_md_trajectory", return_value=mock_result), \
             patch("shalom.plotting.md_plot.MDEnergyPlotter",
                   side_effect=ImportError("no mpl")):
            mock_be.return_value.parse_trajectory.return_value = mock_traj
            rc = _cmd_analyze_md(args)
        assert rc == 0
        assert "matplotlib not installed" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _execute_dft with quality_warnings
# ---------------------------------------------------------------------------


class TestExecuteDftQualityWarnings:
    def test_quality_warnings_printed(self, tmp_path, capsys):
        from unittest.mock import patch
        from shalom.__main__ import _execute_dft
        from shalom.backends.base import DFTResult
        from shalom.backends.runner import ExecutionResult
        import argparse

        args = argparse.Namespace(
            nprocs=1, mpi_command="mpirun", timeout=3600,
            slurm=False, partition="compute", account=None,
            nodes=1, ntasks_per_node=None, walltime="24:00:00",
            qos=None, mem=None, module_loads=[], slurm_extras=[],
            pw_command="pw.x", wsl=False,
        )
        dft = DFTResult(
            energy=-10.0, forces_max=0.05, is_converged=True,
            quality_warnings=["high_force_max"],
        )
        exec_res = ExecutionResult(success=True, return_code=0, wall_time_seconds=5.0)

        with patch("shalom.backends.runner.ExecutionRunner.validate_prerequisites",
                   return_value=[]), \
             patch("shalom.backends.runner.execute_with_recovery",
                   return_value=(exec_res, dft, [])):
            rc = _execute_dft(str(tmp_path), args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "high_force_max" in out
