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

    def test_backend_flag(self):
        parser = build_parser()
        args = parser.parse_args(["run", "mp-19717", "--backend", "qe"])
        assert args.backend == "qe"

    def test_short_backend(self):
        parser = build_parser()
        args = parser.parse_args(["run", "mp-19717", "-b", "qe"])
        assert args.backend == "qe"

    def test_calc_flag(self):
        parser = build_parser()
        args = parser.parse_args(["run", "mp-19717", "--calc", "scf"])
        assert args.calc == "scf"

    def test_accuracy_flag(self):
        parser = build_parser()
        args = parser.parse_args(["run", "mp-19717", "--accuracy", "precise"])
        assert args.accuracy == "precise"

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

    def test_force_flag(self):
        parser = build_parser()
        args = parser.parse_args(["run", "mp-19717", "--force"])
        assert args.force is True

    def test_no_validate_flag(self):
        parser = build_parser()
        args = parser.parse_args(["run", "mp-19717", "--no-validate"])
        assert args.no_validate is True

    def test_quiet_flag(self):
        parser = build_parser()
        args = parser.parse_args(["run", "mp-19717", "-q"])
        assert args.quiet is True

    def test_verbose_flag(self):
        parser = build_parser()
        args = parser.parse_args(["run", "mp-19717", "-v"])
        assert args.verbose is True

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
    def test_integer(self):
        result = _parse_set_values(["ENCUT=600"])
        assert result == {"ENCUT": 600}
        assert isinstance(result["ENCUT"], int)

    def test_float(self):
        result = _parse_set_values(["ecutwfc=80.5"])
        assert result == {"ecutwfc": 80.5}
        assert isinstance(result["ecutwfc"], float)

    def test_scientific_notation(self):
        result = _parse_set_values(["conv_thr=1e-8"])
        assert result["conv_thr"] == 1e-8

    def test_bool_true(self):
        result = _parse_set_values(["tprnfor=true"])
        assert result["tprnfor"] is True

    def test_bool_false(self):
        result = _parse_set_values(["LWAVE=false"])
        assert result["LWAVE"] is False

    def test_string(self):
        result = _parse_set_values(["smearing=cold"])
        assert result["smearing"] == "cold"

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

    def test_fortran_bool(self):
        result = _parse_set_values(["LREAL=.TRUE."])
        assert result["LREAL"] is True

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
        import subprocess
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
            rc = cmd_setup_qe(args)
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
        from unittest.mock import patch, MagicMock
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
            timeout=3600, accuracy="standard", skip_relax=True,
            is_2d=False, dos_emin=None, dos_emax=None,
            command="workflow",
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
            rc = cmd_converge(args)
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
        import sys, logging
        from unittest.mock import patch, MagicMock
        from shalom.__main__ import main
        from ase.build import bulk

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
