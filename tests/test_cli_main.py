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
        assert "[OK]" in out

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
