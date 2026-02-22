"""Tests for shalom.__main__ CLI module."""

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
