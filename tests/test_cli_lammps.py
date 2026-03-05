"""Tests for CLI and MCP LAMMPS/AIMD integration.

Tests verify:
- CLI parser accepts --backend lammps
- CLI parser accepts LAMMPS-specific args (--pair-style, --md-ensemble, etc.)
- CLI parser accepts --calc aimd for VASP
- analyze md subcommand parser
- build_parser does not error
- MCP tool signatures exist
"""

import argparse

import pytest

from shalom.__main__ import build_parser


# ---------------------------------------------------------------------------
# CLI Parser
# ---------------------------------------------------------------------------


class TestCLIParser:
    """Test CLI argument parser accepts LAMMPS/AIMD options."""

    @pytest.fixture
    def parser(self):
        return build_parser()

    def test_backend_lammps_accepted(self, parser):
        args = parser.parse_args(["run", "Fe", "--backend", "lammps"])
        assert args.backend == "lammps"

    def test_backend_vasp_still_works(self, parser):
        args = parser.parse_args(["run", "Fe", "--backend", "vasp"])
        assert args.backend == "vasp"

    def test_backend_qe_still_works(self, parser):
        args = parser.parse_args(["run", "Si", "--backend", "qe"])
        assert args.backend == "qe"

    def test_calc_aimd(self, parser):
        args = parser.parse_args(["run", "Si", "--backend", "vasp", "--calc", "aimd"])
        assert args.calc == "aimd"

    def test_pair_style_arg(self, parser):
        args = parser.parse_args([
            "run", "Fe", "--backend", "lammps", "--pair-style", "eam/alloy",
        ])
        assert args.pair_style == "eam/alloy"

    def test_pair_coeff_arg(self, parser):
        args = parser.parse_args([
            "run", "Fe", "--backend", "lammps",
            "--pair-coeff", "* * Fe.eam.alloy Fe",
        ])
        assert len(args.pair_coeff) == 1
        assert "Fe.eam.alloy" in args.pair_coeff[0]

    def test_md_ensemble_arg(self, parser):
        args = parser.parse_args([
            "run", "Fe", "--backend", "lammps", "--md-ensemble", "npt",
        ])
        assert args.md_ensemble == "npt"

    def test_md_ensemble_invalid(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args([
                "run", "Fe", "--backend", "lammps", "--md-ensemble", "invalid",
            ])

    def test_temperature_arg(self, parser):
        args = parser.parse_args([
            "run", "Fe", "--backend", "lammps", "--temperature", "500",
        ])
        assert args.temperature == 500.0

    def test_md_steps_arg(self, parser):
        args = parser.parse_args([
            "run", "Fe", "--backend", "lammps", "--md-steps", "50000",
        ])
        assert args.md_steps == 50000

    def test_timestep_arg(self, parser):
        args = parser.parse_args([
            "run", "Fe", "--backend", "lammps", "--timestep", "0.5",
        ])
        assert args.timestep == 0.5

    def test_potential_dir_arg(self, parser):
        args = parser.parse_args([
            "run", "Fe", "--backend", "lammps", "--potential-dir", "/opt/lammps",
        ])
        assert args.potential_dir == "/opt/lammps"


# ---------------------------------------------------------------------------
# analyze md subcommand
# ---------------------------------------------------------------------------


class TestAnalyzeMDParser:
    """Test 'analyze md' subcommand parser."""

    @pytest.fixture
    def parser(self):
        return build_parser()

    def test_analyze_md_basic(self, parser):
        args = parser.parse_args(["analyze", "md", "--calc-dir", "/tmp/calc"])
        assert args.analyze_type == "md"
        assert args.calc_dir == "/tmp/calc"

    def test_analyze_md_backend(self, parser):
        args = parser.parse_args([
            "analyze", "md", "--calc-dir", "/tmp", "--backend", "vasp",
        ])
        assert args.backend == "vasp"

    def test_analyze_md_r_max(self, parser):
        args = parser.parse_args([
            "analyze", "md", "--calc-dir", "/tmp", "--r-max", "5.0",
        ])
        assert args.r_max == 5.0

    def test_analyze_md_output(self, parser):
        args = parser.parse_args([
            "analyze", "md", "--calc-dir", "/tmp", "-o", "/out",
        ])
        assert args.output == "/out"


# ---------------------------------------------------------------------------
# Pipeline parser
# ---------------------------------------------------------------------------


class TestPipelineParser:
    """Test pipeline subcommand accepts lammps backend."""

    @pytest.fixture
    def parser(self):
        return build_parser()

    def test_pipeline_lammps_backend(self, parser):
        args = parser.parse_args([
            "pipeline", "Find stable alloy", "--backend", "lammps",
        ])
        assert args.backend == "lammps"


# ---------------------------------------------------------------------------
# CALC_TYPE_ALIASES
# ---------------------------------------------------------------------------


class TestCalcTypeAliases:
    """Test CALC_TYPE_ALIASES includes AIMD and MD."""

    def test_aimd_alias(self):
        from shalom.direct_run import CALC_TYPE_ALIASES
        assert "aimd" in CALC_TYPE_ALIASES
        assert CALC_TYPE_ALIASES["aimd"] == "aimd"

    def test_md_alias(self):
        from shalom.direct_run import CALC_TYPE_ALIASES
        assert "md" in CALC_TYPE_ALIASES
        assert CALC_TYPE_ALIASES["md"] == "aimd"


# ---------------------------------------------------------------------------
# MCP tool existence
# ---------------------------------------------------------------------------


class TestMCPTools:
    """Verify MCP tools for MD are importable."""

    def test_run_md_importable(self):
        mcp = pytest.importorskip("mcp", reason="mcp package not installed")
        from shalom.mcp_server import run_md
        assert callable(run_md)

    def test_analyze_md_trajectory_importable(self):
        mcp = pytest.importorskip("mcp", reason="mcp package not installed")
        from shalom.mcp_server import analyze_md_trajectory
        assert callable(analyze_md_trajectory)


# ---------------------------------------------------------------------------
# Phase 4: direct_run LAMMPS paths, _resolve_calc_type, _write_run_info
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock, patch


class TestResolveCalcType:
    """Test _resolve_calc_type (direct_run.py lines 318-326)."""

    def test_none_defaults_to_relaxation(self):
        from shalom.direct_run import _resolve_calc_type
        assert _resolve_calc_type(None, "vasp") == "relaxation"

    def test_known_alias(self):
        from shalom.direct_run import _resolve_calc_type
        assert _resolve_calc_type("scf", "qe") == "static"

    def test_unknown_defaults_to_relaxation(self):
        from shalom.direct_run import _resolve_calc_type
        result = _resolve_calc_type("imaginary_calc", "vasp")
        assert result == "relaxation"

    def test_aimd_alias(self):
        from shalom.direct_run import _resolve_calc_type
        assert _resolve_calc_type("aimd", "vasp") == "aimd"


class TestWriteRunInfo:
    """Test _write_run_info (direct_run.py lines 282-315)."""

    def test_writes_json_file(self, tmp_path):
        import json
        from shalom.direct_run import _write_run_info
        _write_run_info(
            str(tmp_path), "qe", "scf", {"source": "test"}, {"ecutwfc": 60},
            ["pw.in"], structure_analysis={"formula": "Si"},
        )
        info_path = tmp_path / "run_info.json"
        assert info_path.exists()
        data = json.loads(info_path.read_text())
        assert data["backend"] == "qe"
        assert data["structure_analysis"]["formula"] == "Si"

    def test_oserror_no_crash(self, tmp_path):
        from shalom.direct_run import _write_run_info
        with patch("builtins.open", side_effect=OSError("disk full")):
            # Should not raise
            _write_run_info(
                str(tmp_path), "vasp", "relaxation", None, None, [],
            )


class TestDirectRunLAMMPSPath:
    """Test direct_run with LAMMPS backend (direct_run.py lines 559-584)."""

    def test_lammps_backend_calls_get_lammps_preset(self, tmp_path):
        from shalom.direct_run import direct_run, DirectRunConfig

        config = DirectRunConfig(
            backend_name="lammps",
            calc_type="md",
            output_dir=str(tmp_path),
        )

        mock_lammps_config = MagicMock()
        mock_lammps_config.pair_style = "lj/cut"
        mock_lammps_config.detected_ff = "LJ"
        mock_lammps_config.timestep = 1.0
        mock_lammps_config.is_2d = False
        mock_lammps_config.to_summary_dict.return_value = {}
        mock_lammps_config._detection_log = ["auto LJ"]
        mock_lammps_config.potential_files = []

        mock_backend = MagicMock()
        mock_backend.name = "lammps"
        mock_backend.write_input.return_value = str(tmp_path)

        with patch("shalom.direct_run.get_backend", return_value=mock_backend), \
             patch("shalom.backends.lammps_config.get_lammps_preset", return_value=mock_lammps_config) as mock_preset, \
             patch("shalom.direct_run._resolve_workspace", return_value=str(tmp_path)), \
             patch("shalom.direct_run.ase_read") as mock_read:
            mock_atoms = MagicMock()
            mock_atoms.get_chemical_formula.return_value = "Ar"
            mock_read.return_value = mock_atoms
            result = direct_run("Ar", config)

        mock_preset.assert_called_once()
        assert result.success

    def test_lammps_pair_style_override_clears_potentials(self, tmp_path):
        from shalom.direct_run import direct_run, DirectRunConfig

        config = DirectRunConfig(
            backend_name="lammps",
            calc_type="md",
            output_dir=str(tmp_path),
            user_settings={"pair_style": "lj/cut 10.0", "pair_coeff": "* * 1.0 3.4"},
        )

        mock_lammps_config = MagicMock()
        mock_lammps_config.pair_style = "eam/alloy"
        mock_lammps_config.detected_ff = "EAM"
        mock_lammps_config.timestep = 1.0
        mock_lammps_config.is_2d = False
        mock_lammps_config.to_summary_dict.return_value = {}
        mock_lammps_config._detection_log = []
        mock_lammps_config.potential_files = ["Fe.eam.alloy"]

        mock_backend = MagicMock()
        mock_backend.name = "lammps"
        mock_backend.write_input.return_value = str(tmp_path)

        with patch("shalom.direct_run.get_backend", return_value=mock_backend), \
             patch("shalom.backends.lammps_config.get_lammps_preset", return_value=mock_lammps_config), \
             patch("shalom.direct_run._resolve_workspace", return_value=str(tmp_path)), \
             patch("shalom.direct_run.ase_read") as mock_read:
            mock_atoms = MagicMock()
            mock_atoms.get_chemical_formula.return_value = "Fe"
            mock_read.return_value = mock_atoms
            result = direct_run("Fe", config)

        # potential_files should be cleared when pair_style is overridden
        assert mock_lammps_config.potential_files == []


class TestBuildStructureAnalysis:
    """Test _build_structure_analysis (direct_run.py lines 95-112)."""

    def test_magnetic_elements_detected(self):
        from shalom.direct_run import _build_structure_analysis
        from ase import Atoms
        atoms = Atoms("Fe2O3", positions=[[0]*3]*5,
                       cell=[5, 5, 5], pbc=True)
        result = _build_structure_analysis(atoms)
        assert result["is_magnetic"] is True
        assert "Fe" in result["magnetic_elements"]

    def test_non_magnetic(self):
        from shalom.direct_run import _build_structure_analysis
        from ase import Atoms
        atoms = Atoms("Si2", positions=[[0, 0, 0], [1.35, 1.35, 1.35]],
                       cell=[2.7, 2.7, 2.7], pbc=True)
        result = _build_structure_analysis(atoms)
        assert result["is_magnetic"] is False
        assert "magnetic_elements" not in result
