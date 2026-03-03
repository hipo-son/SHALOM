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
