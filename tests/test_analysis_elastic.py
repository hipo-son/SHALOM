"""Tests for shalom.analysis elastic module.

All tests are pure unit tests — no DFT execution required.
Uses known Si elastic constants for validation.
"""

import json

import numpy as np
import pytest
from unittest.mock import patch

from shalom.analysis._base import ElasticResult
from shalom.analysis.elastic import (
    is_elastic_available,
    analyze_elastic_tensor,
    analyze_stress_strain,
    _ensure_elastic_available,
)


# ---------------------------------------------------------------------------
# Reference data: Si cubic elastic constants (GPa)
# C11=165.7, C12=63.9, C44=79.6
# ---------------------------------------------------------------------------

def _si_tensor() -> list:
    """Return Si cubic elastic tensor as 6x6 nested list (GPa)."""
    c11, c12, c44 = 165.7, 63.9, 79.6
    return [
        [c11, c12, c12, 0.0, 0.0, 0.0],
        [c12, c11, c12, 0.0, 0.0, 0.0],
        [c12, c12, c11, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, c44, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, c44, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, c44],
    ]


def _unstable_tensor() -> list:
    """Return a tensor with a negative eigenvalue (mechanically unstable)."""
    # Make C44 negative → unstable
    c11, c12, c44 = 165.7, 63.9, -10.0
    return [
        [c11, c12, c12, 0.0, 0.0, 0.0],
        [c12, c11, c12, 0.0, 0.0, 0.0],
        [c12, c12, c11, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, c44, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, c44, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, c44],
    ]


# ---------------------------------------------------------------------------
# ElasticResult dataclass
# ---------------------------------------------------------------------------


class TestElasticResult:
    def test_default_fields(self):
        r = ElasticResult(elastic_tensor=np.zeros((6, 6)))
        assert r.bulk_modulus_vrh is None
        assert r.shear_modulus_vrh is None
        assert r.youngs_modulus is None
        assert r.poisson_ratio is None
        assert r.is_stable is False
        assert r.stability_violations == []
        assert r.universal_anisotropy is None
        assert r.compliance_tensor is None
        assert r.raw is None
        assert r.metadata == {}

    def test_elastic_tensor_stored(self):
        t = np.eye(6)
        r = ElasticResult(elastic_tensor=t)
        assert r.elastic_tensor is t

    def test_metadata_mutable(self):
        r = ElasticResult(elastic_tensor=np.zeros((6, 6)))
        r.metadata["source"] = "test"
        assert r.metadata["source"] == "test"


# ---------------------------------------------------------------------------
# Availability guard
# ---------------------------------------------------------------------------


class TestAvailabilityGuard:
    def test_is_elastic_available(self):
        assert is_elastic_available() is True

    def test_ensure_elastic_available_passes(self):
        _ensure_elastic_available()  # should not raise

    def test_ensure_raises_when_unavailable(self):
        with patch("shalom.analysis.elastic._ELASTIC_AVAILABLE", False):
            with pytest.raises(ImportError, match="pymatgen"):
                _ensure_elastic_available()

    def test_is_elastic_available_false(self):
        with patch("shalom.analysis.elastic._ELASTIC_AVAILABLE", False):
            assert is_elastic_available() is False


# ---------------------------------------------------------------------------
# analyze_elastic_tensor
# ---------------------------------------------------------------------------


class TestAnalyzeElasticTensor:
    def test_si_bulk_modulus(self):
        result = analyze_elastic_tensor(_si_tensor())
        # Si VRH bulk modulus ≈ 97.8 GPa
        assert result.bulk_modulus_vrh is not None
        assert abs(result.bulk_modulus_vrh - 97.83) < 1.0

    def test_si_shear_modulus(self):
        result = analyze_elastic_tensor(_si_tensor())
        # Si VRH shear modulus ≈ 66.5 GPa
        assert result.shear_modulus_vrh is not None
        assert abs(result.shear_modulus_vrh - 66.54) < 1.0

    def test_si_youngs_modulus(self):
        result = analyze_elastic_tensor(_si_tensor())
        # Si Young's modulus ≈ 162.7 GPa
        assert result.youngs_modulus is not None
        assert abs(result.youngs_modulus - 162.7) < 1.0

    def test_si_poisson_ratio(self):
        result = analyze_elastic_tensor(_si_tensor())
        # Si Poisson's ratio ≈ 0.223
        assert result.poisson_ratio is not None
        assert abs(result.poisson_ratio - 0.223) < 0.01

    def test_si_anisotropy(self):
        result = analyze_elastic_tensor(_si_tensor())
        assert result.universal_anisotropy is not None
        assert abs(result.universal_anisotropy - 0.244) < 0.01

    def test_si_is_stable(self):
        result = analyze_elastic_tensor(_si_tensor())
        assert result.is_stable is True
        assert result.stability_violations == []

    def test_si_tensor_shape(self):
        result = analyze_elastic_tensor(_si_tensor())
        assert result.elastic_tensor.shape == (6, 6)

    def test_si_compliance_tensor(self):
        result = analyze_elastic_tensor(_si_tensor())
        assert result.compliance_tensor is not None
        assert result.compliance_tensor.shape == (6, 6)

    def test_si_raw_object(self):
        result = analyze_elastic_tensor(_si_tensor())
        assert result.raw is not None
        assert hasattr(result.raw, "k_vrh")

    def test_invalid_shape_5x5(self):
        tensor_5x5 = [[1.0] * 5 for _ in range(5)]
        with pytest.raises(ValueError, match="6x6"):
            analyze_elastic_tensor(tensor_5x5)

    def test_invalid_shape_6x5(self):
        tensor_6x5 = [[1.0] * 5 for _ in range(6)]
        with pytest.raises(ValueError, match="6x6"):
            analyze_elastic_tensor(tensor_6x5)

    def test_invalid_shape_flat(self):
        with pytest.raises((ValueError, Exception)):
            analyze_elastic_tensor([1.0] * 36)

    def test_isotropic_tensor(self):
        """Isotropic material: C11=200, C12=100, C44=50 GPa (C44 = (C11-C12)/2)."""
        c11, c12, c44 = 200.0, 100.0, 50.0
        tensor = [
            [c11, c12, c12, 0, 0, 0],
            [c12, c11, c12, 0, 0, 0],
            [c12, c12, c11, 0, 0, 0],
            [0, 0, 0, c44, 0, 0],
            [0, 0, 0, 0, c44, 0],
            [0, 0, 0, 0, 0, c44],
        ]
        result = analyze_elastic_tensor(tensor)
        # Isotropic: A_U should be ~0
        assert result.universal_anisotropy is not None
        assert abs(result.universal_anisotropy) < 0.01
        assert result.is_stable is True


# ---------------------------------------------------------------------------
# Stability checks
# ---------------------------------------------------------------------------


class TestStability:
    def test_unstable_negative_c44(self):
        result = analyze_elastic_tensor(_unstable_tensor())
        assert result.is_stable is False
        assert len(result.stability_violations) > 0
        assert "Non-positive eigenvalue" in result.stability_violations[0]

    def test_zero_tensor_raises(self):
        """Zero tensor is singular — pymatgen raises LinAlgError."""
        zero = [[0.0] * 6 for _ in range(6)]
        with pytest.raises(Exception):
            analyze_elastic_tensor(zero)

    def test_identity_tensor_stable(self):
        identity = np.eye(6).tolist()
        result = analyze_elastic_tensor(identity)
        assert result.is_stable is True


# ---------------------------------------------------------------------------
# analyze_stress_strain
# ---------------------------------------------------------------------------


class TestAnalyzeStressStrain:
    def test_mismatched_lengths(self):
        stresses = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]
        strains = []
        with pytest.raises(ValueError, match="same length"):
            analyze_stress_strain(stresses, strains)

    def test_empty_inputs(self):
        with pytest.raises(ValueError, match="At least one"):
            analyze_stress_strain([], [])

    def test_invalid_stress_shape(self):
        stresses = [[[1, 0], [0, 1]]]  # 2x2 not 3x3
        strains = [[[0.01, 0, 0], [0, 0, 0], [0, 0, 0]]]
        with pytest.raises(ValueError, match="3x3"):
            analyze_stress_strain(stresses, strains)

    def test_invalid_strain_shape(self):
        stresses = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]
        strains = [[[0.01, 0], [0, 0]]]  # 2x2 not 3x3
        with pytest.raises(ValueError, match="3x3"):
            analyze_stress_strain(stresses, strains)

    def test_returns_elastic_result(self):
        """Smoke test with synthetic data — just verify return type."""
        # Create trivial stress-strain pairs (identity strain, unit stress)
        stresses = [
            [[100, 0, 0], [0, 50, 0], [0, 0, 50]],
            [[50, 0, 0], [0, 100, 0], [0, 0, 50]],
            [[50, 0, 0], [0, 50, 0], [0, 0, 100]],
            [[0, 0, 0], [0, 0, 25], [0, 25, 0]],
            [[0, 0, 25], [0, 0, 0], [25, 0, 0]],
            [[0, 25, 0], [25, 0, 0], [0, 0, 0]],
        ]
        strains = [
            [[0.01, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0.01, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0.01]],
            [[0, 0, 0], [0, 0, 0.01], [0, 0.01, 0]],
            [[0, 0, 0.01], [0, 0, 0], [0.01, 0, 0]],
            [[0, 0.01, 0], [0.01, 0, 0], [0, 0, 0]],
        ]
        result = analyze_stress_strain(stresses, strains)
        assert isinstance(result, ElasticResult)
        assert result.elastic_tensor is not None
        assert result.bulk_modulus_vrh is not None


# ---------------------------------------------------------------------------
# MCP tool integration
# ---------------------------------------------------------------------------


def _import_mcp_tool():
    """Import analyze_elastic from mcp_server, skipping if mcp not installed."""
    try:
        from shalom.mcp_server import analyze_elastic
        return analyze_elastic
    except (ImportError, SystemExit):
        pytest.skip("MCP package not installed")


class TestMCPTool:
    def test_analyze_elastic_tool_success(self):
        """Test the MCP tool with valid input."""
        analyze_elastic = _import_mcp_tool()
        tensor_json = json.dumps(_si_tensor())
        result = analyze_elastic(tensor_json)
        assert result["success"] is True
        assert "bulk_modulus_GPa" in result
        assert abs(result["bulk_modulus_GPa"] - 97.83) < 1.0
        assert result["is_mechanically_stable"] is True

    def test_analyze_elastic_tool_invalid_json(self):
        analyze_elastic = _import_mcp_tool()
        result = analyze_elastic("not valid json")
        assert result["success"] is False
        assert "Invalid JSON" in result["error"]

    def test_analyze_elastic_tool_wrong_shape(self):
        analyze_elastic = _import_mcp_tool()
        result = analyze_elastic(json.dumps([[1, 2], [3, 4]]))
        assert result["success"] is False
        assert "6x6" in result["error"]


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCLI:
    def test_analyze_elastic_cli(self):
        """Test CLI analyze elastic subcommand."""
        from shalom.__main__ import build_parser, cmd_analyze

        parser = build_parser()
        tensor_json = json.dumps(_si_tensor())
        args = parser.parse_args(["analyze", "elastic", "--tensor", tensor_json])
        assert args.command == "analyze"
        assert args.analyze_type == "elastic"

        rc = cmd_analyze(args)
        assert rc == 0

    def test_analyze_no_type(self):
        from shalom.__main__ import build_parser, cmd_analyze

        parser = build_parser()
        args = parser.parse_args(["analyze"])
        rc = cmd_analyze(args)
        assert rc == 1

    def test_analyze_elastic_no_input(self):
        from shalom.__main__ import build_parser, cmd_analyze

        parser = build_parser()
        args = parser.parse_args(["analyze", "elastic"])
        rc = cmd_analyze(args)
        assert rc == 1

    def test_analyze_elastic_invalid_json(self):
        from shalom.__main__ import build_parser, cmd_analyze

        parser = build_parser()
        args = parser.parse_args(["analyze", "elastic", "--tensor", "bad"])
        rc = cmd_analyze(args)
        assert rc == 1

    def test_analyze_elastic_from_file(self, tmp_path):
        """Test loading tensor from a JSON file."""
        from shalom.__main__ import build_parser, cmd_analyze

        tensor_file = tmp_path / "tensor.json"
        tensor_file.write_text(json.dumps(_si_tensor()))

        parser = build_parser()
        args = parser.parse_args(
            ["analyze", "elastic", "--file", str(tensor_file)]
        )
        rc = cmd_analyze(args)
        assert rc == 0


# ---------------------------------------------------------------------------
# Package imports
# ---------------------------------------------------------------------------


class TestPackageImports:
    def test_analysis_init_exports(self):
        from shalom.analysis import (
            ElasticResult,
            analyze_elastic_tensor,
            analyze_stress_strain,
            is_elastic_available,
        )
        assert callable(analyze_elastic_tensor)
        assert callable(analyze_stress_strain)
        assert callable(is_elastic_available)
        assert ElasticResult is not None

    def test_analysis_all(self):
        import shalom.analysis
        assert "ElasticResult" in shalom.analysis.__all__
        assert "analyze_elastic_tensor" in shalom.analysis.__all__
        assert "analyze_stress_strain" in shalom.analysis.__all__
        assert "is_elastic_available" in shalom.analysis.__all__
