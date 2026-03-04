"""Tests for the load_calc_context MCP tool."""

from __future__ import annotations

import json
import os
import sys
import types

import pytest


# ---------------------------------------------------------------------------
# Mock FastMCP so mcp_server can be imported without the mcp package
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _mock_fastmcp(monkeypatch):
    """Mock the mcp.server.fastmcp module so mcp_server can be imported."""

    class FakeFastMCP:
        def __init__(self, name, **kwargs):
            self.name = name

        def tool(self):
            def decorator(fn):
                return fn
            return decorator

        def run(self, **kwargs):
            pass

    mcp_mod = types.ModuleType("mcp")
    mcp_server_mod = types.ModuleType("mcp.server")
    mcp_fastmcp_mod = types.ModuleType("mcp.server.fastmcp")
    mcp_fastmcp_mod.FastMCP = FakeFastMCP
    mcp_mod.server = mcp_server_mod
    mcp_server_mod.fastmcp = mcp_fastmcp_mod

    monkeypatch.setitem(sys.modules, "mcp", mcp_mod)
    monkeypatch.setitem(sys.modules, "mcp.server", mcp_server_mod)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", mcp_fastmcp_mod)
    sys.modules.pop("shalom.mcp_server", None)


def _call_load_calc_context(calc_dir: str):
    """Import and call the load_calc_context tool function directly."""
    import shalom.mcp_server as mod
    return mod.load_calc_context(calc_dir=calc_dir)


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

_RUN_INFO_V2 = {
    "version": 2,
    "timestamp": "2026-03-04T12:00:00",
    "backend": "qe",
    "calc_type": "scf",
    "structure_info": {"source": "ase_bulk", "formula": "Si"},
    "structure_analysis": {
        "formula": "Si",
        "natoms": 2,
        "elements": ["Si"],
        "is_magnetic": False,
        "is_2d": False,
        "is_metal": False,
    },
    "auto_detected": {"ecutwfc": 30.0, "kpoints": [6, 6, 6]},
    "calculation_parameters": {
        "calc_type": "scf",
        "accuracy": "standard",
        "functional": "PBE",
    },
    "detection_log": [
        "ecutwfc=30.0 Ry, ecutrho=240.0 Ry (SSSP)",
        "K-grid [6, 6, 6] from kpr=30.0",
    ],
    "files_generated": ["pw.in"],
}

_RUN_INFO_V1 = {
    "version": 1,
    "timestamp": "2026-02-01T10:00:00",
    "backend": "qe",
    "calc_type": "scf",
    "auto_detected": {"ecutwfc": 30.0},
    "files_generated": ["pw.in"],
}

_RESULTS_SUMMARY = {
    "version": 3,
    "timestamp": "2026-03-04T12:05:00",
    "material_formula": "Si",
    "accuracy": "standard",
    "fermi_energy_eV": 5.234,
    "completed_steps": ["vc_relax", "scf", "bands", "nscf", "dos"],
    "detection_log": ["ecutwfc=30.0 Ry"],
    "structure_analysis": {"formula": "Si", "natoms": 2},
}


class TestLoadCalcContext:
    def test_load_run_info_v2(self, tmp_path):
        """run_info.json v2 is loaded with all fields."""
        (tmp_path / "run_info.json").write_text(
            json.dumps(_RUN_INFO_V2), encoding="utf-8"
        )
        result = _call_load_calc_context(str(tmp_path))
        assert result["success"] is True
        assert result["run_info"]["version"] == 2
        assert result["run_info"]["backend"] == "qe"
        assert result["run_info"]["detection_log"][0].startswith("ecutwfc=")
        assert result["run_info"]["structure_analysis"]["formula"] == "Si"

    def test_load_results_summary_only(self, tmp_path):
        """When only results_summary.json exists."""
        (tmp_path / "results_summary.json").write_text(
            json.dumps(_RESULTS_SUMMARY), encoding="utf-8"
        )
        result = _call_load_calc_context(str(tmp_path))
        assert result["success"] is True
        assert "run_info" not in result
        assert result["results_summary"]["fermi_energy_eV"] == 5.234
        assert result["results_summary"]["detection_log"][0] == "ecutwfc=30.0 Ry"

    def test_load_both_files(self, tmp_path):
        """Both run_info and results_summary present — both returned."""
        (tmp_path / "run_info.json").write_text(
            json.dumps(_RUN_INFO_V2), encoding="utf-8"
        )
        (tmp_path / "results_summary.json").write_text(
            json.dumps(_RESULTS_SUMMARY), encoding="utf-8"
        )
        result = _call_load_calc_context(str(tmp_path))
        assert result["success"] is True
        assert "run_info" in result
        assert "results_summary" in result

    def test_no_context_files(self, tmp_path):
        """Empty directory → success=False, available_files may exist."""
        result = _call_load_calc_context(str(tmp_path))
        assert result["success"] is False
        assert "No context files" in result["error"]

    def test_no_context_but_available_files(self, tmp_path):
        """No context JSON, but other files exist → listed in available_files."""
        (tmp_path / "pw.in").write_text("test", encoding="utf-8")
        (tmp_path / "pw.out").write_text("test", encoding="utf-8")
        result = _call_load_calc_context(str(tmp_path))
        assert result["success"] is False
        assert "pw.in" in result.get("available_files", [])
        assert "pw.out" in result.get("available_files", [])

    def test_nonexistent_dir(self, tmp_path):
        """Non-existent path → success=False."""
        result = _call_load_calc_context(str(tmp_path / "nonexistent"))
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_analysis_results_discovered(self, tmp_path):
        """*_results.json files are listed in analysis_results."""
        (tmp_path / "run_info.json").write_text(
            json.dumps(_RUN_INFO_V2), encoding="utf-8"
        )
        (tmp_path / "phonon_results.json").write_text("{}", encoding="utf-8")
        (tmp_path / "xrd_results.json").write_text("{}", encoding="utf-8")
        result = _call_load_calc_context(str(tmp_path))
        assert result["success"] is True
        assert "phonon_results.json" in result["analysis_results"]
        assert "xrd_results.json" in result["analysis_results"]

    def test_available_files_listed(self, tmp_path):
        """Important files appear in available_files."""
        (tmp_path / "run_info.json").write_text(
            json.dumps(_RUN_INFO_V2), encoding="utf-8"
        )
        (tmp_path / "pw.in").write_text("input", encoding="utf-8")
        (tmp_path / "pw.out").write_text("output", encoding="utf-8")
        (tmp_path / "bands.png").write_bytes(b"PNG")
        result = _call_load_calc_context(str(tmp_path))
        assert "pw.in" in result["available_files"]
        assert "pw.out" in result["available_files"]
        assert "bands.png" in result["available_files"]

    def test_subdirectory_files(self, tmp_path):
        """Files in workflow sub-step dirs are listed with prefix."""
        (tmp_path / "results_summary.json").write_text(
            json.dumps(_RESULTS_SUMMARY), encoding="utf-8"
        )
        scf_dir = tmp_path / "02_scf"
        scf_dir.mkdir()
        (scf_dir / "pw.in").write_text("input", encoding="utf-8")
        (scf_dir / "pw.out").write_text("output", encoding="utf-8")
        result = _call_load_calc_context(str(tmp_path))
        assert result["success"] is True
        available = result.get("available_files", [])
        assert "02_scf/pw.in" in available
        assert "02_scf/pw.out" in available

    def test_malformed_json_graceful(self, tmp_path):
        """Malformed JSON → graceful fallback, other files still collected."""
        (tmp_path / "run_info.json").write_text("{bad json", encoding="utf-8")
        (tmp_path / "results_summary.json").write_text(
            json.dumps(_RESULTS_SUMMARY), encoding="utf-8"
        )
        result = _call_load_calc_context(str(tmp_path))
        # run_info should be skipped, but results_summary should still load
        assert result["success"] is True
        assert "run_info" not in result
        assert "results_summary" in result

    def test_v1_run_info_backward_compat(self, tmp_path):
        """Version 1 run_info.json loads without error."""
        (tmp_path / "run_info.json").write_text(
            json.dumps(_RUN_INFO_V1), encoding="utf-8"
        )
        result = _call_load_calc_context(str(tmp_path))
        assert result["success"] is True
        assert result["run_info"]["version"] == 1
        assert result["run_info"]["backend"] == "qe"

    def test_analysis_results_in_subdirs(self, tmp_path):
        """*_results.json in sub-step dirs are also discovered."""
        (tmp_path / "run_info.json").write_text(
            json.dumps(_RUN_INFO_V2), encoding="utf-8"
        )
        nscf_dir = tmp_path / "04_nscf"
        nscf_dir.mkdir()
        (nscf_dir / "electronic_results.json").write_text("{}", encoding="utf-8")
        result = _call_load_calc_context(str(tmp_path))
        assert "04_nscf/electronic_results.json" in result.get("analysis_results", [])

    def test_json_metadata_not_in_available_files(self, tmp_path):
        """run_info.json and results_summary.json should not appear in available_files."""
        (tmp_path / "run_info.json").write_text(
            json.dumps(_RUN_INFO_V2), encoding="utf-8"
        )
        (tmp_path / "results_summary.json").write_text(
            json.dumps(_RESULTS_SUMMARY), encoding="utf-8"
        )
        (tmp_path / "pw.in").write_text("input", encoding="utf-8")
        result = _call_load_calc_context(str(tmp_path))
        assert result["success"] is True
        available = result.get("available_files", [])
        assert "run_info.json" not in available
        assert "results_summary.json" not in available
        assert "pw.in" in available
