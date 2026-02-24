"""Tests for SHALOM MCP Server tool functions.

Tests the underlying logic of each MCP tool without requiring the MCP SDK.
Each tool function is tested by mocking the SHALOM library calls it wraps.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk


# ---------------------------------------------------------------------------
# Helper: import the tool functions (skip if mcp not installed)
# ---------------------------------------------------------------------------

# The mcp_server module imports FastMCP at module level; if mcp is not installed
# it calls sys.exit(1). We mock that import to test the tool functions.

@pytest.fixture(autouse=True)
def _mock_fastmcp(monkeypatch):
    """Mock the mcp.server.fastmcp module so mcp_server can be imported."""
    import sys
    import types

    # Create a fake FastMCP class
    class FakeFastMCP:
        def __init__(self, name, **kwargs):
            self.name = name
            self._tools = {}

        def tool(self):
            def decorator(fn):
                self._tools[fn.__name__] = fn
                return fn
            return decorator

        def run(self, **kwargs):
            pass

    # Create fake mcp modules
    mcp_mod = types.ModuleType("mcp")
    mcp_server_mod = types.ModuleType("mcp.server")
    mcp_fastmcp_mod = types.ModuleType("mcp.server.fastmcp")
    mcp_fastmcp_mod.FastMCP = FakeFastMCP
    mcp_mod.server = mcp_server_mod
    mcp_server_mod.fastmcp = mcp_fastmcp_mod

    monkeypatch.setitem(sys.modules, "mcp", mcp_mod)
    monkeypatch.setitem(sys.modules, "mcp.server", mcp_server_mod)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", mcp_fastmcp_mod)

    # Remove cached mcp_server module to force re-import with mocked FastMCP
    sys.modules.pop("shalom.mcp_server", None)


def _import_mcp_server():
    """Import mcp_server after mocking FastMCP."""
    import shalom.mcp_server as mod
    return mod


# ===========================================================================
# Test: search_material
# ===========================================================================

class TestSearchMaterial:
    def test_search_by_mp_id(self):
        mod = _import_mcp_server()
        si = bulk("Si")

        mock_result = MagicMock()
        mock_result.mp_id = "mp-149"
        mock_result.formula = "Si"
        mock_result.space_group = "Fd-3m"
        mock_result.energy_above_hull = 0.0
        mock_result.atoms = si

        with patch("shalom.mp_client.is_mp_available", return_value=True), \
             patch("shalom.mp_client.is_mp_id", return_value=True), \
             patch("shalom.mp_client.fetch_by_mp_id", return_value=mock_result):
            result = mod.search_material("mp-149")

        assert result["success"] is True
        assert len(result["results"]) == 1
        assert result["results"][0]["mp_id"] == "mp-149"
        assert result["results"][0]["formula"] == "Si"

    def test_search_by_formula(self):
        mod = _import_mcp_server()
        si = bulk("Si")

        mock_result = MagicMock()
        mock_result.mp_id = "mp-149"
        mock_result.formula = "Si"
        mock_result.space_group = "Fd-3m"
        mock_result.energy_above_hull = 0.0
        mock_result.atoms = si

        with patch("shalom.mp_client.is_mp_available", return_value=True), \
             patch("shalom.mp_client.is_mp_id", return_value=False), \
             patch("shalom.mp_client.search_by_formula", return_value=[mock_result]):
            result = mod.search_material("Si", max_results=3)

        assert result["success"] is True
        assert len(result["results"]) == 1

    def test_mp_not_available(self):
        mod = _import_mcp_server()

        with patch("shalom.mp_client.is_mp_available", return_value=False):
            result = mod.search_material("Si")

        assert result["success"] is False
        assert "mp-api" in result["error"]

    def test_search_error(self):
        mod = _import_mcp_server()

        with patch("shalom.mp_client.is_mp_available", return_value=True), \
             patch("shalom.mp_client.is_mp_id", return_value=True), \
             patch("shalom.mp_client.fetch_by_mp_id",
                   side_effect=ValueError("Not found")):
            result = mod.search_material("mp-999999")

        assert result["success"] is False
        assert "Not found" in result["error"]


# ===========================================================================
# Test: generate_dft_input
# ===========================================================================

class TestGenerateDftInput:
    def test_generate_qe_scf(self):
        mod = _import_mcp_server()

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output_dir = "/tmp/test_output"
        mock_result.files_generated = ["pw.in"]
        mock_result.backend_name = "qe"
        mock_result.structure_info = {"formula": "Si"}
        mock_result.auto_detected = None
        mock_result.error = None

        with patch("shalom.direct_run.direct_run", return_value=mock_result):
            result = mod.generate_dft_input(
                material="Si", backend="qe", calc_type="scf",
            )

        assert result["success"] is True
        assert result["output_dir"] == "/tmp/test_output"
        assert "pw.in" in result["files_generated"]

    def test_generate_error(self):
        mod = _import_mcp_server()

        with patch("shalom.direct_run.direct_run",
                   side_effect=RuntimeError("No structure")):
            result = mod.generate_dft_input(material="XYZ123")

        assert result["success"] is False
        assert "No structure" in result["error"]


# ===========================================================================
# Test: run_workflow
# ===========================================================================

class TestRunWorkflow:
    def test_run_success(self):
        mod = _import_mcp_server()

        mock_wf_result = {
            "fermi_energy": 6.48,
            "bands_png": "/tmp/wf/bands.png",
            "dos_png": "/tmp/wf/dos.png",
            "calc_dirs": {"01_relax": "/tmp/wf/01_relax"},
        }

        with patch.object(mod, "_load_atoms", return_value=bulk("Si")), \
             patch("shalom.workflows.standard.StandardWorkflow") as MockWF:
            MockWF.return_value.run.return_value = mock_wf_result
            result = mod.run_workflow(
                material="Si",
                output_dir="/tmp/wf",
                nprocs=2,
            )

        assert result["success"] is True
        assert result["fermi_energy_eV"] == 6.48
        assert result["bands_png"] == "/tmp/wf/bands.png"

    def test_run_bad_material(self):
        mod = _import_mcp_server()

        with patch.object(mod, "_load_atoms",
                         side_effect=ValueError("Cannot resolve")):
            result = mod.run_workflow(material="ZZZZZ")

        assert result["success"] is False
        assert "Cannot resolve" in result["error"]


# ===========================================================================
# Test: execute_dft
# ===========================================================================

class TestExecuteDft:
    def test_execute_success(self):
        mod = _import_mcp_server()

        mock_exec = MagicMock()
        mock_exec.success = True
        mock_exec.wall_time_seconds = 45.2
        mock_exec.error_message = None

        mock_dft = MagicMock()
        mock_dft.is_converged = True
        mock_dft.energy = -310.5
        mock_dft.forces_max = 0.001
        mock_dft.quality_warnings = []

        with patch("shalom.backends.runner.ExecutionRunner") as MockRunner, \
             patch("shalom.backends.runner.execute_with_recovery",
                   return_value=(mock_exec, mock_dft, [])):
            MockRunner.return_value.validate_prerequisites.return_value = []
            result = mod.execute_dft(
                calc_dir="/tmp/calc", nprocs=4,
            )

        assert result["success"] is True
        assert result["converged"] is True
        assert result["energy_eV"] == -310.5

    def test_prereq_failure(self):
        mod = _import_mcp_server()

        with patch("shalom.backends.runner.ExecutionRunner") as MockRunner:
            MockRunner.return_value.validate_prerequisites.return_value = [
                "pw.x not found"
            ]
            result = mod.execute_dft(calc_dir="/tmp/calc")

        assert result["success"] is False
        assert "pw.x" in result["error"]


# ===========================================================================
# Test: parse_dft_output
# ===========================================================================

class TestParseDftOutput:
    def test_parse_qe(self):
        mod = _import_mcp_server()

        mock_result = MagicMock()
        mock_result.is_converged = True
        mock_result.energy = -155.3
        mock_result.forces_max = 0.01
        mock_result.pressure = None
        mock_result.quality_warnings = []

        with patch("shalom.backends.get_backend") as mock_get:
            mock_get.return_value.parse_output.return_value = mock_result
            result = mod.parse_dft_output("/tmp/calc", backend="qe")

        assert result["success"] is True
        assert result["converged"] is True
        assert result["energy_eV"] == -155.3

    def test_parse_error(self):
        mod = _import_mcp_server()

        with patch("shalom.backends.get_backend") as mock_get:
            mock_get.return_value.parse_output.side_effect = FileNotFoundError("No pw.out")
            result = mod.parse_dft_output("/nonexistent")

        assert result["success"] is False
        assert "pw.out" in result["error"]


# ===========================================================================
# Test: plot_bands
# ===========================================================================

class TestPlotBands:
    def test_plot_success(self, tmp_path):
        mod = _import_mcp_server()

        # Create fake pw.out with Fermi energy
        pw_out = tmp_path / "pw.out"
        pw_out.write_text("the Fermi energy is     6.4841 ev\n")

        mock_bs = MagicMock()
        mock_bs.eigenvalues = np.zeros((10, 5))

        with patch("shalom.backends.qe_parser.find_xml_path",
                   return_value=str(tmp_path / "data.xml")), \
             patch("shalom.backends.qe_parser.parse_xml_bands",
                   return_value=mock_bs), \
             patch("shalom.plotting.band_plot.BandStructurePlotter") as MockPlotter:
            result = mod.plot_bands(
                calc_dir=str(tmp_path),
                output=str(tmp_path / "bands.png"),
            )

        assert result["success"] is True
        assert result["n_kpoints"] == 10
        assert result["n_bands"] == 5
        MockPlotter.return_value.plot.assert_called_once()

    def test_no_xml(self, tmp_path):
        mod = _import_mcp_server()

        with patch("shalom.backends.qe_parser.find_xml_path", return_value=None):
            result = mod.plot_bands(calc_dir=str(tmp_path))

        assert result["success"] is False
        assert "not found" in result["error"]


# ===========================================================================
# Test: plot_dos
# ===========================================================================

class TestPlotDos:
    def test_no_dos_file(self, tmp_path):
        mod = _import_mcp_server()
        result = mod.plot_dos(calc_dir=str(tmp_path))
        assert result["success"] is False
        assert "pwscf.dos" in result["error"]

    def test_plot_success(self, tmp_path):
        mod = _import_mcp_server()

        # Create fake pwscf.dos
        dos_file = tmp_path / "pwscf.dos"
        dos_file.write_text("# header\n-10.0  1.0\n0.0  2.0\n10.0  0.5\n")

        mock_dos = MagicMock()
        mock_dos.energies = np.array([-10.0, 0.0, 10.0])
        mock_dos.fermi_energy = 0.0

        with patch("shalom.backends.qe_parser.parse_dos_file",
                   return_value=mock_dos), \
             patch("shalom.backends.qe_parser.extract_fermi_energy",
                   return_value=5.0), \
             patch("shalom.plotting.dos_plot.DOSPlotter") as MockPlotter:
            result = mod.plot_dos(
                calc_dir=str(tmp_path),
                output=str(tmp_path / "dos.png"),
            )

        assert result["success"] is True
        assert result["fermi_energy_eV"] == 5.0
        MockPlotter.return_value.plot.assert_called_once()


# ===========================================================================
# Test: run_convergence
# ===========================================================================

class TestRunConvergence:
    def test_cutoff_success(self):
        mod = _import_mcp_server()

        mock_result = MagicMock()
        mock_result.converged_value = 50.0
        mock_result.summary.return_value = "Converged at 50 Ry"

        with patch.object(mod, "_load_atoms", return_value=bulk("Si")), \
             patch("shalom.workflows.convergence.CutoffConvergence") as MockConv:
            MockConv.return_value.run.return_value = mock_result
            MockConv.return_value.plot.return_value = "/tmp/conv.png"
            result = mod.run_convergence(
                material="Si",
                test_type="cutoff",
                output_dir="/tmp/conv",
            )

        assert result["success"] is True
        assert result["converged"] is True
        assert result["converged_value"] == 50.0

    def test_invalid_test_type(self):
        mod = _import_mcp_server()

        with patch.object(mod, "_load_atoms", return_value=bulk("Si")):
            result = mod.run_convergence(
                material="Si", test_type="invalid",
            )

        assert result["success"] is False
        assert "invalid" in result["error"].lower()


# ===========================================================================
# Test: check_qe_setup
# ===========================================================================

class TestCheckQeSetup:
    def test_all_present(self, tmp_path):
        mod = _import_mcp_server()

        with patch("shutil.which", side_effect=lambda x: f"/usr/bin/{x}"), \
             patch("shalom.backends.qe_config.SSSP_ELEMENTS", {"Si": {}}), \
             patch("shalom.backends.qe_config.get_pseudo_filename",
                   return_value="Si.upf"):
            # Create pseudo file
            pseudo_dir = tmp_path / "pseudos"
            pseudo_dir.mkdir()
            (pseudo_dir / "Si.upf").write_text("fake")

            result = mod.check_qe_setup(
                pseudo_dir=str(pseudo_dir), elements="Si",
            )

        assert result["success"] is True
        assert result["ready"] is True
        assert len(result["issues"]) == 0

    def test_missing_pw(self):
        mod = _import_mcp_server()

        with patch("shutil.which", return_value=None):
            result = mod.check_qe_setup()

        assert result["success"] is False
        assert any("pw.x" in issue for issue in result["issues"])

    def test_missing_pseudo(self, tmp_path):
        mod = _import_mcp_server()

        pseudo_dir = tmp_path / "pseudos"
        pseudo_dir.mkdir()

        with patch("shutil.which", side_effect=lambda x: f"/usr/bin/{x}"), \
             patch("shalom.backends.qe_config.SSSP_ELEMENTS", {"Fe": {}}), \
             patch("shalom.backends.qe_config.get_pseudo_filename",
                   return_value="Fe.upf"):
            result = mod.check_qe_setup(
                pseudo_dir=str(pseudo_dir), elements="Fe",
            )

        assert result["success"] is False
        assert any("missing" in issue.lower() for issue in result["issues"])


# ===========================================================================
# Test: run_pipeline (MCP tool)
# ===========================================================================

class TestRunPipeline:
    def test_missing_api_key(self, monkeypatch):
        mod = _import_mcp_server()
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        result = mod.run_pipeline(objective="Find HER catalyst")
        assert result["success"] is False
        assert "OPENAI_API_KEY" in result["error"]

    def test_missing_anthropic_key(self, monkeypatch):
        mod = _import_mcp_server()
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        result = mod.run_pipeline(
            objective="Find catalyst", provider="anthropic",
        )
        assert result["success"] is False
        assert "ANTHROPIC_API_KEY" in result["error"]

    def test_pipeline_success(self, monkeypatch):
        mod = _import_mcp_server()
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")

        from shalom.core.schemas import PipelineStatus

        mock_result = MagicMock()
        mock_result.status = PipelineStatus.AWAITING_DFT
        mock_result.objective = "Find HER catalyst"
        mock_result.steps_completed = ["coarse_selection", "fine_selection",
                                       "structure_generation", "awaiting_dft"]
        mock_result.ranked_material = MagicMock()
        mock_result.ranked_material.candidate.material_name = "MoS2"
        mock_result.ranked_material.score = 0.85
        mock_result.ranked_material.ranking_justification = "High score"
        mock_result.candidates = None
        mock_result.structure_path = "/tmp/pipeline/MoS2"
        mock_result.execution_wall_time = None
        mock_result.quality_warnings = []
        mock_result.error_message = None
        mock_result.elapsed_seconds = 12.5
        mock_result.correction_history = None
        mock_result.review_result = None

        with patch("shalom.pipeline.Pipeline") as MockPipeline:
            MockPipeline.return_value.run.return_value = mock_result
            result = mod.run_pipeline(
                objective="Find HER catalyst",
                selector_mode="multi_agent",
            )

        assert result["success"] is True
        assert result["status"] == "awaiting_dft"
        assert result["material"]["name"] == "MoS2"
        assert result["material"]["score"] == 0.85
        assert result["structure_path"] == "/tmp/pipeline/MoS2"

    def test_pipeline_failure(self, monkeypatch):
        mod = _import_mcp_server()
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")

        from shalom.core.schemas import PipelineStatus

        mock_result = MagicMock()
        mock_result.status = PipelineStatus.FAILED_DESIGN
        mock_result.objective = "Bad objective"
        mock_result.steps_completed = []
        mock_result.ranked_material = None
        mock_result.candidates = None
        mock_result.structure_path = None
        mock_result.execution_wall_time = None
        mock_result.quality_warnings = []
        mock_result.error_message = "Coarse selection failed"
        mock_result.elapsed_seconds = 2.1
        mock_result.correction_history = None
        mock_result.review_result = None

        with patch("shalom.pipeline.Pipeline") as MockPipeline:
            MockPipeline.return_value.run.return_value = mock_result
            result = mod.run_pipeline(objective="Bad objective")

        assert result["success"] is False
        assert result["status"] == "failed_design"
        assert "Coarse selection failed" in result["error"]

    def test_guides_to_other_tools(self, monkeypatch):
        """Error message should guide users without API key to other tools."""
        mod = _import_mcp_server()
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        result = mod.run_pipeline(objective="Find catalyst")
        assert "search_material" in result["error"] or "other SHALOM tools" in result["error"]


# ===========================================================================
# Test: _load_atoms helper
# ===========================================================================

class TestLoadAtoms:
    def test_from_structure_file(self, tmp_path):
        mod = _import_mcp_server()

        # Create a fake POSCAR
        si = bulk("Si")
        poscar = tmp_path / "POSCAR"
        si.write(str(poscar), format="vasp")

        atoms = mod._load_atoms(structure_file=str(poscar))
        assert len(atoms) == len(si)

    def test_from_mp(self):
        mod = _import_mcp_server()
        si = bulk("Si")

        mock_result = MagicMock()
        mock_result.atoms = si

        with patch("shalom.mp_client.is_mp_available", return_value=True), \
             patch("shalom.mp_client.fetch_structure", return_value=mock_result):
            atoms = mod._load_atoms(material="mp-149")

        assert len(atoms) == len(si)

    def test_from_ase_bulk_fallback(self):
        mod = _import_mcp_server()

        with patch("shalom.mp_client.is_mp_available", return_value=False):
            atoms = mod._load_atoms(material="Si")

        assert len(atoms) > 0

    def test_empty_raises(self):
        mod = _import_mcp_server()

        with pytest.raises(ValueError, match="Provide either"):
            mod._load_atoms()


# ===========================================================================
# Test: cmd_pipeline (CLI command)
# ===========================================================================

class TestCmdPipeline:
    def test_missing_api_key(self, monkeypatch):
        from shalom.__main__ import cmd_pipeline
        import argparse

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("SHALOM_LLM_BASE_URL", raising=False)

        args = argparse.Namespace(
            objective="Find HER catalyst",
            backend="qe",
            provider="openai",
            model=None,
            material=None,
            steps=None,
            output=None,
            calc="relaxation",
            accuracy="standard",
            execute=False,
            nprocs=1,
            timeout=86400,
            max_loops=1,
            selector_mode="simple",
            base_url=None,
        )
        result = cmd_pipeline(args)
        assert result == 1

    def test_pipeline_success(self, monkeypatch):
        from shalom.__main__ import cmd_pipeline
        from shalom.core.schemas import PipelineStatus
        import argparse

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key")

        mock_result = MagicMock()
        mock_result.status = PipelineStatus.AWAITING_DFT
        mock_result.ranked_material = MagicMock()
        mock_result.ranked_material.candidate.material_name = "MoS2"
        mock_result.ranked_material.score = 0.85
        mock_result.structure_path = "/tmp/pipeline/MoS2"
        mock_result.execution_wall_time = None
        mock_result.quality_warnings = []
        mock_result.error_message = None
        mock_result.elapsed_seconds = 8.5
        mock_result.steps_completed = ["coarse_selection", "fine_selection",
                                       "structure_generation", "awaiting_dft"]

        with patch("shalom.pipeline.Pipeline") as MockPipeline:
            MockPipeline.return_value.run.return_value = mock_result

            args = argparse.Namespace(
                objective="Find HER catalyst",
                backend="qe",
                provider="openai",
                model=None,
                material=None,
                steps=None,
                output=None,
                calc="relaxation",
                accuracy="standard",
                execute=False,
                nprocs=1,
                timeout=86400,
                max_loops=1,
                selector_mode="simple",
                base_url=None,
            )
            result = cmd_pipeline(args)

        assert result == 0

    def test_pipeline_with_material_skip_design(self, monkeypatch):
        from shalom.__main__ import cmd_pipeline
        from shalom.core.schemas import PipelineStatus
        import argparse

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-fake")

        mock_result = MagicMock()
        mock_result.status = PipelineStatus.AWAITING_DFT
        mock_result.ranked_material = MagicMock()
        mock_result.ranked_material.candidate.material_name = "MoS2"
        mock_result.ranked_material.score = 1.0
        mock_result.structure_path = "/tmp/pipeline/MoS2"
        mock_result.execution_wall_time = None
        mock_result.quality_warnings = []
        mock_result.error_message = None
        mock_result.elapsed_seconds = 3.2
        mock_result.steps_completed = ["design_skipped", "structure_generation"]

        with patch("shalom.pipeline.Pipeline") as MockPipeline:
            MockPipeline.return_value.run.return_value = mock_result

            args = argparse.Namespace(
                objective="Calculate MoS2 band structure",
                backend="qe",
                provider="anthropic",
                model="claude-sonnet-4-6",
                material="MoS2",
                steps="simulation",
                output="/tmp/pipeline",
                calc="relaxation",
                accuracy="standard",
                execute=False,
                nprocs=1,
                timeout=86400,
                max_loops=1,
                selector_mode="simple",
                base_url=None,
            )
            result = cmd_pipeline(args)

        assert result == 0

    def test_pipeline_with_base_url_skips_api_key_check(self, monkeypatch):
        """base_url set → should not require API key."""
        from shalom.__main__ import cmd_pipeline
        from shalom.core.schemas import PipelineStatus
        import argparse

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("SHALOM_LLM_BASE_URL", raising=False)

        mock_result = MagicMock()
        mock_result.status = PipelineStatus.AWAITING_DFT
        mock_result.ranked_material = None
        mock_result.structure_path = None
        mock_result.execution_wall_time = None
        mock_result.quality_warnings = []
        mock_result.error_message = None
        mock_result.elapsed_seconds = 1.0
        mock_result.steps_completed = ["design_skipped"]

        with patch("shalom.pipeline.Pipeline") as MockPipeline:
            MockPipeline.return_value.run.return_value = mock_result

            args = argparse.Namespace(
                objective="Test local LLM",
                backend="qe",
                provider="openai",
                model="llama3",
                material="Si",
                steps="simulation",
                output=None,
                calc="relaxation",
                accuracy="standard",
                execute=False,
                nprocs=1,
                timeout=86400,
                max_loops=1,
                selector_mode="simple",
                base_url="http://localhost:11434/v1",
            )
            result = cmd_pipeline(args)

        # Should proceed (return 0) without API key when base_url is set
        assert result == 0


# ===========================================================================
# Test: argparse registration
# ===========================================================================

class TestPipelineArgparse:
    def test_pipeline_parser_exists(self):
        from shalom.__main__ import build_parser
        parser = build_parser()
        args = parser.parse_args(["pipeline", "Find HER catalyst"])
        assert args.command == "pipeline"
        assert args.objective == "Find HER catalyst"

    def test_pipeline_parser_defaults(self):
        from shalom.__main__ import build_parser
        parser = build_parser()
        args = parser.parse_args(["pipeline", "Test objective"])
        assert args.backend == "qe"
        assert args.provider == "openai"
        assert args.model is None
        assert args.accuracy == "standard"
        assert args.execute is False
        assert args.nprocs == 1
        assert args.base_url is None

    def test_pipeline_parser_all_options(self):
        from shalom.__main__ import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "pipeline", "Find catalyst",
            "--backend", "vasp",
            "--provider", "anthropic",
            "--model", "claude-sonnet-4-6",
            "--material", "MoS2",
            "--steps", "simulation,review",
            "-o", "/tmp/out",
            "--calc", "scf",
            "--accuracy", "precise",
            "--execute",
            "--nprocs", "8",
            "--timeout", "3600",
            "--max-loops", "3",
            "--selector-mode", "multi_agent",
            "--base-url", "http://localhost:11434/v1",
        ])
        assert args.backend == "vasp"
        assert args.provider == "anthropic"
        assert args.model == "claude-sonnet-4-6"
        assert args.material == "MoS2"
        assert args.steps == "simulation,review"
        assert args.output == "/tmp/out"
        assert args.execute is True
        assert args.nprocs == 8
        assert args.max_loops == 3
        assert args.base_url == "http://localhost:11434/v1"


# ===========================================================================
# Test: LLMProvider base_url
# ===========================================================================

class TestLLMProviderBaseUrl:
    def test_base_url_from_param(self, monkeypatch):
        """base_url parameter should be passed to OpenAI client."""
        monkeypatch.delenv("SHALOM_LLM_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with patch("openai.OpenAI") as MockClient:
            from shalom.core.llm_provider import LLMProvider
            p = LLMProvider(
                provider_type="openai",
                model_name="llama3",
                base_url="http://localhost:11434/v1",
            )

        assert p.base_url == "http://localhost:11434/v1"
        call_kwargs = MockClient.call_args[1]
        assert call_kwargs["base_url"] == "http://localhost:11434/v1"
        # Should use "local" as API key when no key is set
        assert call_kwargs["api_key"] == "local"

    def test_base_url_from_env(self, monkeypatch):
        """SHALOM_LLM_BASE_URL env var should be picked up."""
        monkeypatch.setenv("SHALOM_LLM_BASE_URL", "http://localhost:8000/v1")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with patch("openai.OpenAI") as MockClient:
            from shalom.core.llm_provider import LLMProvider
            provider = LLMProvider(provider_type="openai", model_name="test")

        assert provider.base_url == "http://localhost:8000/v1"
        call_kwargs = MockClient.call_args[1]
        assert call_kwargs["base_url"] == "http://localhost:8000/v1"

    def test_no_base_url(self, monkeypatch):
        """Without base_url, should use standard OpenAI client."""
        monkeypatch.delenv("SHALOM_LLM_BASE_URL", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("openai.OpenAI") as MockClient:
            from shalom.core.llm_provider import LLMProvider
            provider = LLMProvider(provider_type="openai")

        assert provider.base_url is None
        call_kwargs = MockClient.call_args[1]
        assert "base_url" not in call_kwargs
        assert call_kwargs["api_key"] == "sk-test"

    def test_anthropic_base_url(self, monkeypatch):
        """base_url should also work for Anthropic provider."""
        monkeypatch.delenv("SHALOM_LLM_BASE_URL", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        with patch("shalom.core.llm_provider.Anthropic") as MockClient:
            from shalom.core.llm_provider import LLMProvider
            LLMProvider(
                provider_type="anthropic",
                base_url="http://custom-anthropic:8080",
            )

        call_kwargs = MockClient.call_args[1]
        assert call_kwargs["base_url"] == "http://custom-anthropic:8080"


# ===========================================================================
# Test: SafeExecutor hardened sandbox
# ===========================================================================

class TestSafeExecutorHardened:
    def test_eval_blocked(self):
        """eval() should not be available in sandbox."""
        from shalom.core.sandbox import SafeExecutor
        result = SafeExecutor.execute("x = 1 + 1", timeout_seconds=5)
        assert result["x"] == 2

        # eval should not be in builtins
        with pytest.raises(Exception):
            SafeExecutor.execute("y = eval('1+1')", timeout_seconds=5)

    def test_exec_blocked(self):
        """exec() should not be available in sandbox."""
        from shalom.core.sandbox import SafeExecutor
        with pytest.raises(Exception):
            SafeExecutor.execute("exec('x=1')", timeout_seconds=5)

    def test_compile_blocked(self):
        """compile() should not be available in sandbox."""
        from shalom.core.sandbox import SafeExecutor
        with pytest.raises(Exception):
            SafeExecutor.execute("compile('x=1', '<>', 'exec')", timeout_seconds=5)

    def test_open_blocked(self):
        """open() should not be available in sandbox."""
        from shalom.core.sandbox import SafeExecutor
        with pytest.raises(Exception):
            SafeExecutor.execute("f = open('/etc/passwd')", timeout_seconds=5)

    def test_import_blocked(self):
        """import should still be blocked."""
        from shalom.core.sandbox import SafeExecutor
        with pytest.raises(Exception):
            SafeExecutor.execute("import os", timeout_seconds=5)

    def test_safe_builtins_work(self):
        """Safe builtins (sorted, map, isinstance, hasattr) should work."""
        from shalom.core.sandbox import SafeExecutor
        result = SafeExecutor.execute(
            "result = sorted([3,1,2])\n"
            "mapped = list(map(str, [1,2,3]))\n"
            "is_int = isinstance(42, int)\n"
            "has_x = hasattr(result, '__len__')\n",
            timeout_seconds=5,
        )
        assert result["result"] == [1, 2, 3]
        assert result["mapped"] == ["1", "2", "3"]
        assert result["is_int"] is True
        assert result["has_x"] is True


# ===========================================================================
# Test: Audit logging
# ===========================================================================

class TestAuditLogging:
    def test_log_event_no_env(self, monkeypatch):
        """Without SHALOM_AUDIT_LOG, log_event should be a no-op."""
        monkeypatch.delenv("SHALOM_AUDIT_LOG", raising=False)
        # Reset internal state
        import shalom.core.audit as audit_mod
        audit_mod._initialized = False
        audit_mod._audit_logger = None

        # Should not raise
        audit_mod.log_event("test_action", {"key": "value"})

    def test_log_event_writes_to_file(self, tmp_path, monkeypatch):
        """With SHALOM_AUDIT_LOG set, events should be written to file."""
        import json
        import shalom.core.audit as audit_mod

        log_file = tmp_path / "audit.log"
        monkeypatch.setenv("SHALOM_AUDIT_LOG", str(log_file))
        # Reset internal state to pick up new env var
        audit_mod._initialized = False
        audit_mod._audit_logger = None

        audit_mod.log_event("test_action", {"provider": "openai", "model": "gpt-4o"})

        # Force flush
        logger = audit_mod._get_audit_logger()
        if logger:
            for handler in logger.handlers:
                handler.flush()

        content = log_file.read_text(encoding="utf-8").strip()
        entry = json.loads(content)
        assert entry["action"] == "test_action"
        assert entry["details"]["provider"] == "openai"
        assert "timestamp" in entry

        # Cleanup
        audit_mod._initialized = False
        audit_mod._audit_logger = None


# ===========================================================================
# Test: MCP run_pipeline with base_url
# ===========================================================================

class TestRunPipelineBaseUrl:
    def test_base_url_skips_api_key_check(self, monkeypatch):
        """base_url set → should not require API key."""
        mod = _import_mcp_server()
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("SHALOM_LLM_BASE_URL", raising=False)

        from shalom.core.schemas import PipelineStatus

        mock_result = MagicMock()
        mock_result.status = PipelineStatus.AWAITING_DFT
        mock_result.objective = "Test"
        mock_result.steps_completed = ["design_skipped"]
        mock_result.ranked_material = None
        mock_result.candidates = None
        mock_result.structure_path = None
        mock_result.execution_wall_time = None
        mock_result.quality_warnings = []
        mock_result.error_message = None
        mock_result.elapsed_seconds = 1.0
        mock_result.correction_history = None
        mock_result.review_result = None

        with patch("shalom.pipeline.Pipeline") as MockPipeline:
            MockPipeline.return_value.run.return_value = mock_result
            result = mod.run_pipeline(
                objective="Test local LLM",
                material="Si",
                base_url="http://localhost:11434/v1",
            )

        assert result["success"] is True

    def test_no_key_no_base_url_fails(self, monkeypatch):
        """Without API key AND base_url, should fail."""
        mod = _import_mcp_server()
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("SHALOM_LLM_BASE_URL", raising=False)

        result = mod.run_pipeline(objective="Test")
        assert result["success"] is False
        assert "OPENAI_API_KEY" in result["error"]


# ===========================================================================
# Test: PipelineConfig base_url
# ===========================================================================

class TestPipelineConfigBaseUrl:
    def test_base_url_field(self):
        from shalom.pipeline import PipelineConfig
        config = PipelineConfig(base_url="http://localhost:11434/v1")
        assert config.base_url == "http://localhost:11434/v1"

    def test_base_url_default_none(self):
        from shalom.pipeline import PipelineConfig
        config = PipelineConfig()
        assert config.base_url is None

    def test_base_url_passed_to_llm_provider(self, monkeypatch):
        """Pipeline should pass base_url to LLMProvider."""
        monkeypatch.delenv("SHALOM_LLM_BASE_URL", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        with patch("shalom.pipeline.LLMProvider") as MockProvider:
            from shalom.pipeline import Pipeline, PipelineConfig
            config = PipelineConfig(
                base_url="http://localhost:11434/v1",
                material_name="Si",
                steps=["simulation"],
            )
            Pipeline(objective="test", config=config)

        MockProvider.assert_called_once_with(
            provider_type="openai",
            model_name="gpt-4o",
            base_url="http://localhost:11434/v1",
        )
