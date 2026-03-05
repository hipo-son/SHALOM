"""Tests for SHALOM MCP Server tool functions.

Tests the underlying logic of each MCP tool without requiring the MCP SDK.
Each tool function is tested by mocking the SHALOM library calls it wraps.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
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
        # Actionable guidance in error message
        assert "MP ID" in result["error"] or "structure_file" in result["error"]

    def test_invalid_backend(self):
        mod = _import_mcp_server()
        result = mod.generate_dft_input(material="Si", backend="lammps")
        assert result["success"] is False
        assert "Invalid backend" in result["error"]


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

        with patch.object(mod, "_load_atoms", return_value=(bulk("Si"), "materials_project")), \
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
        assert "warning" not in result  # MP source → no fallback warning

    def test_run_with_new_params(self):
        """New params (mpi_command, pw_executable, etc.) are passed to StandardWorkflow."""
        mod = _import_mcp_server()

        mock_wf_result = {
            "fermi_energy": 5.0,
            "bands_png": None,
            "dos_png": None,
            "calc_dirs": {},
        }

        with patch.object(mod, "_load_atoms", return_value=(bulk("Si"), "structure_file")), \
             patch("shalom.workflows.standard.StandardWorkflow") as MockWF:
            MockWF.return_value.run.return_value = mock_wf_result
            result = mod.run_workflow(
                material="Si",
                output_dir="/tmp/wf",
                mpi_command="srun",
                pw_executable="/opt/qe/pw.x",
                dos_executable="/opt/qe/dos.x",
                npoints_kpath=60,
                dos_deltaE=0.005,
            )

        assert result["success"] is True
        call_kwargs = MockWF.call_args[1]
        assert call_kwargs["mpi_command"] == "srun"
        assert call_kwargs["pw_executable"] == "/opt/qe/pw.x"
        assert call_kwargs["dos_executable"] == "/opt/qe/dos.x"
        assert call_kwargs["npoints_kpath"] == 60
        assert call_kwargs["dos_deltaE"] == 0.005

    def test_run_bad_material(self):
        mod = _import_mcp_server()

        with patch.object(mod, "_load_atoms",
                         side_effect=ValueError("Cannot resolve")):
            result = mod.run_workflow(material="ZZZZZ")

        assert result["success"] is False
        assert "Cannot resolve" in result["error"]

    def test_ase_bulk_fallback_warning(self):
        """ASE bulk fallback should include a warning."""
        mod = _import_mcp_server()

        mock_wf_result = {
            "fermi_energy": 6.0,
            "bands_png": None,
            "dos_png": None,
            "calc_dirs": {},
        }

        with patch.object(mod, "_load_atoms", return_value=(bulk("Si"), "ase_bulk_fallback")), \
             patch("shalom.workflows.standard.StandardWorkflow") as MockWF:
            MockWF.return_value.run.return_value = mock_wf_result
            result = mod.run_workflow(material="Si", output_dir="/tmp/wf")

        assert result["success"] is True
        assert "warning" in result
        assert "ASE bulk" in result["warning"]

    def test_run_workflow_returns_detection_log(self):
        """run_workflow should include detection_log in response."""
        mod = _import_mcp_server()

        mock_wf_result = {
            "fermi_energy": 5.0,
            "bands_png": None,
            "dos_png": None,
            "calc_dirs": {},
            "detection_log": ["ecutwfc=30.0 Ry", "K-grid [6, 6, 6]"],
            "completed_steps": ["vc_relax", "scf"],
            "failed_step": None,
        }

        with patch.object(mod, "_load_atoms", return_value=(bulk("Si"), "materials_project")), \
             patch("shalom.workflows.standard.StandardWorkflow") as MockWF:
            MockWF.return_value.run.return_value = mock_wf_result
            result = mod.run_workflow(material="Si", output_dir="/tmp/wf")

        assert result["success"] is True
        assert result["detection_log"] == ["ecutwfc=30.0 Ry", "K-grid [6, 6, 6]"]
        assert result["completed_steps"] == ["vc_relax", "scf"]
        assert result["failed_step"] is None

    def test_run_workflow_error_includes_detection_log(self):
        """Error responses include detection_log and completed_steps keys."""
        mod = _import_mcp_server()

        with patch.object(mod, "_load_atoms", side_effect=ValueError("bad material")):
            result = mod.run_workflow(material="???", output_dir="/tmp/wf")

        assert result["success"] is False
        assert "detection_log" in result
        assert result["detection_log"] == []
        assert result["completed_steps"] == []
        assert result["failed_step"] is None


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
        mock_result.stress_tensor = None
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
        # Actionable guidance
        assert "execute_dft" in result["error"] or "run_workflow" in result["error"]

    def test_invalid_backend(self):
        mod = _import_mcp_server()
        result = mod.parse_dft_output("/tmp/calc", backend="lammps")
        assert result["success"] is False
        assert "Invalid backend" in result["error"]


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

        with patch.object(mod, "_load_atoms", return_value=(bulk("Si"), "materials_project")), \
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
        assert "warning" not in result

    def test_cutoff_with_kgrid_and_mpi(self):
        """kgrid and mpi_command are forwarded to CutoffConvergence."""
        mod = _import_mcp_server()

        mock_result = MagicMock()
        mock_result.converged_value = 60.0
        mock_result.summary.return_value = "Converged at 60 Ry"

        with patch.object(mod, "_load_atoms", return_value=(bulk("Si"), "structure_file")), \
             patch("shalom.workflows.convergence.CutoffConvergence") as MockConv:
            MockConv.return_value.run.return_value = mock_result
            MockConv.return_value.plot.return_value = None
            result = mod.run_convergence(
                material="Si",
                test_type="cutoff",
                kgrid="4,4,4",
                mpi_command="srun",
            )

        assert result["success"] is True
        call_kwargs = MockConv.call_args[1]
        assert call_kwargs["kgrid"] == [4, 4, 4]
        assert call_kwargs["mpi_command"] == "srun"

    def test_kpoints_with_mpi(self):
        """mpi_command is forwarded to KpointConvergence."""
        mod = _import_mcp_server()

        mock_result = MagicMock()
        mock_result.converged_value = 40.0
        mock_result.summary.return_value = "Converged at 40"

        with patch.object(mod, "_load_atoms", return_value=(bulk("Si"), "materials_project")), \
             patch("shalom.workflows.convergence.KpointConvergence") as MockConv:
            MockConv.return_value.run.return_value = mock_result
            MockConv.return_value.plot.return_value = None
            result = mod.run_convergence(
                material="Si",
                test_type="kpoints",
                ecutwfc=60.0,
                mpi_command="srun",
            )

        assert result["success"] is True
        call_kwargs = MockConv.call_args[1]
        assert call_kwargs["mpi_command"] == "srun"

    def test_ase_bulk_fallback_warning(self):
        """ASE bulk fallback adds warning to convergence result."""
        mod = _import_mcp_server()

        mock_result = MagicMock()
        mock_result.converged_value = 50.0
        mock_result.summary.return_value = "Converged at 50 Ry"

        with patch.object(mod, "_load_atoms", return_value=(bulk("Si"), "ase_bulk_fallback")), \
             patch("shalom.workflows.convergence.CutoffConvergence") as MockConv:
            MockConv.return_value.run.return_value = mock_result
            MockConv.return_value.plot.return_value = None
            result = mod.run_convergence(material="Si", test_type="cutoff")

        assert result["success"] is True
        assert "warning" in result
        assert "ASE bulk" in result["warning"]

    def test_invalid_test_type(self):
        mod = _import_mcp_server()

        with patch.object(mod, "_load_atoms", return_value=(bulk("Si"), "materials_project")):
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

        atoms, source = mod._load_atoms(structure_file=str(poscar))
        assert len(atoms) == len(si)
        assert source == "structure_file"

    def test_from_mp(self):
        mod = _import_mcp_server()
        si = bulk("Si")

        mock_result = MagicMock()
        mock_result.atoms = si

        with patch("shalom.mp_client.is_mp_available", return_value=True), \
             patch("shalom.mp_client.fetch_structure", return_value=mock_result):
            atoms, source = mod._load_atoms(material="mp-149")

        assert len(atoms) == len(si)
        assert source == "materials_project"

    def test_from_ase_bulk_fallback(self):
        mod = _import_mcp_server()

        with patch("shalom.mp_client.is_mp_available", return_value=False):
            atoms, source = mod._load_atoms(material="Si")

        assert len(atoms) > 0
        assert source == "ase_bulk_fallback"

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


# ===========================================================================
# Test: analyze_elastic (MCP tool 11)
# ===========================================================================

class TestAnalyzeElastic:
    """Tests for the analyze_elastic MCP tool."""

    _TENSOR_JSON = (
        "[[165.7,63.9,63.9,0,0,0],[63.9,165.7,63.9,0,0,0],"
        "[63.9,63.9,165.7,0,0,0],[0,0,0,79.6,0,0],"
        "[0,0,0,0,79.6,0],[0,0,0,0,0,79.6]]"
    )

    def test_success(self):
        mod = _import_mcp_server()
        mock_result = MagicMock(
            bulk_modulus_vrh=97.8,
            shear_modulus_vrh=51.1,
            youngs_modulus=130.0,
            poisson_ratio=0.27,
            universal_anisotropy=0.04,
            is_stable=True,
            stability_violations=[],
        )
        with patch("shalom.analysis.elastic.is_elastic_available", return_value=True), \
             patch("shalom.analysis.elastic.analyze_elastic_tensor",
                   return_value=mock_result):
            result = mod.analyze_elastic(elastic_tensor_json=self._TENSOR_JSON)

        assert result["success"] is True
        assert result["bulk_modulus_GPa"] == 97.8
        assert result["is_mechanically_stable"] is True

    def test_unavailable(self):
        mod = _import_mcp_server()
        with patch("shalom.analysis.elastic.is_elastic_available", return_value=False):
            result = mod.analyze_elastic(elastic_tensor_json=self._TENSOR_JSON)

        assert result["success"] is False
        assert "pymatgen" in result["error"]

    def test_invalid_json(self):
        mod = _import_mcp_server()
        with patch("shalom.analysis.elastic.is_elastic_available", return_value=True):
            result = mod.analyze_elastic(elastic_tensor_json="not json")

        assert result["success"] is False
        assert "Invalid JSON" in result["error"]

    def test_exception(self):
        mod = _import_mcp_server()
        with patch("shalom.analysis.elastic.is_elastic_available", return_value=True), \
             patch("shalom.analysis.elastic.analyze_elastic_tensor",
                   side_effect=ValueError("Bad tensor shape")):
            result = mod.analyze_elastic(elastic_tensor_json=self._TENSOR_JSON)

        assert result["success"] is False
        assert "Bad tensor shape" in result["error"]


# ===========================================================================
# Test: analyze_phonon_properties (MCP tool 12)
# ===========================================================================

class TestAnalyzePhononProperties:
    """Tests for the analyze_phonon_properties MCP tool."""

    def test_phonopy_unavailable(self):
        mod = _import_mcp_server()
        with patch("shalom.analysis.phonon.is_phonopy_available", return_value=False):
            result = mod.analyze_phonon_properties(structure_file="POSCAR")

        assert result["success"] is False
        assert "phonopy" in result["error"]

    def test_invalid_supercell(self):
        mod = _import_mcp_server()
        with patch("shalom.analysis.phonon.is_phonopy_available", return_value=True), \
             patch("ase.io.read", return_value=bulk("Si")):
            result = mod.analyze_phonon_properties(
                structure_file="POSCAR", supercell="2,2",
            )

        assert result["success"] is False
        assert "Invalid supercell" in result["error"]

    def test_no_force_data(self):
        mod = _import_mcp_server()
        with patch("shalom.analysis.phonon.is_phonopy_available", return_value=True), \
             patch("ase.io.read", return_value=bulk("Si")):
            result = mod.analyze_phonon_properties(structure_file="POSCAR")

        assert result["success"] is False
        assert "force_sets_file" in result["error"]

    def test_with_force_constants(self):
        mod = _import_mcp_server()
        mock_result = MagicMock(
            is_stable=True,
            min_frequency_THz=1.5,
            imaginary_modes=[],
            n_branches=6,
            thermal_temperatures=np.array([0.0, 100.0, 200.0, 300.0, 400.0]),
            thermal_cv=np.array([0.0, 5.0, 10.0, 15.0, 18.0]),
            thermal_entropy=np.array([0.0, 2.0, 5.0, 8.0, 10.0]),
            thermal_free_energy=np.array([0.0, -1.0, -3.0, -5.0, -7.0]),
        )
        with patch("shalom.analysis.phonon.is_phonopy_available", return_value=True), \
             patch("ase.io.read", return_value=bulk("Si")), \
             patch("phonopy.file_IO.parse_FORCE_CONSTANTS", return_value=MagicMock()), \
             patch("shalom.analysis.phonon.analyze_phonon_from_force_constants",
                   return_value=mock_result):
            result = mod.analyze_phonon_properties(
                structure_file="POSCAR",
                force_constants_file="FORCE_CONSTANTS",
            )

        assert result["success"] is True
        assert result["is_dynamically_stable"] is True
        assert result["cv_300K_J_per_K_mol"] == 15.0

    def test_with_force_sets(self):
        mod = _import_mcp_server()
        mock_ph = MagicMock()
        mock_result = MagicMock(
            is_stable=True,
            min_frequency_THz=2.0,
            imaginary_modes=[],
            n_branches=6,
            thermal_temperatures=None,
            thermal_cv=None,
            thermal_entropy=None,
            thermal_free_energy=None,
        )
        with patch("shalom.analysis.phonon.is_phonopy_available", return_value=True), \
             patch("ase.io.read", return_value=bulk("Si")), \
             patch("phonopy.file_IO.parse_FORCE_SETS", return_value={}), \
             patch("shalom.analysis.phonon.generate_phonon_displacements",
                   return_value=([], mock_ph)), \
             patch("shalom.analysis.phonon._run_phonon_analysis",
                   return_value=mock_result):
            result = mod.analyze_phonon_properties(
                structure_file="POSCAR",
                force_sets_file="FORCE_SETS",
            )

        assert result["success"] is True
        assert result["cv_300K_J_per_K_mol"] is None

    def test_exception(self):
        mod = _import_mcp_server()
        with patch("shalom.analysis.phonon.is_phonopy_available", return_value=True), \
             patch("ase.io.read", side_effect=FileNotFoundError("POSCAR not found")):
            result = mod.analyze_phonon_properties(structure_file="POSCAR")

        assert result["success"] is False
        assert "POSCAR" in result["error"]


# ===========================================================================
# Test: analyze_electronic_structure (MCP tool 13)
# ===========================================================================

class TestAnalyzeElectronicStructure:
    """Tests for the analyze_electronic_structure MCP tool."""

    def test_no_input(self):
        mod = _import_mcp_server()
        result = mod.analyze_electronic_structure()
        assert result["success"] is False
        assert "Provide" in result["error"]

    def test_xml_not_found_in_calc_dir(self, tmp_path):
        mod = _import_mcp_server()
        with patch("shalom.backends.qe_parser.find_xml_path", return_value=None):
            result = mod.analyze_electronic_structure(calc_dir=str(tmp_path))

        assert result["success"] is False
        assert "not found" in result["error"]

    def test_success_with_xml_path(self, tmp_path):
        mod = _import_mcp_server()
        xml_file = str(tmp_path / "data-file-schema.xml")

        # Create a pw.out for fermi energy extraction
        pw_out = tmp_path / "pw.out"
        pw_out.write_text("the Fermi energy is     6.48 ev\n")

        mock_bs = MagicMock(eigenvalues=np.zeros((10, 5)))
        mock_result = MagicMock(
            is_metal=False,
            bandgap_eV=0.61,
            is_direct=False,
            n_occupied_bands=4,
            vbm_energy=5.98,
            cbm_energy=6.59,
            effective_mass_electron=0.54,
            effective_mass_hole=0.19,
            dos_at_fermi=None,
        )

        with patch("shalom.backends.qe_parser.parse_xml_bands", return_value=mock_bs), \
             patch("shalom.backends.qe_parser.extract_fermi_energy", return_value=6.48), \
             patch("shalom.analysis.electronic.analyze_band_structure",
                   return_value=mock_result):
            result = mod.analyze_electronic_structure(bands_xml_path=xml_file)

        assert result["success"] is True
        assert result["bandgap_eV"] == 0.61
        assert result["vbm_energy_eV"] == 5.98
        assert result["effective_mass_electron_me"] == 0.54

    def test_success_with_calc_dir(self, tmp_path):
        mod = _import_mcp_server()
        mock_bs = MagicMock(eigenvalues=np.zeros((5, 3)))
        mock_result = MagicMock(
            is_metal=True, bandgap_eV=0.0, is_direct=False,
            n_occupied_bands=3, vbm_energy=None, cbm_energy=None,
            effective_mass_electron=None, effective_mass_hole=None,
            dos_at_fermi=2.5,
        )
        xml_path = str(tmp_path / "sub" / "data.xml")

        with patch("shalom.backends.qe_parser.find_xml_path", return_value=xml_path), \
             patch("shalom.backends.qe_parser.extract_fermi_energy", return_value=5.0), \
             patch("shalom.backends.qe_parser.parse_xml_bands", return_value=mock_bs), \
             patch("shalom.analysis.electronic.analyze_band_structure",
                   return_value=mock_result):
            result = mod.analyze_electronic_structure(
                calc_dir=str(tmp_path), fermi_energy=5.0,
            )

        assert result["success"] is True
        assert result["is_metal"] is True
        assert result["dos_at_fermi_states_per_eV"] == 2.5

    def test_with_dos_file(self, tmp_path):
        mod = _import_mcp_server()
        xml_file = str(tmp_path / "data.xml")
        dos_file = str(tmp_path / "pwscf.dos")

        mock_bs = MagicMock(eigenvalues=np.zeros((5, 3)))
        mock_dos = MagicMock(fermi_energy=5.0)
        mock_result = MagicMock(
            is_metal=False, bandgap_eV=1.1, is_direct=True,
            n_occupied_bands=4, vbm_energy=None, cbm_energy=None,
            effective_mass_electron=None, effective_mass_hole=None,
            dos_at_fermi=0.0,
        )

        with patch("shalom.backends.qe_parser.parse_xml_bands", return_value=mock_bs), \
             patch("shalom.backends.qe_parser.extract_fermi_energy", return_value=5.0), \
             patch("shalom.backends.qe_parser.parse_dos_file", return_value=mock_dos), \
             patch("shalom.analysis.electronic.analyze_band_structure",
                   return_value=mock_result):
            result = mod.analyze_electronic_structure(
                bands_xml_path=xml_file, dos_file_path=dos_file,
            )

        assert result["success"] is True

    def test_exception(self):
        mod = _import_mcp_server()
        with patch("shalom.backends.qe_parser.find_xml_path",
                   side_effect=OSError("permission denied")):
            result = mod.analyze_electronic_structure(calc_dir="/bad/path")

        assert result["success"] is False
        assert "permission denied" in result["error"]


# ===========================================================================
# Test: analyze_xrd_pattern (MCP tool 14)
# ===========================================================================

class TestAnalyzeXRDPattern:
    """Tests for the analyze_xrd_pattern MCP tool."""

    def test_xrd_unavailable(self):
        mod = _import_mcp_server()
        with patch("shalom.analysis.xrd.is_xrd_available", return_value=False):
            result = mod.analyze_xrd_pattern(structure_file="POSCAR")

        assert result["success"] is False
        assert "pymatgen" in result["error"]

    def test_success(self):
        mod = _import_mcp_server()
        mock_result = MagicMock(
            two_theta=np.array([28.4, 47.3, 56.1]),
            intensities=np.array([100.0, 55.0, 30.0]),
            d_spacings=np.array([3.14, 1.92, 1.64]),
            hkl_indices=[(1, 1, 1), (2, 2, 0), (3, 1, 1)],
            n_peaks=3,
            wavelength_angstrom=1.5406,
        )
        with patch("shalom.analysis.xrd.is_xrd_available", return_value=True), \
             patch("ase.io.read", return_value=bulk("Si")), \
             patch("shalom.analysis.xrd.calculate_xrd", return_value=mock_result):
            result = mod.analyze_xrd_pattern(structure_file="POSCAR")

        assert result["success"] is True
        assert result["n_peaks"] == 3
        assert len(result["top_peaks"]) == 3
        assert result["top_peaks"][0]["two_theta_deg"] == 28.4

    def test_with_output_plot(self, tmp_path):
        mod = _import_mcp_server()
        mock_result = MagicMock(
            two_theta=np.array([28.4]),
            intensities=np.array([100.0]),
            d_spacings=np.array([3.14]),
            hkl_indices=[(1, 1, 1)],
            n_peaks=1,
            wavelength_angstrom=1.5406,
        )
        plot_path = str(tmp_path / "xrd.png")

        with patch("shalom.analysis.xrd.is_xrd_available", return_value=True), \
             patch("ase.io.read", return_value=bulk("Si")), \
             patch("shalom.analysis.xrd.calculate_xrd", return_value=mock_result), \
             patch("shalom.plotting.xrd_plot.XRDPlotter") as MockPlotter:
            result = mod.analyze_xrd_pattern(
                structure_file="POSCAR", output_plot=plot_path,
            )

        assert result["success"] is True
        assert result["plot_path"] == plot_path
        MockPlotter.return_value.plot.assert_called_once()

    def test_plot_import_error(self, tmp_path):
        mod = _import_mcp_server()
        mock_result = MagicMock(
            two_theta=np.array([28.4]),
            intensities=np.array([100.0]),
            d_spacings=np.array([3.14]),
            hkl_indices=[(1, 1, 1)],
            n_peaks=1,
            wavelength_angstrom=1.5406,
        )
        plot_path = str(tmp_path / "xrd.png")

        with patch("shalom.analysis.xrd.is_xrd_available", return_value=True), \
             patch("ase.io.read", return_value=bulk("Si")), \
             patch("shalom.analysis.xrd.calculate_xrd", return_value=mock_result), \
             patch("shalom.plotting.xrd_plot.XRDPlotter",
                   side_effect=ImportError("no matplotlib")):
            result = mod.analyze_xrd_pattern(
                structure_file="POSCAR", output_plot=plot_path,
            )

        assert result["success"] is True
        assert "plot_warning" in result

    def test_exception(self):
        mod = _import_mcp_server()
        with patch("shalom.analysis.xrd.is_xrd_available", return_value=True), \
             patch("ase.io.read", side_effect=FileNotFoundError("No file")):
            result = mod.analyze_xrd_pattern(structure_file="POSCAR")

        assert result["success"] is False


# ===========================================================================
# Test: analyze_symmetry_properties (MCP tool 15)
# ===========================================================================

class TestAnalyzeSymmetryProperties:
    """Tests for the analyze_symmetry_properties MCP tool."""

    def test_spglib_unavailable(self):
        mod = _import_mcp_server()
        with patch("shalom.analysis.symmetry.is_spglib_available", return_value=False):
            result = mod.analyze_symmetry_properties(structure_file="POSCAR")

        assert result["success"] is False
        assert "spglib" in result["error"]

    def test_success(self):
        mod = _import_mcp_server()
        mock_result = MagicMock(
            space_group_number=227,
            space_group_symbol="Fd-3m",
            point_group="m-3m",
            crystal_system="cubic",
            lattice_type="face-centered",
            hall_symbol="-F 4vw 2vw 3",
            n_operations=192,
            is_primitive=True,
            wyckoff_letters=["a", "a"],
            equivalent_atoms=[0, 0],
        )
        with patch("shalom.analysis.symmetry.is_spglib_available", return_value=True), \
             patch("ase.io.read", return_value=bulk("Si")), \
             patch("shalom.analysis.symmetry.analyze_symmetry",
                   return_value=mock_result):
            result = mod.analyze_symmetry_properties(structure_file="POSCAR")

        assert result["success"] is True
        assert result["space_group_number"] == 227
        assert result["crystal_system"] == "cubic"

    def test_custom_symprec(self):
        mod = _import_mcp_server()
        mock_result = MagicMock(
            space_group_number=1, space_group_symbol="P1",
            point_group="1", crystal_system="triclinic",
            lattice_type="primitive", hall_symbol="P 1",
            n_operations=1, is_primitive=True,
            wyckoff_letters=["a"], equivalent_atoms=[0],
        )
        with patch("shalom.analysis.symmetry.is_spglib_available", return_value=True), \
             patch("ase.io.read", return_value=bulk("Si")), \
             patch("shalom.analysis.symmetry.analyze_symmetry",
                   return_value=mock_result) as mock_fn:
            result = mod.analyze_symmetry_properties(
                structure_file="POSCAR", symprec="0.1",
            )

        assert result["success"] is True
        # Verify symprec was passed as float
        assert mock_fn.call_args[1]["symprec"] == 0.1

    def test_exception(self):
        mod = _import_mcp_server()
        with patch("shalom.analysis.symmetry.is_spglib_available", return_value=True), \
             patch("ase.io.read", side_effect=FileNotFoundError("No POSCAR")):
            result = mod.analyze_symmetry_properties(structure_file="POSCAR")

        assert result["success"] is False


# ===========================================================================
# Test: analyze_magnetic_properties (MCP tool 16)
# ===========================================================================

class TestAnalyzeMagneticProperties:
    """Tests for the analyze_magnetic_properties MCP tool."""

    def test_with_site_mags(self):
        mod = _import_mcp_server()
        with patch("shalom.analysis.magnetic.extract_site_magnetization",
                   return_value=[4.5, 4.5, -0.1, -0.1]), \
             patch("shalom.analysis.magnetic.extract_lowdin_charges",
                   return_value=None):
            result = mod.analyze_magnetic_properties(pw_out_path="pw.out")

        assert result["success"] is True
        assert len(result["site_magnetizations_bohr_mag"]) == 4
        assert result["n_atoms"] == 4
        assert result["lowdin_total_charges"] is None

    def test_no_site_mags(self):
        mod = _import_mcp_server()
        with patch("shalom.analysis.magnetic.extract_site_magnetization",
                   return_value=None), \
             patch("shalom.analysis.magnetic.extract_lowdin_charges",
                   return_value=None):
            result = mod.analyze_magnetic_properties(pw_out_path="pw.out")

        assert result["success"] is True
        assert result["site_magnetizations_bohr_mag"] is None

    def test_with_lowdin(self):
        mod = _import_mcp_server()
        with patch("shalom.analysis.magnetic.extract_site_magnetization",
                   return_value=None), \
             patch("shalom.analysis.magnetic.extract_lowdin_charges",
                   return_value={"total_charges": [3.0, 3.0], "spd_charges": []}):
            result = mod.analyze_magnetic_properties(pw_out_path="pw.out")

        assert result["success"] is True
        assert result["lowdin_total_charges"] == [3.0, 3.0]

    def test_with_structure_file(self, tmp_path):
        mod = _import_mcp_server()
        # Create a fake pw.out with total magnetization
        pw_out = tmp_path / "pw.out"
        pw_out.write_text(
            "     total magnetization       =    20.00 Bohr mag/cell\n"
        )

        mock_mag_result = MagicMock(
            total_magnetization=20.0,
            is_magnetic=True,
            is_spin_polarized=True,
            magnetic_elements=["Fe"],
            dominant_moment_element="Fe",
        )

        with patch("shalom.analysis.magnetic.extract_site_magnetization",
                   return_value=None), \
             patch("shalom.analysis.magnetic.extract_lowdin_charges",
                   return_value=None), \
             patch("ase.io.read", return_value=bulk("Fe")), \
             patch("shalom.analysis.magnetic.analyze_magnetism",
                   return_value=mock_mag_result):
            result = mod.analyze_magnetic_properties(
                pw_out_path=str(pw_out),
                structure_file=str(tmp_path / "POSCAR"),
            )

        assert result["success"] is True
        assert result["total_magnetization_bohr_mag"] == 20.0
        assert result["magnetic_elements"] == ["Fe"]

    def test_structure_file_oserror(self, tmp_path):
        mod = _import_mcp_server()
        with patch("shalom.analysis.magnetic.extract_site_magnetization",
                   return_value=None), \
             patch("shalom.analysis.magnetic.extract_lowdin_charges",
                   return_value=None), \
             patch("ase.io.read", side_effect=FileNotFoundError("No POSCAR")):
            result = mod.analyze_magnetic_properties(
                pw_out_path="pw.out",
                structure_file="/nonexistent/POSCAR",
            )

        assert result["success"] is False

    def test_exception(self):
        mod = _import_mcp_server()
        with patch("shalom.analysis.magnetic.extract_site_magnetization",
                   side_effect=OSError("read error")):
            result = mod.analyze_magnetic_properties(pw_out_path="pw.out")

        assert result["success"] is False


# ===========================================================================
# Test: run_md (MCP tool 17)
# ===========================================================================

class TestRunMD:
    """Tests for the run_md MCP tool."""

    def test_lammps_success(self):
        mod = _import_mcp_server()
        mock_result = MagicMock(
            success=True,
            output_dir="/tmp/fe_md",
            files_generated=["data.lammps", "in.lammps"],
            auto_detected={"pair_style": "eam/alloy"},
            structure_analysis={"formula": "Fe"},
            calculation_parameters={"timestep": 0.001},
            detection_log=["EAM detected for Fe"],
            error=None,
        )
        with patch("shalom.direct_run.direct_run", return_value=mock_result):
            result = mod.run_md(material="Fe", backend="lammps")

        assert result["success"] is True
        assert result["output_dir"] == "/tmp/fe_md"
        assert result["auto_detected"]["pair_style"] == "eam/alloy"

    def test_vasp_aimd(self):
        mod = _import_mcp_server()
        mock_result = MagicMock(
            success=True,
            output_dir="/tmp/si_aimd",
            files_generated=["POSCAR", "INCAR", "KPOINTS"],
            auto_detected=None,
            structure_analysis=None,
            calculation_parameters=None,
            detection_log=None,
            error=None,
        )
        with patch("shalom.direct_run.direct_run", return_value=mock_result) as mock_dr:
            result = mod.run_md(material="Si", backend="vasp")

        assert result["success"] is True
        config = mock_dr.call_args[0][1]
        assert config.calc_type == "aimd"

    def test_with_pair_style_override(self):
        mod = _import_mcp_server()
        mock_result = MagicMock(
            success=True, output_dir="/tmp/test",
            files_generated=[], auto_detected=None,
            structure_analysis=None, calculation_parameters=None,
            detection_log=None, error=None,
        )
        with patch("shalom.direct_run.direct_run", return_value=mock_result) as mock_dr:
            result = mod.run_md(
                material="Fe", pair_style="eam/alloy",
                pair_coeff="* * Fe.eam.alloy Fe",
            )

        assert result["success"] is True
        config = mock_dr.call_args[0][1]
        assert config.user_settings["pair_style"] == "eam/alloy"

    def test_exception(self):
        mod = _import_mcp_server()
        with patch("shalom.direct_run.direct_run",
                   side_effect=RuntimeError("No structure")):
            result = mod.run_md(material="XYZ")

        assert result["success"] is False
        assert "No structure" in result["error"]


# ===========================================================================
# Test: analyze_md_trajectory (MCP tool 18)
# ===========================================================================

class TestAnalyzeMDTrajectory:
    """Tests for the analyze_md_trajectory MCP tool."""

    def test_success(self, tmp_path):
        mod = _import_mcp_server()
        mock_traj = MagicMock(n_frames=100, n_atoms=54, source="lammps")
        mock_result = MagicMock(
            avg_temperature=305.2,
            temperature_std=12.3,
            avg_energy=-3.45,
            avg_pressure=1.2,
            diffusion_coefficient=2.5e-5,
            energy_drift_per_atom=1e-6,
            is_equilibrated=True,
            equilibration_step=20,
        )
        mock_result.to_dict.return_value = {"avg_temperature": 305.2}

        with patch("shalom.backends.get_backend") as mock_gb, \
             patch("shalom.analysis.md.analyze_md_trajectory",
                   return_value=mock_result), \
             patch("shalom.analysis._base.save_result_json", return_value=None):
            mock_gb.return_value.parse_trajectory.return_value = mock_traj
            result = mod.analyze_md_trajectory(
                calc_dir=str(tmp_path), backend="lammps",
            )

        assert result["success"] is True
        assert result["n_frames"] == 100
        assert result["avg_temperature_K"] == 305.2
        assert result["is_equilibrated"] is True

    def test_auto_save_json(self, tmp_path):
        mod = _import_mcp_server()
        mock_traj = MagicMock(n_frames=50, n_atoms=10, source="lammps")
        mock_result = MagicMock(
            avg_temperature=300.0, temperature_std=10.0,
            avg_energy=-2.0, avg_pressure=0.0,
            diffusion_coefficient=1e-5,
            energy_drift_per_atom=0.0,
            is_equilibrated=True, equilibration_step=10,
        )
        mock_result.to_dict.return_value = {"test": True}
        json_path = str(tmp_path / "md_analysis_results.json")

        with patch("shalom.backends.get_backend") as mock_gb, \
             patch("shalom.analysis.md.analyze_md_trajectory",
                   return_value=mock_result), \
             patch("shalom.analysis._base.save_result_json",
                   return_value=json_path) as mock_save:
            mock_gb.return_value.parse_trajectory.return_value = mock_traj
            result = mod.analyze_md_trajectory(
                calc_dir=str(tmp_path), backend="lammps",
            )

        assert result["success"] is True
        assert result["results_json_path"] == json_path
        mock_save.assert_called_once()

    def test_save_json_failure(self, tmp_path):
        """save_result_json fails but tool still returns success."""
        mod = _import_mcp_server()
        mock_traj = MagicMock(n_frames=50, n_atoms=10, source="lammps")
        mock_result = MagicMock(
            avg_temperature=300.0, temperature_std=10.0,
            avg_energy=-2.0, avg_pressure=0.0,
            diffusion_coefficient=1e-5,
            energy_drift_per_atom=0.0,
            is_equilibrated=True, equilibration_step=10,
        )
        mock_result.to_dict.return_value = {}

        with patch("shalom.backends.get_backend") as mock_gb, \
             patch("shalom.analysis.md.analyze_md_trajectory",
                   return_value=mock_result), \
             patch("shalom.analysis._base.save_result_json",
                   side_effect=OSError("write error")):
            mock_gb.return_value.parse_trajectory.return_value = mock_traj
            result = mod.analyze_md_trajectory(
                calc_dir=str(tmp_path), backend="lammps",
            )

        assert result["success"] is True
        assert "results_json_path" not in result

    def test_exception(self):
        mod = _import_mcp_server()
        with patch("shalom.backends.get_backend",
                   side_effect=ValueError("Unknown backend")):
            result = mod.analyze_md_trajectory(
                calc_dir="/nonexistent", backend="bad",
            )

        assert result["success"] is False
        assert "Unknown backend" in result["error"]
