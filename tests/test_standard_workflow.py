"""Tests for shalom.workflows.standard.StandardWorkflow."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from ase.build import bulk

from shalom.workflows.standard import StandardWorkflow


# ---------------------------------------------------------------------------
# dos.x input generation (unit test — no pw.x)
# ---------------------------------------------------------------------------


class TestDosInGeneration:
    """Verify that dos.in is written with eV units."""

    def test_dos_in_emin_in_ev(self, tmp_path, make_workflow):
        wf = make_workflow(dos_emin=-20.0, dos_emax=10.0, dos_deltaE=0.01)
        nscf_dir = str(tmp_path / "04_nscf")
        os.makedirs(nscf_dir, exist_ok=True)
        # Monkey-patch _dos_run to avoid actually running dos.x
        wf._dos_run = MagicMock()
        wf._run_dos(nscf_dir, "/fake/scf/tmp")

        dos_in = os.path.join(nscf_dir, "dos.in")
        assert os.path.isfile(dos_in)
        content = open(dos_in).read()

        # Emin/Emax written directly in eV (dos.x expects eV)
        assert "-20.000000" in content
        assert "10.000000" in content

    def test_dos_in_uses_absolute_outdir(self, tmp_path, make_workflow):
        wf = make_workflow()
        nscf_dir = str(tmp_path / "04_nscf")
        os.makedirs(nscf_dir, exist_ok=True)
        wf._dos_run = MagicMock()

        abs_scf_tmp = "/absolute/path/to/scf/tmp"
        wf._run_dos(nscf_dir, abs_scf_tmp)

        content = open(os.path.join(nscf_dir, "dos.in")).read()
        assert abs_scf_tmp in content

    def test_dos_in_prefix_is_shalom(self, tmp_path, make_workflow):
        wf = make_workflow()
        nscf_dir = str(tmp_path / "04_nscf")
        os.makedirs(nscf_dir, exist_ok=True)
        wf._dos_run = MagicMock()
        wf._run_dos(nscf_dir, "/fake/tmp")
        content = open(os.path.join(nscf_dir, "dos.in")).read()
        assert "prefix = 'shalom'" in content


# ---------------------------------------------------------------------------
# outdir absolute path enforcement
# ---------------------------------------------------------------------------


class TestAbsoluteOutdir:
    """Verify that bands/nscf configs use the absolute scf tmp path."""

    def _capture_config_outdir(self, step_method_name, workflow):
        """Run a workflow step and capture the outdir written to pw.in."""
        captured = {}

        def fake_write_input(atoms, calc_dir, config=None):
            if config is not None:
                captured["outdir"] = config.control.get("outdir")
            os.makedirs(calc_dir, exist_ok=True)
            with open(os.path.join(calc_dir, "pw.in"), "w") as f:
                f.write("! placeholder\n")

        with patch("shalom.backends.qe.QEBackend.write_input", side_effect=fake_write_input):
            workflow._pw_run = MagicMock()
            method = getattr(workflow, step_method_name)
            if step_method_name == "_run_bands":
                calc_dir = os.path.join(workflow.output_dir, "03_bands")
                method(calc_dir, workflow.atoms, "/abs/path/scf/tmp")
            elif step_method_name == "_run_nscf":
                calc_dir = os.path.join(workflow.output_dir, "04_nscf")
                method(calc_dir, workflow.atoms, "/abs/path/scf/tmp")
        return captured

    def test_bands_uses_absolute_scf_tmp(self, make_workflow):
        wf = make_workflow()
        captured = self._capture_config_outdir("_run_bands", wf)
        assert captured.get("outdir") == "/abs/path/scf/tmp"

    def test_nscf_uses_absolute_scf_tmp(self, make_workflow):
        wf = make_workflow()
        captured = self._capture_config_outdir("_run_nscf", wf)
        assert captured.get("outdir") == "/abs/path/scf/tmp"


# ---------------------------------------------------------------------------
# Fermi energy priority (NSCF > SCF)
# ---------------------------------------------------------------------------


class TestFermiEnergyPriority:
    def test_nscf_preferred_over_scf(self, tmp_path, make_workflow, setup_fermi_dirs):
        wf = make_workflow()
        scf_dir, nscf_dir = setup_fermi_dirs(include_dos=False)
        ef = wf._get_best_fermi_energy(scf_dir, nscf_dir)
        assert ef == pytest.approx(5.1234)

    def test_falls_back_to_scf_if_nscf_missing(self, tmp_path, make_workflow):
        wf = make_workflow()
        scf_dir = tmp_path / "02_scf"
        nscf_dir = tmp_path / "04_nscf"
        scf_dir.mkdir()
        nscf_dir.mkdir()  # pw.out not written
        (scf_dir / "pw.out").write_text(
            "the Fermi energy is   4.9876 ev\n"
        )
        ef = wf._get_best_fermi_energy(str(scf_dir), str(nscf_dir))
        assert ef == pytest.approx(4.9876)

    def test_returns_none_if_both_missing(self, tmp_path, make_workflow):
        wf = make_workflow()
        scf_dir = tmp_path / "02_scf"
        nscf_dir = tmp_path / "04_nscf"
        scf_dir.mkdir()
        nscf_dir.mkdir()
        ef = wf._get_best_fermi_energy(str(scf_dir), str(nscf_dir))
        assert ef is None


# ---------------------------------------------------------------------------
# StandardWorkflow instantiation
# ---------------------------------------------------------------------------


def test_standard_workflow_defaults():
    si = bulk("Si", "diamond", a=5.43)
    wf = StandardWorkflow(atoms=si, output_dir="/tmp/test_wf")
    assert wf.nprocs == 1
    assert wf.accuracy == "standard"
    assert not wf.skip_relax
    assert not wf.is_2d
    assert wf.dos_emin == pytest.approx(-20.0)
    assert wf.dos_emax == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# _pw_run / _dos_run helpers
# ---------------------------------------------------------------------------


class TestPwRun:
    def test_pw_run_success(self, tmp_path, make_workflow):
        """_pw_run succeeds when ExecutionRunner returns success."""
        wf = make_workflow()
        from unittest.mock import MagicMock, patch

        mock_result = MagicMock()
        mock_result.success = True

        with patch("shalom.backends.runner.ExecutionRunner.run", return_value=mock_result):
            # Should not raise
            wf._pw_run(str(tmp_path))

    def test_pw_run_failure_raises(self, tmp_path, make_workflow):
        """_pw_run raises RuntimeError when runner fails."""
        wf = make_workflow()
        from unittest.mock import MagicMock, patch

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error_message = "pw.x crashed"

        with patch("shalom.backends.runner.ExecutionRunner.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="pw.x failed"):
                wf._pw_run(str(tmp_path))

    def test_pw_run_failure_no_message(self, tmp_path, make_workflow):
        """_pw_run raises RuntimeError even when error_message is None."""
        wf = make_workflow()
        from unittest.mock import MagicMock, patch

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error_message = None

        with patch("shalom.backends.runner.ExecutionRunner.run", return_value=mock_result):
            with pytest.raises(RuntimeError):
                wf._pw_run(str(tmp_path))


class TestDosRun:
    def test_dos_run_success(self, tmp_path, make_workflow):
        """_dos_run succeeds silently."""
        wf = make_workflow()
        from unittest.mock import MagicMock, patch

        mock_result = MagicMock()
        mock_result.success = True

        with patch("shalom.backends.runner.ExecutionRunner.run", return_value=mock_result):
            wf._dos_run(str(tmp_path))  # no exception

    def test_dos_run_failure_only_warns(self, tmp_path, make_workflow):
        """_dos_run logs warning on failure instead of raising."""
        wf = make_workflow()
        from unittest.mock import MagicMock, patch
        import logging

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error_message = "dos.x failed"

        with patch("shalom.backends.runner.ExecutionRunner.run", return_value=mock_result):
            with patch("shalom.workflows.standard.logger") as mock_log:
                wf._dos_run(str(tmp_path))  # should NOT raise
                mock_log.warning.assert_called()


# ---------------------------------------------------------------------------
# _run_vc_relax
# ---------------------------------------------------------------------------


class TestRunVcRelax:
    def test_returns_relaxed_structure(self, tmp_path, make_workflow):
        """When ase_read succeeds, returns the relaxed atoms."""
        wf = make_workflow()

        from unittest.mock import MagicMock, patch
        from ase.build import bulk as ase_bulk

        relaxed = ase_bulk("Si", "diamond", a=5.50)

        with patch("shalom.backends.qe.QEBackend.write_input"), \
             patch.object(wf, "_pw_run"), \
             patch("shalom.workflows.standard.ase_read", return_value=relaxed):
            result = wf._run_vc_relax(str(tmp_path / "01_vc_relax"), wf.atoms)

        assert result is relaxed

    def test_falls_back_to_input_on_read_failure(self, tmp_path, make_workflow):
        """When ase_read fails, returns the original atoms."""
        wf = make_workflow()

        from unittest.mock import patch

        with patch("shalom.backends.qe.QEBackend.write_input"), \
             patch.object(wf, "_pw_run"), \
             patch("shalom.workflows.standard.ase_read", side_effect=Exception("parse error")):
            result = wf._run_vc_relax(str(tmp_path / "01_vc_relax"), wf.atoms)

        assert result is wf.atoms

    def test_precise_accuracy_applied(self, tmp_path, make_workflow):
        """Precise accuracy flag is forwarded to get_qe_preset."""
        wf = make_workflow(accuracy="precise")

        from unittest.mock import patch, MagicMock

        captured_acc = {}

        def fake_preset(calc_type, accuracy, atoms=None):
            captured_acc["acc"] = accuracy
            cfg = MagicMock()
            cfg.pseudo_dir = None
            cfg.control = {}
            cfg.system = {}
            cfg.kpoints = None
            return cfg

        with patch("shalom.workflows.standard.get_qe_preset", side_effect=fake_preset), \
             patch("shalom.backends.qe.QEBackend.write_input"), \
             patch.object(wf, "_pw_run"), \
             patch("shalom.workflows.standard.ase_read", return_value=wf.atoms):
            wf._run_vc_relax(str(tmp_path / "01"), wf.atoms)

        from shalom.backends._physics import AccuracyLevel
        assert captured_acc["acc"] == AccuracyLevel.PRECISE

    def test_pseudo_dir_override_applied(self, tmp_path, make_workflow):
        """pseudo_dir is applied to the config when set."""
        wf = make_workflow(pseudo_dir="/my/pseudos")

        from unittest.mock import patch, MagicMock

        config_obj = MagicMock()
        config_obj.control = {}
        config_obj.system = {}
        config_obj.kpoints = None

        with patch("shalom.workflows.standard.get_qe_preset", return_value=config_obj), \
             patch("shalom.backends.qe.QEBackend.write_input"), \
             patch.object(wf, "_pw_run"), \
             patch("shalom.workflows.standard.ase_read", return_value=wf.atoms):
            wf._run_vc_relax(str(tmp_path / "01"), wf.atoms)

        assert config_obj.pseudo_dir == "/my/pseudos"


# ---------------------------------------------------------------------------
# _run_scf
# ---------------------------------------------------------------------------


class TestRunScf:
    def test_creates_directory_and_runs(self, tmp_path, make_workflow):
        """_run_scf creates calc_dir and calls write_input + _pw_run."""
        wf = make_workflow()

        scf_dir = str(tmp_path / "02_scf")
        from unittest.mock import patch, MagicMock, call

        write_calls = []

        def fake_write(atoms, calc_dir, config=None):
            write_calls.append(calc_dir)

        with patch("shalom.backends.qe.QEBackend.write_input", side_effect=fake_write), \
             patch.object(wf, "_pw_run") as mock_pw:
            wf._run_scf(scf_dir, wf.atoms)

        assert len(write_calls) == 1
        assert write_calls[0] == scf_dir
        mock_pw.assert_called_once_with(scf_dir)

    def test_pseudo_dir_override(self, tmp_path, make_workflow):
        """pseudo_dir is forwarded to config."""
        wf = make_workflow(pseudo_dir="/pseudo")

        from unittest.mock import patch, MagicMock

        config_obj = MagicMock()
        config_obj.control = {}
        config_obj.system = {}
        config_obj.kpoints = None

        with patch("shalom.workflows.standard.get_qe_preset", return_value=config_obj), \
             patch("shalom.backends.qe.QEBackend.write_input"), \
             patch.object(wf, "_pw_run"):
            wf._run_scf(str(tmp_path / "02"), wf.atoms)

        assert config_obj.pseudo_dir == "/pseudo"


# ---------------------------------------------------------------------------
# _run_bands
# ---------------------------------------------------------------------------


class TestRunBandsExtended:
    def test_uses_cached_kpath(self, tmp_path, make_workflow):
        """_run_bands uses self._kpath_cfg when available."""
        wf = make_workflow()

        from unittest.mock import patch, MagicMock
        from shalom.backends.qe_config import QEKPointsConfig

        cached_kpath = QEKPointsConfig(mode="crystal_b", kpath_points=[([0, 0, 0], 1)])
        wf._kpath_cfg = cached_kpath

        config_obj = MagicMock()
        config_obj.control = {}
        config_obj.system = {"nbnd": 20}
        config_obj.kpoints = None

        with patch("shalom.workflows.standard.get_qe_preset", return_value=config_obj), \
             patch("shalom.backends.qe.QEBackend.write_input"), \
             patch.object(wf, "_pw_run"):
            wf._run_bands(str(tmp_path / "03"), wf.atoms, "/fake/scf/tmp")

        assert config_obj.kpoints is cached_kpath

    def test_generates_kpath_when_no_cache(self, tmp_path, make_workflow):
        """_run_bands generates kpath when _kpath_cfg is None."""
        wf = make_workflow()
        wf._kpath_cfg = None

        from unittest.mock import patch, MagicMock
        from shalom.backends.qe_config import QEKPointsConfig

        generated_kpath = QEKPointsConfig(mode="crystal_b", kpath_points=[([0, 0, 0], 1)])

        config_obj = MagicMock()
        config_obj.control = {}
        config_obj.system = {"nbnd": 20}
        config_obj.kpoints = None

        with patch("shalom.workflows.standard.get_qe_preset", return_value=config_obj), \
             patch("shalom.workflows.standard.generate_band_kpath", return_value=generated_kpath), \
             patch("shalom.backends.qe.QEBackend.write_input"), \
             patch.object(wf, "_pw_run"):
            wf._run_bands(str(tmp_path / "03"), wf.atoms, "/fake/scf/tmp")

        assert config_obj.kpoints is generated_kpath

    def test_nbnd_set_in_system(self, tmp_path, make_workflow):
        """_run_bands auto-computes nbnd and adds it to config.system."""
        wf = make_workflow()

        from unittest.mock import patch, MagicMock
        from shalom.backends.qe_config import QEKPointsConfig

        config_obj = MagicMock()
        config_obj.control = {}
        config_obj.system = {}  # no nbnd set
        config_obj.kpoints = None

        with patch("shalom.workflows.standard.get_qe_preset", return_value=config_obj), \
             patch("shalom.workflows.standard.generate_band_kpath",
                   return_value=QEKPointsConfig(mode="crystal_b")), \
             patch("shalom.backends.qe.QEBackend.write_input"), \
             patch.object(wf, "_pw_run"):
            wf._run_bands(str(tmp_path / "03"), wf.atoms, "/fake/scf/tmp")

        assert "nbnd" in config_obj.system
        assert config_obj.system["nbnd"] >= 20


# ---------------------------------------------------------------------------
# _run_nscf
# ---------------------------------------------------------------------------


class TestRunNscfExtended:
    def test_creates_dir_and_runs(self, tmp_path, make_workflow):
        """_run_nscf creates directory, writes input, runs pw.x."""
        wf = make_workflow()

        nscf_dir = str(tmp_path / "04_nscf")
        from unittest.mock import patch, MagicMock

        config_obj = MagicMock()
        config_obj.control = {}
        config_obj.system = {}

        with patch("shalom.workflows.standard.get_qe_preset", return_value=config_obj), \
             patch("shalom.backends.qe.QEBackend.write_input") as mock_write, \
             patch.object(wf, "_pw_run") as mock_pw:
            wf._run_nscf(nscf_dir, wf.atoms, "/abs/scf/tmp")

        mock_write.assert_called_once()
        mock_pw.assert_called_once_with(nscf_dir)
        assert config_obj.control["outdir"] == "/abs/scf/tmp"


# ---------------------------------------------------------------------------
# _plot_bands
# ---------------------------------------------------------------------------


class TestPlotBands:
    def test_returns_none_when_no_xml(self, tmp_path, make_workflow):
        """Returns None when no bands XML found."""
        wf = make_workflow()

        from unittest.mock import patch

        with patch("shalom.workflows.standard.find_xml_path", return_value=None):
            result = wf._plot_bands(str(tmp_path / "03"), fermi=5.0)

        assert result is None

    def test_returns_none_when_no_matplotlib(self, tmp_path, make_workflow):
        """Returns None when matplotlib is not installed (import fails)."""
        wf = make_workflow()

        from unittest.mock import patch
        import sys

        # Remove the band_plot module from sys.modules so the try/except
        # inside _plot_bands triggers ImportError on the local import.
        with patch.dict(sys.modules, {"shalom.plotting.band_plot": None}):
            result = wf._plot_bands(str(tmp_path / "03"), fermi=5.0)

        assert result is None

    def test_plot_bands_full_pipeline(self, tmp_path, make_workflow):
        """_plot_bands calls plotter and returns output path."""
        wf = make_workflow()

        from unittest.mock import patch, MagicMock
        from shalom.backends.base import BandStructureData
        import numpy as np
        from shalom.backends.qe_config import QEKPointsConfig

        fake_bs = BandStructureData(
            eigenvalues=np.zeros((5, 3)),
            kpoint_coords=np.zeros((5, 3)),
            kpath_distances=np.linspace(0, 1, 5),
            high_sym_labels={0: "G", 4: "X"},
            source="qe",
        )

        kpath = QEKPointsConfig(
            mode="crystal_b",
            kpath_points=[([0, 0, 0], 40), ([0.5, 0, 0], 1)],
            kpath_labels={0: "G", 1: "X"},
        )
        wf._kpath_cfg = kpath

        mock_plotter = MagicMock()

        with patch("shalom.workflows.standard.find_xml_path", return_value="/fake/data-file-schema.xml"), \
             patch("shalom.workflows.standard.parse_xml_bands", return_value=fake_bs), \
             patch("shalom.plotting.band_plot.BandStructurePlotter", return_value=mock_plotter):
            result = wf._plot_bands(str(tmp_path / "03"), fermi=5.0)

        mock_plotter.plot.assert_called_once()
        assert result == os.path.join(str(tmp_path), "bands.png")

    def test_plot_bands_gap_collapse(self, tmp_path, make_workflow):
        """Gap at discontinuity label 'X|U' is removed from kpath_distances."""
        wf = make_workflow()

        from unittest.mock import patch, MagicMock
        from shalom.backends.base import BandStructureData
        import numpy as np
        from shalom.backends.qe_config import QEKPointsConfig

        # Simulate 5 k-points with a gap at index 2 (X|U label)
        dist = np.array([0.0, 0.5, 1.0, 1.3, 1.8])  # gap of 0.3 between idx 2→3
        fake_bs = BandStructureData(
            eigenvalues=np.zeros((5, 3)),
            kpoint_coords=np.zeros((5, 3)),
            kpath_distances=dist.copy(),
            high_sym_labels={},
            source="qe",
        )

        # kpath: G at cumulative-idx 0 (npts=2), X|U at cumulative-idx 2 (npts=1), K at 3
        # cumulative indices: G→0, X|U→2, K→3  so gap collapse removes dist[3]-dist[2]
        kpath = QEKPointsConfig(
            mode="crystal_b",
            kpath_points=[([0, 0, 0], 2), ([0.5, 0.5, 0], 1), ([0.5, 0.5, 0.5], 2)],
            kpath_labels={0: "G", 1: "X|U", 2: "K"},
        )
        wf._kpath_cfg = kpath

        captured_bs = {}

        def fake_plotter(bs_arg):
            captured_bs["bs"] = bs_arg
            m = MagicMock()
            m.plot = MagicMock()
            return m

        with patch("shalom.workflows.standard.find_xml_path", return_value="/x.xml"), \
             patch("shalom.workflows.standard.parse_xml_bands", return_value=fake_bs), \
             patch("shalom.plotting.band_plot.BandStructurePlotter", side_effect=fake_plotter):
            wf._plot_bands(str(tmp_path / "03"), fermi=0.0)

        # After gap collapse, the distance at index 3 should be closer to index 2
        result_dist = captured_bs["bs"].kpath_distances
        # The gap (0.3) between index 2 (1.0) and index 3 (1.3) must be removed
        assert result_dist[3] == pytest.approx(result_dist[2], abs=1e-9)


# ---------------------------------------------------------------------------
# _plot_dos
# ---------------------------------------------------------------------------


class TestPlotDos:
    def test_returns_none_when_no_dos_file(self, tmp_path, make_workflow):
        """Returns None when pwscf.dos doesn't exist."""
        wf = make_workflow()

        result = wf._plot_dos(str(tmp_path / "04_nscf"), fermi=5.0)
        assert result is None

    def test_plot_dos_full_pipeline(self, tmp_path, make_workflow):
        """_plot_dos calls DOSPlotter and returns output path."""
        wf = make_workflow()

        nscf_dir = tmp_path / "04_nscf"
        nscf_dir.mkdir()
        (nscf_dir / "pwscf.dos").write_text("# DOS file\n0.0 1.0 0.5\n")

        from unittest.mock import patch, MagicMock
        from shalom.backends.base import DOSData
        import numpy as np

        fake_dos = DOSData(
            energies=np.array([-1.0, 0.0, 1.0]),
            dos=np.array([0.5, 1.0, 0.5]),
            integrated_dos=np.array([0.0, 0.5, 1.0]),
            source="qe",
        )

        mock_plotter = MagicMock()

        with patch("shalom.workflows.standard.parse_dos_file", return_value=fake_dos), \
             patch("shalom.plotting.dos_plot.DOSPlotter", return_value=mock_plotter):
            result = wf._plot_dos(str(nscf_dir), fermi=5.0)

        mock_plotter.plot.assert_called_once()
        assert result == os.path.join(str(tmp_path), "dos.png")
        # Fermi energy was injected
        assert fake_dos.fermi_energy == pytest.approx(5.0)

    def test_plot_dos_none_fermi_not_set(self, tmp_path, make_workflow):
        """When fermi=None, fermi_energy is not set on DOSData."""
        wf = make_workflow()

        nscf_dir = tmp_path / "04_nscf"
        nscf_dir.mkdir()
        (nscf_dir / "pwscf.dos").write_text("# DOS\n0.0 1.0 0.5\n")

        from unittest.mock import patch, MagicMock
        from shalom.backends.base import DOSData
        import numpy as np

        fake_dos = DOSData(
            energies=np.array([0.0]),
            dos=np.array([1.0]),
            integrated_dos=np.array([0.0]),
            source="qe",
        )
        fake_dos.fermi_energy = 99.0  # should stay unchanged

        mock_plotter = MagicMock()

        with patch("shalom.workflows.standard.parse_dos_file", return_value=fake_dos), \
             patch("shalom.plotting.dos_plot.DOSPlotter", return_value=mock_plotter):
            wf._plot_dos(str(nscf_dir), fermi=None)

        assert fake_dos.fermi_energy == pytest.approx(99.0)


# ---------------------------------------------------------------------------
# run() — full workflow integration with all steps mocked
# ---------------------------------------------------------------------------


class TestRunFull:
    def _mock_wf_run(self, wf, tmp_path, minimal_dos_data, setup_fermi_dirs):
        """Run wf.run() with all pw.x/plotting mocked out."""
        from unittest.mock import patch, MagicMock
        from shalom.backends.base import BandStructureData
        import numpy as np

        fake_bs = BandStructureData(
            eigenvalues=np.zeros((3, 3)),
            kpoint_coords=np.zeros((3, 3)),
            kpath_distances=np.linspace(0, 1, 3),
            high_sym_labels={},
            source="qe",
        )
        fake_dos = minimal_dos_data

        setup_fermi_dirs()

        mock_band_plotter = MagicMock()
        mock_dos_plotter = MagicMock()

        with patch.object(wf, "_pw_run"), \
             patch.object(wf, "_dos_run"), \
             patch("shalom.workflows.standard.find_xml_path", return_value="/x.xml"), \
             patch("shalom.workflows.standard.parse_xml_bands", return_value=fake_bs), \
             patch("shalom.workflows.standard.parse_dos_file", return_value=fake_dos), \
             patch("shalom.plotting.band_plot.BandStructurePlotter", return_value=mock_band_plotter), \
             patch("shalom.plotting.dos_plot.DOSPlotter", return_value=mock_dos_plotter):
            return wf.run()

    def test_run_skip_relax_returns_dict(self, tmp_path, make_workflow, minimal_dos_data, setup_fermi_dirs):
        """run() with skip_relax=True returns expected keys."""
        wf = make_workflow(skip_relax=True)
        result = self._mock_wf_run(wf, tmp_path, minimal_dos_data, setup_fermi_dirs)

        assert "atoms" in result
        assert "fermi_energy" in result
        assert "bands_png" in result
        assert "dos_png" in result
        assert "calc_dirs" in result

    def test_run_fermi_energy_from_nscf(self, tmp_path, make_workflow, minimal_dos_data, setup_fermi_dirs):
        """run() extracts Fermi energy from NSCF pw.out (priority)."""
        wf = make_workflow(skip_relax=True)
        result = self._mock_wf_run(wf, tmp_path, minimal_dos_data, setup_fermi_dirs)

        assert result["fermi_energy"] == pytest.approx(5.1234, rel=1e-4)

    def test_run_calc_dirs_has_all_steps(self, tmp_path, make_workflow, minimal_dos_data, setup_fermi_dirs):
        """run() result contains all 4 calc_dir keys."""
        wf = make_workflow(skip_relax=True)
        result = self._mock_wf_run(wf, tmp_path, minimal_dos_data, setup_fermi_dirs)

        assert "vc_relax" in result["calc_dirs"]
        assert "scf" in result["calc_dirs"]
        assert "bands" in result["calc_dirs"]
        assert "nscf" in result["calc_dirs"]

    def test_run_with_relax_calls_vc_relax(self, tmp_path, make_workflow, minimal_dos_data):
        """run() without skip_relax invokes _run_vc_relax."""
        wf = make_workflow(skip_relax=False)

        from unittest.mock import patch, MagicMock
        from shalom.backends.base import BandStructureData
        import numpy as np

        fake_bs = BandStructureData(
            eigenvalues=np.zeros((3, 3)),
            kpoint_coords=np.zeros((3, 3)),
            kpath_distances=np.linspace(0, 1, 3),
            high_sym_labels={},
            source="qe",
        )
        fake_dos = minimal_dos_data

        dos_dir = tmp_path / "04_nscf"
        dos_dir.mkdir(parents=True, exist_ok=True)
        (dos_dir / "pwscf.dos").write_text("# DOS\n0.0 1.0 0.5\n")
        (dos_dir / "pw.out").write_text("the Fermi energy is   5.0000 ev\n")
        scf_dir = tmp_path / "02_scf"
        scf_dir.mkdir(parents=True, exist_ok=True)
        (scf_dir / "pw.out").write_text("")

        vc_relax_called = []

        def fake_vc_relax(calc_dir, atoms):
            vc_relax_called.append(True)
            return atoms

        mock_band_plotter = MagicMock()
        mock_dos_plotter = MagicMock()

        with patch.object(wf, "_run_vc_relax", side_effect=fake_vc_relax), \
             patch.object(wf, "_pw_run"), \
             patch.object(wf, "_dos_run"), \
             patch("shalom.workflows.standard.find_xml_path", return_value="/x.xml"), \
             patch("shalom.workflows.standard.parse_xml_bands", return_value=fake_bs), \
             patch("shalom.workflows.standard.parse_dos_file", return_value=fake_dos), \
             patch("shalom.plotting.band_plot.BandStructurePlotter", return_value=mock_band_plotter), \
             patch("shalom.plotting.dos_plot.DOSPlotter", return_value=mock_dos_plotter):
            wf.run()

        assert len(vc_relax_called) == 1

    def test_run_kpath_cache_populated(self, tmp_path, make_workflow, minimal_dos_data, setup_fermi_dirs):
        """After run(), _kpath_cfg and _calc_atoms are populated."""
        wf = make_workflow(skip_relax=True)
        self._mock_wf_run(wf, tmp_path, minimal_dos_data, setup_fermi_dirs)

        assert wf._kpath_cfg is not None
        assert wf._calc_atoms is not None

    def test_run_png_paths_in_output_dir(self, tmp_path, make_workflow, minimal_dos_data, setup_fermi_dirs):
        """bands.png and dos.png are in the output_dir."""
        wf = make_workflow(skip_relax=True)
        result = self._mock_wf_run(wf, tmp_path, minimal_dos_data, setup_fermi_dirs)

        if result["bands_png"] is not None:
            assert result["bands_png"].startswith(str(tmp_path))
        if result["dos_png"] is not None:
            assert result["dos_png"].startswith(str(tmp_path))


# ---------------------------------------------------------------------------
# Bands XML preservation test
# ---------------------------------------------------------------------------


class TestBandsXMLPreservation:
    """Verify that bands XML is copied to 03_bands/ before NSCF overwrites it."""

    def test_bands_xml_copied_to_bands_dir(self, tmp_path):
        """After _run_bands, XML should be preserved in bands_dir."""
        from shalom.backends.qe_parser import find_xml_path

        bands_dir = str(tmp_path / "03_bands")
        scf_tmp_dir = str(tmp_path / "02_scf" / "tmp")
        os.makedirs(bands_dir, exist_ok=True)
        save_dir = os.path.join(scf_tmp_dir, "shalom.save")
        os.makedirs(save_dir, exist_ok=True)

        # Simulate: bands step wrote XML to scf_tmp_dir
        xml_content = '<?xml version="1.0"?><root>bands_data</root>'
        xml_src = os.path.join(save_dir, "data-file-schema.xml")
        with open(xml_src, "w") as f:
            f.write(xml_content)

        # Execute the copy logic (same as in run() after _run_bands)
        bands_xml_src = find_xml_path(scf_tmp_dir)
        assert bands_xml_src is not None

        import shutil
        bands_xml_dst = os.path.join(bands_dir, "data-file-schema.xml")
        shutil.copy2(bands_xml_src, bands_xml_dst)

        # Verify copy exists and has correct content
        assert os.path.isfile(bands_xml_dst)
        with open(bands_xml_dst) as f:
            assert "bands_data" in f.read()

        # Simulate: NSCF overwrites the original
        with open(xml_src, "w") as f:
            f.write('<?xml version="1.0"?><root>nscf_data</root>')

        # bands_dir copy should still have bands_data
        with open(bands_xml_dst) as f:
            content = f.read()
        assert "bands_data" in content
        assert "nscf_data" not in content


# ---------------------------------------------------------------------------
# dos.in unit defense: eV values, NOT Ry
# ---------------------------------------------------------------------------


class TestDosInUnitDefense:
    """Ensure dos.in values are in eV range, never Ry range."""

    def test_dos_in_not_in_ry_range(self, tmp_path, make_workflow):
        """dos.in Emin/Emax must NOT be in Ry range (validates eV fix)."""
        wf = make_workflow(dos_emin=-20.0, dos_emax=10.0, dos_deltaE=0.01)
        nscf_dir = str(tmp_path / "04_nscf")
        os.makedirs(nscf_dir, exist_ok=True)
        wf._dos_run = MagicMock()
        wf._run_dos(nscf_dir, "/fake/scf/tmp")

        content = open(os.path.join(nscf_dir, "dos.in")).read()

        # Extract numeric Emin/Emax values
        import re
        emin_match = re.search(r"Emin\s*=\s*([-\d.]+)", content)
        emax_match = re.search(r"Emax\s*=\s*([-\d.]+)", content)
        assert emin_match and emax_match

        emin_val = float(emin_match.group(1))
        emax_val = float(emax_match.group(1))

        # Must be eV values (-20, 10), NOT Ry values (-1.47, 0.73)
        assert emin_val == pytest.approx(-20.0)
        assert emax_val == pytest.approx(10.0)
        # Explicitly verify NOT in Ry range
        assert abs(emin_val) > 5.0, "Emin looks like Ry, not eV"
        assert abs(emax_val) > 5.0, "Emax looks like Ry, not eV"


# ---------------------------------------------------------------------------
# Parameter validation tests
# ---------------------------------------------------------------------------


class TestParameterValidation:
    """StandardWorkflow __init__ should reject invalid parameters."""

    @pytest.mark.parametrize("kwargs,match", [
        (dict(accuracy="high"), "accuracy must be"),
        (dict(dos_emin=10.0, dos_emax=-20.0), "dos_emin"),
        (dict(dos_emin=5.0, dos_emax=5.0), "dos_emin"),
        (dict(dos_deltaE=0.0), "dos_deltaE"),
        (dict(dos_deltaE=-0.01), "dos_deltaE"),
    ])
    def test_invalid_params_raise(self, kwargs, match, sample_si_diamond):
        with pytest.raises(ValueError, match=match):
            StandardWorkflow(atoms=sample_si_diamond, output_dir="/tmp/test", **kwargs)

    @pytest.mark.parametrize("accuracy", ["standard", "precise"])
    def test_valid_accuracy_accepted(self, accuracy, sample_si_diamond):
        wf = StandardWorkflow(
            atoms=sample_si_diamond, output_dir="/tmp/test", accuracy=accuracy,
        )
        assert wf.accuracy == accuracy


# ---------------------------------------------------------------------------
# Pre-flight environment validation test
# ---------------------------------------------------------------------------


class TestPreflightValidation:
    """StandardWorkflow._validate_environment warns on missing executables."""

    def test_warns_when_pw_missing(self, tmp_path, caplog, make_workflow):
        import logging
        wf = make_workflow()
        # Mock shutil.which to return None for all executables
        with patch("shutil.which", return_value=None):
            with caplog.at_level(logging.WARNING):
                # Should log warnings, not raise
                wf._validate_environment()
        assert "pw.x" in caplog.text
        assert "dos.x" in caplog.text

    def test_warns_when_pseudo_dir_missing(self, tmp_path, caplog, make_workflow):
        import logging
        wf = make_workflow(pseudo_dir="/nonexistent/pseudo/dir")
        with caplog.at_level(logging.WARNING):
            wf._validate_environment()
        assert "pseudo_dir" in caplog.text


# ---------------------------------------------------------------------------
# StepStatus dataclass tests
# ---------------------------------------------------------------------------


class TestStepStatus:
    """Verify StepStatus dataclass fields and defaults."""

    def test_step_status_fields(self):
        """StepStatus has expected fields with correct defaults."""
        from shalom.workflows.standard import StepStatus
        s = StepStatus(name="scf", step_number=2, success=True)
        assert s.name == "scf"
        assert s.step_number == 2
        assert s.success is True
        assert s.error_message is None
        assert s.elapsed_seconds == 0.0
        assert s.summary == ""

    def test_step_status_with_error(self):
        """StepStatus can store error message."""
        from shalom.workflows.standard import StepStatus
        s = StepStatus(
            name="bands", step_number=3, success=False,
            error_message="pw.x crashed", elapsed_seconds=12.5,
        )
        assert s.success is False
        assert s.error_message == "pw.x crashed"
        assert s.elapsed_seconds == 12.5

    def test_step_status_with_summary(self):
        """StepStatus can store summary string."""
        from shalom.workflows.standard import StepStatus
        s = StepStatus(
            name="vc_relax", step_number=1, success=True,
            summary="E=-10.5 eV, converged",
        )
        assert s.summary == "E=-10.5 eV, converged"

    def test_step_status_skipped(self):
        """StepStatus for a skipped step."""
        from shalom.workflows.standard import StepStatus
        s = StepStatus(name="vc_relax", step_number=1, success=True, summary="skipped")
        assert s.success is True
        assert s.summary == "skipped"

    def test_step_status_resumed(self):
        """StepStatus for a resumed step."""
        from shalom.workflows.standard import StepStatus
        s = StepStatus(name="scf", step_number=2, success=True, summary="resumed")
        assert s.summary == "resumed"


# ---------------------------------------------------------------------------
# Progress helpers tests
# ---------------------------------------------------------------------------


class TestProgressHelpers:
    """Test _log_step_start, _log_step_end, _extract_pw_summary."""

    def test_log_step_start_returns_float(self):
        """_log_step_start returns a monotonic timestamp."""
        si = bulk("Si", "diamond", a=5.43)
        wf = StandardWorkflow(atoms=si, output_dir="/tmp/test")
        t0 = wf._log_step_start(1, 5, "vc-relax")
        assert isinstance(t0, float)
        assert t0 > 0.0

    def test_log_step_start_logs_info(self, caplog):
        """_log_step_start logs the step info message."""
        import logging
        si = bulk("Si", "diamond", a=5.43)
        wf = StandardWorkflow(atoms=si, output_dir="/tmp/test")
        with caplog.at_level(logging.INFO, logger="shalom.workflows.standard"):
            wf._log_step_start(2, 5, "scf")
        assert "[2/5]" in caplog.text
        assert "scf" in caplog.text

    def test_log_step_end_logs_info(self, caplog):
        """_log_step_end logs completion with elapsed time."""
        import logging
        import time
        si = bulk("Si", "diamond", a=5.43)
        wf = StandardWorkflow(atoms=si, output_dir="/tmp/test")
        t0 = time.monotonic() - 2.5  # simulate 2.5 seconds elapsed
        with caplog.at_level(logging.INFO, logger="shalom.workflows.standard"):
            wf._log_step_end(3, 5, "bands", t0, summary="E=-5.0 eV")
        assert "[3/5]" in caplog.text
        assert "bands" in caplog.text
        assert "done" in caplog.text
        assert "E=-5.0 eV" in caplog.text

    def test_log_step_end_minutes_format(self, caplog):
        """_log_step_end uses minutes format for long steps."""
        import logging
        import time
        si = bulk("Si", "diamond", a=5.43)
        wf = StandardWorkflow(atoms=si, output_dir="/tmp/test")
        t0 = time.monotonic() - 125.0  # 2m5s
        with caplog.at_level(logging.INFO, logger="shalom.workflows.standard"):
            wf._log_step_end(1, 5, "vc-relax", t0)
        assert "m" in caplog.text  # should show Nm format

    def test_extract_pw_summary_with_total_energy(self, tmp_path, make_workflow):
        """_extract_pw_summary extracts total energy with correct Ry→eV conversion."""
        pw_out = tmp_path / "pw.out"
        pw_out.write_text(
            "!    total energy              =     -15.85012345 Ry\n"
            "     the Fermi energy is     6.1234 ev\n"
            "     convergence has been achieved in   8 iterations\n"
            "          iteration #   8     ecut=    60.00 Ry\n"
        )
        wf = make_workflow()
        summary = wf._extract_pw_summary(str(pw_out))
        # Verify Ry→eV conversion: -15.850 * 13.6057 ≈ -215.6
        assert "-215." in summary
        assert "eV" in summary
        assert "Ef=6.1234" in summary
        assert "8 SCF iter" in summary
        assert "converged" in summary

    def test_extract_pw_summary_no_file(self, tmp_path, make_workflow):
        """_extract_pw_summary returns empty string for missing file."""
        wf = make_workflow()
        summary = wf._extract_pw_summary(str(tmp_path / "nonexistent.out"))
        assert summary == ""

    def test_extract_pw_summary_empty_file(self, tmp_path, make_workflow):
        """_extract_pw_summary returns empty string for empty file."""
        pw_out = tmp_path / "pw.out"
        pw_out.write_text("")
        wf = make_workflow()
        summary = wf._extract_pw_summary(str(pw_out))
        assert summary == ""

    def test_extract_pw_summary_partial_data(self, tmp_path, make_workflow):
        """_extract_pw_summary handles partial output (no Fermi)."""
        pw_out = tmp_path / "pw.out"
        pw_out.write_text(
            "!    total energy              =     -10.00000000 Ry\n"
            "     convergence has been achieved in   5 iterations\n"
            "          iteration #   5     ecut=    60.00 Ry\n"
        )
        wf = make_workflow()
        summary = wf._extract_pw_summary(str(pw_out))
        assert "eV" in summary
        assert "5 SCF iter" in summary
        assert "converged" in summary
        # No Fermi energy line → no Ef=
        assert "Ef=" not in summary

    def test_extract_pw_summary_not_converged(self, tmp_path, make_workflow):
        """_extract_pw_summary without convergence line omits 'converged'."""
        pw_out = tmp_path / "pw.out"
        pw_out.write_text(
            "!    total energy              =     -10.00000000 Ry\n"
            "     the Fermi energy is     5.0000 ev\n"
            "          iteration #  50     ecut=    60.00 Ry\n"
        )
        wf = make_workflow()
        summary = wf._extract_pw_summary(str(pw_out))
        assert "eV" in summary
        assert "50 SCF iter" in summary
        assert "converged" not in summary


# ---------------------------------------------------------------------------
# Step error handling tests
# ---------------------------------------------------------------------------


class TestStepErrorHandling:
    """Test run() with step failures (mock _pw_run to raise RuntimeError)."""

    def test_scf_failure_returns_failed_step(self, tmp_path, make_workflow):
        """SCF failure sets failed_step='scf' and returns immediately."""
        wf = make_workflow(skip_relax=True)
        with patch.object(wf, "_pw_run", side_effect=RuntimeError("SCF diverged")), \
             patch.object(wf, "_dos_run"):
            result = wf.run()
        assert result["failed_step"] == "scf"
        assert "scf" not in result["completed_steps"]
        # Bands/NSCF/DOS should not have run
        step_names = [s.name for s in result["step_results"]]
        assert "bands" not in step_names
        assert "nscf" not in step_names

    def test_bands_failure_nscf_still_runs(self, tmp_path, make_workflow, minimal_dos_data, setup_fermi_dirs):
        """Bands failure does not prevent NSCF from running."""
        wf = make_workflow(skip_relax=True)
        call_count = [0]

        def failing_pw_run(calc_dir):
            call_count[0] += 1
            if "03_bands" in calc_dir:
                raise RuntimeError("bands pw.x crashed")
            # Otherwise succeed (scf, nscf)

        setup_fermi_dirs()

        fake_dos = minimal_dos_data

        with patch.object(wf, "_pw_run", side_effect=failing_pw_run), \
             patch.object(wf, "_dos_run"), \
             patch("shalom.workflows.standard.parse_dos_file", return_value=fake_dos), \
             patch("shalom.workflows.standard.find_xml_path", return_value=None):
            result = wf.run()

        assert result["failed_step"] == "bands"
        assert "scf" in result["completed_steps"]
        assert "nscf" in result["completed_steps"]
        # Bands should be marked as failed
        bands_status = [s for s in result["step_results"] if s.name == "bands"]
        assert len(bands_status) == 1
        assert bands_status[0].success is False

    def test_nscf_failure_skips_dos(self, tmp_path, make_workflow):
        """NSCF failure causes DOS step to be skipped."""
        wf = make_workflow(skip_relax=True)

        def failing_pw_run(calc_dir):
            if "04_nscf" in calc_dir:
                raise RuntimeError("NSCF failed")

        scf_dir = tmp_path / "02_scf"
        scf_dir.mkdir(parents=True, exist_ok=True)
        (scf_dir / "pw.out").write_text("the Fermi energy is   5.0000 ev\n")

        with patch.object(wf, "_pw_run", side_effect=failing_pw_run), \
             patch.object(wf, "_dos_run"), \
             patch("shalom.workflows.standard.find_xml_path", return_value=None):
            result = wf.run()

        dos_status = [s for s in result["step_results"] if s.name == "dos"]
        assert len(dos_status) == 1
        assert dos_status[0].success is False
        assert "skipped" in (dos_status[0].error_message or "").lower()

    def test_vc_relax_failure_continues(self, tmp_path, make_workflow, minimal_dos_data, setup_fermi_dirs):
        """vc-relax failure is non-fatal: workflow continues with input geometry."""
        wf = make_workflow(skip_relax=False)

        call_count = [0]

        def failing_pw_run(calc_dir):
            call_count[0] += 1
            if "01_vc_relax" in calc_dir:
                raise RuntimeError("vc-relax failed")
            # Otherwise succeed

        setup_fermi_dirs()

        fake_dos = minimal_dos_data

        with patch.object(wf, "_pw_run", side_effect=failing_pw_run), \
             patch.object(wf, "_dos_run"), \
             patch("shalom.workflows.standard.parse_dos_file", return_value=fake_dos), \
             patch("shalom.workflows.standard.find_xml_path", return_value=None):
            result = wf.run()

        # vc_relax failed but workflow continued
        vc_status = [s for s in result["step_results"] if s.name == "vc_relax"]
        assert len(vc_status) == 1
        assert vc_status[0].success is False
        # SCF should still have run
        assert "scf" in result["completed_steps"]

    def test_step_results_in_output(self, tmp_path, make_workflow, minimal_dos_data, setup_fermi_dirs):
        """run() output contains step_results, completed_steps, failed_step keys."""
        wf = make_workflow(skip_relax=True)

        setup_fermi_dirs()

        fake_dos = minimal_dos_data

        with patch.object(wf, "_pw_run"), \
             patch.object(wf, "_dos_run"), \
             patch("shalom.workflows.standard.parse_dos_file", return_value=fake_dos), \
             patch("shalom.workflows.standard.find_xml_path", return_value=None):
            result = wf.run()

        assert "step_results" in result
        assert "completed_steps" in result
        assert "failed_step" in result
        assert isinstance(result["step_results"], list)
        assert isinstance(result["completed_steps"], list)
        assert result["failed_step"] is None  # no failure


# ---------------------------------------------------------------------------
# Resume tests
# ---------------------------------------------------------------------------


class TestResume:
    """Test _save_checkpoint, _load_checkpoint, resume skips."""

    def test_save_checkpoint_creates_file(self, tmp_path, make_workflow):
        """_save_checkpoint writes workflow_state.json."""
        wf = make_workflow()
        os.makedirs(str(tmp_path), exist_ok=True)
        wf._save_checkpoint(["vc_relax", "scf"])
        ckpt_path = os.path.join(str(tmp_path), "workflow_state.json")
        assert os.path.isfile(ckpt_path)

    def test_save_checkpoint_content(self, tmp_path, make_workflow):
        """_save_checkpoint writes correct JSON content."""
        import json
        wf = make_workflow()
        os.makedirs(str(tmp_path), exist_ok=True)
        wf._save_checkpoint(["vc_relax", "scf"])

        ckpt_path = os.path.join(str(tmp_path), "workflow_state.json")
        with open(ckpt_path) as f:
            data = json.load(f)
        assert data["version"] == 1
        assert data["completed_steps"] == ["vc_relax", "scf"]
        assert "timestamp" in data

    def test_load_checkpoint_returns_data(self, tmp_path, make_workflow):
        """_load_checkpoint returns saved data."""
        wf = make_workflow()
        os.makedirs(str(tmp_path), exist_ok=True)
        wf._save_checkpoint(["vc_relax", "scf", "bands"])
        ckpt = wf._load_checkpoint()
        assert ckpt is not None
        assert ckpt["completed_steps"] == ["vc_relax", "scf", "bands"]

    def test_load_checkpoint_returns_none_when_missing(self, tmp_path, make_workflow):
        """_load_checkpoint returns None when no checkpoint file."""
        wf = make_workflow()
        ckpt = wf._load_checkpoint()
        assert ckpt is None

    def test_load_checkpoint_handles_corrupt_json(self, tmp_path, make_workflow):
        """_load_checkpoint returns None for corrupt JSON."""
        wf = make_workflow()
        os.makedirs(str(tmp_path), exist_ok=True)
        ckpt_path = os.path.join(str(tmp_path), "workflow_state.json")
        with open(ckpt_path, "w") as f:
            f.write("{invalid json")
        ckpt = wf._load_checkpoint()
        assert ckpt is None

    def test_resume_skips_completed_steps(self, tmp_path, make_workflow, minimal_dos_data, setup_fermi_dirs):
        """resume=True skips already-completed steps."""
        import json

        # Create a checkpoint saying vc_relax and scf are done
        os.makedirs(str(tmp_path), exist_ok=True)
        ckpt_path = os.path.join(str(tmp_path), "workflow_state.json")
        with open(ckpt_path, "w") as f:
            json.dump({"version": 1, "completed_steps": ["vc_relax", "scf"]}, f)

        wf = make_workflow(skip_relax=False, resume=True)

        pw_run_dirs = []

        def tracking_pw_run(calc_dir):
            pw_run_dirs.append(calc_dir)

        setup_fermi_dirs()

        fake_dos = minimal_dos_data

        with patch.object(wf, "_pw_run", side_effect=tracking_pw_run), \
             patch.object(wf, "_dos_run"), \
             patch("shalom.workflows.standard.parse_dos_file", return_value=fake_dos), \
             patch("shalom.workflows.standard.find_xml_path", return_value=None):
            result = wf.run()

        # vc_relax and scf should NOT have called _pw_run
        for d in pw_run_dirs:
            assert "01_vc_relax" not in d
            assert "02_scf" not in d
        # But bands and nscf should have run
        assert any("03_bands" in d for d in pw_run_dirs)
        assert any("04_nscf" in d for d in pw_run_dirs)

        # step_results should include resumed entries
        resumed_steps = [
            s for s in result["step_results"]
            if s.summary == "resumed"
        ]
        assert len(resumed_steps) == 2  # vc_relax and scf


# ---------------------------------------------------------------------------
# WSL pre-flight validation tests
# ---------------------------------------------------------------------------


class TestValidateEnvironmentWSL:
    """Test _validate_environment with wsl=True."""

    def test_wsl_mode_uses_detect_wsl_executable(self, tmp_path, caplog, make_workflow):
        """WSL mode calls detect_wsl_executable instead of shutil.which."""
        import logging
        wf = make_workflow(wsl=True)
        with patch("shalom.backends.runner.detect_wsl_executable", return_value=False) as mock_detect, \
             caplog.at_level(logging.WARNING, logger="shalom.workflows.standard"):
            wf._validate_environment()

        # detect_wsl_executable should have been called for pw.x and dos.x
        assert mock_detect.call_count == 2
        calls = [c.args[0] for c in mock_detect.call_args_list]
        assert "pw.x" in calls
        assert "dos.x" in calls
        # Warnings logged for both
        assert "pw.x" in caplog.text
        assert "dos.x" in caplog.text

    def test_wsl_mode_no_warning_when_found(self, tmp_path, caplog, make_workflow):
        """WSL mode with executables found does not log warnings."""
        import logging
        wf = make_workflow(wsl=True)
        with patch("shalom.backends.runner.detect_wsl_executable", return_value=True), \
             caplog.at_level(logging.WARNING, logger="shalom.workflows.standard"):
            wf._validate_environment()

        assert "not found" not in caplog.text

    def test_wsl_mode_skips_pseudo_dir_check(self, tmp_path, caplog, make_workflow):
        """WSL mode skips pseudo_dir validation (Windows path cannot be resolved)."""
        import logging
        wf = make_workflow(wsl=True, pseudo_dir="/nonexistent/pseudo_dir")
        with patch("shalom.backends.runner.detect_wsl_executable", return_value=True), \
             caplog.at_level(logging.WARNING, logger="shalom.workflows.standard"):
            wf._validate_environment()

        # pseudo_dir warning should NOT appear in WSL mode
        assert "pseudo_dir" not in caplog.text


# ---------------------------------------------------------------------------
# NSCF k-mesh override tests
# ---------------------------------------------------------------------------


class TestNscfKmeshOverride:
    """Test _run_nscf with custom nscf_kmesh parameter."""

    def test_explicit_nscf_kmesh_applied(self, tmp_path, make_workflow):
        """Explicit nscf_kmesh is used as the k-grid."""
        wf = make_workflow(nscf_kmesh=[8, 8, 8])

        captured_grid = {}

        config_obj = MagicMock()
        config_obj.control = {}
        config_obj.system = {}
        kpts_obj = MagicMock()
        kpts_obj.grid = [4, 4, 4]  # default
        config_obj.kpoints = kpts_obj
        config_obj.is_2d = False

        with patch("shalom.workflows.standard.get_qe_preset", return_value=config_obj), \
             patch("shalom.backends.qe.QEBackend.write_input"), \
             patch.object(wf, "_pw_run"):
            wf._run_nscf(str(tmp_path / "04_nscf"), wf.atoms, "/abs/scf/tmp")

        # Grid should have been set to [8, 8, 8]
        assert config_obj.kpoints.grid == [8, 8, 8]

    def test_default_nscf_kpr_when_no_kmesh(self, tmp_path, make_workflow):
        """When nscf_kmesh is None, DEFAULT_NSCF_KPR is used to compute grid."""
        wf = make_workflow(nscf_kmesh=None)

        config_obj = MagicMock()
        config_obj.control = {}
        config_obj.system = {}
        kpts_obj = MagicMock()
        kpts_obj.grid = [4, 4, 4]
        config_obj.kpoints = kpts_obj
        config_obj.is_2d = False

        with patch("shalom.workflows.standard.get_qe_preset", return_value=config_obj), \
             patch("shalom.backends.qe.QEBackend.write_input"), \
             patch.object(wf, "_pw_run"), \
             patch(
                 "shalom.backends._physics.compute_kpoints_grid",
                 return_value=[6, 6, 6],
             ) as mock_compute:
            wf._run_nscf(str(tmp_path / "04_nscf"), wf.atoms, "/abs/scf/tmp")

        # compute_kpoints_grid was called with DEFAULT_NSCF_KPR
        from shalom.backends._physics import DEFAULT_NSCF_KPR
        mock_compute.assert_called_once()
        call_kwargs = mock_compute.call_args
        assert call_kwargs.kwargs.get("kpr") == DEFAULT_NSCF_KPR or \
            (len(call_kwargs.args) >= 2 and call_kwargs.args[1] == DEFAULT_NSCF_KPR)
        assert config_obj.kpoints.grid == [6, 6, 6]

    def test_nscf_kmesh_parameter_stored(self):
        """nscf_kmesh parameter is stored on the workflow instance."""
        si = bulk("Si", "diamond", a=5.43)
        wf = StandardWorkflow(
            atoms=si, output_dir="/tmp/test",
            nscf_kmesh=[10, 10, 10],
        )
        assert wf.nscf_kmesh == [10, 10, 10]

    def test_nscf_kmesh_none_by_default(self):
        """nscf_kmesh is None by default."""
        si = bulk("Si", "diamond", a=5.43)
        wf = StandardWorkflow(atoms=si, output_dir="/tmp/test")
        assert wf.nscf_kmesh is None

    def test_default_nscf_kpr_constant_exists(self):
        """DEFAULT_NSCF_KPR constant exists in _physics module."""
        from shalom.backends._physics import DEFAULT_NSCF_KPR
        assert isinstance(DEFAULT_NSCF_KPR, float)
        assert DEFAULT_NSCF_KPR == 20.0

    def test_nscf_2d_passes_is_2d_true(self, tmp_path, make_workflow):
        """When is_2d=True, compute_kpoints_grid receives is_2d=True."""
        wf = make_workflow(nscf_kmesh=None)

        config_obj = MagicMock()
        config_obj.control = {}
        config_obj.system = {}
        kpts_obj = MagicMock()
        kpts_obj.grid = [4, 4, 4]
        config_obj.kpoints = kpts_obj
        config_obj.is_2d = True  # 2D structure

        with patch("shalom.workflows.standard.get_qe_preset", return_value=config_obj), \
             patch("shalom.backends.qe.QEBackend.write_input"), \
             patch.object(wf, "_pw_run"), \
             patch(
                 "shalom.backends._physics.compute_kpoints_grid",
                 return_value=[6, 6, 1],
             ) as mock_compute:
            wf._run_nscf(str(tmp_path / "04_nscf"), wf.atoms, "/abs/scf/tmp")

        # is_2d should be passed through
        mock_compute.assert_called_once()
        call_kwargs = mock_compute.call_args
        assert call_kwargs.kwargs.get("is_2d") is True


# ---------------------------------------------------------------------------
# Resume edge cases
# ---------------------------------------------------------------------------


class TestResumeEdgeCases:
    """Additional resume edge case tests."""

    def test_partial_checkpoint_reruns_remaining(self, tmp_path, make_workflow, minimal_dos_data, setup_fermi_dirs):
        """Checkpoint with only vc_relax done → SCF re-runs."""
        import json

        os.makedirs(str(tmp_path), exist_ok=True)
        ckpt_path = os.path.join(str(tmp_path), "workflow_state.json")
        with open(ckpt_path, "w") as f:
            json.dump({"version": 1, "completed_steps": ["vc_relax"]}, f)

        wf = make_workflow(skip_relax=False, resume=True)

        pw_run_dirs = []

        def tracking_pw_run(calc_dir):
            pw_run_dirs.append(calc_dir)

        setup_fermi_dirs()

        fake_dos = minimal_dos_data

        with patch.object(wf, "_pw_run", side_effect=tracking_pw_run), \
             patch.object(wf, "_dos_run"), \
             patch("shalom.workflows.standard.parse_dos_file", return_value=fake_dos), \
             patch("shalom.workflows.standard.find_xml_path", return_value=None):
            result = wf.run()

        # vc_relax should be skipped, but SCF should run
        assert not any("01_vc_relax" in d for d in pw_run_dirs)
        assert any("02_scf" in d for d in pw_run_dirs)
        # Only vc_relax should show as "resumed"
        resumed_steps = [
            s for s in result["step_results"]
            if s.summary == "resumed"
        ]
        assert len(resumed_steps) == 1
        assert resumed_steps[0].name == "vc_relax"

    def test_empty_completed_steps_checkpoint(self, tmp_path, make_workflow, minimal_dos_data, setup_fermi_dirs):
        """Checkpoint with empty completed_steps acts like no checkpoint."""
        import json

        os.makedirs(str(tmp_path), exist_ok=True)
        ckpt_path = os.path.join(str(tmp_path), "workflow_state.json")
        with open(ckpt_path, "w") as f:
            json.dump({"version": 1, "completed_steps": []}, f)

        wf = make_workflow(skip_relax=True, resume=True)

        pw_run_dirs = []

        def tracking_pw_run(calc_dir):
            pw_run_dirs.append(calc_dir)

        setup_fermi_dirs()

        fake_dos = minimal_dos_data

        with patch.object(wf, "_pw_run", side_effect=tracking_pw_run), \
             patch.object(wf, "_dos_run"), \
             patch("shalom.workflows.standard.parse_dos_file", return_value=fake_dos), \
             patch("shalom.workflows.standard.find_xml_path", return_value=None):
            result = wf.run()

        # SCF should have run (not skipped)
        assert any("02_scf" in d for d in pw_run_dirs)

    def test_checkpoint_version_mismatch_ignored(self, tmp_path, make_workflow):
        """Checkpoint with unknown version is still loaded (no validation)."""
        import json
        wf = make_workflow()
        os.makedirs(str(tmp_path), exist_ok=True)
        ckpt_path = os.path.join(str(tmp_path), "workflow_state.json")
        with open(ckpt_path, "w") as f:
            json.dump({"version": 99, "completed_steps": ["vc_relax"]}, f)
        ckpt = wf._load_checkpoint()
        # Currently no version validation — data is returned as-is
        assert ckpt is not None
        assert ckpt["completed_steps"] == ["vc_relax"]


# ---------------------------------------------------------------------------
# C1: dos.x failure does not crash workflow
# ---------------------------------------------------------------------------


class TestDosFailureHandling:
    """Verify dos.x errors are caught and recorded in step_results."""

    def test_dos_failure_does_not_crash_workflow(self, tmp_path, make_workflow, setup_fermi_dirs):
        """OSError in _run_dos is caught; step_results records failure."""
        wf = make_workflow(skip_relax=True)

        setup_fermi_dirs(include_dos=False)

        with patch.object(wf, "_pw_run"), \
             patch.object(wf, "_run_dos", side_effect=OSError("disk full")), \
             patch("shalom.workflows.standard.find_xml_path", return_value=None):
            result = wf.run()

        # dos step should be recorded as failed
        dos_step = [s for s in result["step_results"] if s.name == "dos"]
        assert len(dos_step) == 1
        assert dos_step[0].success is False
        assert "disk full" in dos_step[0].error_message
        # Workflow should still complete (not crash)
        assert result["step_results"] is not None

    def test_dos_failure_does_not_block_plotting(self, tmp_path, make_workflow, setup_fermi_dirs):
        """Even if dos.x fails, band plot is still attempted."""
        wf = make_workflow(skip_relax=True)

        setup_fermi_dirs(include_dos=False)

        plot_bands_called = []

        def mock_plot_bands(*args, **kwargs):
            plot_bands_called.append(True)
            return None

        with patch.object(wf, "_pw_run"), \
             patch.object(wf, "_run_dos", side_effect=RuntimeError("dos crash")), \
             patch.object(wf, "_plot_bands", side_effect=mock_plot_bands), \
             patch("shalom.workflows.standard.find_xml_path", return_value=None):
            result = wf.run()

        # _plot_bands should still have been called (bands succeeded)
        assert len(plot_bands_called) == 1


# ---------------------------------------------------------------------------
# H1: _save_checkpoint OSError handling
# ---------------------------------------------------------------------------


class TestCheckpointOSError:
    """_save_checkpoint should not crash on OSError."""

    def test_save_checkpoint_disk_error_does_not_crash(self, tmp_path, make_workflow):
        """OSError during checkpoint write logs a warning, doesn't raise."""
        wf = make_workflow()

        with patch("builtins.open", side_effect=OSError("disk full")):
            # Should not raise
            wf._save_checkpoint(["scf"])


# ---------------------------------------------------------------------------
# H2: RY_TO_EV constant usage
# ---------------------------------------------------------------------------


class TestRyToEvConstant:
    """_extract_pw_summary uses RY_TO_EV constant, not magic number."""

    def test_ry_to_ev_import(self):
        """RY_TO_EV is imported in standard module."""
        from shalom.workflows import standard
        assert hasattr(standard, "RY_TO_EV")
        assert standard.RY_TO_EV == 13.6057


# ---------------------------------------------------------------------------
# H3: nscf_kmesh validation
# ---------------------------------------------------------------------------


class TestNscfKmeshValidation:
    """nscf_kmesh must be [Nx, Ny, Nz] with positive ints."""

    @pytest.mark.parametrize("kmesh", [
        [6, 6],        # too few
        [6, 6, 6, 6],  # too many
        [6, 0, 6],     # zero
        [6, -1, 6],    # negative
    ])
    def test_nscf_kmesh_invalid_raises(self, kmesh, sample_si_diamond):
        with pytest.raises(ValueError, match="nscf_kmesh must be"):
            StandardWorkflow(
                atoms=sample_si_diamond, output_dir="/tmp/test", nscf_kmesh=kmesh,
            )

    def test_nscf_kmesh_valid_accepted(self, sample_si_diamond):
        wf = StandardWorkflow(
            atoms=sample_si_diamond, output_dir="/tmp/test",
            nscf_kmesh=[8, 8, 8],
        )
        assert wf.nscf_kmesh == [8, 8, 8]


# ---------------------------------------------------------------------------
# H5: Resume missing SCF output reruns
# ---------------------------------------------------------------------------


class TestResumeMissingSCFOutput:
    """Resume detects missing pw.out and re-runs SCF."""

    def test_resume_missing_scf_output_reruns(self, tmp_path, make_workflow, minimal_dos_data):
        """Checkpoint says SCF done but pw.out missing → SCF is re-run."""
        import json

        os.makedirs(str(tmp_path), exist_ok=True)
        ckpt_path = os.path.join(str(tmp_path), "workflow_state.json")
        with open(ckpt_path, "w") as f:
            json.dump({
                "version": 1,
                "completed_steps": ["vc_relax", "scf"],
            }, f)

        wf = make_workflow(skip_relax=False, resume=True)

        pw_run_dirs = []

        def tracking_pw_run(calc_dir):
            pw_run_dirs.append(calc_dir)

        # Create 02_scf dir but do NOT create pw.out
        scf_dir = tmp_path / "02_scf"
        scf_dir.mkdir(parents=True, exist_ok=True)
        # Create nscf dir with output for downstream steps
        nscf_dir = tmp_path / "04_nscf"
        nscf_dir.mkdir(parents=True, exist_ok=True)
        (nscf_dir / "pw.out").write_text("the Fermi energy is   5.0 ev\n")

        fake_dos = minimal_dos_data

        with patch.object(wf, "_pw_run", side_effect=tracking_pw_run), \
             patch.object(wf, "_dos_run"), \
             patch("shalom.workflows.standard.parse_dos_file", return_value=fake_dos), \
             patch("shalom.workflows.standard.find_xml_path", return_value=None):
            # Write pw.out after _pw_run is called (simulate SCF producing output)
            original_side = tracking_pw_run

            def pw_run_then_write(calc_dir):
                original_side(calc_dir)
                if "02_scf" in calc_dir:
                    pw_out_path = os.path.join(calc_dir, "pw.out")
                    with open(pw_out_path, "w") as f:
                        f.write("the Fermi energy is   5.0 ev\n")

            with patch.object(wf, "_pw_run", side_effect=pw_run_then_write):
                result = wf.run()

        # SCF should have been re-run (not just resumed)
        assert any("02_scf" in d for d in pw_run_dirs)


# ---------------------------------------------------------------------------
# H7: AccuracyLevel cached
# ---------------------------------------------------------------------------


class TestAccuracyLevelCached:
    """_accuracy_level attribute is set in __init__."""

    def test_accuracy_level_standard(self):
        from shalom.backends._physics import AccuracyLevel
        si = bulk("Si", "diamond", a=5.43)
        wf = StandardWorkflow(atoms=si, output_dir="/tmp/test", accuracy="standard")
        assert wf._accuracy_level == AccuracyLevel.STANDARD

    def test_accuracy_level_precise(self):
        from shalom.backends._physics import AccuracyLevel
        si = bulk("Si", "diamond", a=5.43)
        wf = StandardWorkflow(atoms=si, output_dir="/tmp/test", accuracy="precise")
        assert wf._accuracy_level == AccuracyLevel.PRECISE
