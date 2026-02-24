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

    def test_dos_in_emin_in_ev(self, tmp_path):
        si = bulk("Si", "diamond", a=5.43)
        wf = StandardWorkflow(
            atoms=si,
            output_dir=str(tmp_path),
            dos_emin=-20.0,
            dos_emax=10.0,
            dos_deltaE=0.01,
        )
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

    def test_dos_in_uses_absolute_outdir(self, tmp_path):
        si = bulk("Si", "diamond", a=5.43)
        wf = StandardWorkflow(atoms=si, output_dir=str(tmp_path))
        nscf_dir = str(tmp_path / "04_nscf")
        os.makedirs(nscf_dir, exist_ok=True)
        wf._dos_run = MagicMock()

        abs_scf_tmp = "/absolute/path/to/scf/tmp"
        wf._run_dos(nscf_dir, abs_scf_tmp)

        content = open(os.path.join(nscf_dir, "dos.in")).read()
        assert abs_scf_tmp in content

    def test_dos_in_prefix_is_shalom(self, tmp_path):
        si = bulk("Si", "diamond", a=5.43)
        wf = StandardWorkflow(atoms=si, output_dir=str(tmp_path))
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

        original_write_input = None

        def fake_write_input(atoms, calc_dir, config=None):
            if config is not None:
                captured["outdir"] = config.control.get("outdir")
            # Create a dummy pw.in so subsequent reads don't fail
            os.makedirs(calc_dir, exist_ok=True)
            with open(os.path.join(calc_dir, "pw.in"), "w") as f:
                f.write("! placeholder\n")

        with patch("shalom.backends.qe.QEBackend.write_input", side_effect=fake_write_input):
            workflow._pw_run = MagicMock()
            method = getattr(workflow, step_method_name)
            if step_method_name == "_run_bands":
                si = bulk("Si", "diamond", a=5.43)
                calc_dir = os.path.join(workflow.output_dir, "03_bands")
                scf_tmp = "/abs/path/scf/tmp"
                method(calc_dir, si, scf_tmp)
            elif step_method_name == "_run_nscf":
                si = bulk("Si", "diamond", a=5.43)
                calc_dir = os.path.join(workflow.output_dir, "04_nscf")
                scf_tmp = "/abs/path/scf/tmp"
                method(calc_dir, si, scf_tmp)
        return captured

    def test_bands_uses_absolute_scf_tmp(self, tmp_path):
        si = bulk("Si", "diamond", a=5.43)
        wf = StandardWorkflow(atoms=si, output_dir=str(tmp_path))
        captured = self._capture_config_outdir("_run_bands", wf)
        assert captured.get("outdir") == "/abs/path/scf/tmp"

    def test_nscf_uses_absolute_scf_tmp(self, tmp_path):
        si = bulk("Si", "diamond", a=5.43)
        wf = StandardWorkflow(atoms=si, output_dir=str(tmp_path))
        captured = self._capture_config_outdir("_run_nscf", wf)
        assert captured.get("outdir") == "/abs/path/scf/tmp"


# ---------------------------------------------------------------------------
# Fermi energy priority (NSCF > SCF)
# ---------------------------------------------------------------------------


class TestFermiEnergyPriority:
    def test_nscf_preferred_over_scf(self, tmp_path):
        si = bulk("Si", "diamond", a=5.43)
        wf = StandardWorkflow(atoms=si, output_dir=str(tmp_path))
        scf_dir = tmp_path / "02_scf"
        nscf_dir = tmp_path / "04_nscf"
        scf_dir.mkdir()
        nscf_dir.mkdir()
        (scf_dir / "pw.out").write_text(
            "the Fermi energy is   5.0000 ev\n"
        )
        (nscf_dir / "pw.out").write_text(
            "the Fermi energy is   5.1234 ev\n"
        )
        ef = wf._get_best_fermi_energy(str(scf_dir), str(nscf_dir))
        assert ef == pytest.approx(5.1234)

    def test_falls_back_to_scf_if_nscf_missing(self, tmp_path):
        si = bulk("Si", "diamond", a=5.43)
        wf = StandardWorkflow(atoms=si, output_dir=str(tmp_path))
        scf_dir = tmp_path / "02_scf"
        nscf_dir = tmp_path / "04_nscf"
        scf_dir.mkdir()
        nscf_dir.mkdir()  # pw.out not written
        (scf_dir / "pw.out").write_text(
            "the Fermi energy is   4.9876 ev\n"
        )
        ef = wf._get_best_fermi_energy(str(scf_dir), str(nscf_dir))
        assert ef == pytest.approx(4.9876)

    def test_returns_none_if_both_missing(self, tmp_path):
        si = bulk("Si", "diamond", a=5.43)
        wf = StandardWorkflow(atoms=si, output_dir=str(tmp_path))
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
    def _make_wf(self, tmp_path):
        si = bulk("Si", "diamond", a=5.43)
        return StandardWorkflow(atoms=si, output_dir=str(tmp_path))

    def test_pw_run_success(self, tmp_path):
        """_pw_run succeeds when ExecutionRunner returns success."""
        wf = self._make_wf(tmp_path)
        from unittest.mock import MagicMock, patch

        mock_result = MagicMock()
        mock_result.success = True

        with patch("shalom.backends.runner.ExecutionRunner.run", return_value=mock_result):
            # Should not raise
            wf._pw_run(str(tmp_path))

    def test_pw_run_failure_raises(self, tmp_path):
        """_pw_run raises RuntimeError when runner fails."""
        wf = self._make_wf(tmp_path)
        from unittest.mock import MagicMock, patch

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error_message = "pw.x crashed"

        with patch("shalom.backends.runner.ExecutionRunner.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="pw.x failed"):
                wf._pw_run(str(tmp_path))

    def test_pw_run_failure_no_message(self, tmp_path):
        """_pw_run raises RuntimeError even when error_message is None."""
        wf = self._make_wf(tmp_path)
        from unittest.mock import MagicMock, patch

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error_message = None

        with patch("shalom.backends.runner.ExecutionRunner.run", return_value=mock_result):
            with pytest.raises(RuntimeError):
                wf._pw_run(str(tmp_path))


class TestDosRun:
    def _make_wf(self, tmp_path):
        si = bulk("Si", "diamond", a=5.43)
        return StandardWorkflow(atoms=si, output_dir=str(tmp_path))

    def test_dos_run_success(self, tmp_path):
        """_dos_run succeeds silently."""
        wf = self._make_wf(tmp_path)
        from unittest.mock import MagicMock, patch

        mock_result = MagicMock()
        mock_result.success = True

        with patch("shalom.backends.runner.ExecutionRunner.run", return_value=mock_result):
            wf._dos_run(str(tmp_path))  # no exception

    def test_dos_run_failure_only_warns(self, tmp_path):
        """_dos_run logs warning on failure instead of raising."""
        wf = self._make_wf(tmp_path)
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
    def test_returns_relaxed_structure(self, tmp_path):
        """When ase_read succeeds, returns the relaxed atoms."""
        si = bulk("Si", "diamond", a=5.43)
        wf = StandardWorkflow(atoms=si, output_dir=str(tmp_path))

        from unittest.mock import MagicMock, patch
        from ase.build import bulk as ase_bulk

        relaxed = ase_bulk("Si", "diamond", a=5.50)

        with patch("shalom.backends.qe.QEBackend.write_input"), \
             patch.object(wf, "_pw_run"), \
             patch("shalom.workflows.standard.ase_read", return_value=relaxed):
            result = wf._run_vc_relax(str(tmp_path / "01_vc_relax"), si)

        assert result is relaxed

    def test_falls_back_to_input_on_read_failure(self, tmp_path):
        """When ase_read fails, returns the original atoms."""
        si = bulk("Si", "diamond", a=5.43)
        wf = StandardWorkflow(atoms=si, output_dir=str(tmp_path))

        from unittest.mock import patch

        with patch("shalom.backends.qe.QEBackend.write_input"), \
             patch.object(wf, "_pw_run"), \
             patch("shalom.workflows.standard.ase_read", side_effect=Exception("parse error")):
            result = wf._run_vc_relax(str(tmp_path / "01_vc_relax"), si)

        assert result is si

    def test_precise_accuracy_applied(self, tmp_path):
        """Precise accuracy flag is forwarded to get_qe_preset."""
        si = bulk("Si", "diamond", a=5.43)
        wf = StandardWorkflow(atoms=si, output_dir=str(tmp_path), accuracy="precise")

        from unittest.mock import patch, MagicMock
        from ase.build import bulk as ase_bulk

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
             patch("shalom.workflows.standard.ase_read", return_value=si):
            wf._run_vc_relax(str(tmp_path / "01"), si)

        from shalom.backends._physics import AccuracyLevel
        assert captured_acc["acc"] == AccuracyLevel.PRECISE

    def test_pseudo_dir_override_applied(self, tmp_path):
        """pseudo_dir is applied to the config when set."""
        si = bulk("Si", "diamond", a=5.43)
        wf = StandardWorkflow(
            atoms=si, output_dir=str(tmp_path), pseudo_dir="/my/pseudos"
        )

        from unittest.mock import patch, MagicMock

        config_obj = MagicMock()
        config_obj.control = {}
        config_obj.system = {}
        config_obj.kpoints = None

        with patch("shalom.workflows.standard.get_qe_preset", return_value=config_obj), \
             patch("shalom.backends.qe.QEBackend.write_input"), \
             patch.object(wf, "_pw_run"), \
             patch("shalom.workflows.standard.ase_read", return_value=si):
            wf._run_vc_relax(str(tmp_path / "01"), si)

        assert config_obj.pseudo_dir == "/my/pseudos"


# ---------------------------------------------------------------------------
# _run_scf
# ---------------------------------------------------------------------------


class TestRunScf:
    def test_creates_directory_and_runs(self, tmp_path):
        """_run_scf creates calc_dir and calls write_input + _pw_run."""
        si = bulk("Si", "diamond", a=5.43)
        wf = StandardWorkflow(atoms=si, output_dir=str(tmp_path))

        scf_dir = str(tmp_path / "02_scf")
        from unittest.mock import patch, MagicMock, call

        write_calls = []

        def fake_write(atoms, calc_dir, config=None):
            write_calls.append(calc_dir)

        with patch("shalom.backends.qe.QEBackend.write_input", side_effect=fake_write), \
             patch.object(wf, "_pw_run") as mock_pw:
            wf._run_scf(scf_dir, si)

        assert len(write_calls) == 1
        assert write_calls[0] == scf_dir
        mock_pw.assert_called_once_with(scf_dir)

    def test_pseudo_dir_override(self, tmp_path):
        """pseudo_dir is forwarded to config."""
        si = bulk("Si", "diamond", a=5.43)
        wf = StandardWorkflow(
            atoms=si, output_dir=str(tmp_path), pseudo_dir="/pseudo"
        )

        from unittest.mock import patch, MagicMock

        config_obj = MagicMock()
        config_obj.control = {}
        config_obj.system = {}
        config_obj.kpoints = None

        with patch("shalom.workflows.standard.get_qe_preset", return_value=config_obj), \
             patch("shalom.backends.qe.QEBackend.write_input"), \
             patch.object(wf, "_pw_run"):
            wf._run_scf(str(tmp_path / "02"), si)

        assert config_obj.pseudo_dir == "/pseudo"


# ---------------------------------------------------------------------------
# _run_bands
# ---------------------------------------------------------------------------


class TestRunBandsExtended:
    def test_uses_cached_kpath(self, tmp_path):
        """_run_bands uses self._kpath_cfg when available."""
        si = bulk("Si", "diamond", a=5.43)
        wf = StandardWorkflow(atoms=si, output_dir=str(tmp_path))

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
            wf._run_bands(str(tmp_path / "03"), si, "/fake/scf/tmp")

        assert config_obj.kpoints is cached_kpath

    def test_generates_kpath_when_no_cache(self, tmp_path):
        """_run_bands generates kpath when _kpath_cfg is None."""
        si = bulk("Si", "diamond", a=5.43)
        wf = StandardWorkflow(atoms=si, output_dir=str(tmp_path))
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
            wf._run_bands(str(tmp_path / "03"), si, "/fake/scf/tmp")

        assert config_obj.kpoints is generated_kpath

    def test_nbnd_set_in_system(self, tmp_path):
        """_run_bands auto-computes nbnd and adds it to config.system."""
        si = bulk("Si", "diamond", a=5.43)
        wf = StandardWorkflow(atoms=si, output_dir=str(tmp_path))

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
            wf._run_bands(str(tmp_path / "03"), si, "/fake/scf/tmp")

        assert "nbnd" in config_obj.system
        assert config_obj.system["nbnd"] >= 20


# ---------------------------------------------------------------------------
# _run_nscf
# ---------------------------------------------------------------------------


class TestRunNscfExtended:
    def test_creates_dir_and_runs(self, tmp_path):
        """_run_nscf creates directory, writes input, runs pw.x."""
        si = bulk("Si", "diamond", a=5.43)
        wf = StandardWorkflow(atoms=si, output_dir=str(tmp_path))

        nscf_dir = str(tmp_path / "04_nscf")
        from unittest.mock import patch, MagicMock

        config_obj = MagicMock()
        config_obj.control = {}
        config_obj.system = {}

        with patch("shalom.workflows.standard.get_qe_preset", return_value=config_obj), \
             patch("shalom.backends.qe.QEBackend.write_input") as mock_write, \
             patch.object(wf, "_pw_run") as mock_pw:
            wf._run_nscf(nscf_dir, si, "/abs/scf/tmp")

        mock_write.assert_called_once()
        mock_pw.assert_called_once_with(nscf_dir)
        assert config_obj.control["outdir"] == "/abs/scf/tmp"


# ---------------------------------------------------------------------------
# _plot_bands
# ---------------------------------------------------------------------------


class TestPlotBands:
    def test_returns_none_when_no_xml(self, tmp_path):
        """Returns None when no bands XML found."""
        si = bulk("Si", "diamond", a=5.43)
        wf = StandardWorkflow(atoms=si, output_dir=str(tmp_path))

        from unittest.mock import patch

        with patch("shalom.workflows.standard.find_xml_path", return_value=None):
            result = wf._plot_bands(str(tmp_path / "03"), fermi=5.0)

        assert result is None

    def test_returns_none_when_no_matplotlib(self, tmp_path):
        """Returns None when matplotlib is not installed (import fails)."""
        si = bulk("Si", "diamond", a=5.43)
        wf = StandardWorkflow(atoms=si, output_dir=str(tmp_path))

        from unittest.mock import patch
        import sys

        # Remove the band_plot module from sys.modules so the try/except
        # inside _plot_bands triggers ImportError on the local import.
        with patch.dict(sys.modules, {"shalom.plotting.band_plot": None}):
            result = wf._plot_bands(str(tmp_path / "03"), fermi=5.0)

        assert result is None

    def test_plot_bands_full_pipeline(self, tmp_path):
        """_plot_bands calls plotter and returns output path."""
        si = bulk("Si", "diamond", a=5.43)
        wf = StandardWorkflow(atoms=si, output_dir=str(tmp_path))

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

    def test_plot_bands_gap_collapse(self, tmp_path):
        """Gap at discontinuity label 'X|U' is removed from kpath_distances."""
        si = bulk("Si", "diamond", a=5.43)
        wf = StandardWorkflow(atoms=si, output_dir=str(tmp_path))

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
    def test_returns_none_when_no_dos_file(self, tmp_path):
        """Returns None when pwscf.dos doesn't exist."""
        si = bulk("Si", "diamond", a=5.43)
        wf = StandardWorkflow(atoms=si, output_dir=str(tmp_path))

        result = wf._plot_dos(str(tmp_path / "04_nscf"), fermi=5.0)
        assert result is None

    def test_plot_dos_full_pipeline(self, tmp_path):
        """_plot_dos calls DOSPlotter and returns output path."""
        si = bulk("Si", "diamond", a=5.43)
        wf = StandardWorkflow(atoms=si, output_dir=str(tmp_path))

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

    def test_plot_dos_none_fermi_not_set(self, tmp_path):
        """When fermi=None, fermi_energy is not set on DOSData."""
        si = bulk("Si", "diamond", a=5.43)
        wf = StandardWorkflow(atoms=si, output_dir=str(tmp_path))

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
    def _mock_wf_run(self, wf, si, tmp_path):
        """Run wf.run() with all pw.x/plotting mocked out."""
        from unittest.mock import patch, MagicMock
        from shalom.backends.base import BandStructureData, DOSData
        import numpy as np

        fake_bs = BandStructureData(
            eigenvalues=np.zeros((3, 3)),
            kpoint_coords=np.zeros((3, 3)),
            kpath_distances=np.linspace(0, 1, 3),
            high_sym_labels={},
            source="qe",
        )
        fake_dos = DOSData(
            energies=np.array([0.0]),
            dos=np.array([1.0]),
            integrated_dos=np.array([0.0]),
            source="qe",
        )

        dos_dir = tmp_path / "04_nscf"
        dos_dir.mkdir(parents=True, exist_ok=True)
        (dos_dir / "pwscf.dos").write_text("# DOS\n0.0 1.0 0.5\n")
        (dos_dir / "pw.out").write_text("the Fermi energy is   5.1234 ev\n")

        scf_dir = tmp_path / "02_scf"
        scf_dir.mkdir(parents=True, exist_ok=True)
        (scf_dir / "pw.out").write_text("the Fermi energy is   5.0000 ev\n")

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

    def test_run_skip_relax_returns_dict(self, tmp_path):
        """run() with skip_relax=True returns expected keys."""
        si = bulk("Si", "diamond", a=5.43)
        wf = StandardWorkflow(
            atoms=si, output_dir=str(tmp_path), skip_relax=True
        )
        result = self._mock_wf_run(wf, si, tmp_path)

        assert "atoms" in result
        assert "fermi_energy" in result
        assert "bands_png" in result
        assert "dos_png" in result
        assert "calc_dirs" in result

    def test_run_fermi_energy_from_nscf(self, tmp_path):
        """run() extracts Fermi energy from NSCF pw.out (priority)."""
        si = bulk("Si", "diamond", a=5.43)
        wf = StandardWorkflow(
            atoms=si, output_dir=str(tmp_path), skip_relax=True
        )
        result = self._mock_wf_run(wf, si, tmp_path)

        assert result["fermi_energy"] == pytest.approx(5.1234, rel=1e-4)

    def test_run_calc_dirs_has_all_steps(self, tmp_path):
        """run() result contains all 4 calc_dir keys."""
        si = bulk("Si", "diamond", a=5.43)
        wf = StandardWorkflow(
            atoms=si, output_dir=str(tmp_path), skip_relax=True
        )
        result = self._mock_wf_run(wf, si, tmp_path)

        assert "vc_relax" in result["calc_dirs"]
        assert "scf" in result["calc_dirs"]
        assert "bands" in result["calc_dirs"]
        assert "nscf" in result["calc_dirs"]

    def test_run_with_relax_calls_vc_relax(self, tmp_path):
        """run() without skip_relax invokes _run_vc_relax."""
        si = bulk("Si", "diamond", a=5.43)
        wf = StandardWorkflow(
            atoms=si, output_dir=str(tmp_path), skip_relax=False
        )

        from unittest.mock import patch, MagicMock
        from shalom.backends.base import BandStructureData, DOSData
        import numpy as np

        fake_bs = BandStructureData(
            eigenvalues=np.zeros((3, 3)),
            kpoint_coords=np.zeros((3, 3)),
            kpath_distances=np.linspace(0, 1, 3),
            high_sym_labels={},
            source="qe",
        )
        fake_dos = DOSData(
            energies=np.array([0.0]),
            dos=np.array([1.0]),
            integrated_dos=np.array([0.0]),
            source="qe",
        )

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

    def test_run_kpath_cache_populated(self, tmp_path):
        """After run(), _kpath_cfg and _calc_atoms are populated."""
        si = bulk("Si", "diamond", a=5.43)
        wf = StandardWorkflow(
            atoms=si, output_dir=str(tmp_path), skip_relax=True
        )
        self._mock_wf_run(wf, si, tmp_path)

        assert wf._kpath_cfg is not None
        assert wf._calc_atoms is not None

    def test_run_png_paths_in_output_dir(self, tmp_path):
        """bands.png and dos.png are in the output_dir."""
        si = bulk("Si", "diamond", a=5.43)
        wf = StandardWorkflow(
            atoms=si, output_dir=str(tmp_path), skip_relax=True
        )
        result = self._mock_wf_run(wf, si, tmp_path)

        if result["bands_png"] is not None:
            assert result["bands_png"].startswith(str(tmp_path))
        if result["dos_png"] is not None:
            assert result["dos_png"].startswith(str(tmp_path))
