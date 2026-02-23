"""Tests for shalom.workflows.standard.StandardWorkflow."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from ase.build import bulk

from shalom.workflows.standard import StandardWorkflow, EV_TO_RY


# ---------------------------------------------------------------------------
# Unit constants
# ---------------------------------------------------------------------------


def test_ev_to_ry_constant():
    """EV_TO_RY should be 1/13.6057 ≈ 0.073499."""
    assert EV_TO_RY == pytest.approx(1.0 / 13.6057, rel=1e-4)


# ---------------------------------------------------------------------------
# dos.x input generation (unit test — no pw.x)
# ---------------------------------------------------------------------------


class TestDosInGeneration:
    """Verify that dos.in is written with Ry units (not eV)."""

    def test_dos_in_emin_in_ry(self, tmp_path):
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

        # Emin = -20 eV × EV_TO_RY ≈ -1.4698 Ry
        expected_emin_ry = -20.0 * EV_TO_RY
        assert f"{expected_emin_ry:.6f}" in content

        # Emax = 10 eV × EV_TO_RY ≈ 0.7349 Ry
        expected_emax_ry = 10.0 * EV_TO_RY
        assert f"{expected_emax_ry:.6f}" in content

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
