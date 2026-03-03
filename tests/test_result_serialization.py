"""Tests for result dataclass serialization (to_dict + save_result_json)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from shalom.analysis._base import (
    ElasticResult,
    ElectronicResult,
    MagneticResult,
    MDResult,
    PhononResult,
    SymmetryResult,
    XRDResult,
    save_result_json,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _assert_json_serializable(d: dict) -> str:
    """Assert dict is JSON-serializable and return JSON string."""
    s = json.dumps(d, default=str)
    assert isinstance(s, str)
    # Round-trip parse
    parsed = json.loads(s)
    assert isinstance(parsed, dict)
    return s


# ---------------------------------------------------------------------------
# ElasticResult
# ---------------------------------------------------------------------------

class TestElasticResultToDict:
    def test_basic(self):
        r = ElasticResult(
            elastic_tensor=np.eye(6) * 100,
            bulk_modulus_vrh=97.8,
            shear_modulus_vrh=50.9,
            youngs_modulus=130.5,
            poisson_ratio=0.28,
            is_stable=True,
            universal_anisotropy=0.0,
        )
        d = r.to_dict()
        _assert_json_serializable(d)
        assert d["bulk_modulus_vrh_GPa"] == 97.8
        assert d["is_stable"] is True
        assert "raw" not in d

    def test_numpy_converted(self):
        r = ElasticResult(elastic_tensor=np.ones((6, 6)))
        d = r.to_dict()
        assert isinstance(d["elastic_tensor_GPa"], list)
        assert isinstance(d["elastic_tensor_GPa"][0], list)

    def test_none_tensor(self):
        r = ElasticResult(elastic_tensor=None)
        d = r.to_dict()
        assert d["elastic_tensor_GPa"] is None
        _assert_json_serializable(d)

    def test_metadata_included(self):
        r = ElasticResult(
            elastic_tensor=None,
            metadata={"source": "test"},
        )
        d = r.to_dict()
        assert d["metadata"] == {"source": "test"}

    def test_compliance_tensor(self):
        r = ElasticResult(
            elastic_tensor=np.eye(6),
            compliance_tensor=np.eye(6) * 0.01,
        )
        d = r.to_dict()
        assert isinstance(d["compliance_tensor_inv_GPa"], list)


# ---------------------------------------------------------------------------
# PhononResult
# ---------------------------------------------------------------------------

class TestPhononResultToDict:
    def test_basic(self):
        r = PhononResult(
            min_frequency_THz=0.5,
            is_stable=True,
            n_atoms=2,
            n_branches=6,
            dos_frequencies=np.linspace(0, 15, 100),
            dos_density=np.ones(100),
            thermal_temperatures=np.array([0, 100, 200, 300]),
            thermal_cv=np.array([0.0, 10.0, 18.0, 23.0]),
        )
        d = r.to_dict()
        _assert_json_serializable(d)
        assert d["is_stable"] is True
        assert d["n_branches"] == 6
        assert len(d["dos_frequencies_THz"]) == 100

    def test_imaginary_modes(self):
        r = PhononResult(
            imaginary_modes=[(0, 1, -0.5), (10, 2, -0.3)],
            is_stable=False,
        )
        d = r.to_dict()
        assert d["n_imaginary_modes"] == 2
        assert d["imaginary_modes"][0]["frequency_THz"] == -0.5

    def test_empty(self):
        r = PhononResult()
        d = r.to_dict()
        _assert_json_serializable(d)

    def test_no_raw_or_force_constants(self):
        r = PhononResult(
            raw="should_be_excluded",
            force_constants=np.zeros((2, 2, 3, 3)),
        )
        d = r.to_dict()
        assert "raw" not in d
        assert "force_constants" not in d

    def test_band_labels_int_keys_converted_to_str(self):
        """JSON requires string keys; int keys must be converted."""
        r = PhononResult(band_labels={0: "Gamma", 10: "X", 20: "L"})
        d = r.to_dict()
        _assert_json_serializable(d)
        assert "0" in d["band_labels"]
        assert "10" in d["band_labels"]
        assert d["band_labels"]["0"] == "Gamma"


# ---------------------------------------------------------------------------
# ElectronicResult
# ---------------------------------------------------------------------------

class TestElectronicResultToDict:
    def test_semiconductor(self):
        r = ElectronicResult(
            bandgap_eV=1.12,
            is_direct=False,
            is_metal=False,
            vbm_energy=-0.5,
            cbm_energy=0.62,
            effective_mass_electron=0.26,
            effective_mass_hole=0.49,
        )
        d = r.to_dict()
        _assert_json_serializable(d)
        assert d["bandgap_eV"] == 1.12
        assert d["is_metal"] is False
        assert "raw" not in d

    def test_metal(self):
        r = ElectronicResult(
            is_metal=True,
            dos_at_fermi=5.3,
        )
        d = r.to_dict()
        assert d["is_metal"] is True
        assert d["dos_at_fermi_states_per_eV"] == 5.3


# ---------------------------------------------------------------------------
# XRDResult
# ---------------------------------------------------------------------------

class TestXRDResultToDict:
    def test_basic(self):
        r = XRDResult(
            two_theta=np.array([28.4, 47.3, 56.1]),
            intensities=np.array([100.0, 55.0, 30.0]),
            d_spacings=np.array([3.14, 1.92, 1.64]),
            hkl_indices=[(1, 1, 1), (2, 2, 0), (3, 1, 1)],
            wavelength="CuKa",
            wavelength_angstrom=1.5406,
            n_peaks=3,
        )
        d = r.to_dict()
        _assert_json_serializable(d)
        assert len(d["two_theta_deg"]) == 3
        assert d["hkl_indices"] == [[1, 1, 1], [2, 2, 0], [3, 1, 1]]
        assert "raw" not in d

    def test_empty(self):
        r = XRDResult()
        d = r.to_dict()
        _assert_json_serializable(d)


# ---------------------------------------------------------------------------
# SymmetryResult
# ---------------------------------------------------------------------------

class TestSymmetryResultToDict:
    def test_basic(self):
        r = SymmetryResult(
            space_group_number=225,
            space_group_symbol="Fm-3m",
            point_group="m-3m",
            crystal_system="cubic",
            lattice_type="F",
            n_operations=192,
            wyckoff_letters=["a"],
            equivalent_atoms=np.array([0]),
        )
        d = r.to_dict()
        _assert_json_serializable(d)
        assert d["space_group_number"] == 225
        assert d["crystal_system"] == "cubic"
        assert isinstance(d["equivalent_atoms"], list)
        assert "raw" not in d


# ---------------------------------------------------------------------------
# MagneticResult
# ---------------------------------------------------------------------------

class TestMagneticResultToDict:
    def test_basic(self):
        r = MagneticResult(
            total_magnetization=4.0,
            is_magnetic=True,
            is_spin_polarized=True,
            site_magnetizations=[3.5, 3.5, -0.1, -0.1],
            magnetic_elements=["Fe"],
            dominant_moment_element="Fe",
        )
        d = r.to_dict()
        _assert_json_serializable(d)
        assert d["total_magnetization_bohr_mag"] == 4.0
        assert d["is_magnetic"] is True
        assert "raw" not in d

    def test_non_magnetic(self):
        r = MagneticResult()
        d = r.to_dict()
        _assert_json_serializable(d)
        assert d["is_magnetic"] is False

    def test_lowdin_charges_nested_dict(self):
        """Nested dicts in lowdin_charges must serialize correctly."""
        r = MagneticResult(
            lowdin_charges={
                "total_charges": [5.8, 5.8, 6.2],
                "spd_charges": [
                    {"s": 0.5, "p": 0.3, "d": 5.0},
                    {"s": 0.5, "p": 0.3, "d": 5.0},
                    {"s": 1.8, "p": 4.4},
                ],
            },
        )
        d = r.to_dict()
        _assert_json_serializable(d)
        assert d["lowdin_charges"]["total_charges"] == [5.8, 5.8, 6.2]

    def test_numpy_site_magnetizations(self):
        """Numpy arrays in site_magnetizations must be converted to lists."""
        r = MagneticResult(
            site_magnetizations=np.array([3.5, 3.5, -0.1]),
            site_charges=np.array([5.8, 5.8, 6.2]),
        )
        d = r.to_dict()
        _assert_json_serializable(d)
        assert isinstance(d["site_magnetizations_bohr_mag"], list)
        assert isinstance(d["site_charges"], list)


# ---------------------------------------------------------------------------
# MDResult
# ---------------------------------------------------------------------------

class TestMDResultToDict:
    def test_full(self):
        r = MDResult(
            rdf_r=np.linspace(0, 10, 200),
            rdf_g=np.ones(200),
            rdf_pairs="all",
            msd_t=np.arange(0, 100, dtype=float),
            msd=np.arange(0, 100, dtype=float) * 0.5,
            diffusion_coefficient=1.5e-5,
            vacf_t=np.arange(0, 50, dtype=float),
            vacf=np.exp(-np.arange(50) / 10),
            avg_temperature=300.0,
            temperature_std=15.0,
            avg_energy=-4.5,
            energy_drift_per_atom=1e-6,
            is_equilibrated=True,
            equilibration_step=100,
        )
        d = r.to_dict()
        _assert_json_serializable(d)
        assert d["diffusion_coefficient_cm2_per_s"] == 1.5e-5
        assert d["thermodynamics"]["avg_temperature_K"] == 300.0
        assert d["equilibration"]["is_equilibrated"] is True
        assert len(d["rdf"]["r_angstrom"]) == 200

    def test_empty(self):
        r = MDResult()
        d = r.to_dict()
        _assert_json_serializable(d)


# ---------------------------------------------------------------------------
# save_result_json
# ---------------------------------------------------------------------------

class TestSaveResultJson:
    def test_saves_file(self, tmp_path):
        data = {"key": "value", "number": 42}
        path = str(tmp_path / "test.json")
        result = save_result_json(data, path)
        assert result is not None
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        assert loaded == data

    def test_creates_parent_dirs(self, tmp_path):
        path = str(tmp_path / "sub" / "dir" / "result.json")
        result = save_result_json({"x": 1}, path)
        assert result is not None
        assert os.path.isfile(path)

    def test_graceful_on_bad_path(self):
        # Non-writable path — should not raise
        result = save_result_json({"x": 1}, "")
        # Empty path causes OSError, returns None
        assert result is None

    def test_numpy_via_default_str(self, tmp_path):
        """Numpy values that slip through should be handled by default=str."""
        data = {"val": np.float64(3.14)}
        path = str(tmp_path / "np.json")
        result = save_result_json(data, path)
        assert result is not None

    def test_round_trip_elastic(self, tmp_path):
        r = ElasticResult(
            elastic_tensor=np.eye(6) * 165,
            bulk_modulus_vrh=97.8,
            is_stable=True,
        )
        path = str(tmp_path / "elastic.json")
        save_result_json(r.to_dict(), path)
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        assert loaded["bulk_modulus_vrh_GPa"] == 97.8
        assert loaded["is_stable"] is True

    def test_round_trip_md(self, tmp_path):
        r = MDResult(
            avg_temperature=300.0,
            diffusion_coefficient=1e-5,
            is_equilibrated=True,
        )
        path = str(tmp_path / "md.json")
        save_result_json(r.to_dict(), path)
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        assert loaded["diffusion_coefficient_cm2_per_s"] == 1e-5


# ---------------------------------------------------------------------------
# Workflow results_summary.json
# ---------------------------------------------------------------------------

class TestWorkflowResultsSummary:
    """Verify that StandardWorkflow._save_results_summary creates correct JSON."""

    def test_creates_file(self, tmp_path):
        from unittest.mock import MagicMock
        from shalom.workflows.standard import StandardWorkflow, StepStatus

        wf = StandardWorkflow.__new__(StandardWorkflow)
        wf.output_dir = str(tmp_path)

        step_results = [
            StepStatus("scf", 1, True, elapsed_seconds=10.5, summary="E=-100 eV"),
            StepStatus("bands", 2, True, elapsed_seconds=20.0),
            StepStatus("dos", 3, False, "timeout", 5.0),
        ]

        atoms_mock = MagicMock()
        atoms_mock.get_chemical_formula.return_value = "Si2"

        result = {
            "atoms": atoms_mock,
            "fermi_energy": 5.23,
            "bands_png": "/path/to/bands.png",
            "dos_png": None,
            "calc_dirs": {"scf": "/scf", "bands": "/bands"},
            "completed_steps": ["scf", "bands"],
            "failed_step": "dos",
        }

        wf._save_results_summary(result, step_results)

        path = tmp_path / "results_summary.json"
        assert path.exists()

        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["version"] == 1
        assert data["material_formula"] == "Si2"
        assert data["fermi_energy_eV"] == 5.23
        assert data["failed_step"] == "dos"
        assert len(data["steps"]) == 3
        assert data["steps"][0]["name"] == "scf"
        assert data["steps"][0]["success"] is True
        assert data["steps"][0]["elapsed_seconds"] == 10.5
        assert data["steps"][2]["error_message"] == "timeout"


# ---------------------------------------------------------------------------
# Direct run run_info.json
# ---------------------------------------------------------------------------

class TestDirectRunInfo:
    """Verify that _write_run_info creates correct JSON."""

    def test_creates_file(self, tmp_path):
        from shalom.direct_run import _write_run_info

        _write_run_info(
            output_dir=str(tmp_path),
            backend_name="qe",
            calc_type="scf",
            structure_info={"formula": "Si", "mp_id": "mp-149"},
            auto_detected={"ecutwfc": 60, "is_2d": False},
            files_generated=["pw.in", "POSCAR"],
        )

        path = tmp_path / "run_info.json"
        assert path.exists()

        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["version"] == 1
        assert data["backend"] == "qe"
        assert data["calc_type"] == "scf"
        assert data["structure_info"]["formula"] == "Si"
        assert data["auto_detected"]["ecutwfc"] == 60
        assert "pw.in" in data["files_generated"]
        assert "timestamp" in data
