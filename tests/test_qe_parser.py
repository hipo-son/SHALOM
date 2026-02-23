"""Tests for shalom.backends.qe_parser."""

from __future__ import annotations

import os

import numpy as np
import pytest

from shalom.backends.qe_parser import (
    HA_TO_EV,
    QE_XML_NS,
    compute_nbnd,
    extract_fermi_energy,
    find_xml_path,
    parse_dos_file,
    parse_xml_bands,
)


# ---------------------------------------------------------------------------
# parse_xml_bands
# ---------------------------------------------------------------------------


class TestParseXmlBands:
    def test_eigenvalue_units_hartree_to_ev(self, mock_bands_xml_path):
        """Eigenvalues from XML (Hartree) must be converted to eV."""
        bs = parse_xml_bands(mock_bands_xml_path, fermi_energy=0.0)
        # First k-point, first band: -0.01716 Ha × 27.2114 = -0.4670 eV
        expected = -0.01716 * HA_TO_EV
        assert abs(bs.eigenvalues[0, 0] - expected) < 1e-3

    def test_nkpts_and_nbands(self, mock_bands_xml_path):
        """Should parse 2 k-points and 3 bands from the mock XML."""
        bs = parse_xml_bands(mock_bands_xml_path)
        assert bs.nkpts == 2
        assert bs.nbands == 3

    def test_kpath_distances_non_negative(self, mock_bands_xml_path):
        """kpath_distances must start at 0 and be monotonically non-decreasing."""
        bs = parse_xml_bands(mock_bands_xml_path)
        assert bs.kpath_distances[0] == pytest.approx(0.0)
        diffs = np.diff(bs.kpath_distances)
        assert np.all(diffs >= 0.0)

    def test_fermi_energy_stored(self, mock_bands_xml_path):
        """Fermi energy passed in must be stored on the result."""
        bs = parse_xml_bands(mock_bands_xml_path, fermi_energy=5.5)
        assert bs.fermi_energy == pytest.approx(5.5)

    def test_not_spin_polarized(self, mock_bands_xml_path):
        """Mock XML has lsda=false."""
        bs = parse_xml_bands(mock_bands_xml_path)
        assert not bs.is_spin_polarized

    def test_file_not_found(self, tmp_path):
        """Raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            parse_xml_bands(str(tmp_path / "nonexistent.xml"))

    def test_source_is_qe(self, mock_bands_xml_path):
        bs = parse_xml_bands(mock_bands_xml_path)
        assert bs.source == "qe"


# ---------------------------------------------------------------------------
# parse_dos_file
# ---------------------------------------------------------------------------


class TestParseDosFile:
    def test_non_spin_columns(self, mock_dos_path):
        """3-column file → non-spin DOS."""
        dos = parse_dos_file(mock_dos_path)
        assert not dos.is_spin_polarized
        assert dos.dos_up is None
        assert dos.dos_down is None

    def test_spin_columns(self, mock_dos_spin_path):
        """4-column file → spin-polarised DOS."""
        dos = parse_dos_file(mock_dos_spin_path)
        assert dos.is_spin_polarized
        assert dos.dos_up is not None
        assert dos.dos_down is not None

    def test_energies_in_ev(self, mock_dos_path):
        """First energy value should be -6.0 eV as written in the fixture."""
        dos = parse_dos_file(mock_dos_path)
        assert dos.energies[0] == pytest.approx(-6.0)

    def test_dos_total_spin(self, mock_dos_spin_path):
        """Total DOS should equal dos_up + dos_down for spin case."""
        dos = parse_dos_file(mock_dos_spin_path)
        np.testing.assert_allclose(dos.dos, dos.dos_up + dos.dos_down)

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_dos_file(str(tmp_path / "missing.dos"))

    def test_source_is_qe(self, mock_dos_path):
        dos = parse_dos_file(mock_dos_path)
        assert dos.source == "qe"

    def test_fermi_energy_defaults_zero(self, mock_dos_path):
        """Parser does not set fermi_energy; caller must set it."""
        dos = parse_dos_file(mock_dos_path)
        assert dos.fermi_energy == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# find_xml_path
# ---------------------------------------------------------------------------


class TestFindXmlPath:
    def test_finds_direct_save(self, tmp_path):
        """Search order 1: {calc_dir}/{prefix}.save/data-file-schema.xml"""
        save_dir = tmp_path / "shalom.save"
        save_dir.mkdir()
        xml = save_dir / "data-file-schema.xml"
        xml.write_text("<root/>")
        found = find_xml_path(str(tmp_path))
        assert found is not None
        assert os.path.isabs(found)
        assert "data-file-schema.xml" in found

    def test_finds_tmp_save(self, tmp_path):
        """Search order 2: {calc_dir}/tmp/{prefix}.save/data-file-schema.xml"""
        save_dir = tmp_path / "tmp" / "shalom.save"
        save_dir.mkdir(parents=True)
        xml = save_dir / "data-file-schema.xml"
        xml.write_text("<root/>")
        found = find_xml_path(str(tmp_path))
        assert found is not None

    def test_returns_none_if_not_found(self, tmp_path):
        result = find_xml_path(str(tmp_path))
        assert result is None


# ---------------------------------------------------------------------------
# extract_fermi_energy
# ---------------------------------------------------------------------------


class TestExtractFermiEnergy:
    def test_extracts_correctly(self, tmp_path):
        pw_out = tmp_path / "pw.out"
        pw_out.write_text(
            "Some output\n"
            "     the Fermi energy is     5.6789 ev\n"
            "More output\n"
        )
        ef = extract_fermi_energy(str(pw_out))
        assert ef == pytest.approx(5.6789)

    def test_case_insensitive(self, tmp_path):
        pw_out = tmp_path / "pw.out"
        pw_out.write_text("the Fermi energy is   3.14 eV\n")
        ef = extract_fermi_energy(str(pw_out))
        assert ef == pytest.approx(3.14)

    def test_returns_none_if_not_found(self, tmp_path):
        pw_out = tmp_path / "pw.out"
        pw_out.write_text("No Fermi energy line here.\n")
        assert extract_fermi_energy(str(pw_out)) is None

    def test_returns_none_for_missing_file(self, tmp_path):
        assert extract_fermi_energy(str(tmp_path / "nonexistent.out")) is None


# ---------------------------------------------------------------------------
# compute_nbnd
# ---------------------------------------------------------------------------


class TestComputeNbnd:
    def test_minimum_20(self):
        """Even for trivial structures, result must be ≥ 20."""
        from ase.build import bulk
        atoms = bulk("Cu", "fcc", a=3.6)
        nbnd = compute_nbnd(atoms, multiplier=1.3)
        assert nbnd >= 20

    def test_increases_with_multiplier(self):
        from ase.build import bulk
        atoms = bulk("Si", "diamond", a=5.43)
        nbnd_low = compute_nbnd(atoms, multiplier=1.0)
        nbnd_high = compute_nbnd(atoms, multiplier=1.5)
        assert nbnd_high >= nbnd_low

    def test_returns_int(self):
        from ase.build import bulk
        atoms = bulk("Fe", "bcc", a=2.87)
        assert isinstance(compute_nbnd(atoms), int)


# ---------------------------------------------------------------------------
# Constants sanity check
# ---------------------------------------------------------------------------


def test_ha_to_ev_constant():
    """HA_TO_EV should be approximately 27.2114."""
    assert HA_TO_EV == pytest.approx(27.2114, rel=1e-4)


def test_qe_xml_ns():
    """QE XML namespace dict must contain the expected URL."""
    assert "qes" in QE_XML_NS
    assert "quantum-espresso.org" in QE_XML_NS["qes"]
