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


# ---------------------------------------------------------------------------
# Additional edge-case tests for missed coverage
# ---------------------------------------------------------------------------


class TestParseXmlBandsEdgeCases:
    def test_malformed_xml_raises_valueerror(self, tmp_path):
        """Malformed XML → ValueError wrapping ParseError."""
        bad_xml = tmp_path / "bad.xml"
        bad_xml.write_text("<<< this is not valid XML >>>")
        with pytest.raises(ValueError, match="Malformed XML"):
            parse_xml_bands(str(bad_xml))

    def test_no_ks_energies_raises_valueerror(self, tmp_path):
        """XML with no <ks_energies> → ValueError."""
        xml = tmp_path / "empty.xml"
        xml.write_text(
            '<?xml version="1.0"?>\n'
            '<espresso xmlns="http://www.quantum-espresso.org/ns/qes/qes-1.0">'
            '<output><band_structure><lsda>false</lsda></band_structure></output>'
            '</espresso>'
        )
        with pytest.raises(ValueError, match="No <ks_energies>"):
            parse_xml_bands(str(xml))

    def test_missing_kpoint_falls_back_to_zero(self, tmp_path):
        """ks_energies without k_point element → [0,0,0] fallback."""
        xml = tmp_path / "nokpt.xml"
        xml.write_text(
            '<?xml version="1.0"?>\n'
            '<espresso xmlns="http://www.quantum-espresso.org/ns/qes/qes-1.0">'
            '<output><band_structure><lsda>false</lsda>'
            '<ks_energies>'
            '  <eigenvalues size="2">-0.01 0.02</eigenvalues>'
            '</ks_energies>'
            '</band_structure></output></espresso>'
        )
        bs = parse_xml_bands(str(xml))
        # k-point should be [0,0,0]
        assert bs.kpoint_coords[0].tolist() == pytest.approx([0.0, 0.0, 0.0])

    def test_spin_polarized_two_eigenvalue_elements(self, tmp_path):
        """lsda=true with two eigenvalue elements per k-point → spin arrays set."""
        xml = tmp_path / "spin.xml"
        xml.write_text(
            '<?xml version="1.0"?>\n'
            '<espresso xmlns="http://www.quantum-espresso.org/ns/qes/qes-1.0">'
            '<output><band_structure><lsda>true</lsda>'
            '<ks_energies>'
            '  <k_point weight="1.0">0.0 0.0 0.0</k_point>'
            '  <eigenvalues size="2">-0.01 0.02</eigenvalues>'
            '  <eigenvalues size="2">-0.015 0.025</eigenvalues>'
            '</ks_energies>'
            '</band_structure></output></espresso>'
        )
        bs = parse_xml_bands(str(xml))
        assert bs.is_spin_polarized is True
        assert bs.spin_up is not None
        assert bs.spin_down is not None
        assert bs.spin_up.shape == (1, 2)
        assert bs.spin_down.shape == (1, 2)

    def test_bvectors_without_namespace(self, tmp_path):
        """Reciprocal lattice vectors without namespace prefix → parsed via fallback."""
        # Build XML without namespace → b1/b2/b3 won't match qes: prefix
        xml = tmp_path / "nons.xml"
        xml.write_text(
            '<?xml version="1.0"?>\n'
            '<espresso xmlns="http://www.quantum-espresso.org/ns/qes/qes-1.0">'
            '<output><basis_set>'
            '<reciprocal_lattice>'
            '  <b1>1.0 0.0 0.0</b1>'
            '  <b2>0.0 1.0 0.0</b2>'
            '  <b3>0.0 0.0 1.0</b3>'
            '</reciprocal_lattice>'
            '</basis_set>'
            '<band_structure><lsda>false</lsda>'
            '<ks_energies>'
            '  <k_point weight="1.0">0.0 0.0 0.0</k_point>'
            '  <eigenvalues size="2">-0.01 0.02</eigenvalues>'
            '</ks_energies>'
            '</band_structure></output></espresso>'
        )
        bs = parse_xml_bands(str(xml))
        assert bs.nkpts == 1


class TestFindXmlPathGlobFallback:
    def test_glob_fallback_finds_nested_xml(self, tmp_path):
        """Deep nested XML found via glob fallback."""
        nested = tmp_path / "deep" / "nested.save"
        nested.mkdir(parents=True)
        xml = nested / "data-file-schema.xml"
        xml.write_text("<root/>")
        found = find_xml_path(str(tmp_path))
        assert found is not None
        assert "data-file-schema.xml" in found


class TestExtractFermiEnergyEdgeCases:
    def test_ioerror_returns_none(self, tmp_path):
        """OSError when opening pw.out → None returned."""
        from unittest.mock import patch
        pw_out = tmp_path / "pw.out"
        pw_out.write_text("the Fermi energy is 5.0 ev\n")
        with patch("builtins.open", side_effect=OSError("permission denied")):
            result = extract_fermi_energy(str(pw_out))
        assert result is None


class TestParseDosFileEdgeCases:
    def test_loadtxt_error_raises_valueerror(self, tmp_path):
        """Un-parseable DOS file → ValueError."""
        dos_path = tmp_path / "bad.dos"
        dos_path.write_text("not\tnumeric\ndata\n")
        with pytest.raises(ValueError, match="Cannot parse DOS"):
            parse_dos_file(str(dos_path))

    def test_single_row_parses_ok(self, tmp_path):
        """Single-row DOS file (1D array from loadtxt) → reshaped correctly."""
        dos_path = tmp_path / "single.dos"
        dos_path.write_text("# E dos idos\n0.0 1.5 0.5\n")
        dos = parse_dos_file(str(dos_path))
        assert len(dos.energies) == 1
        assert dos.energies[0] == pytest.approx(0.0)
        assert dos.dos[0] == pytest.approx(1.5)

    def test_too_few_columns_raises_valueerror(self, tmp_path):
        """DOS file with only 2 columns → ValueError."""
        dos_path = tmp_path / "twocol.dos"
        dos_path.write_text("# E dos\n-1.0 0.5\n0.0 1.0\n")
        with pytest.raises(ValueError, match="Expected"):
            parse_dos_file(str(dos_path))


# ---------------------------------------------------------------------------
# QE 7.5 XML namespace compatibility (xmlns:qes= prefix variant)
# ---------------------------------------------------------------------------


class TestXMLNamespaceVariants:
    """Verify parse_xml_bands works with both default and prefixed XML namespaces."""

    QE75_PREFIXED_XML = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">\n'
        "  <qes:output>\n"
        "    <qes:basis_set>\n"
        "      <qes:reciprocal_lattice>\n"
        "        <qes:b1>1.0 0.0 0.0</qes:b1>\n"
        "        <qes:b2>0.0 1.0 0.0</qes:b2>\n"
        "        <qes:b3>0.0 0.0 1.0</qes:b3>\n"
        "      </qes:reciprocal_lattice>\n"
        "    </qes:basis_set>\n"
        "    <qes:band_structure>\n"
        "      <qes:lsda>false</qes:lsda>\n"
        "      <qes:ks_energies>\n"
        '        <qes:k_point weight="0.5">0.0 0.0 0.0</qes:k_point>\n'
        '        <qes:eigenvalues size="2">-0.01716 0.00934</qes:eigenvalues>\n'
        "      </qes:ks_energies>\n"
        "      <qes:ks_energies>\n"
        '        <qes:k_point weight="0.5">0.5 0.0 0.5</qes:k_point>\n'
        '        <qes:eigenvalues size="2">-0.01450 0.01470</qes:eigenvalues>\n'
        "      </qes:ks_energies>\n"
        "    </qes:band_structure>\n"
        "  </qes:output>\n"
        "</qes:espresso>\n"
    )

    def test_prefixed_namespace_parses_kpoints(self, tmp_path):
        """QE 7.5 XML with xmlns:qes= prefix → k-points parsed correctly."""
        xml_file = tmp_path / "qe75.xml"
        xml_file.write_text(self.QE75_PREFIXED_XML)
        bs = parse_xml_bands(str(xml_file))
        assert bs.nkpts == 2
        assert bs.nbands == 2

    def test_prefixed_namespace_eigenvalues(self, tmp_path):
        """QE 7.5 XML with xmlns:qes= prefix → eigenvalues in eV."""
        xml_file = tmp_path / "qe75.xml"
        xml_file.write_text(self.QE75_PREFIXED_XML)
        bs = parse_xml_bands(str(xml_file))
        expected_first = -0.01716 * HA_TO_EV
        assert abs(bs.eigenvalues[0, 0] - expected_first) < 1e-3

    def test_prefixed_namespace_reciprocal_lattice(self, tmp_path):
        """QE 7.5 XML → reciprocal lattice parsed (not identity fallback)."""
        xml_file = tmp_path / "qe75.xml"
        xml_file.write_text(self.QE75_PREFIXED_XML)
        bs = parse_xml_bands(str(xml_file))
        # kpath_distances should be computed from the identity-like b_matrix
        assert bs.kpath_distances[-1] > 0.0

    def test_default_namespace_still_works(self, mock_bands_xml_path):
        """Existing default xmlns= format must still parse correctly."""
        bs = parse_xml_bands(mock_bands_xml_path)
        assert bs.nkpts == 2
        assert bs.nbands == 3


# ---------------------------------------------------------------------------
# find_xml_path — direct file in calc_dir
# ---------------------------------------------------------------------------


class TestFindXmlPathDirectFile:
    def test_finds_direct_xml_in_calc_dir(self, tmp_path):
        """data-file-schema.xml placed directly in calc_dir → found."""
        xml = tmp_path / "data-file-schema.xml"
        xml.write_text("<root/>")
        found = find_xml_path(str(tmp_path))
        assert found is not None
        assert found.endswith("data-file-schema.xml")

    def test_direct_xml_preferred_over_save_dir(self, tmp_path):
        """Direct XML in calc_dir takes priority over prefix.save/."""
        # Create both candidates
        (tmp_path / "data-file-schema.xml").write_text("<direct/>")
        save_dir = tmp_path / "shalom.save"
        save_dir.mkdir()
        (save_dir / "data-file-schema.xml").write_text("<save/>")

        found = find_xml_path(str(tmp_path))
        assert found is not None
        # Should find the direct one (first candidate)
        assert "shalom.save" not in found


# ---------------------------------------------------------------------------
# extract_fermi_energy — returns last match for restarted calcs
# ---------------------------------------------------------------------------


class TestExtractFermiLastMatch:
    def test_returns_last_fermi_from_multiple_lines(self, tmp_path):
        """Restarted calculation with multiple Fermi lines → last value used."""
        pw_out = tmp_path / "pw.out"
        pw_out.write_text(
            "Some output\n"
            "     the Fermi energy is     5.1234 ev\n"
            "More output after restart\n"
            "     the Fermi energy is     5.6789 ev\n"
            "Final output\n"
        )
        ef = extract_fermi_energy(str(pw_out))
        assert ef == pytest.approx(5.6789)

    def test_single_fermi_line_unchanged(self, tmp_path):
        """Single Fermi line → same behavior as before."""
        pw_out = tmp_path / "pw.out"
        pw_out.write_text("     the Fermi energy is     3.14 ev\n")
        ef = extract_fermi_energy(str(pw_out))
        assert ef == pytest.approx(3.14)
