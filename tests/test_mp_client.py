"""Tests for shalom.mp_client module (all MP API calls mocked)."""

import pytest
from unittest.mock import patch, MagicMock

from shalom.mp_client import (
    is_mp_id,
    fetch_by_mp_id,
    search_by_formula,
    fetch_structure,
    MPStructureResult,
)


# ---------------------------------------------------------------------------
# is_mp_id
# ---------------------------------------------------------------------------


class TestIsMpId:
    def test_valid_mp_id(self):
        assert is_mp_id("mp-19717") is True

    def test_valid_mvc_id(self):
        assert is_mp_id("mvc-12345") is True

    def test_formula_not_mp_id(self):
        assert is_mp_id("Fe2O3") is False

    def test_empty_string(self):
        assert is_mp_id("") is False

    def test_partial_match(self):
        assert is_mp_id("mp-") is False

    def test_whitespace(self):
        assert is_mp_id(" mp-19717 ") is True


# ---------------------------------------------------------------------------
# fetch_by_mp_id (mocked)
# ---------------------------------------------------------------------------


class TestFetchByMpId:
    def test_mp_api_not_installed(self):
        with patch("shalom.mp_client._MP_AVAILABLE", False):
            with pytest.raises(ImportError, match="mp-api"):
                fetch_by_mp_id("mp-19717")

    def test_no_api_key(self):
        with patch("shalom.mp_client._MP_AVAILABLE", True), \
             patch.dict("os.environ", {}, clear=True):
            with pytest.raises(EnvironmentError, match="MP_API_KEY"):
                fetch_by_mp_id("mp-19717")

    def test_successful_fetch(self):
        from ase.build import bulk
        mock_atoms = bulk("Cu", "fcc", a=3.6)

        mock_structure = MagicMock()
        mock_structure.composition.reduced_formula = "Cu"
        mock_structure.volume = 46.66
        mock_structure.__len__ = lambda s: 1

        mock_doc = MagicMock()
        mock_doc.material_id = "mp-19717"
        mock_doc.structure = mock_structure
        mock_doc.energy_above_hull = 0.0

        mock_mpr = MagicMock()
        mock_mpr.__enter__ = MagicMock(return_value=mock_mpr)
        mock_mpr.__exit__ = MagicMock(return_value=False)
        mock_mpr.materials.summary.get_data_by_id.return_value = mock_doc

        with patch("shalom.mp_client._MP_AVAILABLE", True), \
             patch.dict("os.environ", {"MP_API_KEY": "test-key"}), \
             patch("shalom.mp_client.MPRester", create=True, return_value=mock_mpr), \
             patch("shalom.mp_client.AseAtomsAdaptor", create=True) as mock_adaptor:
            mock_adaptor.get_atoms.return_value = mock_atoms
            result = fetch_by_mp_id("mp-19717")

        assert isinstance(result, MPStructureResult)
        assert result.mp_id == "mp-19717"
        assert result.formula == "Cu"
        assert result.atoms is mock_atoms

    def test_invalid_mp_id(self):
        mock_mpr = MagicMock()
        mock_mpr.__enter__ = MagicMock(return_value=mock_mpr)
        mock_mpr.__exit__ = MagicMock(return_value=False)
        mock_mpr.materials.summary.get_data_by_id.side_effect = Exception("Not found")

        with patch("shalom.mp_client._MP_AVAILABLE", True), \
             patch.dict("os.environ", {"MP_API_KEY": "test-key"}), \
             patch("shalom.mp_client.MPRester", create=True, return_value=mock_mpr):
            with pytest.raises(ValueError, match="Could not fetch"):
                fetch_by_mp_id("mp-99999999")


# ---------------------------------------------------------------------------
# search_by_formula (mocked)
# ---------------------------------------------------------------------------


class TestSearchByFormula:
    def test_no_results(self):
        mock_mpr = MagicMock()
        mock_mpr.__enter__ = MagicMock(return_value=mock_mpr)
        mock_mpr.__exit__ = MagicMock(return_value=False)
        mock_mpr.materials.summary.search.return_value = []

        with patch("shalom.mp_client._MP_AVAILABLE", True), \
             patch.dict("os.environ", {"MP_API_KEY": "test-key"}), \
             patch("shalom.mp_client.MPRester", create=True, return_value=mock_mpr):
            with pytest.raises(ValueError, match="No structures found"):
                search_by_formula("XxYyZz")


# ---------------------------------------------------------------------------
# fetch_structure (dispatch)
# ---------------------------------------------------------------------------


class TestFetchStructure:
    def test_dispatches_mp_id(self):
        with patch("shalom.mp_client.fetch_by_mp_id") as mock_fetch:
            mock_fetch.return_value = MPStructureResult(
                atoms=MagicMock(), mp_id="mp-123", formula="Cu",
            )
            fetch_structure("mp-123")
            mock_fetch.assert_called_once_with("mp-123")

    def test_dispatches_formula(self):
        with patch("shalom.mp_client.search_by_formula") as mock_search:
            mock_search.return_value = [MPStructureResult(
                atoms=MagicMock(), mp_id="mp-456", formula="Fe2O3",
            )]
            fetch_structure("Fe2O3")
            mock_search.assert_called_once_with("Fe2O3", max_results=1)


# ---------------------------------------------------------------------------
# Additional edge case tests
# ---------------------------------------------------------------------------


class TestIsMpIdEdgeCases:
    def test_none_returns_false(self):
        assert is_mp_id(None) is False

    def test_non_string_returns_false(self):
        assert is_mp_id(12345) is False

    def test_integer_zero_returns_false(self):
        assert is_mp_id(0) is False


class TestFetchStructureEdgeCases:
    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            fetch_structure("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            fetch_structure("   ")
