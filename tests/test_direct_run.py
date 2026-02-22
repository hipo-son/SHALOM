"""Tests for shalom.direct_run module."""

import os

from unittest.mock import patch
from ase.build import bulk

from shalom.direct_run import (
    direct_run,
    DirectRunConfig,
    CALC_TYPE_ALIASES,
    _resolve_calc_type,
    _auto_output_dir,
)


# ---------------------------------------------------------------------------
# Calc type aliases
# ---------------------------------------------------------------------------


class TestCalcTypeAliases:
    def test_scf_to_static(self):
        assert CALC_TYPE_ALIASES["scf"] == "static"

    def test_relax_to_relaxation(self):
        assert CALC_TYPE_ALIASES["relax"] == "relaxation"

    def test_vc_relax_to_relaxation(self):
        assert CALC_TYPE_ALIASES["vc-relax"] == "relaxation"

    def test_bands_to_band_structure(self):
        assert CALC_TYPE_ALIASES["bands"] == "band_structure"

    def test_resolve_none_default(self):
        assert _resolve_calc_type(None, "vasp") == "relaxation"

    def test_resolve_unknown_default(self):
        assert _resolve_calc_type("unknown", "vasp") == "relaxation"


class TestAutoOutputDir:
    def test_with_mp_id(self):
        d = _auto_output_dir("Cu", "mp-19717", "vasp", "relaxation")
        assert "Cu" in d
        assert "mp-19717" in d
        assert "vasp" in d

    def test_without_mp_id(self):
        d = _auto_output_dir("Cu", None, "qe", "static")
        assert "Cu" in d
        assert "qe" in d


# ---------------------------------------------------------------------------
# direct_run with local structure file
# ---------------------------------------------------------------------------


class TestDirectRunLocalFile:
    def test_vasp_from_local_file(self, tmp_path):
        # Create a structure file
        atoms = bulk("Cu", "fcc", a=3.6)
        from ase.io import write
        poscar = tmp_path / "POSCAR"
        write(str(poscar), atoms, format="vasp")

        output_dir = str(tmp_path / "output")
        config = DirectRunConfig(
            backend_name="vasp",
            structure_file=str(poscar),
            output_dir=output_dir,
        )
        result = direct_run("", config)
        assert result.success is True
        assert result.output_dir == output_dir
        assert "INCAR" in result.files_generated
        assert "POSCAR" in result.files_generated
        assert "KPOINTS" in result.files_generated
        assert "POTCAR.spec" in result.files_generated

    def test_qe_from_local_file(self, tmp_path):
        atoms = bulk("Cu", "fcc", a=3.6)
        from ase.io import write
        poscar = tmp_path / "POSCAR"
        write(str(poscar), atoms, format="vasp")

        output_dir = str(tmp_path / "output")
        config = DirectRunConfig(
            backend_name="qe",
            calc_type="scf",
            structure_file=str(poscar),
            output_dir=output_dir,
        )
        result = direct_run("", config)
        assert result.success is True
        assert "pw.in" in result.files_generated

    def test_nonexistent_file(self, tmp_path):
        config = DirectRunConfig(
            structure_file=str(tmp_path / "nonexistent.vasp"),
            output_dir=str(tmp_path / "out"),
        )
        result = direct_run("", config)
        assert result.success is False
        assert "Failed to read" in result.error

    def test_auto_detected_fields(self, tmp_path):
        atoms = bulk("Cu", "fcc", a=3.6)
        from ase.io import write
        poscar = tmp_path / "POSCAR"
        write(str(poscar), atoms, format="vasp")

        config = DirectRunConfig(
            backend_name="vasp",
            structure_file=str(poscar),
            output_dir=str(tmp_path / "out"),
        )
        result = direct_run("", config)
        assert result.auto_detected is not None
        assert "ENCUT" in result.auto_detected
        assert "kpoints" in result.auto_detected


# ---------------------------------------------------------------------------
# Overwrite protection
# ---------------------------------------------------------------------------


class TestOverwriteProtection:
    def test_existing_dir_blocked(self, tmp_path):
        atoms = bulk("Cu", "fcc", a=3.6)
        from ase.io import write
        poscar = tmp_path / "POSCAR"
        write(str(poscar), atoms, format="vasp")

        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir)
        # Create a dummy file so dir is non-empty
        with open(os.path.join(output_dir, "dummy.txt"), "w") as f:
            f.write("test")

        config = DirectRunConfig(
            structure_file=str(poscar),
            output_dir=output_dir,
            force_overwrite=False,
        )
        result = direct_run("", config)
        assert result.success is False
        assert "--force" in result.error

    def test_force_overwrite(self, tmp_path):
        atoms = bulk("Cu", "fcc", a=3.6)
        from ase.io import write
        poscar = tmp_path / "POSCAR"
        write(str(poscar), atoms, format="vasp")

        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir)
        with open(os.path.join(output_dir, "dummy.txt"), "w") as f:
            f.write("test")

        config = DirectRunConfig(
            structure_file=str(poscar),
            output_dir=output_dir,
            force_overwrite=True,
        )
        result = direct_run("", config)
        assert result.success is True


# ---------------------------------------------------------------------------
# User settings override
# ---------------------------------------------------------------------------


class TestUserSettings:
    def test_vasp_user_settings(self, tmp_path):
        atoms = bulk("Cu", "fcc", a=3.6)
        from ase.io import write
        poscar = tmp_path / "POSCAR"
        write(str(poscar), atoms, format="vasp")

        config = DirectRunConfig(
            backend_name="vasp",
            structure_file=str(poscar),
            output_dir=str(tmp_path / "out"),
            user_settings={"ENCUT": 999},
        )
        result = direct_run("", config)
        assert result.success is True
        with open(os.path.join(result.output_dir, "INCAR"), "r") as f:
            content = f.read()
        assert "999" in content

    def test_qe_user_settings(self, tmp_path):
        atoms = bulk("Cu", "fcc", a=3.6)
        from ase.io import write
        poscar = tmp_path / "POSCAR"
        write(str(poscar), atoms, format="vasp")

        config = DirectRunConfig(
            backend_name="qe",
            structure_file=str(poscar),
            output_dir=str(tmp_path / "out"),
            user_settings={"system.ecutwfc": 999},
        )
        result = direct_run("", config)
        assert result.success is True
        with open(os.path.join(result.output_dir, "pw.in"), "r") as f:
            content = f.read()
        assert "999" in content


# ---------------------------------------------------------------------------
# MP API path (mocked)
# ---------------------------------------------------------------------------


class TestDirectRunMP:
    def test_mp_not_installed(self, tmp_path):
        """When mp-api is not installed, returns clear error."""
        with patch("shalom.direct_run.ase_read", side_effect=ImportError("not used")):
            config = DirectRunConfig(output_dir=str(tmp_path / "out"))
            # This will try to import mp_client which may or may not be importable
            # Patch at the right level
            with patch("shalom.mp_client._MP_AVAILABLE", False):
                result = direct_run("mp-19717", config)
            assert result.success is False
            assert "mp-api" in result.error or "not installed" in result.error.lower() or "Import" in result.error

    def test_no_material_or_file(self, tmp_path):
        """Neither material spec nor --structure → error."""
        config = DirectRunConfig(output_dir=str(tmp_path / "out"), structure_file=None)
        # Pass empty material spec — will try MP API
        # Without MP available, should fail gracefully
        with patch("shalom.mp_client._MP_AVAILABLE", False):
            result = direct_run("", config)
        # Either fails on MP or on "No structure resolved"
        assert result.success is False


# ---------------------------------------------------------------------------
# Additional tests for bug fixes
# ---------------------------------------------------------------------------


class TestDirectRunEmpty:
    def test_empty_material_spec_no_structure(self, tmp_path):
        """Empty material spec without structure → clear error."""
        config = DirectRunConfig(
            output_dir=str(tmp_path / "out"),
            structure_file=None,
        )
        result = direct_run("", config)
        assert result.success is False
        assert "No material specified" in result.error

    def test_whitespace_material_spec(self, tmp_path):
        """Whitespace-only material spec → clear error."""
        config = DirectRunConfig(
            output_dir=str(tmp_path / "out"),
            structure_file=None,
        )
        result = direct_run("   ", config)
        assert result.success is False


class TestAutoOutputDirSanitization:
    def test_special_chars_replaced(self):
        d = _auto_output_dir("Fe<2>O:3", None, "vasp", "relaxation")
        assert "<" not in d
        assert ">" not in d
        assert ":" not in d

    def test_no_special_chars_unchanged(self):
        d = _auto_output_dir("Fe2O3", "mp-123", "vasp", "relaxation")
        assert "Fe2O3" in d
        assert "mp-123" in d


class TestAutoDetectedNoNone:
    def test_vasp_no_none_values(self, tmp_path):
        """auto_detected dict has no None values."""
        atoms = bulk("Cu", "fcc", a=3.6)
        from ase.io import write
        poscar = tmp_path / "POSCAR"
        write(str(poscar), atoms, format="vasp")

        config = DirectRunConfig(
            backend_name="vasp",
            structure_file=str(poscar),
            output_dir=str(tmp_path / "out"),
        )
        result = direct_run("", config)
        assert result.success is True
        for v in result.auto_detected.values():
            assert v is not None
