"""Tests for shalom.direct_run module."""

import os
from pathlib import Path
from unittest.mock import patch
from ase.build import bulk

from shalom.direct_run import (
    direct_run,
    DirectRunConfig,
    CALC_TYPE_ALIASES,
    _resolve_calc_type,
    _auto_output_dir,
    _resolve_workspace,
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


# ---------------------------------------------------------------------------
# Workspace / project directory structure
# ---------------------------------------------------------------------------


class TestResolveWorkspace:
    """_resolve_workspace() priority: arg > $SHALOM_WORKSPACE > Desktop > home."""

    def test_explicit_arg_wins(self, tmp_path):
        ws = str(tmp_path / "my_ws")
        assert _resolve_workspace(ws) == ws

    def test_env_var_used_when_no_arg(self, tmp_path, monkeypatch):
        ws = str(tmp_path / "env_ws")
        monkeypatch.setenv("SHALOM_WORKSPACE", ws)
        assert _resolve_workspace() == ws

    def test_arg_beats_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SHALOM_WORKSPACE", str(tmp_path / "env_ws"))
        explicit = str(tmp_path / "explicit")
        assert _resolve_workspace(explicit) == explicit

    def test_desktop_fallback(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SHALOM_WORKSPACE", raising=False)
        # Pretend home is tmp_path so Desktop sub-path is predictable
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        desktop = tmp_path / "Desktop"
        desktop.mkdir()
        result = _resolve_workspace()
        assert result == str(desktop / "shalom-runs")

    def test_home_fallback_when_no_desktop(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SHALOM_WORKSPACE", raising=False)
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        # Desktop does NOT exist
        result = _resolve_workspace()
        assert result == str(tmp_path / "shalom-runs")


class TestWorkspaceOutputDir:
    """direct_run() places outputs inside workspace and optional project folder."""

    def _cu_poscar(self, tmp_path):
        from ase.io import write
        atoms = bulk("Cu", "fcc", a=3.6)
        p = tmp_path / "POSCAR"
        write(str(p), atoms, format="vasp")
        return str(p)

    def test_workspace_used_as_root(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SHALOM_WORKSPACE", raising=False)
        ws = str(tmp_path / "runs")
        config = DirectRunConfig(
            backend_name="vasp",
            structure_file=self._cu_poscar(tmp_path),
            workspace_dir=ws,
        )
        result = direct_run("", config)
        assert result.success is True
        # Output must be inside the workspace
        assert result.output_dir.startswith(ws)

    def test_project_subfolder_created(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SHALOM_WORKSPACE", raising=False)
        ws = str(tmp_path / "runs")
        config = DirectRunConfig(
            backend_name="vasp",
            structure_file=self._cu_poscar(tmp_path),
            workspace_dir=ws,
            project="my_study",
        )
        result = direct_run("", config)
        assert result.success is True
        expected_project_path = os.path.join(ws, "my_study")
        assert result.output_dir.startswith(expected_project_path)

    def test_explicit_output_dir_bypasses_workspace(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SHALOM_WORKSPACE", raising=False)
        ws = str(tmp_path / "should_not_be_used")
        explicit = str(tmp_path / "explicit_out")
        config = DirectRunConfig(
            backend_name="vasp",
            structure_file=self._cu_poscar(tmp_path),
            workspace_dir=ws,
            output_dir=explicit,
        )
        result = direct_run("", config)
        assert result.success is True
        assert result.output_dir == explicit


# ---------------------------------------------------------------------------
# _write_output_readme — direct unit tests for README generation
# ---------------------------------------------------------------------------


class TestWriteOutputReadme:
    def _call(self, tmp_path, structure_info=None, auto_detected=None, files=None):
        from shalom.direct_run import _write_output_readme
        _write_output_readme(
            output_dir=str(tmp_path),
            backend_name="qe",
            calc_type="scf",
            structure_info=structure_info,
            auto_detected=auto_detected,
            files_generated=files or ["pw.in"],
        )
        readme = tmp_path / "README.md"
        return readme.read_text() if readme.exists() else ""

    def test_with_mp_id_includes_link(self, tmp_path):
        """MP ID → README contains materials project link."""
        info = {"mp_id": "mp-149", "formula": "Si", "source": "materials_project"}
        text = self._call(tmp_path, structure_info=info)
        assert "mp-149" in text
        assert "materialsproject.org" in text

    def test_with_spacegroup(self, tmp_path):
        """spacegroup in structure_info → shown in README."""
        info = {"formula": "Si", "space_group": "Fd-3m"}
        text = self._call(tmp_path, structure_info=info)
        assert "Fd-3m" in text

    def test_with_energy_above_hull(self, tmp_path):
        """energy_above_hull → shown in README."""
        info = {"formula": "Fe2O3", "energy_above_hull": 0.0025}
        text = self._call(tmp_path, structure_info=info)
        assert "E above hull" in text
        assert "0.0025" in text

    def test_local_file_source(self, tmp_path):
        """local file source → --structure placeholder in reproduce command."""
        info = {"formula": "Cu", "source": "local file"}
        text = self._call(tmp_path, structure_info=info)
        assert "--structure" in text

    def test_oswrite_error_gracefully_ignored(self, tmp_path):
        """OSError when writing README → function returns without crashing."""
        from unittest.mock import patch, mock_open
        from shalom.direct_run import _write_output_readme

        m = mock_open()
        m.side_effect = OSError("disk full")
        with patch("builtins.open", m):
            # Should not raise
            _write_output_readme(
                output_dir=str(tmp_path),
                backend_name="qe", calc_type="scf",
                structure_info={"formula": "Si"},
                auto_detected=None,
                files_generated=["pw.in"],
            )

    def test_version_included(self, tmp_path):
        """SHALOM version included in README."""
        text = self._call(tmp_path)
        assert "SHALOM v" in text


# ---------------------------------------------------------------------------
# direct_run — additional error path coverage
# ---------------------------------------------------------------------------


class TestDirectRunErrorPaths:
    def test_no_structure_returns_error(self, tmp_path):
        """Empty material_spec and no structure_file → failure."""
        config = DirectRunConfig(
            backend_name="vasp",
            output_dir=str(tmp_path / "out"),
        )
        result = direct_run("", config)
        assert result.success is False
        assert "No material" in result.error

    def test_write_input_exception_returns_failure(self, tmp_path):
        """backend.write_input raising → failure result."""
        from unittest.mock import patch
        from ase.io import write
        from ase.build import bulk

        si = bulk("Si", "diamond", a=5.43)
        poscar = tmp_path / "Si.vasp"
        write(str(poscar), si, format="vasp")

        config = DirectRunConfig(
            backend_name="vasp",
            structure_file=str(poscar),
            output_dir=str(tmp_path / "out"),
        )
        with patch("shalom.backends.vasp.VASPBackend.write_input",
                   side_effect=IOError("disk error")):
            result = direct_run("", config)
        assert result.success is False
        assert "Failed to write" in result.error

    def test_validation_failure(self, tmp_path):
        """Invalid structure → validation failure result."""
        from unittest.mock import patch, MagicMock
        from ase.build import bulk
        from ase.io import write

        si = bulk("Si", "diamond", a=5.43)
        poscar = tmp_path / "Si.vasp"
        write(str(poscar), si, format="vasp")

        mock_form = MagicMock()
        mock_form.is_valid = False
        mock_form.feedback = "atoms too close"

        config = DirectRunConfig(
            backend_name="vasp",
            structure_file=str(poscar),
            output_dir=str(tmp_path / "out"),
            validate_structure=True,
        )
        with patch("shalom.direct_run.FormFiller.evaluate_atoms", return_value=mock_form):
            result = direct_run("", config)
        assert result.success is False
        assert "validation" in result.error.lower()

    def test_mp_importerror_returns_failure(self, tmp_path):
        """fetch_structure ImportError → failure with error message."""
        from unittest.mock import patch

        config = DirectRunConfig(
            backend_name="vasp",
            output_dir=str(tmp_path / "out"),
        )
        with patch("shalom.mp_client.fetch_structure",
                   side_effect=ImportError("mp-api not installed")):
            result = direct_run("mp-19717", config)
        assert result.success is False

    def test_qe_unknown_calc_type_defaults_scf(self, tmp_path):
        """Unknown QE calc_type falls back to SCF."""
        from ase.build import bulk
        from ase.io import write

        si = bulk("Si", "diamond", a=5.43)
        poscar = tmp_path / "Si.vasp"
        write(str(poscar), si, format="vasp")

        config = DirectRunConfig(
            backend_name="qe",
            calc_type="totally_unknown_calc",
            structure_file=str(poscar),
            output_dir=str(tmp_path / "out"),
        )
        result = direct_run("", config)
        # Should still succeed (defaults to SCF)
        assert result.success is True
