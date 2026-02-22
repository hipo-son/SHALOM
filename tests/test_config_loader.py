"""Tests for the configuration and prompt loader system."""

import pytest

from shalom._config_loader import (
    ShalomConfigurationError,
    clear_cache,
    load_config,
    load_prompt,
)
from shalom._config_schemas import validate_config


# ---------------------------------------------------------------------------
# TestLoadPrompt
# ---------------------------------------------------------------------------


class TestLoadPrompt:
    """Tests for load_prompt() function."""

    def setup_method(self):
        clear_cache()

    def test_load_existing_prompt(self):
        """coarse_selector prompt loads successfully."""
        prompt = load_prompt("coarse_selector")
        assert "Coarse Selector" in prompt
        assert len(prompt) > 50

    def test_load_all_prompts(self):
        """All 11 defined prompts load without error."""
        names = [
            "coarse_selector", "fine_selector", "geometry_generator", "review_agent",
            "eval_confidence_rule",
            "eval_stability", "eval_target_property", "eval_dft_feasibility",
            "eval_synthesizability", "eval_novelty", "eval_environmental_cost",
        ]
        for name in names:
            prompt = load_prompt(name)
            assert isinstance(prompt, str)
            assert len(prompt) > 0

    def test_prompt_not_empty(self):
        """Loaded prompts are never empty strings."""
        prompt = load_prompt("review_agent")
        assert prompt.strip() != ""

    def test_missing_prompt_raises(self):
        """Non-existent prompt name raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="No prompt found"):
            load_prompt("nonexistent_agent_xyz")

    def test_caching(self):
        """Two calls return the same object (str is immutable, cached)."""
        p1 = load_prompt("coarse_selector")
        p2 = load_prompt("coarse_selector")
        assert p1 is p2  # Same cached object

    def test_version_tag_present(self):
        """Prompts contain version tags."""
        prompt = load_prompt("coarse_selector")
        assert "[v1.0.0]" in prompt

    def test_review_agent_v2(self):
        """Review agent has v2.0.0 tag."""
        prompt = load_prompt("review_agent")
        assert "[v2.0.0]" in prompt


# ---------------------------------------------------------------------------
# TestLoadConfig
# ---------------------------------------------------------------------------


class TestLoadConfig:
    """Tests for load_config() function."""

    def setup_method(self):
        clear_cache()

    def test_load_existing_config(self):
        """potcar_mapping config loads successfully."""
        cfg = load_config("potcar_mapping")
        assert "vasp_recommended" in cfg
        assert isinstance(cfg["vasp_recommended"], dict)

    def test_load_all_configs(self):
        """All 9 defined configs load without error."""
        names = [
            "potcar_mapping", "enmax_values", "magnetic_elements",
            "hubbard_u", "metallic_elements", "incar_presets",
            "error_patterns", "correction_strategies", "evaluator_weights",
        ]
        for name in names:
            cfg = load_config(name)
            assert cfg is not None

    def test_potcar_has_expected_keys(self):
        """POTCAR mapping contains essential elements."""
        cfg = load_config("potcar_mapping")
        potcars = cfg["vasp_recommended"]
        for el in ["H", "C", "N", "O", "Fe", "Cu", "Ti", "W"]:
            assert el in potcars, f"Missing element {el}"

    def test_potcar_version_metadata(self):
        """POTCAR mapping includes dataset version (Issue E)."""
        cfg = load_config("potcar_mapping")
        assert cfg.get("potcar_version") == "54"

    def test_hubbard_u_has_functional(self):
        """Hubbard U config includes functional tag (Issue F)."""
        cfg = load_config("hubbard_u")
        assert cfg.get("functional") == "PBE"

    def test_missing_config_raises(self):
        """Non-existent config name raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="No config found"):
            load_config("nonexistent_config_xyz")

    def test_deepcopy_prevents_cache_poisoning(self):
        """Mutating returned dict does NOT affect cached value (Issue A)."""
        cfg1 = load_config("evaluator_weights")
        original_stability = cfg1["weights"]["stability"]
        cfg1["weights"]["stability"] = 999.0  # Mutate!

        cfg2 = load_config("evaluator_weights")
        assert cfg2["weights"]["stability"] == original_stability  # Unaffected

    def test_enmax_values_structure(self):
        """ENMAX values are element->float mappings."""
        cfg = load_config("enmax_values")
        assert isinstance(cfg.get("O"), (int, float))
        assert cfg["O"] == 400.0

    def test_error_patterns_structure(self):
        """Error patterns are list of dicts with pattern/type/severity."""
        cfg = load_config("error_patterns")
        assert isinstance(cfg, list)
        assert len(cfg) == 11
        for entry in cfg:
            assert "pattern" in entry
            assert "type" in entry
            assert "severity" in entry

    def test_correction_strategies_structure(self):
        """Correction strategies have expected error types."""
        cfg = load_config("correction_strategies")
        assert "SCF_UNCONVERGED" in cfg
        assert "BRMIX" in cfg
        assert "ZBRENT" in cfg
        assert len(cfg["SCF_UNCONVERGED"]) == 5


# ---------------------------------------------------------------------------
# TestSchemaValidation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    """Tests for Pydantic schema validation (Issue D)."""

    def test_potcar_valid(self):
        """Valid POTCAR mapping passes validation."""
        from shalom._defaults import CONFIGS

        data = CONFIGS["potcar_mapping"]
        result = validate_config("potcar_mapping", data)
        assert "vasp_recommended" in result

    def test_potcar_missing_element(self):
        """POTCAR mapping missing required element raises error."""
        from pydantic import ValidationError

        data = {
            "potcar_version": "54",
            "vasp_recommended": {"Si": "Si"},  # Missing H, C, N, O, Fe, Cu, Ti
        }
        with pytest.raises(ValidationError, match="Missing common elements"):
            validate_config("potcar_mapping", data)

    def test_hubbard_u_valid(self):
        """Valid Hubbard U config passes validation."""
        from shalom._defaults import CONFIGS

        data = CONFIGS["hubbard_u"]
        result = validate_config("hubbard_u", data)
        assert "values" in result

    def test_hubbard_u_missing_oxygen(self):
        """Anion elements without O raises error."""
        from pydantic import ValidationError

        data = {
            "functional": "PBE",
            "values": {"Fe": {"L": 2, "U": 5.3, "J": 0.0}},
            "anion_elements": ["S", "Se"],  # Missing O!
        }
        with pytest.raises(ValidationError, match="must include 'O'"):
            validate_config("hubbard_u", data)

    def test_no_schema_passthrough(self):
        """Config without a schema passes through unchanged."""
        data = {"key": "value"}
        result = validate_config("evaluator_weights", data)
        assert result == data


# ---------------------------------------------------------------------------
# TestClearCache
# ---------------------------------------------------------------------------


class TestClearCache:
    """Tests for cache clearing."""

    def test_clear_cache(self):
        """clear_cache() allows fresh reload."""
        p1 = load_prompt("coarse_selector")
        clear_cache()
        p2 = load_prompt("coarse_selector")
        # After cache clear, new object (but same content)
        assert p1 == p2

    def test_prompt_content_matches_default(self):
        """External file content matches _defaults.py fallback."""
        from shalom._defaults import PROMPTS

        for name, default_text in PROMPTS.items():
            clear_cache()
            loaded = load_prompt(name)
            assert loaded == default_text, (
                f"Prompt '{name}' mismatch between loaded and default"
            )


# ---------------------------------------------------------------------------
# TestShalomConfigurationError
# ---------------------------------------------------------------------------


class TestShalomConfigurationError:
    """Tests for the custom exception class."""

    def test_is_exception_subclass(self):
        """ShalomConfigurationError is a proper Exception."""
        exc = ShalomConfigurationError("test error")
        assert isinstance(exc, Exception)
        assert str(exc) == "test error"


# ---------------------------------------------------------------------------
# Additional coverage: fallback paths and edge cases
# ---------------------------------------------------------------------------


class TestConfigLoaderFallback:
    """Tests for fallback and error paths in _config_loader."""

    def setup_method(self):
        clear_cache()

    def test_yaml_unavailable_uses_defaults(self, monkeypatch):
        """When PyYAML is not available, built-in defaults are used."""
        import shalom._config_loader as loader

        original = loader._YAML_AVAILABLE
        monkeypatch.setattr(loader, "_YAML_AVAILABLE", False)
        clear_cache()
        try:
            cfg = load_config("evaluator_weights")
            assert "weights" in cfg
        finally:
            monkeypatch.setattr(loader, "_YAML_AVAILABLE", original)
            clear_cache()

    def test_yaml_syntax_error_raises(self, monkeypatch):
        """Malformed YAML raises ShalomConfigurationError."""
        import importlib.resources
        import shalom._config_loader as loader

        clear_cache()

        # Monkey-patch to return invalid YAML
        def fake_files(pkg):
            mock = type("FakeFiles", (), {
                "__truediv__": lambda self, name: type("FakePath", (), {
                    "__truediv__": lambda self2, name2: type("FakeFile", (), {
                        "read_text": lambda self3, encoding="utf-8": ":{bad yaml:\n  - [",
                    })(),
                })(),
            })()
            return mock

        monkeypatch.setattr(importlib.resources, "files", fake_files)
        try:
            with pytest.raises(ShalomConfigurationError, match="Failed to parse"):
                loader._load_config_cached.__wrapped__("bad_config_test")
        finally:
            clear_cache()

    def test_prompt_fallback_warning(self, monkeypatch):
        """When prompt file is not found, warning is issued with default."""
        import importlib.resources
        import warnings
        import shalom._config_loader as loader

        clear_cache()

        def fake_files(pkg):
            class FakeRef:
                def __truediv__(self, name):
                    return self
                def read_text(self, encoding="utf-8"):
                    raise FileNotFoundError("not found")
            return FakeRef()

        monkeypatch.setattr(importlib.resources, "files", fake_files)
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = loader._load_prompt_cached.__wrapped__("coarse_selector")
                assert len(w) >= 1
                assert "built-in default" in str(w[0].message)
                assert len(result) > 0
        finally:
            clear_cache()
