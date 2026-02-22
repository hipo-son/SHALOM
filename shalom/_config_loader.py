"""Configuration and prompt loader with caching and fallback.

Design decisions (professor feedback):
- deepcopy on config return: prevents cache poisoning (Issue A)
- Fail-Fast on YAML syntax errors: prevents silent fallback (Issue B)
- CRLF normalization: ensures cross-platform prompt equality (Issue C)
- Pydantic schema validation: catches typos in critical physics configs (Issue D)
- Custom exception wrapping: user-friendly error messages (결함 3)
"""

from __future__ import annotations

import copy
import importlib.resources
import logging
import warnings
from functools import lru_cache
from typing import Any, Dict

logger = logging.getLogger(__name__)

try:
    import yaml

    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


class ShalomConfigurationError(Exception):
    """Raised when a SHALOM configuration file is invalid.

    Wraps yaml.YAMLError and pydantic.ValidationError with context
    about which file failed and why, so users get actionable messages
    instead of raw parser tracebacks (결함 3 from professor feedback).
    """


@lru_cache(maxsize=32)
def _load_prompt_cached(name: str) -> str:
    """Internal cached loader — returns immutable str, safe to cache."""
    try:
        ref = importlib.resources.files("shalom") / "prompts" / f"{name}.md"
        text = ref.read_text(encoding="utf-8")
        # Issue C: CRLF normalization for Windows compatibility
        text = text.replace("\r\n", "\n").strip()
        if text:
            return text
    except (FileNotFoundError, TypeError, ModuleNotFoundError, OSError):
        pass

    # Fallback to built-in defaults
    from shalom._defaults import PROMPTS

    if name in PROMPTS:
        warnings.warn(
            f"Prompt file 'prompts/{name}.md' not found, using built-in default.",
            stacklevel=3,
        )
        return PROMPTS[name]
    raise FileNotFoundError(f"No prompt found for '{name}'")


def load_prompt(name: str) -> str:
    """Load a prompt from shalom/prompts/{name}.md.

    Falls back to ``_defaults.PROMPTS[name]`` if file not found.
    Strings are immutable, so caching is safe without deepcopy.
    """
    return _load_prompt_cached(name)


@lru_cache(maxsize=32)
def _load_config_cached(name: str) -> Dict[str, Any]:
    """Internal cached loader — callers get deepcopy via load_config()."""
    if _YAML_AVAILABLE:
        try:
            ref = importlib.resources.files("shalom") / "config" / f"{name}.yaml"
            raw = ref.read_text(encoding="utf-8")
        except (FileNotFoundError, TypeError, ModuleNotFoundError, OSError):
            raw = None

        if raw is not None:
            # Issue B: Fail-Fast — YAML syntax errors must NOT fall through.
            # Wrap in ShalomConfigurationError for user-friendly messages.
            try:
                data = yaml.safe_load(raw)
            except yaml.YAMLError as exc:
                raise ShalomConfigurationError(
                    f"Failed to parse 'config/{name}.yaml'. "
                    f"Check YAML syntax (indentation, colons, etc.).\n"
                    f"Original error: {exc}"
                ) from exc

            if data is not None:
                # Issue D: Pydantic schema validation for critical configs
                from shalom._config_schemas import validate_config

                try:
                    data = validate_config(name, data)
                except Exception as exc:
                    raise ShalomConfigurationError(
                        f"Schema validation failed for 'config/{name}.yaml'. "
                        f"Check required keys and data types.\n"
                        f"Original error: {exc}"
                    ) from exc
                return data

    # Fallback to built-in defaults
    from shalom._defaults import CONFIGS

    if name in CONFIGS:
        if not _YAML_AVAILABLE:
            logger.debug("PyYAML not installed, using built-in defaults for '%s'", name)
        else:
            warnings.warn(
                f"Config file 'config/{name}.yaml' not found, using built-in default.",
                stacklevel=3,
            )
        return CONFIGS[name]
    raise FileNotFoundError(f"No config found for '{name}'")


def load_config(name: str) -> Dict[str, Any]:
    """Load a YAML config from shalom/config/{name}.yaml.

    Issue A: Returns a deep copy to prevent cache poisoning.
    Callers may freely mutate the returned dict.
    """
    return copy.deepcopy(_load_config_cached(name))


def clear_cache() -> None:
    """Clear all cached prompts and configs (for testing)."""
    _load_prompt_cached.cache_clear()
    _load_config_cached.cache_clear()
