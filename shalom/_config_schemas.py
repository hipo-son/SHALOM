"""Pydantic schemas for validating YAML config files.

Only critical physics configs are validated (POTCAR, Hubbard U).
Simple configs (metallic_elements, evaluator_weights) are validated
by the consuming code's existing logic.

Issue D (professor feedback): prevents typos like "Fepv" from silently
producing wrong VASP calculations.
"""

from __future__ import annotations

from typing import Dict, List, Type

from pydantic import BaseModel, field_validator


class PotcarMappingConfig(BaseModel):
    """Validates potcar_mapping.yaml structure."""

    potcar_version: str = "54"
    vasp_recommended: Dict[str, str]
    mp_default_overrides: Dict[str, str] = {}

    @field_validator("vasp_recommended")
    @classmethod
    def must_have_common_elements(cls, v: Dict[str, str]) -> Dict[str, str]:
        required = {"H", "C", "N", "O", "Fe", "Cu", "Ti"}
        missing = required - set(v.keys())
        if missing:
            raise ValueError(f"Missing common elements in POTCAR mapping: {missing}")
        return v


class HubbardUEntry(BaseModel):
    """Single element Hubbard U parameters."""

    L: int
    U: float
    J: float = 0.0


class HubbardUConfig(BaseModel):
    """Validates hubbard_u.yaml structure.

    Issue F: functional-dependent U values.
    """

    functional: str = "PBE"
    values: Dict[str, HubbardUEntry]
    anion_elements: List[str]

    @field_validator("anion_elements")
    @classmethod
    def must_have_oxygen(cls, v: List[str]) -> List[str]:
        if "O" not in v:
            raise ValueError("anion_elements must include 'O'")
        return v


# Registry of configs that have a Pydantic schema.
_SCHEMA_MAP: Dict[str, Type[BaseModel]] = {
    "potcar_mapping": PotcarMappingConfig,
    "hubbard_u": HubbardUConfig,
}


def validate_config(name: str, data: dict) -> dict:
    """Validate config data against its Pydantic schema if one exists.

    Returns the validated (and possibly coerced) data as a dict.
    Raises ``pydantic.ValidationError`` if schema check fails.
    Configs without a schema are passed through unchanged.
    """
    schema_cls = _SCHEMA_MAP.get(name)
    if schema_cls is None:
        return data
    validated = schema_cls.model_validate(data)
    return validated.model_dump()
