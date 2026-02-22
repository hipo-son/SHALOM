"""Built-in demo scenarios for the SHALOM material discovery pipeline."""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Scenario:
    """A demo scenario configuration."""

    name: str
    objective: str
    selector_mode: str  # "simple" or "multi_agent"
    estimated_api_calls: int
    estimated_cost_usd: float
    description: str


SCENARIOS: Dict[str, Scenario] = {
    "smoke_test": Scenario(
        name="Binary Oxide",
        objective=(
            "Find the most promising binary metal oxide for photocatalytic "
            "water splitting under visible light (bandgap 1.8-2.5 eV)."
        ),
        selector_mode="simple",
        estimated_api_calls=3,
        estimated_cost_usd=0.03,
        description="Quick smoke test: Coarse + Fine + Geometry (simple mode, 3 API calls).",
    ),
    "her_catalyst": Scenario(
        name="HER Electrocatalyst",
        objective=(
            "Design a non-precious-metal electrocatalyst for the hydrogen evolution "
            "reaction (HER) in acidic media, targeting low overpotential and high "
            "exchange current density via optimal hydrogen adsorption free energy."
        ),
        selector_mode="multi_agent",
        estimated_api_calls=9,
        estimated_cost_usd=0.12,
        description="Multi-agent evaluation with 6 specialist evaluators.",
    ),
    "2d_semiconductor": Scenario(
        name="2D TMD Semiconductor",
        objective=(
            "Identify a 2D transition metal dichalcogenide (TMD) with a direct "
            "bandgap of 1.5-2.0 eV suitable for optoelectronic applications, "
            "considering exfoliation energy and air stability."
        ),
        selector_mode="multi_agent",
        estimated_api_calls=9,
        estimated_cost_usd=0.12,
        description="2D materials discovery with structure-aware VASP configuration.",
    ),
    "thermoelectric": Scenario(
        name="Thermoelectric Material",
        objective=(
            "Find a high-performance thermoelectric material with ZT > 1 near "
            "room temperature, preferring earth-abundant and non-toxic elements. "
            "Consider Seebeck coefficient, electrical conductivity, and low "
            "lattice thermal conductivity."
        ),
        selector_mode="multi_agent",
        estimated_api_calls=9,
        estimated_cost_usd=0.12,
        description="Thermoelectric optimization balancing multiple properties.",
    ),
    "battery_cathode": Scenario(
        name="Li-ion Cathode",
        objective=(
            "Design a novel lithium-ion battery cathode material with high "
            "theoretical capacity (>200 mAh/g), good structural stability during "
            "cycling, and a voltage plateau between 3.0-4.5V vs Li/Li+."
        ),
        selector_mode="multi_agent",
        estimated_api_calls=9,
        estimated_cost_usd=0.12,
        description="Battery cathode design with safety and cost considerations.",
    ),
}
