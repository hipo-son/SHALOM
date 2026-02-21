from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class MaterialCandidate(BaseModel):
    """Rough material candidate schema selected by the Coarse Selector."""

    material_name: str = Field(
        description="Common name or chemical formula of the material (e.g., 'Copper', 'MoS2')."
    )
    elements: List[str] = Field(
        description="List of element symbols composing the material (e.g., ['Cu'], ['Mo', 'S'])."
    )
    reasoning: str = Field(
        description=(
            "Scientific/physical rationale for selecting this candidate "
            "(e.g., d-band theory, electronegativity differences)."
        )
    )
    expected_properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Approximate expected properties (e.g., {'bandgap': '1.5~2.0 eV'}).",
    )


class RankedMaterial(BaseModel):
    """Final candidate schema evaluated and ranked by the Fine Selector."""

    candidate: MaterialCandidate = Field(description="The evaluated material candidate.")
    score: float = Field(
        ge=0.0, le=1.0, description="Alignment score with the target objective (0.0 to 1.0)."
    )
    ranking_justification: str = Field(
        description="Detailed justification for the assigned score and ranking."
    )


class StructureReviewForm(BaseModel):
    """Form to validate the initial structure (POSCAR equivalent) generated in the Simulation Layer."""

    file_path: Optional[str] = Field(
        default=None, description="Relative or absolute path of the analyzed structure file."
    )
    num_atoms: int = Field(description="Total number of atoms in the structure.")
    cell_volume: float = Field(description="Unit cell volume (Angstrom^3).")
    minimum_distance: float = Field(
        description="Shortest interatomic distance (Angstrom). Used for overlap detection."
    )
    vacuum_thickness: Optional[float] = Field(
        default=None,
        description="Vacuum layer thickness for surface/2D materials (Angstrom).",
    )
    is_valid: bool = Field(
        description="Whether this structure is physically valid for a VASP calculation."
    )
    feedback: str = Field(
        description="Specific feedback on what to fix if the structure has issues."
    )


class AgentMessage(BaseModel):
    """Base message schema used for inter-agent communication."""

    sender: str = Field(description="Name of the sending agent.")
    receiver: str = Field(description="Name of the receiving agent.")
    content: str = Field(description="Natural-language message body.")
    payload: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional structured data to transmit (JSON)."
    )


class ReviewResult(BaseModel):
    """Final evaluation and feedback schema from the Review Layer after analyzing OUTCAR."""

    is_successful: bool = Field(
        description="Whether the materials simulation successfully achieved the target objective."
    )
    energy: Optional[float] = Field(default=None, description="Calculated final energy (eV).")
    forces_max: Optional[float] = Field(
        default=None, description="Maximum calculated force (eV/A)."
    )
    feedback_for_design: str = Field(
        description=(
            "Feedback to be sent back to the Design Layer "
            "(e.g., failure reasons or suggestions for next targets)."
        )
    )
