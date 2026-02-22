from enum import Enum
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


class EvaluationDetails(BaseModel):
    """Multi-agent evaluation metadata attached to a RankedMaterial."""

    evaluations: List["EvaluationResponse"] = Field(
        default_factory=list,
        description="Evaluation responses from each specialist evaluator.",
    )
    veto_reasons: List[str] = Field(
        default_factory=list,
        description="Reasons candidates were vetoed during this evaluation.",
    )
    micro_loop_retries: int = Field(
        default=0, ge=0,
        description="Number of design micro-loop retries before finding a non-vetoed candidate.",
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
    evaluation_details: Optional[EvaluationDetails] = Field(
        default=None,
        description="Multi-agent evaluation metadata (populated by MultiAgentFineSelector).",
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
        description="Whether this structure is physically valid for a DFT calculation."
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
    """Final evaluation and feedback schema from the Review Layer after analyzing DFT output."""

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
    bandgap: Optional[float] = Field(
        default=None, description="Band gap in eV (None if metallic or not computed)."
    )
    magnetization: Optional[float] = Field(
        default=None, description="Total magnetization in Bohr magneton."
    )
    entropy_per_atom: Optional[float] = Field(
        default=None,
        description="Electronic entropy T*S per atom in eV. Warns if > 1 meV/atom.",
    )
    suggested_incar_changes: Optional[Dict[str, Any]] = Field(
        default=None,
        description="INCAR parameter changes suggested by the reviewer for next iteration.",
    )
    correction_summary: Optional[str] = Field(
        default=None,
        description="Summary of error corrections applied during the calculation.",
    )
    physics_warnings: Optional[List[str]] = Field(
        default=None,
        description="List of physics validation warnings (e.g., ISIF=4 trap, +U limitation).",
    )


# ---------------------------------------------------------------------------
# Multi-Agent Evaluation Schemas
# ---------------------------------------------------------------------------


class CandidateScore(BaseModel):
    """Single candidate evaluation by a specialist evaluator."""

    material_name: str = Field(
        description="Name of the evaluated material."
    )
    score: float = Field(
        ge=0.0, le=1.0, description="Score from this perspective."
    )
    confidence: float = Field(
        ge=0.0, le=1.0, default=0.5,
        description=(
            "Evaluator confidence in this score. "
            "0.0 = pure guess, 1.0 = literature/database-backed. "
            "Must be <= 0.5 unless citing ICSD, Materials Project, or published references."
        ),
    )
    justification: str = Field(
        description="Reasoning for this score with specific physical/chemical evidence."
    )


class EvaluationResponse(BaseModel):
    """A specialist evaluator's scores for all candidates."""

    perspective: str = Field(description="Evaluation perspective name.")
    scores: List[CandidateScore] = Field(
        description="Scores for each candidate from this perspective."
    )


# ---------------------------------------------------------------------------
# Pipeline Schemas
# ---------------------------------------------------------------------------


class PipelineStatus(str, Enum):
    """Status of a pipeline execution."""

    COMPLETED = "completed"
    AWAITING_DFT = "awaiting_dft"
    FAILED_DESIGN = "failed_design"
    FAILED_SIMULATION = "failed_simulation"
    FAILED_REVIEW = "failed_review"


class PipelineResult(BaseModel):
    """Complete result of an end-to-end pipeline run."""

    status: PipelineStatus = Field(
        description="Final status of the pipeline execution."
    )
    objective: str = Field(
        description="The target objective that was provided."
    )
    iteration: int = Field(
        default=1, description="Current iteration in the closed loop."
    )

    # Design Layer outputs
    candidates: Optional[List[MaterialCandidate]] = Field(
        default=None, description="Candidates from the Coarse Selector."
    )
    ranked_material: Optional[RankedMaterial] = Field(
        default=None, description="Winner from the Fine Selector."
    )

    # Simulation Layer outputs
    structure_generated: bool = Field(
        default=False,
        description="Whether a valid structure was successfully generated.",
    )
    structure_path: Optional[str] = Field(
        default=None, description="Path to the generated DFT input file."
    )

    # Review Layer outputs
    review_result: Optional[ReviewResult] = Field(
        default=None,
        description="Review of the DFT output, if DFT was executed.",
    )

    # Metadata
    error_message: Optional[str] = Field(
        default=None, description="Error details if the pipeline failed at any stage."
    )
    steps_completed: List[str] = Field(
        default_factory=list,
        description="Ordered list of successfully completed pipeline steps.",
    )


# Resolve forward references for EvaluationDetails -> EvaluationResponse
EvaluationDetails.model_rebuild()
