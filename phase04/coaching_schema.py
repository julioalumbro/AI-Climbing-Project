# define the structured data models used to validate generated coaching responses.
from pydantic import BaseModel, Field
from typing import Literal

# describe a single coaching suggestion and its supporting fields.
class CoachingRecommendation(BaseModel):
    observation: str = Field(
        description="What was observed in the climber's movement."
    )

    suggested_adjustment: str = Field(
        description="An actionable climbing technique suggestion"
    )

    supporting_evidence: list[str] = Field(
        description=(
            "Claims from the evidence report supporting "
            "this recommendation."
        )
    )

    uncertainty: str = Field(
        description=(
            "Limitations or uncertainties affecting "
            "this recommendation."
        )
    )

# describe the complete structured coaching response.
class CoachingAnalysis(BaseModel):
    movement_summary: str = Field(
        description="A concise summary of the analyzed movement."
    )

    recommendations: list[CoachingRecommendation] = Field(
        description="Evidence-grounded climbing recommendations."
    )

    needs_review: list[str] = Field(
        description=(
            "Conflicting or insufficient evidence that "
            "should not be treated as established fact."
        )
    )

    overall_confidence: Literal[
        "low", "medium", "high"
    ] = Field(
        description=(
            "Qualitative confidence in the coaching analysis, "
            "considering evidence quality and disagreements."
        )
    )

if __name__ == "__main__":
    print("Coaching schema loaded successfully!")
    print(CoachingAnalysis.model_json_schema())