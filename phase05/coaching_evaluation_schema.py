from pydantic import BaseModel, Field
from typing import Literal

# define the score and rationale required for an individual evaluation criterion.
class EvaluationCriterion(BaseModel):
    score: int = Field(ge=1, le=5)
    reasoning: str
    issues: list[str]

# represent the evaluation of a coaching recommendation and its supporting evidence.
class RecommendationEvaluation(BaseModel):
    recommendation_index: int = Field(ge=0)

    evidence_accuracy: EvaluationCriterion
    reasoning_quality: EvaluationCriterion
    actionability: EvaluationCriterion
    climbing_relevance: EvaluationCriterion
    uncertainty_handling: EvaluationCriterion

    suggested_improvement: str

# validate the structured quality assessment returned by the evaluator.
class CoachingQualityEvaluation(BaseModel):
    recommendation_evaluations: list[RecommendationEvaluation]

    overall_strengths: list[str]
    overall_weaknesses: list[str]

    overall_confidence: Literal["low", "medium", "high"]
    requires_human_review: bool