from pydantic import BaseModel, Field
from typing import Literal

class EvaluationCriterion(BaseModel):
    score: int = Field(ge=1, le=5)
    reasoning: str
    issues: list[str]

class RecommendationEvaluation(BaseModel):
    recommendation_index: int = Field(ge=0)

    evidence_accuracy: EvaluationCriterion
    reasoning_quality: EvaluationCriterion
    actionability: EvaluationCriterion
    climbing_relevance: EvaluationCriterion
    uncertainty_handling: EvaluationCriterion

    suggested_improvement: str

class CoachingQualityEvaluation(BaseModel):
    recommendation_evaluations: list[RecommendationEvaluation]

    overall_strengths: list[str]
    overall_weaknesses: list[str]

    overall_confidence: Literal["low", "medium", "high"]
    requires_human_review: bool