
from pydantic import ValidationError

from coaching_evaluation_schema import (
    CoachingQualityEvaluation
)


# confirm that a correctly structured coaching evaluation passes schema validation.
def test_valid_coaching_evaluation():
    criterion = {
        "score": 4,
        "reasoning": "The recommendation is mostly supported.",
        "issues": []
    }

    evaluation = {
        "recommendation_evaluations": [
            {
                "recommendation_index": 0,
                "evidence_accuracy": criterion,
                "reasoning_quality": criterion,
                "actionability": criterion,
                "climbing_relevance": criterion,
                "uncertainty_handling": criterion,
                "suggested_improvement": (
                    "Specify when the climber should adjust "
                    "their arm position."
                )
            }
        ],
        "overall_strengths": [
            "References measured joint movements."
        ],
        "overall_weaknesses": [
            "The adjustment lacks precise timing."
        ],
        "overall_confidence": "medium",
        "requires_human_review": False
    }

    result = CoachingQualityEvaluation.model_validate(evaluation)

    assert len(result.recommendation_evaluations) == 1
    assert result.recommendation_evaluations[0].actionability.score == 4

    print("Valid coaching evaluation schema test passed!")


# confirm that out-of-range scores fail schema validation.
def test_invalid_evaluation_score():
    criterion = {
        "score": 6,
        "reasoning": "Invalid score.",
        "issues": []
    }

    evaluation = {
        "recommendation_evaluations": [
            {
                "recommendation_index": 0,
                "evidence_accuracy": criterion,
                "reasoning_quality": criterion,
                "actionability": criterion,
                "climbing_relevance": criterion,
                "uncertainty_handling": criterion,
                "suggested_improvement": "Be more specific."
            }
        ],
        "overall_strengths": [],
        "overall_weaknesses": [],
        "overall_confidence": "medium",
        "requires_human_review": True
    }

    try:
        CoachingQualityEvaluation.model_validate(evaluation)
        assert False, "Expected validation to fail."
    except ValidationError:
        pass

    print("Invalid evaluation score test passed!")


if __name__ == "__main__":
    test_valid_coaching_evaluation()
    test_invalid_evaluation_score()
