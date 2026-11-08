
import json
from unittest.mock import MagicMock

from gemini_coaching_evaluator import (
    generate_gemini_coaching_evaluation,
    calculate_evaluation_fingerprint
)

# validate the evaluator response handling with a mocked model result.
def test_mock_gemini_evaluation():
    criterion = {
        "score": 4,
        "reasoning": "The recommendation is mostly supported.",
        "issues": []
    }

    mock_evaluation = {
        "recommendation_evaluations": [
            {
                "recommendation_index": 0,
                "evidence_accuracy": criterion,
                "reasoning_quality": criterion,
                "actionability": criterion,
                "climbing_relevance": criterion,
                "uncertainty_handling": criterion,
                "suggested_improvement": (
                    "Specify when to adjust arm position."
                )
            }
        ],
        "overall_strengths": [
            "Uses measured movement evidence."
        ],
        "overall_weaknesses": [
            "The proposed adjustment needs more detail."
        ],
        "overall_confidence": "medium",
        "requires_human_review": False
    }

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value.text = (
        json.dumps(mock_evaluation)
    )

    evidence = {
        "evaluations": [
            {
                "claim": "Left elbow angle change",
                "movement": "flexion"
            }
        ]
    }

    coaching = {
        "recommendations": [
            {
                "observation": "The climber flexed their left elbow.",
                "suggested_adjustment": (
                    "Try keeping the left arm more extended."
                ),
                "supporting_evidence": ["Left elbow movement"],
                "uncertainty": "Hold positions are unknown."
            }
        ]
    }

    result = generate_gemini_coaching_evaluation(
        evidence,
        coaching,
        client=mock_client
    )

    assert len(result.recommendation_evaluations) == 1

    assert (
        result.recommendation_evaluations[0]
        .actionability.score == 4
    )

    mock_client.models.generate_content.assert_called_once()

    print("Mock Gemini coaching evaluation test passed!")


# verify that the cache fingerprint reflects evaluation inputs.
def test_evaluation_fingerprint():
    evidence = {
        "evaluations": [
            {"claim": "Left elbow movement", "status": "measured"}
        ]
    }

    coaching = {
        "recommendations": [
            {"observation": "Left elbow flexion occurred."}
        ]
    }

    original = calculate_evaluation_fingerprint(
        evidence, coaching
    )

    repeated = calculate_evaluation_fingerprint(
        evidence, coaching
    )

    assert original == repeated

    changed_coaching = {
        "recommendations": [
            {"observation": "Left elbow extension occurred."}
        ]
    }

    changed = calculate_evaluation_fingerprint(
        evidence, changed_coaching
    )

    assert original != changed

    changed_version = calculate_evaluation_fingerprint(
        evidence,
        coaching,
        evaluator_version="2"
    )

    assert original != changed_version

    print("Evaluation fingerprint test passed!")


if __name__ == "__main__":
    test_mock_gemini_evaluation()
    test_evaluation_fingerprint()
