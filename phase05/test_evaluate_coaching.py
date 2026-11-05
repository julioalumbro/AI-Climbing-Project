
from evaluate_coaching import (
    evaluate_evidence_alignment,
    evaluate_uncertainty_handling,
    extract_joint_movement_ground_truth,
    evaluate_joint_movement_descriptions
)


def test_evidence_alignment():
    """
    Verify that existing evidence references pass and
    nonexistent references are identified.
    """

    evidence = {
        "reconciliations": [
            {"claim": "Left elbow movement", "status": "supported"},
            {"claim": "Hip movement direction", "status": "supported"}
        ]
    }

    coaching = {
        "recommendations": [
            {
                "supporting_evidence": [
                    "Left elbow movement",
                    "Hip movement direction"
                ]
            },
            {
                "supporting_evidence": [
                    "Nonexistent movement claim"
                ]
            }
        ]
    }

    results = evaluate_evidence_alignment(evidence, coaching)

    assert results[0]["evidence_alignment_passed"] is True
    assert results[0]["missing_claims"] == []

    assert results[1]["evidence_alignment_passed"] is False
    assert results[1]["missing_claims"] == [
        "Nonexistent movement claim"
    ]

    print("Evidence alignment evaluation test passed!")

def test_evidence_reliability():
    """
    Verify that supported, contradicted, and unverified
    evidence references are classified correctly.
    """

    evidence = {
        "reconciliations": [
            {"claim": "Left elbow movement", "status": "supported"},
            {"claim": "Right elbow movement", "status": "contradicted"}
        ],
        "visual_observations": [
            {
                "claim": "Target foot foothold contact",
                "status": "not_independently_verified"
            }
        ]
    }

    coaching = {
        "recommendations": [
            {"supporting_evidence": ["Left elbow movement"]},
            {"supporting_evidence": ["Right elbow movement"]},
            {"supporting_evidence": ["Target foot foothold contact"]},
            {"supporting_evidence": ["Unknown claim"]}
        ]
    }

    results = evaluate_evidence_alignment(evidence, coaching)

    assert results[0]["evidence_alignment_passed"] is True
    assert results[0]["supported_claims"] == ["Left elbow movement"]

    assert results[1]["evidence_alignment_passed"] is False
    assert results[1]["contradicted_claims"] == ["Right elbow movement"]

    assert results[2]["evidence_alignment_passed"] is False
    assert results[2]["unverified_claims"] == [
        "Target foot foothold contact"
    ]

    assert results[3]["evidence_alignment_passed"] is False
    assert results[3]["missing_claims"] == ["Unknown claim"]

    print("Evidence reliability evaluation test passed!")



def test_uncertainty_handling():
    evidence = {
        "reconciliations": [
            {
                "claim": "Right elbow movement",
                "status": "contradicted"
            }
        ]
    }

    coaching = {
        "recommendations": [
            {
                "supporting_evidence": ["Right elbow movement"],
                "uncertainty": ""
            },
            {
                "supporting_evidence": ["Right elbow movement"],
                "uncertainty": (
                    "Gemini and MediaPipe disagree about "
                    "the right elbow movement."
                )
            }
        ]
    }

    results = evaluate_uncertainty_handling(evidence, coaching)

    assert results[0]["uncertainty_check_passed"] is False
    assert results[1]["uncertainty_check_passed"] is True

    print("Uncertainty handling evaluation test passed!")


def test_extract_joint_movement_ground_truth():
    evidence = {
        "evaluations": [
            {
                "claim": "Left elbow angle change",
                "first_angle": 106.58,
                "last_angle": 72.06,
                "angle_change_degrees": -34.53,
                "movement": "flexion"
            },
            {
                "claim": "Right elbow angle change",
                "first_angle": 122.79,
                "last_angle": 69.20,
                "angle_change_degrees": -53.59,
                "movement": "flexion"
            },
            {
                "claim": "Horizontal hip movement",
                "direction": "image-right"
            }
        ]
    }

    results = extract_joint_movement_ground_truth(evidence)

    assert results["left elbow"]["movement"] == "flexion"
    assert results["right elbow"]["movement"] == "flexion"

    assert results["left elbow"]["first_angle"] == 106.58
    assert results["left elbow"]["last_angle"] == 72.06

    assert "horizontal hip movement" not in results

    print("Joint movement extraction test passed!")


def test_joint_movement_descriptions():
    evidence = {
        "evaluations": [
            {
                "claim": "Left elbow angle change",
                "first_angle": 106.58,
                "last_angle": 72.06,
                "angle_change_degrees": -34.53,
                "movement": "flexion"
            },
            {
                "claim": "Right elbow angle change",
                "first_angle": 122.79,
                "last_angle": 69.20,
                "angle_change_degrees": -53.59,
                "movement": "flexion"
            }
        ]
    }

    coaching = {
        "recommendations": [
            {
                "observation": (
                    "The climber demonstrated left elbow flexion."
                )
            },
            {
                "observation": (
                    "The climber demonstrated right elbow extension."
                )
            },
            {
                "observation": (
                    "The climber shifted their hips toward image-right."
                )
            }
        ]
    }

    results = evaluate_joint_movement_descriptions(
        evidence,
        coaching
    )

    # Correct description
    assert results[0]["description_check_passed"] is True
    assert results[0]["checked_movement_count"] == 1

    # Contradictory description
    assert results[1]["description_check_passed"] is False
    assert len(results[1]["contradictions"]) == 1
    assert results[1]["contradictions"][0]["joint"] == "right elbow"

    # No joint movement description detected
    assert results[2]["checked_movement_count"] == 0

    print("Joint movement description test passed!")


if __name__ == "__main__":
    test_evidence_alignment()
    test_evidence_reliability()
    test_uncertainty_handling()
    test_extract_joint_movement_ground_truth()
    test_joint_movement_descriptions()
