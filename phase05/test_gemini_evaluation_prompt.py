# test to check that prompt includes eval criteria, safeguards, + evidence and coaching data
from gemini_evaluation_prompt import (
    build_gemini_evaluation_prompt
)


# verify that the generated prompt includes evidence, coaching text, and evaluation safeguards.
def test_gemini_evaluation_prompt():
    evidence = {
        "evaluations": [
            {
                "claim": "Left elbow angle change",
                "movement": "flexion",
                "angle_change_degrees": -34.53
            }
        ],
        "reconciliations": [
            {
                "claim": "Left elbow movement",
                "status": "supported"
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
                "uncertainty": (
                    "Hold positions and wall angle are unknown."
                )
            }
        ]
    }

    prompt = build_gemini_evaluation_prompt(
        evidence,
        coaching
    )

    assert "EVIDENCE ACCURACY" in prompt
    assert "REASONING QUALITY" in prompt
    assert "ACTIONABILITY" in prompt
    assert "CLIMBING RELEVANCE" in prompt
    assert "UNCERTAINTY HANDLING" in prompt

    assert "Left elbow angle change" in prompt
    assert "The climber flexed their left elbow." in prompt

    assert "Do not invent hold positions" in prompt
    assert "Do not assume the original coaching is correct" in prompt

    print("Gemini evaluation prompt test passed!")


if __name__ == "__main__":
    test_gemini_evaluation_prompt()
