
import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv

from evaluate_coaching import (
    evaluate_evidence_alignment,
    evaluate_uncertainty_handling,
    evaluate_joint_movement_descriptions
)

from gemini_coaching_evaluator import (
    generate_gemini_coaching_evaluation,
    calculate_evaluation_fingerprint,
    load_cached_evaluation,
    save_cached_evaluation
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

EVIDENCE_PATH = (
    PROJECT_ROOT / "phase04" / "evidence_report_v2.json"
)

COACHING_PATH = (
    PROJECT_ROOT
    / "phase04"
    / "coaching_outputs"
    / "climbingTestDeepBentElbow"
    / "left_release_return"
    / "frames_332_373"
    / "coaching_analysis_v2.json"
)

CACHE_PATH = (
    PROJECT_ROOT
    / "phase05"
    / "evaluation_outputs"
    / "climbingTestDeepBentElbow"
    / "left_release_return"
    / "frames_332_373"
    / "coaching_quality_evaluation.json"
)

MODEL_NAME = "gemini-3.8-flash"
EVALUATOR_VERSION = "1"


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def run_local_evaluations(evidence, coaching):
    """
    Run deterministic coaching evaluation checks.
    These do not call Gemini.
    """

    alignment = evaluate_evidence_alignment(
        evidence, coaching
    )

    uncertainty = evaluate_uncertainty_handling(
        evidence, coaching
    )

    descriptions = evaluate_joint_movement_descriptions(
        evidence, coaching
    )

    print("\n=== Local Coaching Evaluation ===")

    for index in range(len(coaching["recommendations"])):
        print(f"\nRecommendation {index + 1}")

        print(
            "Evidence alignment:",
            alignment[index]["evidence_alignment_passed"]
        )

        print(
            "Uncertainty handling:",
            uncertainty[index]["uncertainty_check_passed"]
        )

        print(
            "Description consistency:",
            descriptions[index]["description_check_passed"]
        )

        print(
            "Descriptions checked:",
            descriptions[index]["checked_movement_count"]
        )

        print(
            "Contradictions:",
            descriptions[index]["contradictions"]
        )


def run_gemini_evaluation(
    evidence,
    coaching,
    *,
    generate=False,
    cache_path=CACHE_PATH
):
    """
    Load a cached evaluation or explicitly generate one.
    """

    fingerprint = calculate_evaluation_fingerprint(
        evidence,
        coaching,
        model_name=MODEL_NAME,
        evaluator_version=EVALUATOR_VERSION
    )

    cached = load_cached_evaluation(
        cache_path,
        fingerprint
    )

    if cached is not None:
        print("\nLoaded verified Gemini evaluation from cache.")
        return cached

    if not generate:
        print(
            "\nNo valid Gemini evaluation cache found. "
            "Use --generate to request one."
        )
        return None

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set."
        )

    from google import genai

    client = genai.Client(api_key=api_key)

    print("\nRequesting Gemini coaching evaluation...")

    evaluation = generate_gemini_coaching_evaluation(
        evidence,
        coaching,
        client=client,
        model_name=MODEL_NAME
    )

    save_cached_evaluation(
        cache_path,
        fingerprint,
        evaluation
    )

    print("Gemini evaluation validated and cached.")

    return evaluation


def print_detailed_gemini_evaluation(evaluation):
    """
    Print detailed Gemini evaluation results, including
    scores, reasoning, issues, and suggested improvements.

    Uses an existing validated evaluation.
    Does not make API calls.
    """

    criteria = [
        ("Evidence Accuracy", "evidence_accuracy"),
        ("Reasoning Quality", "reasoning_quality"),
        ("Actionability", "actionability"),
        ("Climbing Relevance", "climbing_relevance"),
        ("Uncertainty Handling", "uncertainty_handling")
    ]

    print("\n=== Detailed Gemini Coaching Evaluation ===")

    for item in evaluation.recommendation_evaluations:
        print(f"\n{'=' * 55}")
        print(f"RECOMMENDATION {item.recommendation_index + 1}")
        print(f"{'=' * 55}")

        for label, attribute in criteria:
            criterion = getattr(item, attribute)

            print(f"\n{label.upper()}: {criterion.score}/5")
            print(f"Reasoning: {criterion.reasoning}")

            if criterion.issues:
                print("Issues:")

                for issue in criterion.issues:
                    print(f"  - {issue}")
            else:
                print("Issues: None identified")

        print("\nSUGGESTED IMPROVEMENT:")
        print(item.suggested_improvement)

    print(f"\n{'=' * 55}")
    print("OVERALL EVALUATION")
    print(f"{'=' * 55}")

    print("\nStrengths:")
    for strength in evaluation.overall_strengths:
        print(f"  - {strength}")

    print("\nWeaknesses:")
    for weakness in evaluation.overall_weaknesses:
        print(f"  - {weakness}")

    print(
        "\nOverall confidence:",
        evaluation.overall_confidence
    )

    print(
        "Requires human review:",
        evaluation.requires_human_review
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate AI climbing coaching quality."
    )

    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate a Gemini evaluation if no valid cache exists."
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=EVIDENCE_PATH,
        help="Path to the evidence report JSON."
    )

    parser.add_argument(
        "--coaching",
        type=Path,
        default=COACHING_PATH,
        help="Path to the coaching analysis JSON."
    )

    parser.add_argument(
        "--cache",
        type=Path,
        default=CACHE_PATH,
        help="Path to the Gemini evaluation cache JSON."
    )

    args = parser.parse_args()

    evidence = load_json(args.evidence)
    coaching = load_json(args.coaching)

    run_local_evaluations(evidence, coaching)

    evaluation = run_gemini_evaluation(
        evidence,
        coaching,
        generate=args.generate,
        cache_path=args.cache
    )

    if evaluation is not None:
        print_detailed_gemini_evaluation(evaluation)


if __name__ == "__main__":
    main()
