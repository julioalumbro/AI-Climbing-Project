
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RELATIVE_DIR = Path(
    "climbingTestDeepBentElbow"
) / "left_release_return" / "frames_332_373"

BASELINE_DIR = (
    PROJECT_ROOT / "phase05" / "comparison_outputs" / RELATIVE_DIR
)

EVALUATION_DIR = (
    PROJECT_ROOT / "phase05" / "evaluation_outputs" / RELATIVE_DIR
)

COACHING_DIR = (
    PROJECT_ROOT / "phase04" / "coaching_outputs" / RELATIVE_DIR
)

CRITERIA = [
    "evidence_accuracy",
    "reasoning_quality",
    "actionability",
    "climbing_relevance",
    "uncertainty_handling"
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def compare_scores(baseline, updated):
    baseline_items = {
        item["recommendation_index"]: item
        for item in baseline["evaluation"]["evaluation"]["recommendation_evaluations"]
    }

    updated_items = {
        item["recommendation_index"]: item
        for item in updated["evaluation"]["evaluation"]["recommendation_evaluations"]
    }

    if baseline_items.keys() != updated_items.keys():
        raise ValueError(
            "Baseline and updated evaluations contain different "
            "recommendation indices."
        )

    for index in sorted(baseline_items):
        old = baseline_items[index]
        new = updated_items[index]

        print(f"\n=== Recommendation {index + 1} ===")
        print(f"{'Criterion':<24} {'Baseline':>10} {'Updated':>10} {'Change':>10}")
        print("-" * 58)

        for criterion in CRITERIA:
            old_score = old[criterion]["score"]
            new_score = new[criterion]["score"]
            difference = new_score - old_score

            print(
                f"{criterion.replace('_', ' ').title():<24}"
                f"{old_score:>10}"
                f"{new_score:>10}"
                f"{difference:>+10}"
            )

        print("\nOriginal recommendation:")
        print(baseline["coaching"]["recommendations"][index]["suggested_adjustment"])

        print("\nUpdated recommendation:")
        print(updated["coaching"]["recommendations"][index]["suggested_adjustment"])

        print("\nEvaluator's suggested improvement:")
        print(new["suggested_improvement"])


def main():
    baseline = {
        "coaching": load_json(
            BASELINE_DIR / "baseline_coaching.json"
        ),
        "evaluation": load_json(
            BASELINE_DIR / "baseline_evaluation.json"
        )
    }

    updated = {
        "coaching": load_json(
            COACHING_DIR / "coaching_analysis_v2.json"
        ),
        "evaluation": load_json(
            EVALUATION_DIR / "coaching_quality_evaluation.json"
        )
    }

    compare_scores(baseline, updated)


if __name__ == "__main__":
    main()
