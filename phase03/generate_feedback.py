# interpret and communicate features of phase01-02
# goal is to translate measurements into factual descriptions of climbing
# movements, w/out judging whether the technique is good/bad yet
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

# loading the feature dataset dynamically
# using a path relative to the script
FEATURE_FILE = PROJECT_DIR / 'phase02' / 'movement_features.json'

# function to save feedback
# use ensure_ascii = false so deg symbol is stored directly instead of being escaped
# encoding = utf-8 converts chars into bytes --> ensure_ascii = false tells json.dump to keep
# non-ASCII characters rather than escaping them
def saveMovementFeedback(feedback_records, output_path):
    output = {
        "total_movements": len(feedback_records),
        "feedback": feedback_records,
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4, ensure_ascii=False)

    print(f"\nMovement feedback saved to: {output_path}")

def loadMovementFeatures(filepath):
    with open(filepath) as file:
        data = json.load(file)

    return data["features"]

# uses actual direciton of hip velocity instead of relying soley on the diff between the two avgs
def generateMovementSummary(movement):
    foot = movement["foot"]
    movement_type = movement.get("movement_type")

    left_change = movement.get("left_elbow_change")
    right_change = movement.get("right_elbow_change")


    hip_before = movement.get("hip_velocity_before")
    hip_during = movement.get("hip_velocity_during")

    summary = []

    # movement identification
    if movement_type is not None:
        summary.append(
            f"Verified {movement_type.replace('_', ' ' )} "
            f"involving the {foot} foot."
        )
    else:
        summary.append(
            f"Detected candidate moving involving the {foot} foot."
        )

    # elbow movement
    for arm, change in [
        ("left", left_change),
        ("right", right_change)
    ]:
        if change is None:
            summary.append(
                f"{arm.capitalize()} elbow movement could not "
                f"be reliably measured."
            )
        elif change < 0:
            summary.append(
                f"The {arm} elbow became more bent, with "
                f"an average angle decrease of {abs(change):.2f}°."
            )
        elif change > 0:
            summary.append(
                f"The {arm} elbow became straighter, with a "
                f"{abs(change):.2f}° increase in average angle."
            )
        else:
            summary.append(
                f"The {arm} elbow's average angle did not change."

            )

    # hip movement
    if hip_before is not None and hip_during is not None:
        if hip_before >= 0 > hip_during:
            summary.append(
                "Average hip movement transitioned from downward "
                "or stationary to upward in image coordinates"
            )
        elif hip_before < 0 and hip_during < 0:
            summary.append(
                "The hips continued moving upward on average "
                "during the foot movement."
            )
        elif hip_before > 0 and hip_during > 0:
            summary.append(
                "The hips continue moving downward on average "
                "during the foot movement."
            )
        else:
            summary.append(
                "Average hip vertical movement remained near zero."
            )
    else:
        summary.append(
            "Hip movement could not be reliably measured."
        )
    return " ".join(summary)
# function that converts elbow measurements into structured observations
def generateElbowObservations(movement):
    observations = []

    for arm in ["left", "right"]:
        change = movement.get(f"{arm}_elbow_change")

        if change is None:
            observation = "unavailable"
        elif change < 0:
            observation = "increased_flexion"
        elif change > 0:
            observation = "increased_extension"
        else:
            observation = "no_average_change"

        observations.append({
            "body_part": f"{arm}_elbow",
            "observation": observation,
            "angle_change": change,
        })
    return observations

def generateHipObservations(movement):
    before = movement.get("hip_velocity_before")
    during = movement.get("hip_velocity_during")

    if before is None or during is None:
        return {
            "body_part": "hips",
            "observation": "unavailable",
            "velocity_before": before,
            "velocity_during": during,
            "velocity_change": None
        }

    def getDirection(velocity):
        if velocity < 0:
            return "upward"
        elif velocity > 0:
            return "downward"
        else:

            return "stationary"

    before_direction = getDirection(before)
    during_direciton = getDirection(during)

    if before_direction == during_direciton:
        observation = f"continued_{during_direciton}"
    else:
        observation = f"{before_direction}_to_{during_direciton}"
    return {
        "body_part": "hips",
        "observation": observation,
        "velocity_before": before,
        "velocity_during": during,
        "velocity_change": during - before
    }

# combination that brings movement summ, elbow obs, amd hip obs together:
def buildMovementFeedback(movement):
    return {
        "source": movement["source"],
        "foot": movement["foot"],
        "movement_type": movement.get("movement_type"),

        "frame_ranges": {
            "first_event": {
                "start": movement["first_event_start"],
                "end": movement["first_event_end"]
            },
            "second_event": {
                "start": movement["second_event_start"],
                "end": movement["second_event_end"]
            }
        },

        "summary": generateMovementSummary(movement),

        "observations": {
            "elbows": generateElbowObservations(movement),
            "hips": generateHipObservations(movement)
        }
    }

if __name__ == "__main__":
    movements = loadMovementFeatures(FEATURE_FILE)

    print("\n--- Phase03: Movement Feedback ---")
    print("Total movements:", len(movements))

    print("\n--- Generated Movement Summaries ---")

    for movement in movements:
        summary = generateMovementSummary(movement)

        print(
            f"\n{movement['source']} | "
            f"{movement['foot']} foot"
        )

        print(summary)
    # observation testing
    # elbows
    print("\n--- Structured Elbow Observations ---")

    for movement in movements:
        print(
            f"\n{movement['source']} | "
            f"{movement['foot']} foot"
        )

        for observation in generateElbowObservations(movement):
            print(observation)
    # hips
    print("\n--- Structured Hip Observations ---")

    for movement in movements:
        print(
            f"\n{movement['source']} | "
            f"{movement['foot']} foot"
        )

        print(generateHipObservations(movement))
    # unified movement feedback test
    print("\n--- Unified Movement Feedback ---")

    for movement in movements:
        feedback = buildMovementFeedback(movement)
        print(json.dumps(feedback, indent=4))

    # saving feedback into json file
    if __name__ == "__main__":
        movements = loadMovementFeatures(FEATURE_FILE)

        feedback_records = [
            buildMovementFeedback(movement)
            for movement in movements
        ]

        OUTPUT_FILE = BASE_DIR / "movement_feedback.json"

        saveMovementFeedback(feedback_records, OUTPUT_FILE)

        print("\n--- Phase 03 Feedback Summary ---")
        print("Total movements:", len(feedback_records))

        for record in feedback_records:
            print(
                f"{record['source']} | "
                f"{record['foot']} foot | "
                f"{record['movement_type']}"
            )