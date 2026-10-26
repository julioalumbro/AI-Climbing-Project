import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

ANALYSIS_FILES = sorted(
    path.name
    for path in BASE_DIR.glob("*_movement_analysis.json")
)

if not ANALYSIS_FILES:
    raise FileNotFoundError(
        "No movement analysis files found in phase02"
    )

def loadMovementAnalysis(filenames):
    if filenames is None:
        print("No filenames provided")
        return

    movements = []
    for filename in filenames:

        with open(filename, "r") as f:
            data = json.load(f)

        for movement in data["movement_analyses"]:
            movements.append({
                "source": data["source_video_metrics"],
                **movement
            })
    return movements

def extractMovementFeatures(movement):
    elbow_changes = movement["elbow_changes"]
    hip_movement = movement["hip_movement"]


    hip_before = getReliableAverage(
        hip_movement["before"],
        "average_velocity"
    )

    hip_during = getReliableAverage(
        hip_movement["during"],
        "average_velocity"
    )
    return {
        "source": movement["source"],
        "foot": movement["foot"],
        "movement_type": movement["verified_movement_type"],

        "left_elbow_change": getReliableElbowChange(
            movement, "left"
        ),

        "right_elbow_change": getReliableElbowChange(
            movement, "right"
        ),

        "hip_velocity_before": hip_before,
        "hip_velocity_during": hip_during,

        "hip_velocity_change": (
            hip_during - hip_before
            if hip_before is not None and hip_during is not None
            else None
        ),
        "first_event_start": movement["first_event"]["start_frame"],
        "first_event_end": movement["first_event"]["end_frame"],
        "second_event_start": movement["second_event"]["start_frame"],
        "second_event_end": movement["second_event"]["end_frame"],
    }
# helper func: only returns a measurement when its tracking coverage meets a min of 80
def getReliableAverage(metrics, key, min_coverage=80):
    if metrics is None:
        return None

    coverage = metrics.get("tracking_coverage", 0)
    value = metrics.get(key)

    if coverage < min_coverage or value is None:
        return None

    return value

def getReliableElbowChange(movement, arm, min_coverage = 80):
    positioning = movement["elbow_positioning"][arm]
    changes = movement["elbow_changes"].get(arm)

    if changes is None:
        return None

    for period in ("before", "during", "after"):
        coverage = positioning[period]["tracking_coverage"]

        if coverage < min_coverage:
            return None

    return changes["overall_change"]

def validateMovementFeatures(features):
    required_fields = [
        "source",
        "foot",
        "movement_type",
        "first_event_start",
        "first_event_end",
        "second_event_start",
        "second_event_end",
        "left_elbow_change",
        "right_elbow_change",
        "hip_velocity_before",
        "hip_velocity_during",
        "hip_velocity_change"
    ]

    missing_fields = [
        field for field in required_fields
        if field not in features
    ]

    if missing_fields:
        raise ValueError(
            f"Missing required fields: {missing_fields}"
        )

    if features["foot"] not in ("left", "right"):
        raise ValueError("Invalid foot value")

    if features["first_event_start"] > features["first_event_end"]:
        raise ValueError("Invalid first event frame range")

    if features["second_event_start"] > features["second_event_end"]:
        raise ValueError("Invalid second event frame range")

    if features["first_event_end"] >= features["second_event_start"]:
        raise ValueError("Movement events overlap or are out of order")

    return True

if __name__ == "__main__":
    movements = loadMovementAnalysis(ANALYSIS_FILES)
    feature_dataset = []

    for movement in movements:
        features = extractMovementFeatures(movement)

        validateMovementFeatures(features)

        feature_dataset.append(features)

    print(
        f"\nValidated {len(feature_dataset)} movement records."
    )

    output_path = BASE_DIR / "movement_features.json"


    with open(output_path, "w") as file:
        json.dump(
            {
                "total_movements": len(feature_dataset),
                "features": feature_dataset
            },
            file,
            indent=4
        )
    print(f"\nFeature dataset saved to: {output_path}")
    print("\n--- Cross-Video Movement Comparison---")
    print("Total Movements:", len(movements))

    for movement in movements:
        print(
            f"{movement['source']} | "
            f"{movement['foot']} | "
            f"{movement['verified_movement_type']}"
        )
    print("\n--- Movement Feature Comparison ---")

    for movement in movements:
        features = extractMovementFeatures(movement)

        print(
            f"\n{features['source']} | "
            f"{features['foot']} foot | "
            f"{features['movement_type']}"
        )

        for key, value in features.items():
            if key in ("source", "foot", "movement_type"):
                continue

            if value is None:
                print(f"{key}: Unavailable")
            else:
                print(f"{key}: {value:.4f}")