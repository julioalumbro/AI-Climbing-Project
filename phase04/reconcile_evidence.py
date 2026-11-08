# compare measured joint and body movement with visual observations to assess evidence support.
# used to help determine if observations from gemini are supported by measureable movement data

import json
from pathlib import Path
import argparse

# ============================================================
# 1. file paths and version configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# change to 1 or 2 to select an analysis version.
ANALYSIS_VERSION = 2

if ANALYSIS_VERSION == 1:
    combined_filename = "combined_movement_analysis.json"
    evidence_filename = "evidence_report.json"

elif ANALYSIS_VERSION == 2:
    combined_filename = "combined_movement_analysis_v2.json"
    evidence_filename = "evidence_report_v2.json"

else:
    raise ValueError(
        f"Unsupported analysis version: {ANALYSIS_VERSION}"
    )

parser = argparse.ArgumentParser(
    description="Reconcile MediaPipe and Gemini movement evidence."
)

parser.add_argument(
    "--combined",
    type=Path,
    default=BASE_DIR / combined_filename,
    help="Path to the combined movement analysis JSON."
)

parser.add_argument(
    "--output",
    type=Path,
    default=BASE_DIR / evidence_filename,
    help="Path for the generated evidence report."
)

args = parser.parse_args()

COMBINED_PATH = args.combined
OUTPUT_PATH = args.output

print(f"Analysis version: {ANALYSIS_VERSION}")
print(f"Combined input: {COMBINED_PATH}")
print(f"Evidence output: {OUTPUT_PATH}")

if not COMBINED_PATH.is_file():
    raise FileNotFoundError(
        f"Combined analysis not found: {COMBINED_PATH}\n"
        "Make sure you have generated the selected "
        "analysis version first."
    )

with open(COMBINED_PATH, "r", encoding="utf-8") as file:
    combined_data = json.load(file)

frames = sorted(
    combined_data["frames"],
    key=lambda record: record["frame"]
)

# inspect knee-angle measurements for meaningful flexion changes.
def evaluate_knee_flexion(frames, side):
    """
    Measure knee-angle changes for the specified leg.

    Decreasing angle = flexion
    Increasing angle = extension
    """

    if side not in ("left", "right"):
        raise ValueError("Side must be 'left' or 'right'.")

    if len(frames) < 2:
        return {
            "claim": f"{side.capitalize()} knee angle change",
            "status": "insufficient_evidence",
            "reason": "At least two frames are required."
        }

    first = frames[0]["mediapipe"]
    last = frames[-1]["mediapipe"]

    valid_key = f"{side}_leg_tracking_valid"
    angle_key = f"{side}_knee_angle"

    if not (
        first.get(valid_key, False)
        and last.get(valid_key, False)
    ):
        return {
            "claim": f"{side.capitalize()} knee angle change",
            "status": "insufficient_evidence",
            "reason": "Leg tracking is invalid."
        }

    first_angle = first.get(angle_key)
    last_angle = last.get(angle_key)

    if first_angle is None or last_angle is None:
        return {
            "claim": f"{side.capitalize()} knee angle change",
            "status": "insufficient_evidence",
            "reason": "Knee measurements are missing."
        }

    angle_change = last_angle - first_angle

    threshold = 15.0  # temporary heuristic

    if angle_change < -threshold:
        movement = "flexion"
    elif angle_change > threshold:
        movement = "extension"
    else:
        movement = "minimal_angle_change"

    return {
        "claim": f"{side.capitalize()} knee angle change",
        "status": "measured",
        "first_frame": frames[0]["frame"],
        "last_frame": frames[-1]["frame"],
        "first_angle": round(first_angle, 2),
        "last_angle": round(last_angle, 2),
        "angle_change_degrees": round(angle_change, 2),
        "movement": movement
    }


# compare knee movement claims with available measurements.
def reconcile_knee_movement(frames):
    """
    Compare Gemini's structured target-knee movement
    against MediaPipe's measured knee-angle change.
    """

    side = combined_data["target_foot"]

    knee_result = evaluate_knee_flexion(frames, side)

    if knee_result["status"] != "measured":
        return {
            "claim": f"{side.capitalize()} knee movement",
            "status": "insufficient_evidence",
            "reason": "MediaPipe knee measurement unavailable."
        }

    gemini_movement = combined_data.get(
        "gemini_target_knee_movement"
    )

    mediapipe_movement = knee_result["movement"]

    if gemini_movement is None:
        status = "inconclusive"
        reason = "Structured Gemini knee movement unavailable."

    elif gemini_movement == "unclear":
        status = "inconclusive"
        reason = "Gemini could not determine knee movement."

    elif gemini_movement == mediapipe_movement:
        status = "supported"
        reason = "Both sources report the same knee movement."

    else:
        status = "contradicted"
        reason = "Sources report different knee movements."

    return {
        "claim": f"{side.capitalize()} knee movement",
        "status": status,
        "gemini_movement": gemini_movement,
        "mediapipe_movement": mediapipe_movement,
        "angle_change_degrees": knee_result[
            "angle_change_degrees"
        ],
        "reason": reason
    }



# summarize hip movement evidence from tracked frames.
def evaluate_hip_movement(frames):
    """
    Measure horizontal hip displacement and classify
    its direction using normalized image coordinates.
    """

    if len(frames) < 2:
        return {
            "claim": "Horizontal hip movement",
            "status": "insufficient_evidence",
            "reason": "At least two frames are required."
        }

    first = frames[0]["mediapipe"]
    last = frames[-1]["mediapipe"]

    first_hip_x = first.get("hip_x")
    last_hip_x = last.get("hip_x")

    if first_hip_x is None or last_hip_x is None:
        return {
            "claim": "Horizontal hip movement",
            "status": "insufficient_evidence",
            "reason": "Hip coordinates are missing."
        }

    displacement = last_hip_x - first_hip_x

    threshold = 0.05  # temporary normalized-coordinate threshold

    if displacement > threshold:
        direction = "image-right"
    elif displacement < -threshold:
        direction = "image-left"
    else:
        direction = "minimal_horizontal_change"

    return {
        "claim": "Horizontal hip movement",
        "status": "measured",
        "first_frame": frames[0]["frame"],
        "last_frame": frames[-1]["frame"],
        "first_hip_x": round(first_hip_x, 4),
        "last_hip_x": round(last_hip_x, 4),
        "horizontal_displacement": round(displacement, 4),
        "direction": direction
    }



# check whether measured hip motion supports a directional observation.
def reconcile_hip_direction(frames):
    """
    Compare Gemini's structured hip direction
    with MediaPipe's measured direction.
    """

    hip_result = evaluate_hip_movement(frames)

    if hip_result["status"] != "measured":
        return {
            "claim": "Hip movement direction",
            "status": "insufficient_evidence",
            "reason": "MediaPipe hip measurement unavailable."
        }

    gemini_direction = combined_data.get(
        "gemini_hip_horizontal_direction"
    )

    mediapipe_direction = hip_result["direction"]

    if gemini_direction is None:
        status = "inconclusive"
        reason = "Structured Gemini direction unavailable."

    elif gemini_direction == "unclear":
        status = "inconclusive"
        reason = "Gemini could not determine hip direction."

    elif gemini_direction == mediapipe_direction:
        status = "supported"
        reason = "Both sources report the same direction."

    else:
        status = "contradicted"
        reason = "Sources report different directions."

    return {
        "claim": "Hip movement direction",
        "status": status,
        "gemini_direction": gemini_direction,
        "mediapipe_direction": mediapipe_direction,
        "horizontal_displacement": hip_result[
            "horizontal_displacement"
        ],
        "reason": reason
    }






# evaluate elbow-angle changes for the requested arm.
def evaluate_elbow_flexion(frames, side):
    """
    Measure elbow-angle changes for either arm.

    A decreasing angle indicates flexion.
    An increasing angle indicates extension.
    """

    if side not in ("left", "right"):
        raise ValueError("Side must be 'left' or 'right'.")

    if len(frames) < 2:
        return {
            "claim": f"{side.capitalize()} elbow angle change",
            "status": "insufficient_evidence",
            "reason": "At least two frames are required."
        }

    first = frames[0]["mediapipe"]
    last = frames[-1]["mediapipe"]

    valid_key = f"{side}_arm_valid"
    angle_key = f"{side}_elbow_angle"

    if not (
        first.get(valid_key, False)
        and last.get(valid_key, False)
    ):
        return {
            "claim": f"{side.capitalize()} elbow angle change",
            "status": "insufficient_evidence",
            "reason": "Arm tracking is invalid."
        }

    first_angle = first.get(angle_key)
    last_angle = last.get(angle_key)

    if first_angle is None or last_angle is None:
        return {
            "claim": f"{side.capitalize()} elbow angle change",
            "status": "insufficient_evidence",
            "reason": "Elbow measurements are missing."
        }

    angle_change = last_angle - first_angle

    threshold = 15.0  # temporary heuristic

    if angle_change < -threshold:
        movement = "flexion"
    elif angle_change > threshold:
        movement = "extension"
    else:
        movement = "minimal_angle_change"

    return {
        "claim": f"{side.capitalize()} elbow angle change",
        "status": "measured",
        "first_frame": frames[0]["frame"],
        "last_frame": frames[-1]["frame"],
        "first_angle": round(first_angle, 2),
        "last_angle": round(last_angle, 2),
        "angle_change_degrees": round(angle_change, 2),
        "movement": movement
    }


# check an elbow movement observation against joint measurements.
def reconcile_elbow_movement(frames, side):
    """
    Compare Gemini's structured elbow movement
    against MediaPipe's measured elbow-angle change.
    """

    if side not in ("left", "right"):
        raise ValueError("Side must be 'left' or 'right'.")

    elbow_result = evaluate_elbow_flexion(frames, side)

    if elbow_result["status"] != "measured":
        return {
            "claim": f"{side.capitalize()} elbow movement",
            "status": "insufficient_evidence",
            "reason": "MediaPipe elbow measurement unavailable."
        }

    gemini_movement = combined_data.get(
        f"gemini_{side}_elbow_movement"
    )

    mediapipe_movement = elbow_result["movement"]

    if gemini_movement is None:
        status = "inconclusive"
        reason = "Structured Gemini elbow movement unavailable."

    elif gemini_movement == "unclear":
        status = "inconclusive"
        reason = "Gemini could not determine elbow movement."

    elif gemini_movement == mediapipe_movement:
        status = "supported"
        reason = "Both sources report the same elbow movement."

    else:
        status = "contradicted"
        reason = "Sources report different elbow movements."

    return {
        "claim": f"{side.capitalize()} elbow movement",
        "status": status,
        "gemini_movement": gemini_movement,
        "mediapipe_movement": mediapipe_movement,
        "angle_change_degrees": elbow_result[
            "angle_change_degrees"
        ],
        "reason": reason
    }




# compile reconciled movement evidence into a structured report.
def build_evidence_report(frames):
    """
    Build a structured report containing:
    1. MediaPipe measurements
    2. Gemini visual observations
    3. Cross-model reconciliation results
    """

    evaluations = [
        evaluate_knee_flexion(
            frames,
            combined_data["target_foot"]
        ),
        evaluate_hip_movement(frames),
        evaluate_elbow_flexion(frames, "left"),
        evaluate_elbow_flexion(frames, "right")
    ]

    reconciliations = [
        reconcile_hip_direction(frames),
        reconcile_knee_movement(frames),
        reconcile_elbow_movement(frames, "left"),
        reconcile_elbow_movement(frames, "right")
    ]

    return {
        "schema_version": combined_data.get("schema_version", 1),
        "movement_type": combined_data["movement_type"],
        "video_file": combined_data["video_file"],
        "analyzed_frames": [
            record["frame"] for record in frames
        ],
        "evaluations": evaluations,
        "visual_observations": collect_visual_observations(frames),
        "reconciliations": reconciliations,
        "gemini_movement_interpretation": combined_data[
            "gemini_movement_interpretation"
        ],
        "gemini_uncertainties": combined_data[
            "gemini_uncertainties"
        ]
    }



# gather model observations that need measurement reconciliation.
def collect_visual_observations(frames):
    """
    Collect Gemini's per-frame foothold-contact observations
    without treating them as independently verified facts.
    """

    observations = []

    for record in frames:
        gemini = record["gemini"]

        observations.append({
            "frame": record["frame"],
            "claim": "Target foot foothold contact",
            "observed_value": gemini["foothold_contact"],
            "foot_position": gemini["foot_position"],
            "source": "gemini",
            "status": "not_independently_verified"
        })

    return observations

# want to verify that reconciliation logic handles agreement and disagreements
# between mediapipe and gemini
# exercise evidence checks on representative frame data.
def test_reconciliation_logic(frames):
    """
    Test reconciliation using temporary Gemini values.
    Does not modify saved JSON files.
    """

    test_cases = [
        {
            "name": "Matching knee movement",
            "field": "gemini_target_knee_movement",
            "value": "flexion",
            "expected": "supported",
            "evaluate": lambda: reconcile_knee_movement(frames)
        },
        {
            "name": "Conflicting knee movement",
            "field": "gemini_target_knee_movement",
            "value": "extension",
            "expected": "contradicted",
            "evaluate": lambda: reconcile_knee_movement(frames)
        },
        {
            "name": "Matching left elbow movement",
            "field": "gemini_left_elbow_movement",
            "value": "flexion",
            "expected": "supported",
            "evaluate": lambda: reconcile_elbow_movement(
                frames, "left"
            )
        },
        {
            "name": "Matching hip direction",
            "field": "gemini_hip_horizontal_direction",
            "value": "image-right",
            "expected": "supported",
            "evaluate": lambda: reconcile_hip_direction(frames)
        }
    ]

    for test in test_cases:
        field = test["field"]
        original_value = combined_data.get(field)

        try:
            combined_data[field] = test["value"]
            result = test["evaluate"]()

            assert result["status"] == test["expected"], (
                f"Expected {test['expected']}, "
                f"got {result['status']}"
            )

            print(f"PASS: {test['name']}")

        finally:
            if original_value is None:
                combined_data.pop(field, None)
            else:
                combined_data[field] = original_value



result = evaluate_knee_flexion(frames, combined_data["target_foot"])

print("\nKnee Flexion Evidence Evaluation:")
print(json.dumps(result, indent=4))

hip_result = evaluate_hip_movement(frames)

print("\nHip Movement Evidence Evaluation:")
print(json.dumps(hip_result, indent=4))

for side in ("left", "right"):
    elbow_result = evaluate_elbow_flexion(frames, side)

    print(f"\n{side.capitalize()} Elbow Evidence Evaluation:")
    print(json.dumps(elbow_result, indent=4))

# saves report

report = build_evidence_report(frames)

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=4, ensure_ascii=False)

print(f"\nEvidence report saved to: {OUTPUT_PATH}")

print("\nEvidence Summary:")

for evaluation in report["evaluations"]:
    print(
        f"{evaluation.get('claim', 'Unknown claim')}: "
        f"{evaluation['status']}"
    )

print("\nGemini Visual Observations:")

for observation in report["visual_observations"]:
    print(
        f"Frame {observation['frame']}: "
        f"Foothold contact = {observation['observed_value']} "
        f"({observation['status']})"
    )


print("\nCross-Model Reconciliation:")

for reconciliation in report["reconciliations"]:
    print(
        f"{reconciliation['claim']}: "
        f"{reconciliation['status']}"
    )
    print(f"Reason: {reconciliation['reason']}")

# legacy reconciliation tests depend on specific mediapipe measurements.
# do not run them automatically on every video.
RUN_LEGACY_TESTS = False

if RUN_LEGACY_TESTS:
    print("\nLegacy Reconciliation Tests:")
    test_reconciliation_logic(frames)






