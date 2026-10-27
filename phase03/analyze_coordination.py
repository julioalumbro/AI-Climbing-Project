import json
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent
FEEDBACK_FILE = BASE_DIR / "movement_feedback.json"

def loadMovementFeedback(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data["feedback"]

# function checks two things: (1) is the elbow flexed? (2) were the hips moving at an upward motion
def analyzeElbowHipCoordination(movement):
    elbows = movement["observations"]["elbows"]
    hips = movement["observations"]["hips"]

    hip_observation = hips["observation"]

    upward_patterns = {
        "downward_to_upward",
        "stationary_to_upward",
        "continued_upward"
    }

    hip_upward = hip_observation in upward_patterns

    flexing_arms = [
        elbow["body_part"]
        for elbow in elbows
        if elbow["observation"] == "increased_flexion"
    ]

    if hip_observation == "unavailable":
        pattern = "insufficient_hip_data"
    elif hip_upward:
        pattern = "elbow_flexion_with_upward_hip_movement"
    else:
        pattern = "elbow_flexion_without_upward_hip_movement"

    return {
        "pattern": pattern,
        "flexing_arms": flexing_arms,
        "hip_observation": hip_observation
    }
# use default dict list to initialize a key before appending vals
# python automatically creates an empty list when trying to access a missing key
# as more videos are analyzed, new classifications can be added automatically
def compareCoordinationPatterns(movements):
    comparisons = defaultdict(list)

    for movement in movements:
        movement_type = movement.get("movement_type") or "unverified"
        coordination = analyzeElbowHipCoordination(movement)

        comparisons[movement_type].append({
            "source": movement["source"],
            "foot": movement["foot"],
            "pattern": coordination["pattern"],
            "flexing_arms": coordination["flexing_arms"],
            "hip_observation": coordination["hip_observation"],
        })

    return dict(comparisons)
# load phase02 movement analyses and match them onto phase03 feedback records
def loadDetailedMovementAnalyses(directory):
    analyses = []

    for filepath in sorted(directory.glob("*_movement_analysis.json")):
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)

        source = data["source_video_metrics"]

        for movement in data["movement_analyses"]:
            analyses.append({
                "source": source,
                **movement
            })
    return analyses

# finds all matching movement files between phase02 and phase03
def findMatchingMovement(feedback, detailed_analyses):

    feedback_first = feedback["frame_ranges"]["first_event"]
    feedback_second = feedback["frame_ranges"]["second_event"]

    for movement in detailed_analyses:
        if movement["source"] != feedback["source"]:
            continue
        if movement["foot"] != feedback["foot"]:
            continue

        first_event = movement["first_event"]
        second_event = movement["second_event"]


        if (
            first_event["start_frame"] == feedback_first["start"]
            and first_event["end_frame"] == feedback_first["end"]
            and second_event["start_frame"] == feedback_second["start"]
            and second_event["end_frame"] == feedback_second["end"]
        ):
            # only returns smt if detailed_analysis and feedback are the same frames
            return movement
    return None
# extracts avg foot speed and tracking coverage for each window: before, during, after
# determines whether each measurement meets our min coverage requirement
# calculates speed diffs only when both windows have reliable measurements
# threshold is subject to change
def analyzeFootStability(movement, min_coverage = 80.0):
    stability = movement["foot_stability"]

    windows = {
        "before": stability["pre_event_speed"],
        "between": stability["between_event_speed"],
        "after": stability["post_event_speed"]
    }

    observations = {}

    for name, data in windows.items():
        speed = data["average_speed"]
        coverage = data["tracking_coverage"]

        reliable = (
            speed is not None
            and coverage >= min_coverage
            and data["valid_frames"] > 0
        )
        observations[name] = {
            "average_speed": speed,
            "tracking_coverage": coverage,
            "reliable": reliable
        }

    before = observations["before"]
    between = observations["between"]
    after = observations["after"]

    if before["reliable"] and between["reliable"]:
        before_to_between = (
            between["average_speed"] - before["average_speed"]
        )
    else:
        before_to_between = None

    if between["reliable"] and after["reliable"]:
        between_to_after = (
            after["average_speed"] - between["average_speed"]
        )
    else:
        between_to_after = None

    return {
        "windows": observations,
        "speed_changes": {
            "before_to_between": before_to_between,
            "between_to_after": between_to_after,
        }
    }

# combining Foot Stability and Elbow-Hip Coordination funcitons
def buildCoordinationAnalysis(feedback, detailed_analyses):
    matched_movement = findMatchingMovement(
        feedback, detailed_analyses
    )

    elbow_hip = analyzeElbowHipCoordination(feedback)

    foot_stability = (
        analyzeFootStability(matched_movement)
        if matched_movement is not None
        else None
    )

    return {
        "source": feedback["source"],
        "foot": feedback["foot"],
        "movement_type": feedback.get("movement_type"),
        "frame_ranges": feedback["frame_ranges"],
        "elbow_hip_coordination": elbow_hip,
        "foot_stability": foot_stability,
    }
# save coordination analysis results onto json file
def saveCoordinationAnalysis(analyses, output_path):

    output = {
        "total_movements": len(analyses),
        "coordination_analyses": analyses
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4, ensure_ascii=False)

    print(f"\nCoordination analysis saved to: {output_path}")

if __name__ == "__main__":
    movements = loadMovementFeedback(FEEDBACK_FILE)

    PHASE02_DIR = BASE_DIR.parent / "phase02"
    detailed_analyses = loadDetailedMovementAnalyses(PHASE02_DIR)

    coordination_analyses = [
        buildCoordinationAnalysis(movement, detailed_analyses)
        for movement in movements
    ]

    OUTPUT_FILE = BASE_DIR / "coordination_analysis.json"

    saveCoordinationAnalysis(coordination_analyses, OUTPUT_FILE)

    print("\n--- Coordination Analysis Summary ---")

    for analysis in coordination_analyses:
        print(
            f"{analysis['source']} | "
            f"{analysis['foot']} foot | "
            f"{analysis['elbow_hip_coordination']['pattern']}"
        )