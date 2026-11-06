
COVERAGE_THRESHOLD = 0.5

# service does three thigns:
# Selects relevant metrics, evaluates coverage, generates cautious feedback

def format_measurement(value):
    if value is None:
        return "unavailable"
    return str(round(value, 3))


def evaluate_metric(metric: dict, label: str) -> dict:
    coverage = metric.get("coverage")
    samples = metric.get("samples", 0)

    if coverage is None or samples == 0:
        return {
            "status": "insufficient",
            "message": f"No valid {label} measurements were available."
        }

    if coverage < COVERAGE_THRESHOLD:
        return {
            "status": "limited",
            "message": (
                f"{label} had limited tracking coverage "
                f"({coverage:.1%})."
            )
        }

    return {
        "status": "supported",
        "message": (
            f"{label}: mean={format_measurement(metric.get('mean', metric.get('mean_angle')))}, "
            f"min={format_measurement(metric.get('min', metric.get('min_angle')))}, "
            f"max={format_measurement(metric.get('max', metric.get('max_angle')))} "
            f"(coverage={coverage:.1%})."
        )
    }


def select_relevant_metrics(movement: dict) -> list[tuple[str, str]]:
    movement_type = movement["movement_type"].lower()
    body_parts = " ".join(movement.get("body_parts", [])).lower()

    if any(word in movement_type for word in ("dynamic", "deadpoint", "dyno")):
        return [
            ("hip_vertical_velocity", "Hip vertical velocity"),
            ("left_foot_speed", "Left foot speed"),
            ("right_foot_speed", "Right foot speed"),
            ("left_elbow", "Left elbow angle"),
            ("right_elbow", "Right elbow angle")
        ]

    if any(word in movement_type for word in ("hand", "reach")):
        if "left" in body_parts and "right" not in body_parts:
            return [("left_elbow", "Left elbow angle")]

        if "right" in body_parts and "left" not in body_parts:
            return [("right_elbow", "Right elbow angle")]

        return [
            ("left_elbow", "Left elbow angle"),
            ("right_elbow", "Right elbow angle")
        ]

    if "foot" in movement_type or "feet" in movement_type:
        if "left" in body_parts and "right" not in body_parts:
            return [("left_foot_speed", "Left foot speed")]

        if "right" in body_parts and "left" not in body_parts:
            return [("right_foot_speed", "Right foot speed")]

        return [
            ("left_foot_speed", "Left foot speed"),
            ("right_foot_speed", "Right foot speed")
        ]

    return [("hip_vertical_velocity", "Hip vertical velocity")]


def generate_hybrid_feedback(matched_movements: list[dict]) -> list[dict]:
    feedback_results = []

    for match in matched_movements:
        movement = match["gemini_movement"]
        mediapipe = match["mediapipe"]

        relevant_metrics = select_relevant_metrics(movement)
        metric_results = []

        for metric_key, label in relevant_metrics:
            metric = mediapipe.get(metric_key)

            if metric is None:
                metric_results.append({
                    "metric": metric_key,
                    "status": "insufficient",
                    "message": f"{label} was not available."
                })
                continue

            evaluation = evaluate_metric(metric, label)

            metric_results.append({
                "metric": metric_key,
                **evaluation
            })

        statuses = [result["status"] for result in metric_results]

        if statuses and all(status == "supported" for status in statuses):
            evidence_status = "measurement_supported"
        elif "supported" in statuses:
            evidence_status = "partially_supported"
        else:
            evidence_status = "insufficient_measurement_data"

        feedback_results.append({
            "movement_type": movement["movement_type"],
            "start_seconds": movement["start_seconds"],
            "end_seconds": movement["end_seconds"],
            "observation": movement["observation"],
            "gemini_confidence": movement["confidence"],
            "evidence_status": evidence_status,
            "metric_evaluations": metric_results,
            "feedback": (
                "Relevant movement measurements are available for review. "
                "These measurements describe body motion but do not, "
                "by themselves, establish climbing efficiency."
                if evidence_status == "measurement_supported"
                else
                "Some relevant measurements are missing or have limited "
                "tracking coverage. Technique conclusions should be deferred "
                "until the movement can be reviewed with better evidence."
            )
        })

    return feedback_results



if __name__ == "__main__":
    import json
    from pathlib import Path

    from phase06.backend.movement_matching_service import (
        match_gemini_movements
    )

    gemini_path = Path(
        "phase06/backend/gemini_movements.json"
    )

    metrics_path = Path(
        "phase06/backend/results/e31406068ee74acfbf820381058431a7/metrics.json"
    )

    with gemini_path.open("r", encoding="utf-8") as file:
        gemini_result = json.load(file)

    with metrics_path.open("r", encoding="utf-8") as file:
        mediapipe_metrics = json.load(file)

    matched_movements = match_gemini_movements(
        gemini_result,
        mediapipe_metrics
    )

    feedback_results = generate_hybrid_feedback(
        matched_movements
    )

    print("\n=== Hybrid Feedback Results ===")

    for index, result in enumerate(feedback_results, start=1):
        print(
            f"\nMovement {index}: "
            f"{result['movement_type']}"
        )
        print(
            f"Time: {result['start_seconds']}s "
            f"- {result['end_seconds']}s"
        )
        print(f"Observation: {result['observation']}")
        print(f"Evidence: {result['evidence_status']}")

        for metric in result["metric_evaluations"]:
            print(f"  [{metric['status']}] {metric['message']}")

        print(f"Feedback: {result['feedback']}")

    print(f"\nTotal feedback results: {len(feedback_results)}")
