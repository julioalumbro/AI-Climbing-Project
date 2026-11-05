from math import isclose

# Initial engineering threshold for filtering short detections.
# Not a validated threshold for climbing efficiency.
DEFAULT_MIN_BEND_DURATION = 0.3

# implementing it this way b/c:
#  validate measurements before using them
#  separate observations from interpretations
#  use conditional coaching language
#  preserve timestamps
def generate_feedback(metrics: dict, min_bend_duration: float = DEFAULT_MIN_BEND_DURATION) -> list[dict]:
    """

    Generate evidence-based observations from arm-bend intervals.

    Only completed intervals with consistent timestamps are used.

    """
    feedback = []

    for arm_key, arm_name in [
        ("left_arm", "left"),
        ("right_arm", "right"),
    ]:
        arm_data = metrics.get(arm_key, {})
        intervals = arm_data.get("deep_bend_intervals", [])

        for interval in intervals:
            start = interval.get("start")
            end = interval.get("end")
            duration = interval.get("duration")
            status = interval.get("status")

            # Skip interrupted or unfinished intervals
            if status != "completed":
                continue

            # Skip missing or invalid measurements
            if not all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    for value in (start, end, duration)
            ):
                continue

            if start < 0 or end < start or duration < 0:
                continue

            # Confirm recorded duration matches timestamps
            if not isclose(
                    end - start,
                    duration,
                    abs_tol=0.02
            ):
                continue

            # Ignore very short bends
            if duration < min_bend_duration:
                continue

            feedback.append({
                "type": "sustained_elbow_bend",
                "body_part": f"{arm_name}_elbow",
                "start_seconds": round(start, 2),
                "end_seconds": round(end, 2),
                "duration_seconds": round(duration, 2),
                "observation": (
                    f"A deep {arm_name} elbow bend was "
                    f"recorded for {duration:.2f} seconds."
                ),
                "interpretation": (
                    "This may represent a lock-off, "
                    "active pulling, or another "
                    "intentional climbing movement."
                ),
                "recommendation": (
                    "Review this movement and consider "
                    "whether greater foot support or a "
                    "straighter arm was possible without "
                    "compromising control."
                ),
                "evidence_status": "measurement_supported",
                "limitations": [
                    "Elbow angle alone cannot establish inefficient technique.",
                    "Handhold position and foot loading were not verified."
                ]
            })

    feedback.sort(key=lambda item: item["start_seconds"])

    return feedback

if __name__ == "__main__":
    import json
    from pathlib import Path

    metrics_path = Path(
        "phase06/backend/results/7241a3af11b7451599003a0522920e80/metrics.json"
    )

    with metrics_path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    for threshold in [0.1, 0.2, 0.3, 0.5]:
        feedback = generate_feedback(
            metrics,
            min_bend_duration=threshold
        )

        print(f"\nThreshold: {threshold}s")
        print(f"Feedback entries: {len(feedback)}")

        for item in feedback:
            print(
                f"  {item['body_part']}: "
                f"{item['start_seconds']}s - "
                f"{item['end_seconds']}s "
                f"({item['duration_seconds']}s)"
            )