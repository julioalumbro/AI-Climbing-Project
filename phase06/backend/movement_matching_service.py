
from math import isfinite
from statistics import mean


def match_gemini_movements(
    gemini_result: dict,
    mediapipe_metrics: dict
) -> list[dict]:
    """
    Match Gemini movement intervals with timestamped
    MediaPipe elbow measurements.
    """

    frames = mediapipe_metrics["frames"]
    matched_movements = []

    for movement in gemini_result["movements"]:
        start = movement["start_seconds"]
        end = movement["end_seconds"]

        # Find frames within the Gemini movement interval.
        interval_frames = [
            frame
            for frame in frames
            if isinstance(frame.get("timestamp_seconds"), (int, float))
            and isfinite(frame["timestamp_seconds"])
            and start <= frame["timestamp_seconds"] <= end
        ]

        # Count frames where MediaPipe inference actually ran.
        inference_frames = [
            frame
            for frame in interval_frames
            if frame.get("inference_performed") is True
        ]

        def summarize_elbow(angle_key: str, valid_key: str) -> dict:
            samples = []

            for frame in interval_frames:
                angle = frame.get(angle_key)

                if (
                    frame.get(valid_key) is True
                    and isinstance(angle, (int, float))
                    and not isinstance(angle, bool)
                    and isfinite(angle)
                    and 0 <= angle <= 180
                ):
                    samples.append({
                        "timestamp_seconds": frame["timestamp_seconds"],
                        "angle": round(angle, 2)
                    })

            angles = [sample["angle"] for sample in samples]
            total_inferences = len(inference_frames)

            coverage = (
                len(samples) / total_inferences
                if total_inferences > 0
                else None
            )

            return {
                "samples": len(samples),
                "inference_frames": total_inferences,
                "coverage": round(coverage, 3) if coverage is not None else None,
                "min_angle": min(angles) if angles else None,
                "max_angle": max(angles) if angles else None,
                "mean_angle": round(mean(angles), 2) if angles else None,
                "time_series": samples
            }

        def summarize_metric(metric_key: str, valid_key: str | None = None) -> dict:
            samples = []

            for frame in inference_frames:
                value = frame.get(metric_key)

                if valid_key is not None and frame.get(valid_key) is not True:
                    continue

                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and isfinite(value)
                ):
                    samples.append({
                        "timestamp_seconds": frame["timestamp_seconds"],
                        "value": round(value, 4)
                    })

            values = [sample["value"] for sample in samples]
            total_inferences = len(inference_frames)

            coverage = (
                len(samples) / total_inferences
                if total_inferences > 0
                else None
            )

            return {
                "samples": len(samples),
                "inference_frames": total_inferences,
                "coverage": round(coverage, 3) if coverage is not None else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "mean": round(mean(values), 4) if values else None,
                "time_series": samples
            }

        matched_movements.append({
            "gemini_movement": movement,

            "mediapipe": {
                "frames_in_interval": len(interval_frames),
                "inference_frames": len(inference_frames),

                "right_elbow": summarize_elbow(
                    "right_elbow_angle",
                    "right_arm_valid"
                ),
                "left_elbow": summarize_elbow(
                    "left_elbow_angle",
                    "left_arm_valid"
                ),

                "hip_vertical_velocity": summarize_metric(
                    "hip_vertical_velocity"
                ),

                "left_foot_speed": summarize_metric(
                    "left_foot_speed",
                    "left_leg_tracking_valid"
                ),
                "right_foot_speed": summarize_metric(
                    "right_foot_speed",
                    "right_leg_tracking_valid"
                ),

                "left_foot_vx": summarize_metric(
                    "left_foot_vx",
                    "left_leg_tracking_valid"
                ),
                "left_foot_vy": summarize_metric(
                    "left_foot_vy",
                    "left_leg_tracking_valid"
                ),
                "right_foot_vx": summarize_metric(
                    "right_foot_vx",
                    "right_leg_tracking_valid"
                ),
                "right_foot_vy": summarize_metric(
                    "right_foot_vy",
                    "right_leg_tracking_valid"
                )
            }
        })

    return matched_movements






if __name__ == "__main__":
    import json
    from pathlib import Path

    # Load saved Gemini observations.
    gemini_path = Path(
        "phase06/backend/gemini_movements.json"
    )

    # Load MediaPipe metrics from the same climbing video.
    metrics_path = Path(
        "phase06/backend/results/e31406068ee74acfbf820381058431a7/metrics.json"
    )

    with gemini_path.open("r", encoding="utf-8") as file:
        gemini_result = json.load(file)

    with metrics_path.open("r", encoding="utf-8") as file:
        mediapipe_metrics = json.load(file)

    # Match all Gemini movements with MediaPipe measurements.
    matched_movements = match_gemini_movements(
        gemini_result,
        mediapipe_metrics
    )

    print("\n=== Movement Matching Summary ===")

    for index, match in enumerate(matched_movements, start=1):
        movement = match["gemini_movement"]
        mediapipe = match["mediapipe"]

        print(f"\nMovement {index}: {movement['movement_type']}")
        print(
            f"Time: {movement['start_seconds']}s "
            f"- {movement['end_seconds']}s"
        )
        print(f"Observation: {movement['observation']}")

        print(
            f"Frames: {mediapipe['frames_in_interval']} "
            f"| Inferences: {mediapipe['inference_frames']}"
        )

        # Print elbow measurements.
        for side in ("left", "right"):
            elbow = mediapipe[f"{side}_elbow"]

            print(
                f"{side.capitalize()} elbow: "
                f"{elbow['samples']} samples | "
                f"Coverage: {elbow['coverage']} | "
                f"Mean angle: {elbow['mean_angle']}"
            )

        # Print hip and foot measurements.
        for metric_name in (
                "hip_vertical_velocity",
                "left_foot_speed",
                "right_foot_speed"
        ):
            metric = mediapipe[metric_name]

            print(
                f"{metric_name}: "
                f"{metric['samples']} samples | "
                f"Coverage: {metric['coverage']} | "
                f"Min: {metric['min']} | "
                f"Max: {metric['max']} | "
                f"Mean: {metric['mean']}"
            )

    print(f"\nTotal movements matched: {len(matched_movements)}")
