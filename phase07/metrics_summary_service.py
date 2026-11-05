# preserving three categories of info
# video, detected events, movement samples
def summarize_metrics(metrics: dict) -> dict:
    frames = metrics.get("frames", [])

    summary = {
        "total_frames": len(frames),
        "video_duration_seconds": None,
        "left_arm": metrics.get("left_arm", {}),
        "right_arm": metrics.get("right_arm", {}),
        "movement_samples": []
    }

    if not frames:
        return summary

    summary["video_duration_seconds"] = frames[-1].get(
        "timestamp_seconds"
    )

    # Sample up to approximately 20 frames across the video.
    sample_step = max(1, len(frames) // 20)

    for frame in frames[::sample_step][:20]:
        summary["movement_samples"].append({
            "timestamp_seconds": frame.get("timestamp_seconds"),
            "hip_vertical_velocity": frame.get("hip_vertical_velocity"),
            "left_elbow_angle": frame.get("left_elbow_angle"),
            "right_elbow_angle": frame.get("right_elbow_angle"),
            "left_knee_angle": frame.get("left_knee_angle"),
            "right_knee_angle": frame.get("right_knee_angle")
        })

    return summary

if __name__ == "__main__":
    import json
    from pathlib import Path

    metrics_path = (Path(__file__).resolve().parents[1]
                    /"phase06"
                    /"backend"
                    /"results"
                )

    result_files = list(metrics_path.glob("*/metrics.json"))

    if not result_files:
        print("No metrics files found.")
    else:
        with open(result_files[-1], "r") as file:
            metrics = json.load(file)

        summary = summarize_metrics(metrics)

        print(json.dumps(summary, indent=4))
