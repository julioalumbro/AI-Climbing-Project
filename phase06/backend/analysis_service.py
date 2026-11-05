
import json
import shutil
import subprocess
import time
from pathlib import Path
from uuid import UUID

from phase02.video_processor import process_climbing_video
from phase06.backend.feedback_service import generate_feedback

# Backend directories
BACKEND_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BACKEND_DIR / "uploads"
RESULTS_DIR = BACKEND_DIR / "results"

ALLOWED_EXTENSIONS = (".mp4", ".mov", ".webm")


def convert_to_h264(input_path: Path, output_path: Path) -> None:
    """
    Convert an OpenCV-generated MP4 to browser-compatible H.264.
    """

    # Check whether FFmpeg is installed
    ffmpeg_path = shutil.which("ffmpeg")

    if ffmpeg_path is None:
        raise RuntimeError(
            "FFmpeg was not found. Install FFmpeg and restart PyCharm."
        )

    command = [
        ffmpeg_path,
        "-y",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path)
    ]

    # Run FFmpeg and wait for conversion to finish
    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True
    )


def analyze_uploaded_video(upload_id: str) -> dict:
    """
    Process an uploaded video, save its metrics, and generate
    a browser-compatible skeleton-overlay video.
    """

    # Step 1: Validate upload ID
    try:
        validated_id = UUID(hex=upload_id).hex
    except (ValueError, AttributeError, TypeError):
        raise ValueError("Invalid upload ID")

    # Step 2: Find uploaded video
    video_path = None

    for extension in ALLOWED_EXTENSIONS:
        candidate = UPLOAD_DIR / f"{validated_id}{extension}"

        if candidate.is_file():
            video_path = candidate
            break

    if video_path is None:
        raise FileNotFoundError(
            f"No uploaded video found for ID: {validated_id}"
        )

    # Step 3: Create results directory
    result_dir = RESULTS_DIR / validated_id
    result_dir.mkdir(parents=True, exist_ok=True)

    # OpenCV output (temporary)
    raw_video_path = result_dir / "annotated_raw.mp4"

    # Browser-compatible output (final)
    annotated_video_path = result_dir / "annotated.mp4"

    metrics_path = result_dir / "metrics.json"

    processing_start = time.perf_counter()

    # Step 4: Run MediaPipe processing
    metrics = process_climbing_video(
        video_path=video_path,
        display_video=False,
        annotated_output_path=raw_video_path
    )

    processing_time = time.perf_counter() - processing_start

    print(
        f"[Performance] MediaPipe + OpenCV: "
        f"{processing_time:.2f} seconds"
    )


    # Generate evidence-based coaching feedback
    feedback = generate_feedback(metrics)

    conversion_start = time.perf_counter()
    # Step 5: Convert video to H.264
    try:
        convert_to_h264(
            input_path=raw_video_path,
            output_path=annotated_video_path
        )

        conversion_time = time.perf_counter() - conversion_start
        print(
            f"[Performance] FFmpeg conversion: "
            f"{conversion_time:.2f} seconds"
        )

    finally:
        # Delete the temporary OpenCV video after conversion
        # or if conversion fails
        raw_video_path.unlink(missing_ok=True)

    # Step 6: Save JSON metrics
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    # Step 7: Return analysis response
    return {
        "upload_id": validated_id,
        "status": "completed",
        "frames_processed": len(metrics["frames"]),
        "left_arm_summary": metrics["left_arm"]["summary"],
        "right_arm_summary": metrics["right_arm"]["summary"],
        "metrics_url": f"/api/results/{validated_id}/metrics",
        "annotated_video_url": f"/api/results/{validated_id}/video",
        "feedback": feedback
    }
