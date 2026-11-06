
import json
import time
from pathlib import Path
from uuid import UUID

from phase02.video_processor import process_climbing_video
from phase06.backend.feedback_service import generate_feedback
from phase06.backend.gemini_video_service import analyze_climbing_video
from phase06.backend.movement_matching_service import match_gemini_movements
from phase06.backend.hybrid_feedback_service import generate_hybrid_feedback

# Backend directories
BACKEND_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BACKEND_DIR / "uploads"
RESULTS_DIR = BACKEND_DIR / "results"

ALLOWED_EXTENSIONS = (".mp4", ".mov", ".webm")


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
        annotated_output_path=annotated_video_path
    )

    processing_time = time.perf_counter() - processing_start

    print(
        f"[Performance] MediaPipe + FFmpeg: "
        f"{processing_time:.2f} seconds"
    )


    # Generate evidence-based coaching feedback
    feedback = generate_feedback(metrics)

    # Step 5: Run Gemini + MediaPipe hybrid analysis
    hybrid_feedback = []
    hybrid_status = "completed"
    hybrid_error = None

    try:
        # Analyze the original uploaded video with Gemini.
        gemini_result = analyze_climbing_video(video_path)

        # Match Gemini movements with MediaPipe measurements.
        matched_movements = match_gemini_movements(
            gemini_result,
            metrics
        )

        # Generate evidence-aware hybrid feedback.
        hybrid_feedback = generate_hybrid_feedback(
            matched_movements
        )

        # Save Gemini observations for future review.
        gemini_path = result_dir / "gemini_movements.json"

        with gemini_path.open("w", encoding="utf-8") as file:
            json.dump(gemini_result, file, indent=4)

        # Save hybrid feedback.
        hybrid_path = result_dir / "hybrid_feedback.json"

        with hybrid_path.open("w", encoding="utf-8") as file:
            json.dump(hybrid_feedback, file, indent=4)

        print(
            f"[Hybrid Analysis] Generated feedback for "
            f"{len(hybrid_feedback)} movements."
        )

    except Exception as exc:
        hybrid_status = "failed"
        hybrid_error = str(exc)

        print(f"[Hybrid Analysis] Failed: {exc}")



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
        "feedback": feedback,
        "hybrid_status": hybrid_status,
        "hybrid_error": hybrid_error,
        "hybrid_feedback": hybrid_feedback
    }

