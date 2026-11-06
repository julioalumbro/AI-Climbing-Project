import os
import time
from pathlib import Path
import json
import cv2
from google import genai
from google.genai import types

def validate_gemini_movements(result: dict, video_duration: float) -> None:
    if not isinstance(result, dict):
        raise ValueError("Gemini result must be a dictionary.")

    movements = result.get("movements")

    if not isinstance(movements, list):
        raise ValueError("'movements' must be a list.")

    if not isinstance(video_duration, (int, float)) or video_duration <= 0:
        raise ValueError("Video duration must be positive.")

    required_fields = {
        "start_seconds",
        "end_seconds",
        "movement_type",
        "body_parts",
        "observation",
        "confidence",
    }

    previous_start = -1

    for index, movement in enumerate(movements):
        if not isinstance(movement, dict):
            raise ValueError(f"Movement {index} must be a dictionary.")

        missing = required_fields - movement.keys()

        if missing:
            raise ValueError(
                f"Movement {index} is missing fields: {sorted(missing)}"
            )

        start = movement["start_seconds"]
        end = movement["end_seconds"]

        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
        ):
            raise ValueError(f"Movement {index} has invalid timestamps.")

        if not (0 <= start < end <= video_duration):
            raise ValueError(
                f"Movement {index} has timestamps outside the video duration."
            )

        if start < previous_start:
            raise ValueError(
                f"Movement {index} is not in chronological order."
            )

        previous_start = start

        if not isinstance(movement["movement_type"], str) or not movement["movement_type"].strip():
            raise ValueError(f"Movement {index} has an invalid movement type.")

        if not isinstance(movement["observation"], str) or not movement["observation"].strip():
            raise ValueError(f"Movement {index} has an invalid observation.")

        body_parts = movement["body_parts"]

        if (
            not isinstance(body_parts, list)
            or not all(
                isinstance(part, str) and part.strip()
                for part in body_parts
            )
        ):
            raise ValueError(f"Movement {index} has invalid body parts.")

        if movement["confidence"] not in ("high", "medium", "low"):
            raise ValueError(f"Movement {index} has invalid confidence.")


def get_video_duration(video_path: str | Path) -> float:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        if fps <= 0 or frame_count <= 0:
            raise RuntimeError("Could not determine video duration.")

        return frame_count / fps

    finally:
        cap.release()

def analyze_climbing_video(video_path: str | Path) -> dict:
    """Analyze original climbing footage using Gemini."""

    video_path = Path(video_path)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {video_path}")

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )
    client = genai.Client(api_key=api_key)

    uploaded_video = client.files.upload(file=str(video_path))

    try:
        while uploaded_video.state.name == "PROCESSING":
            time.sleep(2)
            uploaded_video = client.files.get(name=uploaded_video.name)

        if uploaded_video.state.name != "ACTIVE":
            raise RuntimeError(
                f"Gemini video processing failed: {uploaded_video.state.name}"
            )

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=[
                uploaded_video,
                types.Part.from_text(
                    text=(
                        "Analyze this rock climbing video and identify "
                        "observable climbing movements in chronological order. "

                        "For each movement, provide approximate start and end "
                        "timestamps in seconds, the movement type, relevant "
                        "body parts, a concise observation, and a qualitative "
                        "confidence level (high, medium, or low). "

                        "Focus on hand movements, foot placements, body "
                        "positioning, and movement sequences. "

                        "Only describe movements that are visible. "
                        "Do not assume that a movement is inefficient simply "
                        "because an arm is bent or the climber pauses. "
                        "Do not invent holds, movements, or force measurements. "
                        "Use low confidence when the movement is partially "
                        "obscured or uncertain. "

                        "Return the movements in chronological order. "
                        "If no movements can be identified reliably, "
                        "return an empty movements list."
                    )
                ),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "movements": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "start_seconds": {"type": "NUMBER"},
                                    "end_seconds": {"type": "NUMBER"},
                                    "movement_type": {"type": "STRING"},
                                    "body_parts": {
                                        "type": "ARRAY",
                                        "items": {"type": "STRING"}
                                    },
                                    "observation": {"type": "STRING"},
                                    "confidence": {
                                        "type": "STRING",
                                        "enum": ["high", "medium", "low"]
                                    }
                                },
                                "required": [
                                    "start_seconds",
                                    "end_seconds",
                                    "movement_type",
                                    "body_parts",
                                    "observation",
                                    "confidence"
                                ]
                            }
                        }
                    },
                    "required": ["movements"]
                }
            )
        )

        if not response.text:
            raise RuntimeError("Gemini returned no text response.")

        try:
            result = json.loads(response.text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Gemini returned invalid JSON.") from exc

        video_duration = get_video_duration(video_path)

        validate_gemini_movements(result, video_duration)

        return result

    finally:
        if uploaded_video.name:
            client.files.delete(name=uploaded_video.name)




if __name__ == "__main__":
    video_path = Path("climbingTestTension.mov")

    result = analyze_climbing_video(video_path)

    output_path = Path(
        "phase06/backend/gemini_movements.json"
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    print("\n=== Gemini Structured Video Analysis ===")
    print(json.dumps(result, indent=4))

    print("\nGemini JSON validation passed!")
    print(f"Gemini movements saved to: {output_path}")
