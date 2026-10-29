import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors

import json
from datetime import datetime, timezone
from gemini_schema import GeminiMovementAnalysis


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

load_dotenv(PROJECT_ROOT / ".env")
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI API key was not found in .env")

client = genai.Client(api_key=api_key)

FRAME_DIR = (
    BASE_DIR
    / "vlm_frames"
    / "climbingTestDeepBentElbow_left_release_return"
)

frame_numbers = [332, 358, 373]

prompt = """
Analyze these three chronological frames from an indoor bouldering video.

Movement type: left_release_return
Target foot: anatomical LEFT foot, not image-left.

For each provided frame:
1. Describe the target foot's visible position.
2. Determine whether it visibly contacts a climbing hold.
3. Describe the visible body and arm positioning.
4. Identify any uncertainties.

Then summarize the movement across all three frames.

Separate direct visual observations from interpretations.
Do not assume foothold contact, weight distribution, or movement intention.
Use "unclear" when contact cannot be determined.

Return one observation for each provided frame.
Use the original frame numbers supplied with the images.
"""

contents = [prompt]

for frame_number in frame_numbers:
    image_path = FRAME_DIR / f"original_frame_{frame_number}.jpg"

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    contents.append(f"Original video frame {frame_number}:")
    contents.append(
        types.Part.from_bytes(
            data=image_path.read_bytes(),
            mime_type="image/jpeg"
        )
    )
try:
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=4096,
            response_mime_type="application/json",
            response_schema=GeminiMovementAnalysis
        )
    )

except errors.ServerError as error:
    if error.code == 503:
        print(
            "Gemini is temporarily unavailable due to high demand. "
            "Please try again later."
        )
    else:
        print(f"Gemini server error: {error}")

    raise SystemExit(1)

except errors.ClientError as error:
    print(f"Gemini API client error: {error}")
    raise SystemExit(1)

print("\nFinish reason:")
print(response.candidates[0].finish_reason)

print("\nStructured Gemini Analysis:\n")

analysis = response.parsed

if analysis is None:
    raise ValueError(
        "Gemini did not return a valid structured response."
    )


# Verify that Gemini returned the requested frames
returned_frames = [frame.frame for frame in analysis.frames]

if returned_frames != frame_numbers:
    raise ValueError(
        f"Frame mismatch! "
        f"Expected: {frame_numbers}, "
        f"Received: {returned_frames}"
    )

# Verify that Gemini analyzed the correct foot
if analysis.target_foot != "left":
    raise ValueError(
        f"Incorrect target foot: {analysis.target_foot}"
    )

print("Structured analysis validation passed!")


print(analysis.model_dump_json(indent=4))

# Save structured Gemini analysis
output_path = BASE_DIR / "gemini_structured_analysis.json"

analysis_record = {
    "video_file": "climbingTestDeepBentElbow.mov",
    "movement_type": "left_release_return",
    "analyzed_frames": frame_numbers,
    "model": "gemini-3.8-flash",
    "analysis_type": "multi_frame",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "finish_reason": response.candidates[0].finish_reason.value,
    "token_usage": {
        "prompt_tokens": response.usage_metadata.prompt_token_count,
        "reasoning_tokens": response.usage_metadata.thoughts_token_count,
        "response_tokens": response.usage_metadata.candidates_token_count,
        "total_tokens": response.usage_metadata.total_token_count
    },
    "analysis": analysis.model_dump(mode="json")
}

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(analysis_record, file, indent=4, ensure_ascii=False)

print(f"\nSaved analysis to: {output_path}")
