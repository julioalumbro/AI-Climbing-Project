# experiment with frame-based multimodal analysis and inspect model responses.
import os
from pathlib import Path
import argparse

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


parser = argparse.ArgumentParser(
    description="Generate structured Gemini movement analysis."
)

parser.add_argument(
    "--generate",
    action="store_true",
    help="Explicitly authorize a Gemini API request."
)


parser.add_argument(
    "--video",
    default="climbingTestDeepBentElbow.mov"
)

parser.add_argument(
    "--movement",
    default="left_release_return"
)

parser.add_argument(
    "--target-foot",
    choices=["left", "right"],
    default="left"
)

parser.add_argument(
    "--frames",
    type=int,
    nargs=3,
    default=[332, 358, 373]
)

parser.add_argument(
    "--output",
    type=Path,
    default=BASE_DIR / "gemini_structured_analysis_v2.json"
)


args = parser.parse_args()

if not args.generate:
    print(
        "Gemini generation is disabled by default.\n"
        "Use --generate to explicitly authorize an API request."
    )
    raise SystemExit(0)

if args.output.exists():
    raise FileExistsError(
        f"Output already exists: {args.output}\n"
        "Choose a different --output path to preserve existing analyses."
    )



client = genai.Client(api_key=api_key)

FRAME_DIR = (
    BASE_DIR
    / "vlm_frames"
    / f"{Path(args.video).stem}_{args.target_foot}_{args.movement}"
)

frame_numbers = args.frames

# these instructions matter bc i want to make sure gemini and mediapipe
# evaluat the same movement over the same interval
prompt = f"""
Analyze these three chronological frames from an indoor bouldering video.

Movement type: {args.movement}
Target foot: anatomical {args.target_foot.upper()} foot,
not image-left or image-right.

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


STRUCTURED MOVEMENT CLASSIFICATION

Analyze the movement between the FIRST and LAST supplied frames.

Use the climber's anatomical left and right when referring
to specific arms or legs.

For hip_horizontal_direction:
- image-left: the hip midpoint moves toward the left side
  of the image.
- image-right: the hip midpoint moves toward the right side
  of the image.
- minimal_horizontal_change: little net horizontal movement.
- unclear: insufficient visual evidence.

For target_knee_movement:
- flexion: the target knee becomes more bent.
- extension: the target knee becomes straighter.
- minimal_angle_change: little visible change in knee angle.
- unclear: insufficient visual evidence.

For left_elbow_movement and right_elbow_movement:
- flexion: the elbow becomes more bent.
- extension: the elbow becomes straighter.
- minimal_angle_change: little visible change in elbow angle.
- unclear: insufficient visual evidence.

IMPORTANT:
- Compare the same climber across the supplied frames.
- Do not confuse anatomical left/right with image left/right.
- Base classifications only on visible movement.
- Do not infer joint movement from assumed weight distribution.
- If a joint is occluded or difficult to assess, use "unclear".
- Do not invent numerical angles or displacement measurements.
- Report uncertainties rather than guessing.

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
    if error.code == 402:
        print(
            "Gemini API prepaid credits are depleted. "
            "Check your Google AI Studio billing settings."
        )
    elif error.code == 429:
        print(
            "Gemini API rate limit or quota exceeded."
        )
    else:
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

# new: validate gemini v2 structured classifications
print("\nGemini V2 Structured Classifications:")

print("Hip direction:", analysis.hip_horizontal_direction)
print("Target knee:", analysis.target_knee_movement)
print("Left elbow:", analysis.left_elbow_movement)
print("Right elbow:", analysis.right_elbow_movement)

required_fields = {
    "hip_horizontal_direction": analysis.hip_horizontal_direction,
    "target_knee_movement": analysis.target_knee_movement,
    "left_elbow_movement": analysis.left_elbow_movement,
    "right_elbow_movement": analysis.right_elbow_movement
}

missing_fields = [
    name
    for name, value in required_fields.items()
    if value is None
]

if missing_fields:
    raise ValueError(
        f"Gemini V2 omitted required movement classifications: "
        f"{missing_fields}"
    )

print("All four structured fields are present!")
# continue with your existing validation
# and json-saving code below.


# verify that gemini returned the requested frames
returned_frames = [frame.frame for frame in analysis.frames]

if returned_frames != frame_numbers:
    raise ValueError(
        f"Frame mismatch! "
        f"Expected: {frame_numbers}, "
        f"Received: {returned_frames}"
    )

# verify that gemini analyzed the correct foot
if analysis.target_foot != args.target_foot:
    raise ValueError(
        f"Incorrect target foot: {analysis.target_foot}"
    )

if analysis.movement_type != args.movement:
    raise ValueError(
        f"Incorrect movement type: {analysis.movement_type}"
    )

print("Structured analysis validation passed!")


print(analysis.model_dump_json(indent=4))

# save structured gemini analysis
output_path = args.output

# meta data dictionary
analysis_record = {
    "video_file": args.video,
    "movement_type": args.movement,
    "analyzed_frames": frame_numbers,
    "model": "gemini-3.8-flash",
    "analysis_type": "multi_frame",
    "schema_version": 2,
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


with open(output_path, "x", encoding="utf-8") as file:
    json.dump(analysis_record, file, indent=4, ensure_ascii=False)


print(f"\nSaved analysis to: {output_path}")
