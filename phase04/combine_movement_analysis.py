import json
from pathlib import Path

from gemini_schema import GeminiMovementAnalysis

BASE_DIR = Path(__file__).resolve().parent
GEMINI_PATH = BASE_DIR / "gemini_structured_analysis.json"

# load saved gemini response
with open(GEMINI_PATH) as file:
    gemini_data = json.load(file)


#what im trying to achieve here:
# json reads the saved Gemini output
# pydantic reconstructs the GeminiMovementAnalysis obj and validates the structure
# really important here bc I can take the prev saved Gemini output and turn it back into a
# validated python obj without contacting the Gemini API // spend more credits/tokens

# extract the structured analysis
analysis = GeminiMovementAnalysis.model_validate(
    gemini_data["analysis"]
)

print("Gemini analysis loaded successfully!")
print(f"Movement type: {analysis.movement_type}")
print(f"Target foot: {analysis.target_foot}")
print(f"Frames analyzed: {[frame.frame for frame in analysis.frames]}")

# Access individual frame observations
for frame in analysis.frames:
    print(f"\nFrame {frame.frame}")
    print(f"Foot position: {frame.foot_position}")
    print(f"Foothold contact: {frame.foothold_contact}")

# Load MediaPipe measurements from .json file

PROJECT_ROOT = BASE_DIR.parent
MEDIAPIPE_PATH = PROJECT_ROOT / "phase02" / "metrics.json"

with open(MEDIAPIPE_PATH, "r", encoding="utf-8") as file:
    mediapipe_data = json.load(file)

# Create a lookup using original frame numbers
frames_by_number = {
    record["frame"]: record
    for record in mediapipe_data["frames"]
}

# Combine the two sources
combined_records = []

for observation in analysis.frames:
    frame_number = observation.frame

    if frame_number not in frames_by_number:
        print(f"Warning: No MediaPipe data for frame {frame_number}")
        continue

    mediapipe_record = frames_by_number[frame_number]

    combined_record = {
        "frame": frame_number,
        "timestamp_seconds": mediapipe_record["timestamp_seconds"],
        "mediapipe": mediapipe_record,
        "gemini": observation.model_dump(mode="json")
    }

    combined_records.append(combined_record)

print(f"\nSuccessfully combined {len(combined_records)} frames!")

for record in combined_records:
    print(f"\nFrame {record['frame']}")
    print(f"Timestamp: {record['timestamp_seconds']:.3f}s")
    print(
        f"Left elbow angle: "
        f"{record['mediapipe']['left_elbow_angle']}"
    )
    print(
        f"Left knee angle: "
        f"{record['mediapipe']['left_knee_angle']}"
    )
    print(
        f"Gemini foothold contact: "
        f"{record['gemini']['foothold_contact']}"
    )


# Verify that every Gemini frame was matched
if len(combined_records) != len(analysis.frames):
    raise ValueError(
        "Some Gemini frames could not be matched "
        "with MediaPipe records."
    )

# Save the combined movement analysis
output_path = BASE_DIR / "combined_movement_analysis.json"

combined_analysis = {
    "video_file": gemini_data["video_file"],
    "movement_type": analysis.movement_type,
    "target_foot": analysis.target_foot,
    "frame_count": len(combined_records),
    "frames": combined_records,
    "gemini_movement_summary": analysis.movement_summary,
    "gemini_movement_interpretation": analysis.movement_interpretation,
    "gemini_uncertainties": analysis.uncertainties
}

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(combined_analysis, file, indent=4, ensure_ascii=False)

print(f"\nCombined analysis saved to: {output_path}")

