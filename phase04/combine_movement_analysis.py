
import json
from pathlib import Path
import argparse
from gemini_schema import GeminiMovementAnalysis


# ============================================================
# 1. FILE PATHS AND VERSION CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Change this to 1 or 2 to select an analysis version.
ANALYSIS_VERSION = 2

if ANALYSIS_VERSION == 1:
    gemini_filename = "gemini_structured_analysis.json"
    combined_filename = "combined_movement_analysis.json"

elif ANALYSIS_VERSION == 2:
    gemini_filename = "gemini_structured_analysis_v2.json"
    combined_filename = "combined_movement_analysis_v2.json"

else:
    raise ValueError(
        f"Unsupported analysis version: {ANALYSIS_VERSION}"
    )


parser = argparse.ArgumentParser(
    description="Combine Gemini and MediaPipe movement analysis."
)

parser.add_argument(
    "--gemini",
    type=Path,
    default=BASE_DIR / gemini_filename,
    help="Path to the structured Gemini analysis JSON."
)

parser.add_argument(
    "--mediapipe",
    type=Path,
    default=PROJECT_ROOT / "phase02" / "metrics.json",
    help="Path to the MediaPipe metrics JSON."
)

parser.add_argument(
    "--output",
    type=Path,
    default=BASE_DIR / combined_filename,
    help="Path for the combined analysis JSON."
)

args = parser.parse_args()

GEMINI_PATH = args.gemini
MEDIAPIPE_PATH = args.mediapipe
OUTPUT_PATH = args.output



print(f"Analysis version: {ANALYSIS_VERSION}")
print(f"Gemini input: {GEMINI_PATH}")
print(f"MediaPipe input: {MEDIAPIPE_PATH}")
print(f"Combined output: {OUTPUT_PATH}")


# ============================================================
# 2. VERIFY INPUT FILES EXIST
# ============================================================

if not GEMINI_PATH.is_file():
    raise FileNotFoundError(
        f"Gemini analysis file not found: {GEMINI_PATH}\n"
        "Make sure the selected analysis version "
        "has already been generated."
    )

if not MEDIAPIPE_PATH.is_file():
    raise FileNotFoundError(
        f"MediaPipe metrics file not found: {MEDIAPIPE_PATH}"
    )


# ============================================================
# 3. LOAD AND VALIDATE GEMINI ANALYSIS
# ============================================================

# Load saved Gemini JSON without making another API request.
with open(GEMINI_PATH, "r", encoding="utf-8") as file:
    gemini_data = json.load(file)

# Reconstruct and validate the structured Gemini response.
analysis = GeminiMovementAnalysis.model_validate(
    gemini_data["analysis"]
)

print("\nGemini analysis loaded successfully!")
print(f"Movement type: {analysis.movement_type}")
print(f"Target foot: {analysis.target_foot}")
print(
    f"Frames analyzed: "
    f"{[frame.frame for frame in analysis.frames]}"
)

# Display structured movement classifications.
# Older analyses may return None for these fields.
print("\nGemini Structured Movement Classifications:")
print(
    f"Hip direction: "
    f"{analysis.hip_horizontal_direction}"
)
print(
    f"Target knee movement: "
    f"{analysis.target_knee_movement}"
)
print(
    f"Left elbow movement: "
    f"{analysis.left_elbow_movement}"
)
print(
    f"Right elbow movement: "
    f"{analysis.right_elbow_movement}"
)


# ============================================================
# 4. LOAD MEDIAPIPE MEASUREMENTS
# ============================================================

with open(MEDIAPIPE_PATH, "r", encoding="utf-8") as file:
    mediapipe_data = json.load(file)

# Create a lookup table using original video frame numbers.
frames_by_number = {
    record["frame"]: record
    for record in mediapipe_data["frames"]
}

print(
    f"\nLoaded {len(frames_by_number)} "
    "MediaPipe frame records."
)


# ============================================================
# 5. COMBINE GEMINI AND MEDIAPIPE DATA
# ============================================================

combined_records = []

for observation in analysis.frames:
    frame_number = observation.frame

    if frame_number not in frames_by_number:
        print(
            f"Warning: No MediaPipe data for frame "
            f"{frame_number}"
        )
        continue

    mediapipe_record = frames_by_number[frame_number]

    combined_record = {
        "frame": frame_number,
        "timestamp_seconds": mediapipe_record[
            "timestamp_seconds"
        ],
        "mediapipe": mediapipe_record,
        "gemini": observation.model_dump(mode="json")
    }

    combined_records.append(combined_record)


# ============================================================
# 6. VALIDATE COMBINED RECORDS
# ============================================================

if len(combined_records) != len(analysis.frames):
    raise ValueError(
        "Some Gemini frames could not be matched "
        "with MediaPipe records."
    )

# Ensure chronological order.
combined_records.sort(key=lambda record: record["frame"])

print(
    f"\nSuccessfully combined "
    f"{len(combined_records)} frames!"
)

for record in combined_records:
    print(f"\nFrame {record['frame']}")
    print(
        f"Timestamp: "
        f"{record['timestamp_seconds']:.3f}s"
    )
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


# ============================================================
# 7. BUILD COMBINED ANALYSIS
# ============================================================

combined_analysis = {
    "schema_version": gemini_data.get(
        "schema_version", 1
    ),
    "video_file": gemini_data["video_file"],
    "movement_type": analysis.movement_type,
    "target_foot": analysis.target_foot,
    "frame_count": len(combined_records),
    "frames": combined_records,

    # Gemini's natural-language observations.
    "gemini_movement_summary": analysis.movement_summary,
    "gemini_movement_interpretation": (
        analysis.movement_interpretation
    ),
    "gemini_uncertainties": analysis.uncertainties,

    # Gemini's structured movement classifications.
    "gemini_hip_horizontal_direction": (
        analysis.hip_horizontal_direction
    ),
    "gemini_target_knee_movement": (
        analysis.target_knee_movement
    ),
    "gemini_left_elbow_movement": (
        analysis.left_elbow_movement
    ),
    "gemini_right_elbow_movement": (
        analysis.right_elbow_movement
    )
}


# ============================================================
# 8. SAVE COMBINED ANALYSIS
# ============================================================

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(
        combined_analysis,
        file,
        indent=4,
        ensure_ascii=False
    )

print(
    f"\nCombined analysis saved to: {OUTPUT_PATH}"
)
