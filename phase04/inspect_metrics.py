import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

metrics_path = PROJECT_ROOT / "phase02" / "metrics.json"

# I need to inspect here b/c I need to find out if MediaPipe measurements are
# stored as:
# a list of frame records
# a dict containing a list of frame records
# separate movement-event records w/ timestmaps
with open(metrics_path, "r", encoding="utf-8") as file:
    data = json.load(file)

print("Root type:", type(data).__name__)

if isinstance(data, dict):
    print("Top-level keys:", list(data.keys()))

elif isinstance(data, list):
    print("Number of records:", len(data))
    print("First record keys:", list(data[0].keys()))


# were checking that frames field is a list of dictionaries --> each dict = one original video frame
# we can create a lookup dictionary to retrieve MediaPipe measurements
frames = data["frames"]

print("Frames type:", type(frames).__name__)
print("Total frame records:", len(frames))

if isinstance(frames, list) and frames:
    print("First frame keys:", list(frames[0].keys()))

    target_frames = {332, 358, 373}

    for record in frames:
        if record.get("frame") in target_frames:
            print(
                f"Frame {record['frame']} found! "
                f"Timestamp: {record.get('timestamp_seconds')}"
            )