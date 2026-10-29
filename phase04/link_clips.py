import json
from pathlib import Path

from phase04.extract_clips import PROJECT_ROOT

BASE_DIR = Path(__file__).resolve().parent
PROEJCT_ROOT = BASE_DIR.parent

RECOMMENDATIONS_FILE = (PROEJCT_ROOT
                       / "phase03"
                       / "unified_coaching_recommendations.json")


CLIPS_DIR = BASE_DIR / "clips"

OUTPUT_FILE = BASE_DIR / "video_coaching_records.json"

def linkClipsToRecommendations():
    with open(RECOMMENDATIONS_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    recommendations = data["recommendations"]

    linked_records = []

    for movement in recommendations:
        clip_name = (
            f"{Path(movement['video_file']).stem}_"
            f"{movement['foot']}_"
            f"{movement['movement_type']}.mp4"
        )

        clip_path = CLIPS_DIR / clip_name

        if not clip_path.is_file():
            print(f"WARNING: Missing clip: {clip_name}")
            continue
        record = movement.copy()
        record["clip_file"] = str(
            clip_path.relative_to(PROJECT_ROOT)
        )

        linked_records.append(record)

    return linked_records



if __name__ == "__main__":

    records = linkClipsToRecommendations()

    output = {
        "total_movements": len(records),
        "records": records
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    print("\n--- Linked Video Coaching Records ---")
    print(f"Total linked records: {len(records)}")

    for record in records:
        print(
            f"{record['foot']} | "
            f"{record['movement_type']} -> "
            f"{record['clip_file']}"
        )
