# create json req containing images I want to analyze and questions I want it to answer
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def loadJSON(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

def buildVLMRequest(vlm_record, frame_record):
    return {
        "source": vlm_record["source"],
        "video_file": vlm_record["video_file"],
        "clip_file": vlm_record["clip_file"],
        "movement": vlm_record["movement"],
        "images": frame_record["extracted_frames"],
        "questions": vlm_record["vlm_questions"],
    }
# matching function to map vlm records and the correct frames
def findMatchingFrameRecord(vlm_record, frame_records):
    for frame_record in frame_records:
        if (
            vlm_record["source"] == frame_record["source"]
            and vlm_record["clip_file"] == frame_record["clip_file"]
            and vlm_record["movement"] == frame_record["movement"]
        ):
            return frame_record

    raise ValueError(
        f"No matching frame record found for: "
        f"{vlm_record['clip_file']}"
    )

# save file
# stores reqs and calculates total number of images across all movements
def saveVLMRequests(requests, output_path):
    output_data = {
        "total_requests": len(requests),
        "total_images": sum(
            len(request["images"])
            for request in requests
        ),
        "requests": requests
    }

    with open(output_path, "w") as file:
        json.dump(output_data, file, indent=4)

    print(f"\nVLM requests saved to: {output_path}")


if __name__ == "__main__":
    vlm_data = loadJSON(BASE_DIR / "vlm_inputs.json")
    frame_data = loadJSON(BASE_DIR / "vlm_frame_references.json")

    vlm_records = vlm_data["vlm_inputs"]
    frame_records = frame_data["movements"]

    print("VLM records:", len(vlm_records))
    print("Frame records:", len(frame_records))

    # Test using the first movement
    request = buildVLMRequest(
        vlm_records[0],
        frame_records[0]
    )

    print("\n--- Matched VLM Requests ---")

    vlm_requests = []

    for vlm_record in vlm_records:
        frame_record = findMatchingFrameRecord(
            vlm_record,
            frame_records
        )

        request = buildVLMRequest(
            vlm_record,
            frame_record
        )

        vlm_requests.append(request)

    output_path = BASE_DIR / "vlm_requests.json"

    saveVLMRequests(vlm_requests, output_path)