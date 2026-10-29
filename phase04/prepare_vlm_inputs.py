import json
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

RECORDS_FILE = BASE_DIR / "video_coaching_records.json"


def loadVideoCoachingRecords():
    with open(RECORDS_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data["records"]


def generateVLMQuestions(record):
    coaching_focus = record["coaching_focus"]

    questions = []

    if coaching_focus == "foot_contact_and_body_position":
        questions.append(
            "After the foot returns, does it visibly establish "
            "contact with a foothold? Describe what can be observed."
        )
    elif coaching_focus == "foot_repositioning_and_weight_transfer":
        questions.append(
            "After the foot returns, does it continue repositioning? "
            "Describe any visible adjustments."
        )
    elif coaching_focus == "verify_movement_in_video":
        questions.append(
            "During the heel-hook readjustment, is the heel visibly "
            "engaged with a hold? Describe what can be observed, "
            "including any occlusion or uncertainty."
        )

    return questions

def generateUpperBodyVLMQuestion(record):
    upper_context = record["upper_body_context"]

    if upper_context == "bilateral_elbow_flexion":
        return (
            "During the upward movement, how do the climber's "
            "arms and legs appear to coordinate? Describe visible "
            "elbow bending, foot positioning, and body movement. "
            "Do not assume elbow flexion indicates excessive arm reliance."
        )

    elif upper_context == "partial_elbow_evidence":
        return (
            "During this movement, how does the climber's upper body "
            "move relative to their hips and feet? Describe visible "
            "arm positioning and weight transfer cues. "
            "Do not assume the untracked elbow was inactive."
        )

    return (
        "Describe the climber's visible upper-body positioning "
        "during this movement. Indicate if the footage is insufficient "
        "to evaluate coordination."
    )

def buildVLMInputRecord(record):
    foot_questions = generateVLMQuestions(record)
    upper_body_question = generateUpperBodyVLMQuestion(record)

    return {
        "source": record["source"],
        "video_file": record["video_file"],
        # use posix to convert windows-style paths into foward-slash paths
        "clip_file": Path(record["clip_file"]).as_posix(),

        "movement": {
            "foot": record["foot"],
            "movement_type": record["movement_type"],
            "frame_ranges": record["frame_ranges"]
        },

        "observations": {
            "coaching_focus": record["coaching_focus"],
            "upper_body_context": record["upper_body_context"],
        },

        "vlm_questions": {
            "foot": foot_questions,
            "upper_body": upper_body_question
        },

        "existing_recommendations": record["recommendations"],

        "requires_video_review": record["requires_video_review"]
    }


# save VLM inputs
def saveVLMInputs(vlm_inputs, output_path):
    output = {
        "total_movements": len(vlm_inputs),
        "vlm_inputs": vlm_inputs
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    print(f"VLM inputs saved to: {output_path}")


# uses original vids fps and detected movement ranges to build clip boundaires
def calculateClipMetadata(record, padding_seconds=1.0):
 project_root = BASE_DIR.parent
 video_path = project_root / record["video_file"]

 cap = cv2.VideoCapture(str(video_path))

 if not cap.isOpened():
     raise FileNotFoundError(f"Cannot open video: {video_path}")

 fps = cap.get(cv2.CAP_PROP_FPS)
 total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


 cap.release()

 if fps <= 0 or total_frames <= 0:
     raise ValueError(f"Invalid video metadata: {video_path}")

 frame_ranges = record["movement"]["frame_ranges"]

 first_event = frame_ranges["first_event"]
 second_event = frame_ranges["second_event"]

 movement_start = min(
     first_event["start"],
     second_event["start"]
 )

 movement_end = max(
     first_event["end"],
     second_event["end"]
 )

 padding_frames = round(padding_seconds * fps)

 clip_start = max(0, movement_start - padding_frames)
 clip_end = min(total_frames - 1, movement_end + padding_frames)

 return {
     "start_frame": clip_start,
     "end_frame": clip_end,
     "fps": fps,
     "total_frames": clip_end - clip_start + 1,
     "duration_seconds": (clip_end - clip_start + 1) / fps
 }



if __name__ == "__main__":
    records = loadVideoCoachingRecords()

    print("\n--- Generated VLM Questions ---")

    for record in records:
        questions = generateVLMQuestions(record)

        print(f"\n{record['source']} | {record['foot']} foot")
        print(f"Clip: {record['clip_file']}")

        for question in questions:
            print(f"Question: {question}")

    print("\n--- Upper-Body VLM Questions ---")

    for record in records:
        question = generateUpperBodyVLMQuestion(record)

        print(f"\n{record['source']} | {record['foot']} foot")
        print(f"Question: {question}")

    print("\n--- Structured VLM Input Records ---")

    vlm_inputs = []

    for record in records:
        vlm_input = buildVLMInputRecord(record)

        vlm_input["clip_metadata"] = calculateClipMetadata(vlm_input)

        vlm_inputs.append(vlm_input)

        print(
            f"\n{record['source']} | "
            f"{record['foot']} foot"
        )

        OUTPUT_FILE = BASE_DIR / "vlm_inputs.json"

        saveVLMInputs(vlm_inputs, OUTPUT_FILE)

        print(json.dumps(vlm_input, indent=4))

    print("\n--- VLM Clip Metadata ---")

    for vlm_input in vlm_inputs:
        metadata = calculateClipMetadata(vlm_input)

        print(
            f"\n{vlm_input['source']} | "
            f"{vlm_input['movement']['foot']} foot"
        )

        print(json.dumps(metadata, indent=4))