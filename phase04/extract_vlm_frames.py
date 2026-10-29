import cv2
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

def extractFrame(clip_path, frame_number, output_path):
    cap = cv2.VideoCapture(str(clip_path))

    if not cap.isOpened():
        raise FileNotFoundError(
            f"Could not open video file: {clip_path}"
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    success, frame = cap.read()
    cap.release()

    if not success:
        raise ValueError(
            f"Could not read frame {frame_number}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Could not save frame: {output_path}")

    return output_path

# used to select start, midpoint, and end of each event
# now we don't have to hardcode frames
def selectEventFrames(vlm_record, context_seconds=0.25):
    frame_ranges = vlm_record["movement"]["frame_ranges"]
    clip_metadata = vlm_record["clip_metadata"]

    fps = clip_metadata["fps"]
    clip_start = clip_metadata["start_frame"]
    clip_end = clip_metadata["end_frame"]

    context_frames = round(context_seconds * fps)

    selected_frames = []

    for event_name in ["first_event", "second_event"]:
        event = frame_ranges[event_name]

        start = event["start"]
        end = event["end"]
        midpoint = (start + end) // 2

        before = max(clip_start, start - context_frames)
        after = min(clip_end, end + context_frames)

        selected_frames.extend([
            before,
            start,
            midpoint,
            end,
            after
        ])

    return sorted(set(selected_frames))


def extractSelectedFrames(vlm_record):
    clip_path = PROJECT_ROOT / vlm_record["clip_file"]

    clip_start_frame = vlm_record["clip_metadata"]["start_frame"]
    total_frames = vlm_record["clip_metadata"]["total_frames"]

    selected_frames = selectEventFrames(vlm_record)

    output_dir = BASE_DIR / "vlm_frames" / clip_path.stem

    extracted_frames = []

    for original_frame in selected_frames:
        clip_frame = original_frame - clip_start_frame

        if not 0 <= clip_frame <= total_frames:
            raise ValueError(
                f"Original frame {original_frame} is outside the clip"
            )

        output_path = output_dir / f"original_frame_{original_frame}.jpg"

        saved_path = extractFrame(
            clip_path,
            frame_number=clip_frame,
            output_path=output_path
        )

        fps = vlm_record["clip_metadata"]["fps"]

        extracted_frames.append({
            "original_frame": original_frame,
            "clip_frame": clip_frame,
            "clip_timestamp_seconds": round(clip_frame / fps, 3),
            "image_file": saved_path.relative_to(PROJECT_ROOT).as_posix()
        })

    return extracted_frames


def saveExtractedFrameReferences(records, output_path):
    output_data = {
        "total_movements": len(records),
        "total_frames": sum(
        len(record["extracted_frames"])
        for record in records
    ),
        "movements": records
    }

    with open(output_path, "w") as file:
        json.dump(output_data, file, indent=4)

    print(f"\nFrame references saved to: {output_path}")



if __name__ == "__main__":
    vlm_inputs_path = BASE_DIR / "vlm_inputs.json"

    with open(vlm_inputs_path, "r") as file:
        data = json.load(file)
    frame_records = []

    for record in data["vlm_inputs"]:
        extracted_frames = extractSelectedFrames(record)

        frame_records.append({
            "source": record["source"],
            "video_file": record["video_file"],
            "clip_file": record["clip_file"],
            "movement": record["movement"],
            "clip_metadata": record["clip_metadata"],
            "extracted_frames": extracted_frames
        })

    output_path = BASE_DIR / "vlm_frame_references.json"

    saveExtractedFrameReferences(frame_records, output_path)