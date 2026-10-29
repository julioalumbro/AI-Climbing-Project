import cv2
import json
from pathlib import Path

# project directories
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# input recommendations
RECOMMENDATIONS_FILE = (
    PROJECT_ROOT
    / "phase03"
    / "unified_coaching_recommendations.json"
)

# output directory
CLIPS_DIR = BASE_DIR / "clips"
CLIPS_DIR.mkdir(parents=True, exist_ok=True)

def loadRecommendations():
    with open(RECOMMENDATIONS_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data["recommendations"]

# use OpenCV's video capture to grab fps, total frames, width, height
def inspectVideo(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"ERROR: could not open {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height
    }
# function to add one extra second to the first_event and second_event
# used to allow VLM to get more context
def calculateClipBoundaries(movement, video_info, padding_seconds=1.0):
    fps = video_info["fps"]
    total_frames = video_info["total_frames"]

    first_event = movement["frame_ranges"]["first_event"]
    second_event = movement["frame_ranges"]["second_event"]

    # find full movement window
    movement_start = min(
        first_event["start"],
        second_event["start"]
    )

    movement_end = max(
        first_event["end"],
        second_event["end"]
    )

    # convert padding from seconds to frames
    padding_frames = round(padding_seconds * fps)

    # clamp boundaries onto video
    clip_start = max(0, movement_start - padding_frames)
    clip_end = min(total_frames - 1, movement_end + padding_frames)

    return {
        "start_frame": clip_start,
        "end_frame": clip_end,
        "start_time": clip_start / fps,
        "end_time": (clip_end + 1) / fps,
        "duration": (clip_end - clip_start + 1) / fps
    }

# exporting movement clips with OpenCV
# opens climbing video
# creates an MP4 video writer
# goes to the start of the movement clip, writes every frame through til the ending frame inclusive
# checks whether the expected number of frames was written
def extractMovementClip(video_path, output_path, boundaries):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"ERROR: Could not open {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        print(f"ERROR: Could not create {output_path}")
        cap.release()
        return False

    start_frame = boundaries["start_frame"]
    end_frame = boundaries["end_frame"]

    # Jump to the first frame of the clip
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames_written = 0

    for frame_index in range(start_frame, end_frame + 1):
        success, frame = cap.read()

        if not success:
            print(f"WARNING: Could not read frame {frame_index}")
            break

        writer.write(frame)
        frames_written += 1

    cap.release()
    writer.release()

    # verify exported video
    check_cap = cv2.VideoCapture(str(video_path))

    if not check_cap.isOpened():
        print("Error: Exported video could not be opened.")
    else:
        output_frames = int(check_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        output_fps = check_cap.get(cv2.CAP_PROP_FPS)

        readable, first_frame = check_cap.read()

        print(f"Output frame count: {output_frames}")
        print(f"Output FPS: {output_fps}")
        print(f"First frame readable: {readable}")

    check_cap.release()

    expected_frames = end_frame - start_frame + 1

    print(f"Exported: {output_path.name}")
    print(f"Frames written: {frames_written}/{expected_frames}")

    return frames_written == expected_frames

if __name__ == "__main__":
    recommendations = loadRecommendations()

    print("\n--- Source Video Inspection ---")

    # avoid insepcting the same video twice
    unique_videos = {
        movement["video_file"]
        for movement in recommendations
    }

    for video_file in sorted(unique_videos):

        video_path = PROJECT_ROOT / video_file

        info = inspectVideo(video_path)

        print(f"\nVideo: {video_file}")

        if info is None:
            continue

        print(f"FPS: {info['fps']}")
        print(f"Total Frames: {info['total_frames']}")
        print(f"Resolution: {info['width']} x {info['height']}")
        if info["fps"] > 0:
            print(
                f"Duration: "
                f"{info['total_frames'] / info['fps']:.2f} seconds "
            )
    print("\n--- Movement Clip Boundaries ---")

    for movement in recommendations:

        video_path = PROJECT_ROOT / movement["video_file"]
        video_info = inspectVideo(video_path)

        if video_info is None:
            continue

        boundaries = calculateClipBoundaries(movement, video_info)

        print(f"\n{movement['source']} | {movement['foot']} foot")
        print(f"Movement: {movement['movement_type']}")
        print(
            f"Frames: {boundaries['start_frame']} "
            f"to {boundaries['end_frame']}"
        )
        print(
            f"Time: {boundaries['start_time']:.2f}s "
            f"to {boundaries['end_time']:.2f}s"
        )
        print(f"Clip duration: {boundaries['duration']:.2f}s")


    print("\n--- Exporting All Movement Clips ---")

    for movement in recommendations:
        video_path = PROJECT_ROOT / movement["video_file"]
        video_info = inspectVideo(video_path)

        if video_info is None:
            continue

        boundaries = calculateClipBoundaries(movement, video_info)

        clip_name = (
            f"{Path(movement['video_file']).stem}_"
            f"{movement['foot']}_"
            f"{movement['movement_type']}.mp4"
        )

        output_path = CLIPS_DIR / clip_name

        success = extractMovementClip(
            video_path,
            output_path,
            boundaries
        )

        print(f"Extraction successful: {success}\n")