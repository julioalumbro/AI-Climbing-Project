"""Reusable Phase 02 pose and movement-metrics processor.

Run from the project root:
    python -m phase02.video_processor
"""

from pathlib import Path
import json
import math
import time
import shutil
import subprocess

import cv2
import mediapipe as mp

from phase02.angle_utils import calculateAngle, classifyElbowPosition
from phase02.velocity_utils import calculateFootVelocity, calculateFootSpeed
from phase02.duration_utils import trackDeepBendDuration, summarizeDeepBendIntervals

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "phase01-opencv_videoloading" / "pose_landmarker_full.task"
TARGET_INFERENCE_FPS = 30.0
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

SKELETON_CONNECTIONS = (
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
    (27, 29), (29, 31), (24, 26), (26, 28), (28, 30), (30, 32),
)


def _visible(point, threshold=0.5):
    return (
        point is not None
        and point.visibility is not None
        and point.visibility >= threshold
        and point.x is not None
        and point.y is not None
        and math.isfinite(point.x)
        and math.isfinite(point.y)
    )


def _pixel(point, width, height):
    return (point.x * width, point.y * height)


def _angle(a, b, c, width, height):
    return calculateAngle(_pixel(a, width, height), _pixel(b, width, height), _pixel(c, width, height))


def _draw_skeleton(frame, landmarks, width, height):
    for start, end in SKELETON_CONNECTIONS:
        p1, p2 = landmarks[start], landmarks[end]
        if not (_visible(p1) and _visible(p2)):
            continue
        cv2.line(frame, (int(p1.x * width), int(p1.y * height)),
                 (int(p2.x * width), int(p2.y * height)), (0, 255, 0), 2)
    for landmark in landmarks:
        if not _visible(landmark):
            continue
        cv2.circle(frame, (int(landmark.x * width), int(landmark.y * height)),
                   4, (0, 0, 255), -1)


def _finish_interval(start, missing_start, last_timestamp, intervals):
    if start is None or last_timestamp is None:
        return
    end = missing_start if missing_start is not None else last_timestamp
    intervals.append({
        "start": start,
        "end": end,
        "duration": max(0.0, end - start),
        "status": "interrupted" if missing_start is not None else "video_ended",
    })

# starts FFmpeg as a separate process and preps to receive raw video frames directly from python
def start_ffmpeg_writer(output_path, width, height, fps):
    ffmpeg_path = shutil.which("ffmpeg")

    if ffmpeg_path is None:
        raise RuntimeError("FFmpeg was not found.")

    command = [
        ffmpeg_path,
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path)
    ]

    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL # FFmpegs detailed error msgs wont be captured
    )



def process_climbing_video(
    video_path: str | Path,
    display_video: bool = False,
    annotated_output_path: str | Path | None = None,
) -> dict:
    """Process a video and return per-frame movement metrics.

    Optionally write a silent MP4 with a pose skeleton. No Gemini calls occur.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"MediaPipe model does not exist: {MODEL_PATH}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video: {video_path}")

    writer = None
    output_path = Path(annotated_output_path) if annotated_output_path is not None else None
    metrics_log = []
    previous_hip_y = previous_hip_timestamp = None
    previous_left_foot_dx = previous_left_foot_dy = previous_left_foot_timestamp = None
    previous_right_foot_dx = previous_right_foot_dy = previous_right_foot_timestamp = None
    left_deep_bent_start = left_missing_start = None
    right_deep_bent_start = right_missing_start = None
    left_deep_bent_intervals = []
    right_deep_bent_intervals = []
    last_processed_timestamp = None
    frame_number = 0

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Inference sampling
        inference_interval = max(1, round(fps / TARGET_INFERENCE_FPS))
        last_landmarks = None

        print(f"[Performance] Inference interval: every {inference_interval} frame(s)")
        if width <= 0 or height <= 0:
            raise RuntimeError("Video dimensions are invalid")

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            writer = start_ffmpeg_writer(
                output_path=output_path,
                width=width,
                height=height,
                fps=fps
            )

            if writer.stdin is None:
                raise RuntimeError("Could not open FFmpeg input pipe.")

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=1,
        )

        # performance tracking
        total_inference_time = 0
        inference_count = 0
        total_writer_time = 0
        writer_count = 0
        with PoseLandmarker.create_from_options(options) as landmarker:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                current_timestamp = frame_number / fps
                timestamp_ms = int(frame_number * 1000 / fps)
                last_processed_timestamp = current_timestamp

                # Only run MediaPipe on selected frames
                should_infer = frame_number % inference_interval == 0

                result = None

                if should_infer:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb_frame
                    )

                    inference_start = time.perf_counter()
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                    inference_time = time.perf_counter() - inference_start

                    total_inference_time += inference_time
                    inference_count += 1

                    last_landmarks = (
                        result.pose_landmarks[0]
                        if result.pose_landmarks
                        else None
                    )

                hip_x = hip_y = hip_velocity_y = None
                left_arm_valid = right_arm_valid = False
                left_elbow_angle = right_elbow_angle = None
                left_elbow_position = right_elbow_position = None
                left_leg_tracking_valid = right_leg_tracking_valid = False
                left_knee_angle = right_knee_angle = None
                left_hip_angle = right_hip_angle = None
                left_foot_dx = left_foot_dy = None
                right_foot_dx = right_foot_dy = None

                if should_infer and last_landmarks is not None:
                    landmarks = last_landmarks
                    left_shoulder, right_shoulder = landmarks[11], landmarks[12]
                    left_elbow, right_elbow = landmarks[13], landmarks[14]
                    left_wrist, right_wrist = landmarks[15], landmarks[16]
                    left_hip, right_hip = landmarks[23], landmarks[24]
                    left_knee, right_knee = landmarks[25], landmarks[26]
                    left_ankle, right_ankle = landmarks[27], landmarks[28]
                    left_heel, right_heel = landmarks[29], landmarks[30]
                    left_foot_index, right_foot_index = landmarks[31], landmarks[32]

                    left_arm_valid = all(_visible(p) for p in (left_shoulder, left_elbow, left_wrist))
                    right_arm_valid = all(_visible(p) for p in (right_shoulder, right_elbow, right_wrist))
                    if left_arm_valid:
                        left_elbow_angle = _angle(left_shoulder, left_elbow, left_wrist, width, height)
                        left_elbow_position = classifyElbowPosition(left_elbow_angle)
                    if right_arm_valid:
                        right_elbow_angle = _angle(right_shoulder, right_elbow, right_wrist, width, height)
                        right_elbow_position = classifyElbowPosition(right_elbow_angle)

                    if _visible(left_hip) and _visible(right_hip):
                        hip_x = (left_hip.x + right_hip.x) / 2
                        hip_y = (left_hip.y + right_hip.y) / 2
                        if previous_hip_y is not None and previous_hip_timestamp is not None:
                            delta_t = current_timestamp - previous_hip_timestamp
                            if delta_t > 0:
                                hip_velocity_y = (hip_y - previous_hip_y) / delta_t
                        previous_hip_y = hip_y
                        previous_hip_timestamp = current_timestamp

                    left_leg_tracking_valid = all(_visible(p) for p in (
                        left_hip, left_knee, left_ankle, left_heel, left_foot_index
                    ))
                    right_leg_tracking_valid = all(_visible(p) for p in (
                        right_hip, right_knee, right_ankle, right_heel, right_foot_index
                    ))

                    if left_leg_tracking_valid:
                        left_knee_angle = _angle(left_hip, left_knee, left_ankle, width, height)
                    if right_leg_tracking_valid:
                        right_knee_angle = _angle(right_hip, right_knee, right_ankle, width, height)
                    if left_leg_tracking_valid and left_arm_valid:
                        left_hip_angle = _angle(left_shoulder, left_hip, left_knee, width, height)
                    if right_leg_tracking_valid and right_arm_valid:
                        right_hip_angle = _angle(right_shoulder, right_hip, right_knee, width, height)

                    if hip_x is not None and hip_y is not None:
                        if left_leg_tracking_valid:
                            left_foot_dx = left_foot_index.x - hip_x
                            left_foot_dy = left_foot_index.y - hip_y
                        if right_leg_tracking_valid:
                            right_foot_dx = right_foot_index.x - hip_x
                            right_foot_dy = right_foot_index.y - hip_y

                    if writer is not None or display_video:
                        _draw_skeleton(frame, landmarks, width, height)

                # Draw the most recently detected skeleton on skipped frames
                if (
                        not should_infer
                        and last_landmarks is not None
                        and (writer is not None or display_video)
                ):
                    _draw_skeleton(frame, last_landmarks, width, height)

                left_foot_vx = left_foot_vy = None
                right_foot_vx = right_foot_vy = None
                if should_infer:
                    # Existing foot velocity and speed calculations
                    (left_foot_vx, left_foot_vy,
                     previous_left_foot_dx, previous_left_foot_dy,
                     previous_left_foot_timestamp) = calculateFootVelocity(
                        left_foot_dx, left_foot_dy, current_timestamp,
                        previous_left_foot_dx, previous_left_foot_dy,
                        previous_left_foot_timestamp,
                    )
                    (right_foot_vx, right_foot_vy,
                     previous_right_foot_dx, previous_right_foot_dy,
                     previous_right_foot_timestamp) = calculateFootVelocity(
                        right_foot_dx, right_foot_dy, current_timestamp,
                        previous_right_foot_dx, previous_right_foot_dy,
                        previous_right_foot_timestamp,
                    )
                    left_foot_speed = calculateFootSpeed(left_foot_vx, left_foot_vy)
                    right_foot_speed = calculateFootSpeed(right_foot_vx, right_foot_vy)

                # Only update elbow-bend tracking on analyzed frames
                if should_infer:
                    left_deep_bent_start, left_missing_start = trackDeepBendDuration(
                        left_elbow_position, current_timestamp,
                        left_deep_bent_start, left_missing_start, left_deep_bent_intervals,
                    )

                    right_deep_bent_start, right_missing_start = trackDeepBendDuration(
                        right_elbow_position, current_timestamp,
                        right_deep_bent_start, right_missing_start, right_deep_bent_intervals,
                    )

                metrics_log.append({
                    "frame": frame_number,
                    "timestamp_seconds": current_timestamp,
                    "hip_x": hip_x,
                    "hip_y": hip_y,
                    "hip_vertical_velocity": hip_velocity_y,
                    "left_arm_valid": left_arm_valid,
                    "left_elbow_angle": left_elbow_angle,
                    "left_elbow_position": left_elbow_position,
                    "right_arm_valid": right_arm_valid,
                    "right_elbow_angle": right_elbow_angle,
                    "right_elbow_position": right_elbow_position,
                    "left_leg_tracking_valid": left_leg_tracking_valid,
                    "right_leg_tracking_valid": right_leg_tracking_valid,
                    "left_knee_angle": left_knee_angle,
                    "right_knee_angle": right_knee_angle,
                    "left_hip_angle": left_hip_angle,
                    "right_hip_angle": right_hip_angle,
                    "left_foot_dx": left_foot_dx,
                    "left_foot_dy": left_foot_dy,
                    "right_foot_dx": right_foot_dx,
                    "right_foot_dy": right_foot_dy,
                    "left_foot_vx": left_foot_vx,
                    "left_foot_vy": left_foot_vy,
                    "right_foot_vx": right_foot_vx,
                    "right_foot_vy": right_foot_vy,
                    "left_foot_speed": left_foot_speed,
                    "right_foot_speed": right_foot_speed,
                    "inference_performed": should_infer
                })

                if writer is not None:
                    writer_start = time.perf_counter()

                    try:
                        writer.stdin.write(frame.tobytes())
                    except BrokenPipeError as exc:
                        raise RuntimeError("FFmpeg stopped accepting video frames.") from exc

                    total_writer_time += time.perf_counter() - writer_start
                    writer_count += 1

                if display_video:
                    scale = min(1200 / width, 900 / height, 1.0)
                    display_frame = cv2.resize(frame, None, fx=scale, fy=scale)
                    cv2.imshow("AI Climbing Coach - Pose Detection", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        frame_number += 1
                        break
                frame_number += 1


            if inference_count > 0:
                average_inference_time = total_inference_time / inference_count

                print(
                    f"[Performance] MediaPipe inference total: "
                    f"{total_inference_time:.2f} seconds"
                )

                print(
                    f"[Performance] MediaPipe inference average: "
                    f"{average_inference_time * 1000:.2f} ms/frame"
                )

                print(
                    f"[Performance] Frames analyzed: "
                    f"{inference_count}"
                )

            if writer_count > 0:
                average_writer_time = total_writer_time / writer_count

                print(
                    f"[Performance] FFmpeg frame streaming total: "
                    f"{total_writer_time:.2f} seconds"
                )

                print(
                    f"[Performance] FFmpeg frame streaming average: "
                    f"{average_writer_time * 1000:.2f} ms/frame"
                )

                print(f"[Performance] Frames written: {writer_count}")

        _finish_interval(left_deep_bent_start, left_missing_start,
                         last_processed_timestamp, left_deep_bent_intervals)
        _finish_interval(right_deep_bent_start, right_missing_start,
                         last_processed_timestamp, right_deep_bent_intervals)
        return {
            "frames": metrics_log,
            "left_arm": {
                "deep_bend_intervals": left_deep_bent_intervals,
                "summary": summarizeDeepBendIntervals(left_deep_bent_intervals),
            },
            "right_arm": {
                "deep_bend_intervals": right_deep_bent_intervals,
                "summary": summarizeDeepBendIntervals(right_deep_bent_intervals),
            },
        }
    finally:
        cap.release()

        if writer is not None:
            if writer.stdin is not None:
                writer.stdin.close()

            return_code = writer.wait()

            if return_code != 0:
                raise RuntimeError(
                    f"FFmpeg video encoding failed with exit code {return_code}"
                )

        if display_video:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    test_video = BASE_DIR.parent / "climbingTestTension.mov"
    annotated_video = BASE_DIR / "test_annotated.mp4"
    result = process_climbing_video(
        test_video, display_video=False, annotated_output_path=annotated_video
    )
    print(f"Processed {len(result['frames'])} frames")
    print(f"Left arm: {result['left_arm']['summary']}")
    print(f"Right arm: {result['right_arm']['summary']}")
    print(f"Annotated video saved to: {annotated_video}")
