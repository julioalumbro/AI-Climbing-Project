import cv2
import mediapipe as mp
from pathlib import Path
import json
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from angle_utils import calculateAngle, classifyElbowPosition
from velocity_utils import calculateFootVelocity, calculateFootSpeed
from phase02.duration_utils import trackDeepBendDuration, summarizeDeepBendIntervals

# step 1: set up mediapipe classes
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# step 2: set up file paths
BASE_DIR = Path(__file__).resolve().parent

model_path = str(
    BASE_DIR.parent
    / "phase01-opencv_videoloading"
    / "pose_landmarker_full.task"
)

video_path = str(BASE_DIR.parent / "climbingTestTension.mov")

# step 3: configure the pose landmarker
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1
)

# step 4: open the video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

frame_number = 0

# store metrics for each processed frame
metrics_log = []

# previous valid hip measurements
previous_hip_y = None
previous_hip_timestamp = None

# previous left foot measurements
previous_left_foot_dx = None
previous_left_foot_dy = None
previous_left_foot_timestamp = None

# previous right foot measurements
previous_right_foot_dx = None
previous_right_foot_dy = None
previous_right_foot_timestamp = None

left_deep_bent_start: float | None = None
left_deep_bent_intervals = []
avg_left_deep_bent_duration = None
left_missing_start: float | None = None

right_deep_bent_start: float | None = None
right_deep_bent_intervals = []
avg_right_deep_bent_duration = None
right_missing_start: float | None = None

last_processed_timestamp: float | None = None


# step 5: create the pose landmarker
with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        # convert opencv bgr image to rgb
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # convert image into mediapipe format
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # calculate timestamps
        timestamp_ms = int(frame_number * 1000 / fps)
        current_timestamp = frame_number / fps
        last_processed_timestamp = current_timestamp

        # detect pose landmarks
        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # initialize frame-specific variables
        # hips
        hip_tracking_valid = False
        hip_x = None
        hip_y = None
        hip_velocity_y = None
        left_hip_visibility: float | int | None = None
        right_hip_visibility: float | int | None = None

        # arms
        left_arm_valid = False
        left_elbow_angle = None
        right_arm_valid = False
        right_elbow_angle = None

        left_elbow_position = None
        left_deep_bent_duration = None
        right_elbow_position = None
        right_deep_bent_duration = None

        # legs (ankle, heel, foot index)
        # left leg landmarks

        left_hip: NormalizedLandmark | None = None
        left_knee = None
        left_ankle = None
        left_heel = None
        left_foot_index = None

        # right leg landmarks
        right_hip: NormalizedLandmark | None = None
        right_knee = None
        right_ankle = None
        right_heel = None
        right_foot_index = None

        # knee and ankle coordinates (reset for each frame)
        left_knee_x: float | int | None = None
        left_knee_y: float | int | None = None
        left_ankle_x: float | int | None = None
        left_ankle_y: float | int | None = None
        right_knee_x: float | int | None = None
        right_knee_y: float | int | None = None
        right_ankle_x: float | int | None = None
        right_ankle_y: float | int | None = None

        # lower-body tracking validity
        left_leg_tracking_valid = False
        right_leg_tracking_valid = False
        left_knee_angle = None
        right_knee_angle = None

        # lower-body hip angle calculations
        left_hip_angle = None
        right_hip_angle = None

        # foot positioning relative to hips
        left_foot_dx = None
        left_foot_dy = None

        right_foot_dx = None
        right_foot_dy = None

        # foot velocity calc
        left_foot_vx = None
        left_foot_vy = None
        right_foot_vx = None
        right_foot_vy = None


        # step 6: process pose landmarks
        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]

            # retrieve hip landmarks
            left_hip = landmarks[23]
            right_hip = landmarks[24]

            # retrieve left arm landmarks
            left_shoulder = landmarks[11]
            left_elbow = landmarks[13]
            left_wrist = landmarks[15]

            # retrieve right arm landmarks
            right_shoulder = landmarks[12]
            right_elbow = landmarks[14]
            right_wrist = landmarks[16]

            # retrieve left and right leg landmarks
            # left
            # hip has already been init
            left_knee = landmarks[25]
            left_ankle = landmarks[27]
            left_heel = landmarks[29]
            left_foot_index = landmarks[31]

            # right
            # hip has already been init
            right_knee = landmarks[26]
            right_ankle = landmarks[28]
            right_heel = landmarks[30]
            right_foot_index = landmarks[32]

            right_ankle = landmarks[28]
            right_heel = landmarks[30]
            right_foot = landmarks[32]

            if frame_number % 60 == 0:
                print(f"\nFrame: {frame_number}")
                print(f"Right ankle visibility: {right_ankle.visibility:.3f}")
                print(f"Right heel visibility: {right_heel.visibility:.3f}")
                print(f"Right foot visibility: {right_foot.visibility:.3f}")
                print(f"Right foot X: {right_foot.x:.3f}")
                print(f"Right foot Y: {right_foot.y:.3f}")

            # ----------------------------------
            # right arm visibility and validation
            # ----------------------------------

            right_shoulder_visibility = right_shoulder.visibility
            right_elbow_visibility = right_elbow.visibility
            right_wrist_visibility = right_wrist.visibility

            right_shoulder_x = right_shoulder.x
            right_shoulder_y = right_shoulder.y
            right_elbow_x = right_elbow.x
            right_elbow_y = right_elbow.y
            right_wrist_x = right_wrist.x
            right_wrist_y = right_wrist.y

            right_arm_valid = (

                right_shoulder_visibility is not None
                and right_elbow_visibility is not None
                and right_wrist_visibility is not None
                and right_shoulder_visibility >= 0.5
                and right_elbow_visibility >= 0.5
                and right_wrist_visibility >= 0.5

                and right_shoulder_x is not None
                and right_shoulder_y is not None
                and right_elbow_x is not None
                and right_elbow_y is not None
                and right_wrist_x is not None
                and right_wrist_y is not None
            )


            if right_arm_valid:
                height, width = frame.shape[:2]

                # get the x and y pixels using the left and right coordinates
                right_shoulder_point = (
                    right_shoulder_x * width,
                    right_shoulder_y * height
                )
                # e.g. elbow_point = (650, 240) --> format for calculateangle()
                right_elbow_point = (
                    right_elbow_x * width,
                    right_elbow_y * height
                )

                right_wrist_point = (
                    right_wrist_x * width,
                    right_wrist_y * height
                )

                right_elbow_angle = calculateAngle(
                    right_shoulder_point,
                    right_elbow_point,
                    right_wrist_point
                )

                right_elbow_position = classifyElbowPosition(right_elbow_angle)

                if right_elbow_angle is not None:
                    cv2.putText(
                        # image were drawing
                        frame,
                        # text value (the angle rounded to deg)
                        f"{right_elbow_angle:.1f} deg",
                        # the location of the drawing
                        (int(right_elbow_point[0]) + 15, int(right_elbow_point[1]) - 15),
                        # font
                        cv2.FONT_HERSHEY_SIMPLEX,
                        # font scale (float|int)
                        0.8,
                        # color
                        (255, 255, 255),
                        # thickness
                        3
                    )

            # ----------------------------------
            # left arm visibility and validation
            # ----------------------------------

            left_shoulder_visibility = left_shoulder.visibility
            left_elbow_visibility = left_elbow.visibility
            left_wrist_visibility = left_wrist.visibility

            left_shoulder_x = left_shoulder.x
            left_shoulder_y = left_shoulder.y

            left_elbow_x = left_elbow.x
            left_elbow_y = left_elbow.y

            left_wrist_x = left_wrist.x
            left_wrist_y = left_wrist.y

            left_arm_valid = (
                left_shoulder_visibility is not None
                and left_elbow_visibility is not None
                and left_wrist_visibility is not None
                and left_shoulder_visibility >= 0.5
                and left_elbow_visibility >= 0.5
                and left_wrist_visibility >= 0.5

                # coordinate checks
                and left_shoulder_x is not None
                and left_shoulder_y is not None
                and left_elbow_x is not None
                and left_elbow_y is not None
                and left_wrist_x is not None
                and left_wrist_y is not None
            )


            if left_arm_valid:
                height, width = frame.shape[:2]

                # get the x and y pixels using the left and right coordinates
                left_shoulder_point = (
                    left_shoulder_x * width,
                    left_shoulder_y * height
                )
                # e.g. elbow_point = (650, 240) --> format for calculateangle()
                left_elbow_point = (
                    left_elbow_x * width,
                    left_elbow_y * height
                )

                left_wrist_point = (
                    left_wrist_x * width,
                    left_wrist_y * height
                )

                left_elbow_angle = calculateAngle(
                        left_shoulder_point,
                        left_elbow_point,
                        left_wrist_point
                )

                left_elbow_position = classifyElbowPosition(left_elbow_angle)

                    # drawing the left_elbow_angle onto our video
                    # we have it originally saved in json but now we want to display it
                    # rmb movement away from the camera can affect the measurements
                if left_elbow_angle is not None:
                    cv2.putText(
                        # image were drawing
                        frame,
                        # text value (the angle rounded to deg)
                        f"{left_elbow_angle:.1f} deg",
                        # the location of the drawing
                        (int(left_elbow_point[0]) + 15, int(left_elbow_point[1]) - 15),
                        # font
                        cv2.FONT_HERSHEY_SIMPLEX,
                        # font scale (float|int)
                        0.8,
                        # color
                        (255, 255, 255),
                        # thickness
                        3
                    )

            # ----------------------------------
            # hip visibility and validation
            # ----------------------------------

            left_hip_visibility = left_hip.visibility
            right_hip_visibility = right_hip.visibility

            left_hip_visible = (
                    left_hip_visibility is not None
                    and left_hip_visibility >= 0.5
            )

            right_hip_visible = (
                    right_hip_visibility is not None
                    and right_hip_visibility >= 0.5
            )

            # store hip coordinates
            left_hip_x = left_hip.x
            left_hip_y = left_hip.y

            right_hip_x = right_hip.x
            right_hip_y = right_hip.y

            # store knee and ankle coordinates
            left_knee_x, left_knee_y = left_knee.x, left_knee.y
            left_ankle_x, left_ankle_y = left_ankle.x, left_ankle.y
            right_knee_x, right_knee_y = right_knee.x, right_knee.y
            right_ankle_x, right_ankle_y = right_ankle.x, right_ankle.y

            # validate hip tracking
            hip_tracking_valid = (
                    left_hip_visible
                    and right_hip_visible
                    and left_hip_x is not None
                    and left_hip_y is not None
                    and right_hip_x is not None
                    and right_hip_y is not None
            )

            # ----------------------------------
            # hip position and velocity
            # ----------------------------------

            if (
                hip_tracking_valid
                and left_hip_x is not None and left_hip_y is not None
                and right_hip_x is not None and right_hip_y is not None
            ):
                # calculate hip midpoint
                hip_x = (left_hip_x + right_hip_x) / 2
                hip_y = (left_hip_y + right_hip_y) / 2

                # calculate vertical hip velocity
                if (
                        previous_hip_y is not None
                        and previous_hip_timestamp is not None
                ):
                    hip_delta_y = hip_y - previous_hip_y
                    hip_delta_t = (
                            current_timestamp - previous_hip_timestamp
                    )

                    if hip_delta_t > 0:
                        hip_velocity_y = hip_delta_y / hip_delta_t

                # update previous valid hip measurements
                previous_hip_y = hip_y
                previous_hip_timestamp = current_timestamp

            # ----------------------------------
            # legs + foot-hip positioning
            # ----------------------------------

            # all func returns true only if every condition is satisfied
            left_leg_tracking_valid = all(
                isinstance(point, NormalizedLandmark)
                and point.visibility is not None
                and point.visibility >= 0.5
                and point.x is not None
                and point.y is not None
                for point in [
                    left_hip,
                    left_knee,
                    left_ankle,
                    left_heel,
                    left_foot_index
                ]
            )

            right_leg_tracking_valid = all(
                isinstance(point, NormalizedLandmark)
                and point.visibility is not None
                and point.visibility >= 0.5
                and point.x is not None
                and point.y is not None
                for point in [
                    right_hip,
                    right_knee,
                    right_ankle,
                    right_heel,
                    right_foot_index
                ]
            )
            # knee angles: explicitly narrow optional coordinate types.
            if (
                left_leg_tracking_valid
                and left_hip_x is not None and left_hip_y is not None
                and left_knee_x is not None and left_knee_y is not None
                and left_ankle_x is not None and left_ankle_y is not None
            ):
                left_knee_angle = calculateAngle(
                    [left_hip_x, left_hip_y],
                    [left_knee_x, left_knee_y],
                    [left_ankle_x, left_ankle_y]
                )

            if (
                right_leg_tracking_valid
                and right_hip_x is not None and right_hip_y is not None
                and right_knee_x is not None and right_knee_y is not None
                and right_ankle_x is not None and right_ankle_y is not None
            ):
                right_knee_angle = calculateAngle(
                    [right_hip_x, right_hip_y],
                    [right_knee_x, right_knee_y],
                    [right_ankle_x, right_ankle_y]
                )

            # hip angles: shoulder -> hip -> knee.
            if (
                left_leg_tracking_valid and left_arm_valid
                and left_shoulder_x is not None and left_shoulder_y is not None
                and left_hip_x is not None and left_hip_y is not None
                and left_knee_x is not None and left_knee_y is not None
            ):
                left_hip_angle = calculateAngle(
                    [left_shoulder_x, left_shoulder_y],
                    [left_hip_x, left_hip_y],
                    [left_knee_x, left_knee_y]
                )

            if (
                right_leg_tracking_valid and right_arm_valid
                and right_shoulder_x is not None and right_shoulder_y is not None
                and right_hip_x is not None and right_hip_y is not None
                and right_knee_x is not None and right_knee_y is not None
            ):
                right_hip_angle = calculateAngle(
                    [right_shoulder_x, right_shoulder_y],
                    [right_hip_x, right_hip_y],
                    [right_knee_x, right_knee_y]
                )

            # foot positions relative to the validated hip midpoint.
            if hip_x is not None and hip_y is not None:
                if (
                    left_leg_tracking_valid
                    and left_foot_index is not None
                    and left_foot_index.x is not None
                    and left_foot_index.y is not None
                ):
                    left_foot_dx = left_foot_index.x - hip_x
                    left_foot_dy = left_foot_index.y - hip_y

                if (
                    right_leg_tracking_valid
                    and right_foot_index is not None
                    and right_foot_index.x is not None
                    and right_foot_index.y is not None
                ):
                    right_foot_dx = right_foot_index.x - hip_x
                    right_foot_dy = right_foot_index.y - hip_y


            # ----------------------------------
            # draw pose landmarks
            # ----------------------------------

            height, width = frame.shape[:2]

            for landmark in landmarks:
                x = landmark.x
                y = landmark.y

                if x is None or y is None:
                    continue

                pixel_x = int(x * width)
                pixel_y = int(y * height)

                cv2.circle(
                    frame,
                    (pixel_x, pixel_y),
                    4,
                    (0, 0, 255),
                    -1
                )

            # draw skeleton connections
            connections = [
                (11, 12),
                (11, 13), (13, 15),
                (12, 14), (14, 16),
                (11, 23), (12, 24),
                (23, 24),
                (23, 25), (25, 27),
                (24, 26), (26, 28)
            ]

            for start, end in connections:
                p1 = landmarks[start]
                p2 = landmarks[end]

                x1 = p1.x
                y1 = p1.y
                x2 = p2.x
                y2 = p2.y

                if (
                        x1 is None
                        or y1 is None
                        or x2 is None
                        or y2 is None
                ):
                    continue

                cv2.line(
                    frame,
                    (int(x1 * width), int(y1 * height)),
                    (int(x2 * width), int(y2 * height)),
                    (0, 255, 0),
                    2
                )

        # left foot velocity calc
        (
            left_foot_vx,
            left_foot_vy,
            previous_left_foot_dx,
            previous_left_foot_dy,
            previous_left_foot_timestamp
        ) = calculateFootVelocity(
            left_foot_dx,
            left_foot_dy,
            current_timestamp,
            previous_left_foot_dx,
            previous_left_foot_dy,
            previous_left_foot_timestamp
        )

        # right foot velocity
        (
            right_foot_vx,
            right_foot_vy,
            previous_right_foot_dx,
            previous_right_foot_dy,
            previous_right_foot_timestamp
        ) = calculateFootVelocity(
            right_foot_dx,
            right_foot_dy,
            current_timestamp,
            previous_right_foot_dx,
            previous_right_foot_dy,
            previous_right_foot_timestamp
        )

        # get overall foot velocity
        left_foot_speed = calculateFootSpeed(
            left_foot_vx,
            left_foot_vy
        )

        right_foot_speed = calculateFootSpeed(
            right_foot_vx,
            right_foot_vy
        )

        left_deep_bent_start, left_missing_start = trackDeepBendDuration(
            left_elbow_position,
            current_timestamp,
            left_deep_bent_start,
            left_missing_start,
            left_deep_bent_intervals
        )
        right_deep_bent_start, right_missing_start = trackDeepBendDuration(
            right_elbow_position,
            current_timestamp,
            right_deep_bent_start,
            right_missing_start,
            right_deep_bent_intervals
        )

        # step 7: save metrics for every processed frame
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
        })
        # step 8: display video
        height, width = frame.shape[:2]
        max_display_width = 1200
        max_display_height = 900
        scale = min(max_display_width / width, max_display_height / height, 1.0)

        display_frame = cv2.resize(
            frame,
            None,
            fx=scale,
            fy=scale
        )

        cv2.imshow(
            "AI Climbing Coach - Pose Detection",
            display_frame
        )

        # press q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            frame_number += 1
            break

        frame_number += 1

    # step 1: finalize any unfinished deep-bend intervals
    # this handles videos ending while the climber is still deeply bent
    # or while elbow tracking is temporarily lost.

    # left arm
    if left_deep_bent_start is not None and last_processed_timestamp is not None:
        end = (
            left_missing_start
            if left_missing_start is not None
            else last_processed_timestamp
        )

        left_deep_bent_intervals.append({
            "start": left_deep_bent_start,
            "end": end,
            "duration": end - left_deep_bent_start,
            "status": "interrupted" if left_missing_start is not None else "video_ended"
        })

        left_deep_bent_start = None
        left_missing_start = None

    # right arm
    if right_deep_bent_start is not None and last_processed_timestamp is not None:
        end = (
            right_missing_start
            if right_missing_start is not None
            else last_processed_timestamp
        )

        right_deep_bent_intervals.append({
            "start": right_deep_bent_start,
            "end": end,
            "duration": end - right_deep_bent_start,
            "status": "interrupted" if right_missing_start is not None else "video_ended"
        })

        right_deep_bent_start = None
        right_missing_start = None

    # step 2: calculate summary statistics
    left_summary = summarizeDeepBendIntervals(left_deep_bent_intervals)
    right_summary = summarizeDeepBendIntervals(right_deep_bent_intervals)

    output_data = {
        "frames": metrics_log,
        "left_arm": {
            "deep_bend_intervals": left_deep_bent_intervals,
            "summary": left_summary
        },
        "right_arm": {
            "deep_bend_intervals": right_deep_bent_intervals,
            "summary": right_summary
        }
    }

# step 9: release resources
cap.release()
cv2.destroyAllWindows()

# step 10: export metrics to json
output_path = BASE_DIR / "metrics_tension.json"

with open(output_path, "w") as f:
    json.dump(output_data, f, indent=4)

print(f"Saved {len(metrics_log)} frames of metrics to {output_path}")
print("Video processing complete!")

