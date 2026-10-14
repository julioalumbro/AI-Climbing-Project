
import cv2
import mediapipe as mp
from pathlib import Path
import json

# Step 1: Set up MediaPipe classes
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Step 2: Set up file paths
BASE_DIR = Path(__file__).resolve().parent

model_path = str(BASE_DIR / "pose_landmarker_full.task")
video_path = str(BASE_DIR.parent / "climbingTestKilter.MOV")

# Step 3: Configure the pose landmarker
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1
)

# Step 4: Open the video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

frame_number = 0

# Store metrics for each processed frame
metrics_log = []

# Previous valid hip measurements
previous_hip_y = None
previous_timestamp = None

# Step 5: Create the pose landmarker
with PoseLandmarker.create_from_options(options) as landmarker:

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        # Convert OpenCV BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert image into MediaPipe format
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Calculate timestamps
        timestamp_ms = int(frame_number * 1000 / fps)
        current_timestamp = frame_number / fps

        # Detect pose landmarks
        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # DEBUG: Check frame 971 regardless of pose detection
        if frame_number == 971:
            print("Reached frame 971!")
            print(f"Pose detected: {bool(result.pose_landmarks)}")

        # Initialize frame-specific variables
        # These reset every frame to prevent stale data
        tracking_valid = False

        hip_x = None
        hip_y = None
        hip_velocity_y = None

        left_hip_visibility = None
        right_hip_visibility = None

        # Step 6: Process pose landmarks
        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]

            # Get left and right hip landmarks
            left_hip = landmarks[23]
            right_hip = landmarks[24]

            # Store visibility scores for JSON logging
            left_hip_visibility = left_hip.visibility
            right_hip_visibility = right_hip.visibility

            # Check whether both hips are sufficiently visible
            left_visible = (
                left_hip_visibility is not None
                and left_hip_visibility >= 0.5
            )

            right_visible = (
                right_hip_visibility is not None
                and right_hip_visibility >= 0.5
            )

            # Store hip coordinates in local variables
            left_x = left_hip.x
            left_y = left_hip.y
            right_x = right_hip.x
            right_y = right_hip.y

            # Tracking is valid only if both hips are visible
            # and all four coordinates exist
            tracking_valid = (
                left_visible
                and right_visible
                and left_x is not None
                and left_y is not None
                and right_x is not None
                and right_y is not None
            )

            # DEBUG: Inspect hip landmarks at frame 971
            if frame_number == 971:
                print(f"Tracking valid: {tracking_valid}")
                print(f"Left hip: x={left_x}, y={left_y}")
                print(f"Right hip: x={right_x}, y={right_y}")
                print(f"Left visibility: {left_hip_visibility}")
                print(f"Right visibility: {right_hip_visibility}")

            # Calculate metrics only for valid tracking
            # Individual checks help PyCharm narrow coordinate types
            if (
                left_visible
                and right_visible
                and left_x is not None
                and left_y is not None
                and right_x is not None
                and right_y is not None
            ):
                # Calculate hip midpoint
                hip_x = (left_x + right_x) / 2
                hip_y = (left_y + right_y) / 2

                # Calculate vertical hip velocity
                if (
                    previous_hip_y is not None
                    and previous_timestamp is not None
                ):
                    delta_y = hip_y - previous_hip_y
                    delta_t = current_timestamp - previous_timestamp

                    if delta_t > 0:
                        hip_velocity_y = delta_y / delta_t

                # Update previous valid measurements
                previous_hip_y = hip_y
                previous_timestamp = current_timestamp

            # Get frame dimensions for drawing
            height, width = frame.shape[:2]

            # Draw landmark points
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

            # Draw skeleton connections
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

        # Step 7: Save metrics for EVERY processed frame
        # This must remain outside if result.pose_landmarks
        metrics_log.append({
            "frame": frame_number,
            "timestamp_seconds": current_timestamp,
            "hip_x": hip_x,
            "hip_y": hip_y,
            "hip_vertical_velocity": hip_velocity_y,
            "left_hip_visibility": left_hip_visibility,
            "right_hip_visibility": right_hip_visibility,
            "tracking_valid": tracking_valid
        })

        # Step 8: Display video
        height, width = frame.shape[:2]

        # Resize video for display without changing aspect ratio
        scale = min(800 / width, 600 / height, 1.0)

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

        # Press q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            frame_number += 1
            break

        frame_number += 1

# Step 9: Release resources
cap.release()
cv2.destroyAllWindows()

# Step 10: Export metrics to JSON
output_path = BASE_DIR / "metrics.json"

with open(output_path, "w") as f:
    json.dump(metrics_log, f, indent=4)

print(f"Saved {len(metrics_log)} frames of metrics to {output_path}")
print("Video processing complete!")
