
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

input_path = BASE_DIR / "clips" / "climbingTestDeepBentElbow_left_release_return.mp4"
output_path = BASE_DIR / "clips" / "left_foot_gemini_test.mp4"

cap = cv2.VideoCapture(str(input_path))

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

target_fps = 15
scale = 480 / width

new_width = 480
new_height = int(height * scale)

writer = cv2.VideoWriter(
    str(output_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    target_fps,
    (new_width, new_height)
)

frame_index = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    # Sample frames while maintaining the original duration
    if int(frame_index * target_fps / fps) > int(
        (frame_index - 1) * target_fps / fps
    ):
        resized = cv2.resize(frame, (new_width, new_height))
        writer.write(resized)

    frame_index += 1

cap.release()
writer.release()

print(f"Saved: {output_path}")
test = cv2.VideoCapture(str(output_path))

print("Video opened:", test.isOpened())
print("Frame count:", test.get(cv2.CAP_PROP_FRAME_COUNT))
print("FPS:", test.get(cv2.CAP_PROP_FPS))

test.release()
