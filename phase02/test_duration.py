from duration_utils import trackDeepBendDuration


# testing short/long tracking interruption
intervals = []
deep_bent_start = None
missing_start = None


# short tracking interruption
# test_frames = [
#     ("deeply bent", 1.00),
#     ("deeply bent", 1.10),
#     (none, 1.20),
#     (none, 1.25),
#     ("deeply bent", 1.30),
#     ("deeply bent", 1.40),
#     ("straight", 1.50)
# ]

# long tracking interruption
# test_frames = [
#      ("deeply bent", 1.00),
#       ("deeply bent", 1.10),
#       (none, 1.20),
#      (none, 1.30),
#      (none, 1.40),
#       ("deeply bent", 1.50),
#       ("straight", 1.60)
# ]

# edge case- elbow is alr deeply bent,
# mediapipe loses tracking
# when tracking returns, your elbow is straight
test_frames = [
    ("Deeply Bent", 1.00),
    ("Deeply Bent", 1.10),
    (None, 1.20),
    ("Straight", 1.30)
]

for position, timestamp in test_frames:
    deep_bent_start, missing_start = trackDeepBendDuration(
        position,
        timestamp,
        deep_bent_start,
        missing_start,
        intervals
    )

# test for edge case
assert len(intervals) == 1
assert intervals[0]["status"] == "interrupted"
assert abs(intervals[0]["start"] - 1.0) < 0.000001
assert abs(intervals[0]["end"] - 1.2) < 0.000001
assert abs(intervals[0]["duration"] - 0.2) < 0.000001

print("Test 3 passed!")