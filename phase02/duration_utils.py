import math

def trackDeepBendDuration(
        elbow_position,
        current_timestamp,
        deep_bent_start,
        missing_start,
        deep_bent_intervals,
        tolerance=0.15
    ):
    # How long a climber maintains an elbow position
    if elbow_position == "Deeply Bent":
        if deep_bent_start is None:
            deep_bent_start = current_timestamp

        missing_start = None

    elif elbow_position in ("Straight", "Slightly Bent"):

        if deep_bent_start is not None:
            end = missing_start if missing_start is not None else current_timestamp


            deep_bent_intervals.append({
                "start": deep_bent_start,
                "end": end,
                "duration": end - deep_bent_start,
                "status": "interrupted" if missing_start is not None else "completed"
            })

            deep_bent_start = None
            missing_start = None
    else:
        # tracking interruption
        if deep_bent_start is not None:

            if missing_start is None:
                missing_start = current_timestamp

            if current_timestamp - missing_start > tolerance:
                deep_bent_intervals.append({
                    "start": missing_start,
                    "end": missing_start,
                    "duration": missing_start - deep_bent_start,
                    "status": "interrupted"
                })

                deep_bent_start = None
                missing_start = None

    return deep_bent_start, missing_start


# returns a dictionary of the longest duration, total duration, and average duration of a deeply bent elbow
def summarizeDeepBendIntervals(deep_bent_intervals):
    longest_duration = 0
    average_duration = 0
    total_duration = 0

    # grab the longest duration of deep bent interval
    for interval in deep_bent_intervals:
        duration = interval["duration"]
        longest_duration = max(duration, longest_duration)
        total_duration += duration

    if deep_bent_intervals:
        average_duration = (
            total_duration / len(deep_bent_intervals)
        )

    return {
        "longest_duration": longest_duration,
        "average_duration": average_duration,
        "total_duration": total_duration
    }


test_intervals = [
    {"start": 2.0, "end": 3.2, "duration": 1.2},
    {"start": 5.0, "end": 7.5, "duration": 2.5},
    {"start": 9.0, "end": 9.8, "duration": 0.8}
]

if __name__ == "__main__":

    intervals = []
    deep_bent_start = None
    missing_start = None

    test_frames = [
        ("Deeply Bent", 1.00),
        ("Deeply Bent", 1.10),
        (None, 1.20),
        (None, 1.30),
        (None, 1.40),
        ("Deeply Bent", 1.50),
        ("Straight", 1.60)
    ]

    for position, timestamp in test_frames:
        deep_bent_start, missing_start = trackDeepBendDuration(
            position,
            timestamp,
            deep_bent_start,
            missing_start,
            intervals
        )

    print(intervals)