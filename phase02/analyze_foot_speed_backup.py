import json
import statistics
import math
import sys

import numpy as np

# event labels for foot classifications
# each dict key contains: foot, start_frame, end_frame
EVENT_LABELS = {
    ("left", 347, 369): "foot_release",
    ("right", 351, 364): "foot_release",
    ("left", 388, 400): "foot_return",
    ("right", 379, 382): "foot_return",
    ("right", 168, 169): "foot_readjustment",
    ("right", 431, 433): "foot_return",  # Newly verified
    ("right", 474, 486): "foot_release"
}
def summarizeSpeeds(json_path):
    if not json_path:
        return None

    with open(json_path, "r") as file:
        data = json.load(file)

    frames = data["frames"]

    left_speeds = []
    right_speeds = []

    for frame in frames:
        left_speed = frame.get("left_foot_speed")
        right_speed = frame.get("right_foot_speed")

        if left_speed is not None:
            left_speeds.append(left_speed)

        if right_speed is not None:
            right_speeds.append(right_speed)

    print("First Frame:", frames[0])

    # P50 = typical foot speed
    # P90 = 90% of observed speed are at or below this val
    # P95 = top 5% of observed speeds
    # P99 top 1% of observed speeds
    left_pctiles = (
        np.percentile(left_speeds, [50, 90, 95, 99])
        if left_speeds else None
    )
    right_pctiles = (
        np.percentile(right_speeds, [50, 90, 95, 99])
        if right_speeds else None
    )

    print("Total Frames:", len(frames))
    print("Left valid speeds:", len(left_speeds))
    print("Right valid speeds:", len(right_speeds))


    return {
        "left_foot_speed_count": len(left_speeds),
        "left_foot_speed_mean":
            statistics.mean(left_speeds)
            if left_speeds else None,
        "left_foot_speed_median": statistics.median(left_speeds),
        "left_foot_speed_max": max(left_speeds),
        "left_foot_speed_std": (
            statistics.stdev(left_speeds)
            if len(left_speeds) > 1 else 0.0),
        "right_foot_speed_count": len(right_speeds),
        "right_foot_speed_mean":
            statistics.mean(right_speeds)
            if right_speeds else None,
        "right_foot_speed_median": statistics.median(right_speeds),
        "right_foot_speed_max": max(right_speeds),
        "right_foot_speed_std": (
            statistics.stdev(right_speeds)
            if len(right_speeds) > 1 else 0.0
        ),
        "total_frames": len(frames),
        "left_total_tracking_coverage": len(left_speeds) / len(frames) * 100
        if frames else 0,
        "right_total_tracking_coverage": len(right_speeds) / len(frames) * 100
        if frames else 0,
        "left_foot_percentiles": (
            left_pctiles.tolist()
        if left_pctiles is not None else None
        ),
        "right_foot_percentiles": (
            right_pctiles.tolist()
        if right_pctiles is not None else None
        )
    }

def findHighSpeedFrames(frames, speed_key, threshold):
    high_speed_frames = []
    for index, frame in enumerate(frames):
        speed = frame.get(speed_key)

        if speed is not None and speed >threshold:
            high_speed_frames.append({
                "frame_index": index,
                "speed": speed
            })

    return high_speed_frames

def groupHighSpeedFrames(high_speed_frames, max_gap = 2):
    if not high_speed_frames:
        return None
    events = []
    current_event = [high_speed_frames[0]]

    for frame in high_speed_frames[1:]:
        prev_frame = current_event[-1]

        missing_frames = (
            frame["frame_index"]
            - prev_frame["frame_index"]
            - 1
        )

        if missing_frames <= max_gap:
            current_event.append(frame)
        else:
            events.append(current_event)
            current_event = [frame]

    events.append(current_event)

    return events

def summarizeMovementEvents(grouped_events, frames, foot, fps):
    summaries = []
    for event in grouped_events:
        start_frame = event[0]["frame_index"]
        end_frame = event[-1]["frame_index"]

        peak_speed = max(
            frame["speed"] for frame in event
        )

        displacement = calculateEventDisplacement(
            frames,
            start_frame,
            end_frame,
            foot
        )

        path_length = calculateEventPathLength(
            frames,
            start_frame,
            end_frame,
            foot
        )

        path_efficiency = calculatePathEfficiency(
            displacement["net displacement"] if displacement else None,
            path_length
        )

        summaries.append({
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time": start_frame / fps,
            "end_time": end_frame / fps,
            "frame count": len(event),
            "duration": (end_frame - start_frame + 1) / fps,
            "peak_speed": peak_speed,
            "displacement": displacement,
            "path_length": path_length,
            "path_efficiency": path_efficiency,
        })

    return summaries

def inspectFootMovement(frames, start_frame, end_frame, foot):
    for index in range(start_frame, end_frame + 1):
        frame = frames[index]

        print({
            "frame": frame["frame"],
            "dx": frame.get(f"{foot}_foot_dx"),
            "dy": frame.get(f"{foot}_foot_dy"),
            "vx": frame.get(f"{foot}_foot_vx"),
            "vy": frame.get(f"{foot}_foot_vy"),
            "speed": frame.get(f"{foot}_foot_speed")
        })

def calculateEventDisplacement(frames, start_frame, end_frame, foot):
    if start_frame <= 0:
        return None

    start = frames[start_frame - 1]
    end = frames[end_frame]

    start_dx = start.get(f"{foot}_foot_dx")
    start_dy = start.get(f"{foot}_foot_dy")

    end_dx = end.get(f"{foot}_foot_dx")
    end_dy = end.get(f"{foot}_foot_dy")

    if None in (start_dx, start_dy, end_dx, end_dy):
        return None

    delta_x = end_dx - start_dx
    delta_y = end_dy - start_dy

    net_displacement = math.hypot(delta_x, delta_y )
    return {
        "delta_x": delta_x,
        "delta_y": delta_y,
        "net displacement": net_displacement
    }

def calculateEventPathLength(frames, start_frame, end_frame, foot):
    total_distance = 0.0

    for i in range(start_frame, end_frame + 1):
        previous = frames[i -1]
        current = frames[i]

        prev_dx = previous.get(f"{foot}_foot_dx")
        prev_dy = previous.get(f"{foot}_foot_dy")

        curr_dx = current.get(f"{foot}_foot_dx")
        curr_dy = current.get(f"{foot}_foot_dy")

        if None in (prev_dx, prev_dy, curr_dx, curr_dy):
            return None

        step_distance = math.hypot(
            curr_dx - prev_dx,
            curr_dy - prev_dy)

        total_distance += step_distance

    return total_distance

def calculatePathEfficiency(net_displacement, path_length):
    if net_displacement is None or path_length is None:
        return None

    if path_length == 0:
        return None

    return net_displacement / path_length

# define specific events for the event_labels
# iterate through summarized events
# construct a key w/ foot + frame range
# look that up in event labels, if there is a verified label, assign it,
# otherwise assign "unknown"
def labelMovementEvents(events, foot):
    if events is None or foot is None:
        return None

    for event in events:
        start_frame = event["start_frame"]
        end_frame = event["end_frame"]

        event_key = (foot, start_frame, end_frame)

        event["event_type"] = EVENT_LABELS.get(
            event_key,
            "unknown"
        )

    return events

def groupEventsByType(events):
    if events is None:
        return None
    grouped = {}
    for event in events:
        event_type = event["event_type"]

        if event_type not in grouped:
            grouped[event_type] = []

        grouped[event_type].append(event)

    return grouped

# we want statistics on the type of events
def summarizeEventsByType(grouped_events):
    summaries = {}

    for event_type, events in grouped_events.items():
        if event_type == "unknown":
            continue

        peak_speeds = []
        displacements = []
        durations = []

        for event in events:
            peak_speeds.append(event["peak_speed"])
            durations.append(event["duration"])


            displacement = event["displacement"]

            if displacement is not None:
                displacements.append(displacement["net displacement"]
                )

        summaries[event_type] = {
            "count": len(events),
            "avg_peak_speed": sum(peak_speeds) / len(peak_speeds),
            "avg_duration": sum(durations) / len(durations),
            "avg_displacement": (
                sum(displacements) / len(displacements)
                if displacements else None
            )
        }

    return summaries

def classifyMovementEvent(event):
    displacement = event["displacement"]

    if displacement is None:
        return "unknown"

    net_displacement = displacement["net displacement"]
    duration = event["duration"]

    # starting thresholds, will change
    if net_displacement >= 0.15 and duration >= 0.10:
        return "large_movement"

    return "small_movement"
# temp function to find overlapping events
def eventsOverlap(event1, event2, max_gap = 0):
    return (
        event1["start_frame"] <= event2["end_frame"] + max_gap
        and event2["start_frame"] <= event1["end_frame"] + max_gap
    )
# automatic function that finds overlapping events
def findOverlappingFootEvents(left_events, right_events):
    overlapping_events = []

    for left_event in left_events:
        for right_event in right_events:
            if eventsOverlap(left_event, right_event):
                overlapping_events.append( {
                    "left_event": left_event,
                    "right_event": right_event
                } )

    return overlapping_events
# calculates the amount frames in between two events
# if events overlap then ret 0
def calculateEventGap(event1, event2):
    if eventsOverlap(event1, event2):
        return 0
    # case1 : event1 happens before event2
    if event1["end_frame"] < event2["start_frame"]:
        return event2[ "start_frame"] - event1["end_frame"] - 1
    # case2: event2 happens before event1
    return event1["start_frame"] - event2["end_frame"] - 1


# we want to be able to group individual movements into one potential dynamic movement sequence
# the idea with this function: sort events by start frame to process them chronologically
# set an empty group list, current group as a list of the first elm in sorted events
# set a current end
# iterate through the sorted events: set the current group start/end frames (boundaries)
# ge th the gap between the group boundary vs the current event
# check if the gap is less/equal than/to the fixed max gap (set to 15 here can be changed later)
def groupNearbyEvents(events, max_gap = 15):
    if not events:
        return []

    sorted_events = sorted(events, key=lambda event: event["start_frame"]
    )

    groups = []
    current_group = [sorted_events[0]]
    current_end = sorted_events[0]["end_frame"]

    for event in sorted_events[1:]:
        group_boundary = {
            "start_frame": current_group[0]["start_frame"],
            "end_frame": current_end
        }

        gap = calculateEventGap(group_boundary, event)

        if gap <= max_gap:
            current_group.append(event)
            current_end = max(current_end, event["end_frame"])
        else:
            groups.append(current_group)
            current_group = [event]
            current_end = event["end_frame"]

    groups.append(current_group)
    return groups

def summarizeMovementGroup(group):
    release_events = []
    return_events = []
    other_events = []

    for event in group:
        if event["event_type"] == "foot_release":
            release_events.append(event)
        elif event["event_type"] == "foot_return":
            return_events.append(event)
        else:
            other_events.append(event)

    return {
        "start_frame": min(event["start_frame"] for event in group),
        "end_frame": max(event["end_frame"] for event in group),
        "release_counts": len(release_events),
        "return_count": len(return_events),
        "other_count": len(other_events),
    }
# idea is to grab release and return events. We want to let our model know that releases come before returns
def identifyMovementPhases(group):
    # create release and return []
    release_events = []
    return_events = []

    for event in group:
        if event["event_type"] == "foot_release":
            release_events.append(event)
        elif event["event_type"] == "foot_return":
            return_events.append(event)
        # case where there is no release or return
        # return incomplete sequence
    if not release_events or not return_events:
        return "incomplete_sequence"

    first_release = min(event["start_frame"] for event in release_events)
    first_return = min(event["start_frame"] for event in return_events)

    if first_release < first_return:
        return "release_then_recovery"

    return "unexpected_order"

# calculates the amt of time foot release and returns take
# find release events
# locate returns for the same foot
# calculate interval using frame numbers + fps
def pairFootReleaseAndReturn(group, fps=60):
    pairs = []

    sorted_events = sorted(
        group,
        key=lambda event: event["start_frame"]
    )

    used_returns = set()

    for release in sorted_events:
        if release["event_type"] != "foot_release":
            continue
        for i, recovery in enumerate(sorted_events):
            # we alr used this return event
            if i in used_returns:
                continue
            # this event_type isnt a return
            if recovery["event_type"] != "foot_return":
                continue
            # we need to make sure we're pairing release and recovery on the same feet
            if release["foot"] != recovery["foot"]:
                continue
            # release needs to come before recovery
            if recovery["start_frame"] <= release["end_frame"]:
                continue
            # otherwise grab the interval of frames
            interval_frames = (
                recovery["start_frame"] - release["end_frame"]
            )

            interval_seconds = interval_frames / fps

            pairs.append({
                "foot": release["foot"],
                "release_start": release["start_frame"],
                "release_end": release["end_frame"],
                "return_start": recovery["start_frame"],
                "return_end": recovery["end_frame"],
                "interval_frames": interval_frames,
                "interval_seconds": interval_seconds
            })

            used_returns.add(i)
            break
    return pairs


def summarizeRecoveryPairs(pairs):
    if not pairs:
        return None

    intervals = [
        pair["interval_seconds"]
        for pair in pairs
    ]

    return {
        "paired_count": len(pairs),
        "average_interval": sum(intervals) / len(intervals),
        "shorted_interval": min(intervals),
        "longest_interval": max(intervals),
    }

# now were onto creating a classifier for climbing movements
def classifyMovementDirection(event):
    displacement = event.get("displacement")

    if displacement is None:
        return "unknown"
    dx = displacement.get("delta_x")
    dy = displacement.get("delta_y")

    if dx is None or dy is None:
        return "unknown"

    if abs(dx) > abs(dy):
        return "rightward" if dx > 0 else "leftward"

    return "downward" if dy > 0 else "upward"

# check whether the horizontal displacement changes sign
def detectDirectionReversal(release_event, return_event):
    release_displacement = release_event.get("displacement")
    return_displacement = return_event.get("displacement")

    if release_displacement is None or return_displacement is None:
        return None

    release_dx = release_displacement.get("delta_x")
    return_dx = return_displacement.get("delta_x")

    if release_dx is None or return_dx is None:
        return None

    return release_dx * return_dx < 0

def detectMovementCycle(event1, event2, max_gap=30, min_displacement = 0.05):
    if event1["foot"] != event2["foot"]:
        return False

    if event2["start_frame"] <= event1["end_frame"]:
        return False

    gap = calculateEventGap(event1, event2)

    if gap > max_gap:
        return False

    displacement1 = event1.get("displacement")
    displacement2 = event2.get("displacement")

    if displacement1 is None or displacement2 is None:
        return False

    if displacement1["net displacement"] < min_displacement:
        return False
    if displacement2["net displacement"] < min_displacement:
        return False


    return detectDirectionReversal(event1, event2) is True

# function that automates detectMovementCycle
# iterates through sorted events and calls detectMovementCucle on each possible event pair
# greedy pairing algorithm --> chooses first eligile partner
# not necessarily the ebst possible pairing
def findMovementCycles(events, max_gap=30, min_displacement=0.05):
    cycles = []

    sorted_events = sorted(
        events, key=lambda event: event["start_frame"]
    )

    used_events = set()

    for i,event1 in enumerate(sorted_events):
        if i in used_events:
            continue
        # grabbing second event
        for j in range(i + 1, len(sorted_events)):
            if j in used_events:
                continue

            event2 = sorted_events[j]

            if detectMovementCycle(
                    event1,
                    event2,
                    max_gap = max_gap,
                    min_displacement = min_displacement):
                # append to cycle
                cycles.append({
                    "foot": event1["foot"],
                    "first_event": event1,
                    "second_event": event2,
                    "interval_seconds": (
                        event2["start_frame"] - event1["end_frame"]
                    ) / 60
                })

                used_events.add(i)
                used_events.add(j)
                break
    return cycles

# eval function --> returns TP, FP, FN
def evaluateMovementCycles(predicted_cycles, verified_pairs):
    predicted = {
        (
            cycle["foot"],
            cycle["first_event"]["start_frame"],
            cycle["second_event"]["start_frame"]
        )

        for cycle in predicted_cycles
    }

    actual = {
        (
            pair["foot"],
            pair["release_start"],
            pair["return_start"]
        )
        for pair in verified_pairs
    }

    # finds pairs in both sets
    tp = len(predicted & actual)
    # finds predictions missing from verified pairs
    fp = len(predicted - actual)
    # finds verified pairs detector missed
    fn = len(actual - predicted)

    return {
        "true positives": tp,
        "false positives": fp,
        "false negatives": fn,
    }
# evaluation metrics (precision, recall/fn, f1)
def calculateEvaluationMetrics(evaluation):
    tp = evaluation["true positives"]
    fp = evaluation["false positives"]
    fn = evaluation["false negatives"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }
# right now we use two thresholds to calculate movement cycles
# we wanna test how diff thresholds values affect how accurate our model is
def testMovementThresholds(events, verified_pairs):
    gap_values = [10,15,20,30,45]
    displacement_values = [0.03, 0.05, 0.08, 0.10]

    res = []

    for gap in gap_values:
        for displacement in displacement_values:

            cycles = findMovementCycles(
                events,
                max_gap = gap,
                min_displacement = displacement
            )

            evaluation = evaluateMovementCycles(
                cycles,
                verified_pairs
            )

            metrics = calculateEvaluationMetrics(evaluation)

            res.append({
                "max_gap": gap,
                "min_displacement": displacement,
                "detected_cycles": len(cycles),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"]
            })
    return res

# examines events
def inspectMovementEvent(event):
    displacement = event["displacement"]

    print(f"\nFoot: {event['foot']}")
    print(f"Frames: {event['start_frame']}-{event['end_frame']}")
    print(f"Manual label: {event['event_type']}")
    print(f"Duration: {event['duration']:.3f}s")
    print(f"Peak speed: {event['peak_speed']:.3f}")
    print(f"Delta X: {displacement['delta_x']:.4f}")
    print(f"Delta Y: {displacement['delta_y']:.4f}")
    print(f"Net displacement: {displacement['net displacement']:.4f}")
    print(f"Direction: {classifyMovementDirection(event)}")

#  make another mesureable feature. Right now the detector identifies direction reversing movement pairs
# not exactly release-return cycles
def calculateMovementDisplacementRatio(event1, event2):
    displacement1 = event1["displacement"]["net displacement"]
    displacement2 = event2["displacement"]["net displacement"]

    # cant divide by 0
    if displacement2 == 0:
        return None
    return displacement1 / displacement2


if __name__ == "__main__":
    with open("metrics_tension.json", "r") as file:
        data = json.load(file)

    frames = data["frames"]

    summary = summarizeSpeeds("metrics.json")

    left_p95 = summary["left_foot_percentiles"][2]
    right_p95 = summary["right_foot_percentiles"][2]

    left_events = findHighSpeedFrames(
        frames,
        "left_foot_speed",
        left_p95
    )
    right_events= findHighSpeedFrames(
        frames,
        "right_foot_speed",
        right_p95
    )

    left_groups = groupHighSpeedFrames(left_events)
    right_groups = groupHighSpeedFrames(right_events)

    left_summaries = summarizeMovementEvents(left_groups, frames, "left", 60)
    right_summaries = summarizeMovementEvents(right_groups, frames, "right", 60)

    left_summaries = labelMovementEvents(left_summaries, "left")
    right_summaries = labelMovementEvents(right_summaries, "right")

    overlapping_events = findOverlappingFootEvents(left_summaries, right_summaries)

    for event in left_summaries:
        event["foot"] = "left"
    for event in right_summaries:
        event["foot"] = "right"
    all_events = sorted(left_summaries + right_summaries, key=lambda event: event["start_frame"]
                    )
    movement_groups = groupNearbyEvents(all_events, max_gap = 15)
    print("Total input events:", len(all_events))
    print("Total grouped events:", sum(len(groups) for groups in movement_groups))

    for i, group in enumerate(movement_groups, start=1):
        summary = summarizeMovementGroup(group)
        phase = identifyMovementPhases(group)
        pairs = pairFootReleaseAndReturn(group, fps=60)

        print(f"\nMovement Group {i}")
        print(summary)
        print("Phase:", phase)
        print("Pairs found:", len(pairs))
        print("Pairs:", pairs)

        for pair in pairs:
            print(
                f"{pair['foot']} foot: "
                f"release {pair['release_start']}-{pair['release_end']} -> "
                f"return {pair['return_start']}-{pair['return_end']} | "
                f"interval: {pair['interval_seconds']:.3f}s"
            )
        recover_summary = summarizeRecoveryPairs(pairs)

        if recover_summary is not None:
            print("Recovery Summary:", recover_summary)

    for event in all_events:
        if event["event_type"] == "unknown":
            continue

        direction = classifyMovementDirection(event)

        print(
            event["foot"],
            event["event_type"],
            "->",
            direction
        )


    left_release = next(e for e in all_events if e["foot"] == "left" and e["start_frame"] == 347)
    left_return = next(e for e in all_events if e["foot"] == "left" and e["start_frame"] == 388)

    right_release = next(e for e in all_events if e["foot"] == "right" and e["start_frame"] == 351)
    right_return = next(e for e in all_events if e["foot"] == "right" and e["start_frame"] == 379)

    print("Left reversal:", detectDirectionReversal(left_release, left_return))
    print("Right reversal:", detectDirectionReversal(right_release, right_return))
    print(
        "Left movement cycle:",
        detectMovementCycle(left_release, left_return)
    )

    print(
        "Right movement cycle:",
        detectMovementCycle(right_release, right_return)
    )
    grouped_by_type = groupEventsByType(all_events)

    type_summaries = summarizeEventsByType(grouped_by_type)

    movement_cycles = findMovementCycles(all_events)

    print("\nAutomatically Detected Movement Cycles:")

    for cycle in movement_cycles:
        first = cycle["first_event"]
        second = cycle["second_event"]

        print(
            f"{cycle['foot']} foot: "
            f"{first['start_frame']}-{first['end_frame']} -> "
            f"{second['start_frame']}-{second['end_frame']} | "
            f"interval: {cycle['interval_seconds']:.3f}s"
        )

    verified_pairs = []

    for group in movement_groups:
        verified_pairs.extend(
            pairFootReleaseAndReturn(group, fps=60)
        )

    evaluation = evaluateMovementCycles(
        movement_cycles,
        verified_pairs
    )

    print("Evaluation:", evaluation)

    # eval metric function testing
    metrics = calculateEvaluationMetrics(evaluation)

    print("\nMovement Cycle Evaluation Metrics:")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")
    # testing different thresholds
    baseline_cycles = findMovementCycles(
        all_events,
        max_gap=30,
        min_displacement=0.05
    )

    expanded_cycles = findMovementCycles(
        all_events,
        max_gap=45,
        min_displacement=0.03
    )

    baseline_pairs = {
        (
            cycle["foot"],
            cycle["first_event"]["start_frame"],
            cycle["second_event"]["start_frame"]
        )
        for cycle in baseline_cycles
    }

    print("\nAdditional Movement Cycle Candidates:")

    for cycle in expanded_cycles:
        pair = (
            cycle["foot"],
            cycle["first_event"]["start_frame"],
            cycle["second_event"]["start_frame"]
        )

        if pair not in baseline_pairs:
            print(
                f"Foot: {cycle['foot']} | "
                f"First: {cycle['first_event']['start_frame']}-"
                f"{cycle['first_event']['end_frame']} | "
                f"Second: {cycle['second_event']['start_frame']}-"
                f"{cycle['second_event']['end_frame']}"
            )
    first_event = next(
        e for e in all_events
        if e["foot"] == "right" and e["start_frame"] == 431
    )

    second_event = next(
        e for e in all_events
        if e["foot"] == "right" and e["start_frame"] == 474
    )

    inspectMovementEvent(first_event)
    inspectMovementEvent(second_event)

    # release-return ratio checks
    left_ratio = calculateMovementDisplacementRatio(
        left_release, left_return
    )

    right_ratio = calculateMovementDisplacementRatio(
        right_release, right_return
    )

    reverse_ratio = calculateMovementDisplacementRatio(
        first_event, second_event
    )

    print(f"Left release-return ratio: {left_ratio:.3f}")
    print(f"Right release-return ratio: {right_ratio:.3f}")
    print(f"Right return-release ratio: {reverse_ratio:.3f}")
