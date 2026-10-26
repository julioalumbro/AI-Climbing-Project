import math

def calculateFootVelocity(
    foot_dx,
    foot_dy,
    current_timestamp,
    previous_dx,
    previous_dy,
    previous_timestamp,
    max_gap=0.5
):
    vx = None
    vy = None

    # missing tracking = reset previous measurements
    if foot_dx is None or foot_dy is None:
        return None, None, None, None, None

    # else calculate vel if previous measurement exist
    if (
        previous_dx is not None
        and previous_dy is not None
        and previous_timestamp is not None
    ):
        delta_t = current_timestamp - previous_timestamp
        # 0.05 second limit is a starting safeguard against calc
        # velocity across long tracking gaps, not a foot-cut detection threshold
        if 0 < delta_t <= 0.05:
            vx = (
                (foot_dx - previous_dx) / delta_t
            )

            vy = (
                (foot_dy - previous_dy) / delta_t
            )

    return vx, vy, foot_dx, foot_dy, current_timestamp


def calculateFootSpeed(vx,vy):
    if vx is None or vy is None:
        return None

    return math.hypot(vx, vy)

if __name__ == "__main__":

    res = []

    test_frames = [
        (0.0, 0.0),
        (0.3, 0.4),
        (-0.3, 0.4),
        (-0.6, -0.8),
        (None, -0.4),
    ]

    for vx, vy in test_frames:
        speed = calculateFootSpeed(vx, vy)
        res.append(speed)

    print(res)