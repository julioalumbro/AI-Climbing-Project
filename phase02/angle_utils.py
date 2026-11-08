import math

# calculate the angle at the middle landmark using two vectors.
def calculateAngle(A, B, C):

    # step 1: calculate vectors ba and bc
    # ba = ax - bx, ay - by
    BA = A[0] - B[0], A[1] - B[1]
    BC = C[0] - B[0], C[1] - B[1]
    # step 2: calculate their dot product
    dotProd = (BA[0] * BC[0]) + (BA[1] * BC[1])
    # step 3: calculate their magnitudes
    vectMagnitudeBA = math.sqrt(BA[0]**2 + BA[1]**2)
    vectMagnitudeBC = math.sqrt(BC[0]**2 + BC[1]**2)

    # edge case to prevent dividing by 0
    if vectMagnitudeBA == 0 or vectMagnitudeBC == 0:
        return None

    # step 4: calculate the cosine value
    cosAngle = dotProd / (vectMagnitudeBA * vectMagnitudeBC)

    # clamp cosine to the valid range [-1, 1]
    cosAngle = max(-1, min(1, cosAngle))

    # calculate the angle in radians
    angle = math.acos(cosAngle)

    # step 5: return the angle in degrees
    return math.degrees(angle)

if __name__ == "__main__":
    # test cases for common angles
    print(calculateAngle((0, 1), (0, 0), (1, 0)))
    # expected: 90.0

    print(calculateAngle((-1, 0), (0, 0), (1, 0)))
    # expected: 180.0

    print(calculateAngle((1, 0), (0, 0), (1, 1)))
    # expected: 45.0

# classifies whether an elbow is partially bent, straight, or deeply bet
# classify elbow posture from its measured joint angle.
def classifyElbowPosition(angle):
    if angle is None:
        return None
    if 150 <= angle <= 180:
        return "Straight"
    elif 90 <= angle < 150:
        return "Slightly Bent"
    elif 0 <= angle < 90:
        return "Deeply Bent"
    else:
        return None