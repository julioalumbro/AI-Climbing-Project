import math

def calculateAngle(A, B, C):

    # Step 1: Calculate vectors BA and BC
    # BA = Ax - Bx, Ay - By
    BA = A[0] - B[0], A[1] - B[1]
    BC = C[0] - B[0], C[1] - B[1]
    # Step 2: Calculate their dot product
    dotProd = (BA[0] * BC[0]) + (BA[1] * BC[1])
    # Step 3: Calculate their magnitudes
    vectMagnitudeBA = math.sqrt(BA[0]**2 + BA[1]**2)
    vectMagnitudeBC = math.sqrt(BC[0]**2 + BC[1]**2)

    # edge case to prevent dividing by 0
    if vectMagnitudeBA == 0 or vectMagnitudeBC == 0:
        return None

    # Step 4: Calculate the cosine value
    cosAngle = dotProd / (vectMagnitudeBA * vectMagnitudeBC)

    # Clamp cosine to the valid range [-1, 1]
    cosAngle = max(-1, min(1, cosAngle))

    # Calculate the angle in radians
    angle = math.acos(cosAngle)

    # Step 5: Return the angle in degrees
    return math.degrees(angle)


# test cases for common angles
print(calculateAngle((0, 1), (0, 0), (1, 0)))
# Expected: 90.0

print(calculateAngle((-1, 0), (0, 0), (1, 0)))
# Expected: 180.0

print(calculateAngle((1, 0), (0, 0), (1, 1)))
# Expected: 45.0

# classifies whether an elbow is partially bent, straight, or deeply bet
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