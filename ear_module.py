# modules/ear_module.py

import numpy as np
def euclidean_distance(point1, point2):
    """
    Compute the Euclidean distance between two 2D points.
    """
    return np.linalg.norm(point1 - point2)

def calculate_ear(eye):
    """
    Calculate the Eye Aspect Ratio (EAR) for one eye.

    Parameters:
        eye (np.array): 6-point eye landmarks.

    Returns:
        float: EAR value.
    """
    if len(eye) != 6:
        raise ValueError("Eye landmarks must contain 6 points")

    # Vertical distances
    A = euclidean_distance(eye[1], eye[5])
    B = euclidean_distance(eye[2], eye[4])

    # Horizontal distance
    C = euclidean_distance(eye[0], eye[3])

    # Avoid division by zero
    if C == 0:
        return 0.0

    # EAR formula
    ear = (A + B) / (2.0 * C)
    return ear

# Debugging support
if __name__ == "__main__":
    # Dummy eye points for testing (hexagon shape)
    dummy_eye = np.array([
        [1.0, 2.0],  # p1
        [1.5, 1.0],  # p2
        [2.5, 1.0],  # p3
        [3.0, 2.0],  # p4
        [2.5, 3.0],  # p5
        [1.5, 3.0]   # p6
    ])
    ear_value = calculate_ear(dummy_eye)
    print(f"EAR for dummy eye: {ear_value:.3f}")
