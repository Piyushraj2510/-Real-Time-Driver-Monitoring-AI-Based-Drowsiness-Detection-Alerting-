# modules/mar_module.py

import numpy as np

def euclidean_distance(p1, p2):
    """
    Calculate Euclidean distance between two points.
    """
    return np.linalg.norm(p1 - p2)

def calculate_mar(mouth):
    """
    Calculate the Mouth Aspect Ratio (MAR).

    Parameters:
        mouth (np.array): Array of 8 (x, y) landmarks outlining the mouth.

    Returns:
        float: MAR value.
    """
    if len(mouth) != 8:
        raise ValueError("Mouth landmarks must contain 8 points")

    # Vertical distances (top to bottom)
    A = euclidean_distance(mouth[2], mouth[3])  # center vertical
    B = euclidean_distance(mouth[4], mouth[6])  # inner left
    C = euclidean_distance(mouth[5], mouth[7])  # inner right

    # Horizontal distance (left to right)
    D = euclidean_distance(mouth[0], mouth[1])

    if D == 0:
        return 0.0

    mar = (A + B + C) / (3.0 * D)
    return mar

# Debug test
if __name__ == "__main__":
    # Dummy mouth coordinates
    dummy_mouth = np.array([
        [1.0, 2.0], [5.0, 2.0],   # left-right
        [3.0, 1.0], [3.0, 3.0],   # top-bottom center
        [2.0, 1.5], [4.0, 1.5],   # top inner
        [2.0, 2.5], [4.0, 2.5]    # bottom inner
    ])
    mar_value = calculate_mar(dummy_mouth)
    print(f"MAR for dummy mouth: {mar_value:.3f}")