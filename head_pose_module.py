# modules/head_pose_module.py

import cv2
import numpy as np

def estimate_head_pose(landmarks, image_shape):
    """
    Estimate head pose angles (pitch, yaw, roll) using facial landmarks.

    Parameters:
        landmarks (np.array): 6-point (x, y) array of facial keypoints
            [nose_tip, chin, left_eye, right_eye, left_mouth, right_mouth]
        image_shape (tuple): shape of the image frame (h, w, c)

    Returns:
        tuple: (pitch, yaw, roll) in degrees
    """
    if len(landmarks) != 6:
        raise ValueError("Head pose landmarks must contain 6 points")

    # 3D model points (based on a generic face model)
    model_points = np.array([
        [0.0, 0.0, 0.0],          # Nose tip
        [0.0, -63.6, -12.5],      # Chin
        [-43.3, 32.7, -26.0],     # Left eye left corner
        [43.3, 32.7, -26.0],      # Right eye right corner
        [-28.9, -28.9, -24.1],    # Left Mouth corner
        [28.9, -28.9, -24.1]      # Right mouth corner
    ])

    # Camera internals
    height, width = image_shape[:2]
    focal_length = width
    center = (width / 2, height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    # Solve PnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, landmarks, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0, 0.0

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Extract Euler angles (in radians)
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)

    pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    yaw = np.arctan2(-rotation_matrix[2, 0], sy)
    roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    # Convert to degrees
    pitch = np.degrees(pitch)
    yaw = np.degrees(yaw)
    roll = np.degrees(roll)

    return pitch, yaw, roll

# Self-test example
if __name__ == "__main__":
    dummy_points = np.array([
        [320, 240], [320, 480],
        [240, 200], [400, 200],
        [260, 300], [380, 300]
    ], dtype=np.float64)

    img_shape = (480, 640, 3)
    pitch, yaw, roll = estimate_head_pose(dummy_points, img_shape)
    print(f"Pitch: {pitch:.2f}, Yaw: {yaw:.2f}, Roll: {roll:.2f}")