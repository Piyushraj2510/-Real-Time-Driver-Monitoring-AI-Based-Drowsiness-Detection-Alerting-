# main.py

# ---------------------- IMPORTS ----------------------
# Standard libraries
import cv2
import numpy as np
import pygame
from datetime import datetime
import time
import random

# Custom modules
from ear_module import calculate_ear  # For Eye Aspect Ratio
from mar_module import calculate_mar  # For Mouth Aspect Ratio
from head_pose_module import estimate_head_pose  # For head orientation
from utils import (
    init_mediapipe, smooth_signal, draw_metrics,
    update_alert_level, play_alert, log_event,
    save_log, reset_counters
)

# ---------------------- INITIALIZATION ----------------------

# Start webcam
cap = cv2.VideoCapture(0)

# Initialize sound system for playing alerts
pygame.mixer.init()

# Initialize MediaPipe FaceMesh model for facial landmark detection
face_mesh = init_mediapipe()

# Initialize internal state (blink/yawn counters, alert levels, etc.)
state = {}
reset_counters(state)

# For performance measurement
start_time = time.time()
frame_count = 0

# Random motivational messages
messages = [
    "Stay sharp! 🚗",
    "Eyes on the road 👀",
    "Keep focused 🧠",
    "Drive safe ✨",
    "Alertness saves lives!"
]

# ---------------------- MAIN LOOP ----------------------
try:
    while True:
        # Capture frame from webcam
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Convert the image to RGB format for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe to detect facial landmarks
        result = face_mesh.process(rgb)

        # If a face is detected
        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0]
            h, w = frame.shape[:2]  # Get frame dimensions

            # Convert MediaPipe landmarks to pixel coordinates
            points = np.array([
                (int(p.x * w), int(p.y * h)) for p in landmarks.landmark
            ])

            # Extract coordinates of eyes and mouth landmarks
            left_eye = points[[362, 385, 387, 263, 373, 380]]
            right_eye = points[[33, 160, 158, 133, 153, 144]]
            mouth = points[[78, 308, 81, 13, 311, 402, 14, 178]]
            head_pts = points[[1, 152, 33, 263, 61, 291]]  # Nose, chin, eyes, mouth corners

            # ---------------------- FEATURE EXTRACTION ----------------------

            # Calculate Eye Aspect Ratio (EAR) to detect blinking
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            ear = (left_ear + right_ear) / 2

            # Calculate Mouth Aspect Ratio (MAR) to detect yawning
            mar = calculate_mar(mouth)

            # Estimate head pose (pitch, yaw, roll angles)
            pitch, yaw, roll = estimate_head_pose(head_pts.astype(np.float64), frame.shape)

            # Smooth EAR and MAR using a moving average to reduce noise
            ear = smooth_signal(ear, state['ear_smooth'])
            mar = smooth_signal(mar, state['mar_smooth'])

            # ---------------------- DROWSINESS DETECTION ----------------------

            # Get current time and check if user is drowsy
            now = datetime.now()
            drowsy_flag, state = update_alert_level(ear, mar, state, now)

            # If drowsiness persists long enough, play alert and log the event
            if drowsy_flag:
                play_alert(state['current_level'])
                log_event(state, ear, mar, pitch, yaw, roll)

            # Draw visual feedback on screen (EAR, MAR, drowsiness level, etc.)
            draw_metrics(
                frame, ear, mar, yaw,
                state['blink_counter'],
                state['yawn_counter'],
                state['current_level'],
                state['current_color']
            )

            # Display a motivational message every 5 seconds
            if int(time.time()) % 5 == 0:
                cv2.putText(
                    frame,
                    random.choice(messages),
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # Display the video frame with detection results
        cv2.imshow("Drowsiness Detection", frame)

        # Press 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ---------------------- CLEANUP ----------------------
finally:
    cap.release()              # Release webcam
    cv2.destroyAllWindows()    # Close display window
    save_log(state)            # Save all drowsiness events to a CSV log

    # Print session summary
    duration = time.time() - start_time
    print(f"\n[INFO] Session Duration: {duration:.2f} seconds")
    print(f"[INFO] Total Frames Processed: {frame_count}")
    print(f"[INFO] Logged Events: {len(state['log'])}")
