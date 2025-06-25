# modules/config.py

"""
Configuration file for Drowsiness Detection System
"""

import cv2

CONFIG = {
    # EAR & MAR thresholds
    "EAR_THRESHOLD": 0.25,
    "MAR_THRESHOLD": 0.6,

    # Consecutive frame threshold before action
    "CONSEC_FRAMES": 20,

    # Frame rate (used for timing calculations)
    "FPS": 20,

    # Alert audio path per level
    "alert_sounds": {
        1: "alert_level1.mp3",
        2: "alert_level2.mp3",
        3: "alert_level3.mp3",
        4: "alert_level4.mp3",
        5: "alert_level5.mp3"
    },

    # Alert messages per level
    "alert_messages": {
        1: "⚠️ Mild drowsiness detected. Stay alert!",
        2: "⚠️ Warning: You’re getting drowsy!",
        3: "⛔ Drowsiness increasing. Take a break!",
        4: "❗Critical drowsiness! Immediate action advised!",
        5: "🚨 Danger: You must stop and rest now."
    },

    # Drowsiness level mapping by time in seconds
    "SECONDS_PER_LEVEL": {
        2: ("Level 1", (0, 255, 255)),      # Yellow
        4: ("Level 2", (0, 165, 255)),     # Orange
        6: ("Level 3", (0, 140, 255)),     # Deep Orange
        8: ("Level 4", (0, 0, 255)),       # Red
        10: ("Level 5", (128, 0, 128))     # Purple
    },

    # Display font settings
    "font": cv2.FONT_HERSHEY_SIMPLEX,
    "font_scale": 0.7,
    "font_thickness": 2,

    # Output log file
    "log_file": "drowsiness_full_log.csv"
}
