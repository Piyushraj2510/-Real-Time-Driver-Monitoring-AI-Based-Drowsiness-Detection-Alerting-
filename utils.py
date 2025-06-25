# modules/utils.py

import pygame
import csv
from datetime import datetime
from config import CONFIG
import cv2

def init_mediapipe():
    """Initialize MediaPipe FaceMesh solution."""
    import mediapipe as mp
    return mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

def smooth_signal(value, buffer, k=5):
    """Apply moving average smoothing to a signal."""
    buffer.append(value)
    if len(buffer) > k:
        buffer.pop(0)
    return sum(buffer) / len(buffer)

def is_blinking(ear):
    """Check if EAR is below threshold indicating blink."""
    return ear < CONFIG["EAR_THRESHOLD"]

def is_yawning(mar):
    """Check if MAR is above threshold indicating yawn."""
    return mar > CONFIG["MAR_THRESHOLD"]

def play_alert(level=1):
    """Play alert sound based on drowsiness level."""
    pygame.mixer.music.load(CONFIG["alert_sounds"].get(level, "alert_level1.mp3"))
    pygame.mixer.music.play()

def draw_metrics(frame, ear, mar, yaw, blinks, yawns, level, color):
    """Display EAR, MAR, yaw, and counters with alert level message on frame."""
    cv2.putText(frame, f"EAR: {ear:.2f}  MAR: {mar:.2f}  Yaw: {yaw:.1f}", (10, 30),
                CONFIG["font"], CONFIG["font_scale"], color, CONFIG["font_thickness"])
    cv2.putText(frame, f"Blinks: {blinks}  Yawns: {yawns}", (10, 60),
                CONFIG["font"], CONFIG["font_scale"], color, CONFIG["font_thickness"])
    if level:
        message = CONFIG["alert_messages"].get(level, "")
        cv2.putText(frame, message, (10, 90),
                    CONFIG["font"], CONFIG["font_scale"], color, CONFIG["font_thickness"])

def update_alert_level(ear, mar, state, now):
    """
    Update alert level based on duration of drowsiness symptoms.
    Returns a flag indicating whether alert should be triggered.
    """
    drowsy_flag = False
    if is_blinking(ear) or is_yawning(mar):
        if not state['start_drowsy_time']:
            state['start_drowsy_time'] = now
        duration = (now - state['start_drowsy_time']).total_seconds()
        for seconds, (level_name, color) in sorted(CONFIG['SECONDS_PER_LEVEL'].items()):
            if duration >= seconds:
                state['current_level'] = list(CONFIG['SECONDS_PER_LEVEL'].keys()).index(seconds) + 1
                state['current_color'] = color
                drowsy_flag = True
    else:
        # Reset if alert condition clears
        state['start_drowsy_time'] = None
        state['current_level'] = None
        state['current_color'] = (255, 255, 255)
    return drowsy_flag, state

def log_event(state, ear, mar, pitch, yaw, roll):
    """Log drowsiness-related data into memory for CSV export."""
    state['log'].append({
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'EAR': round(ear, 2),
        'MAR': round(mar, 2),
        'Pitch': round(pitch, 2),
        'Yaw': round(yaw, 2),
        'Roll': round(roll, 2),
        'Level': state['current_level']
    })

def save_log(state):
    """Save all logged drowsiness events to a CSV file."""
    with open(CONFIG['log_file'], mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=[
            'timestamp', 'EAR', 'MAR', 'Pitch', 'Yaw', 'Roll', 'Level'])
        writer.writeheader()
        for entry in state['log']:
            writer.writerow(entry)

def reset_counters(state):
    """Reset all counters and buffers to initial state."""
    state.update({
        'blink_counter': 0,
        'yawn_counter': 0,
        'drowsy_counter': 0,
        'start_drowsy_time': None,
        'eye_closed': False,
        'ear_smooth': [],
        'mar_smooth': [],
        'current_level': None,
        'current_color': (255, 255, 255),
        'log': [],
        'frame_log': []
    })
