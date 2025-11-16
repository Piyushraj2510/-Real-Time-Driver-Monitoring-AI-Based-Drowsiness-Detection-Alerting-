import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from ear_module import calculate_ear
from mar_module import calculate_mar
from head_pose_module import estimate_head_pose
from utils import (
    init_mediapipe, smooth_signal, draw_metrics,
    update_alert_level, log_event, reset_counters
)

st.set_page_config(page_title="Driver Drowsiness Detection", layout="wide")
st.title("😴 Real-Time AI Driver Drowsiness Detection")
st.markdown("### EAR · MAR · Head Pose · Alert Levels")

# ---------------------- VIDEO PROCESSOR ----------------------
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = init_mediapipe()
        self.state = {}
        reset_counters(self.state)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0]
            pts = np.array([(int(p.x*w), int(p.y*h)) for p in landmarks.landmark])

            # Extract regions
            left_eye = pts[[362,385,387,263,373,380]]
            right_eye = pts[[33,160,158,133,153,144]]
            mouth = pts[[78,308,81,13,311,402,14,178]]
            head_pts = pts[[1,152,33,263,61,291]]

            # Compute features
            ear = (calculate_ear(left_eye)+calculate_ear(right_eye)) / 2
            mar = calculate_mar(mouth)
            pitch, yaw, roll = estimate_head_pose(head_pts.astype(np.float64), img.shape)

            # Smooth signals
            ear = smooth_signal(ear, self.state["ear_smooth"])
            mar = smooth_signal(mar, self.state["mar_smooth"])

            # Update alert levels
            drowsy, self.state = update_alert_level(ear, mar, self.state, None)

            # Log if needed
            if drowsy:
                log_event(self.state, ear, mar, pitch, yaw, roll)

            # Draw UI
            draw_metrics(
                img, ear, mar, yaw,
                self.state['blink_counter'],
                self.state['yawn_counter'],
                self.state['current_level'],
                self.state['current_color']
            )

        return img


# ---------------------- START STREAM ----------------------
webrtc_streamer(
    key="drowsiness-app",
    video_processor_factory=DrowsinessProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.warning("⚠️ Note: Audio alerts (pygame) do not work in a browser. Use visual alerts instead.")
