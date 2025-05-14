import streamlit as st

# ⛔ Pastikan ini paling atas sebelum semua pemanggilan st lainnya
st.set_page_config(page_title="Emotion Camera", layout="wide")

from streamlit_webrtc import webrtc_streamer
from face_classification.src.video_camera import EmotionProcessor
import threading

# lanjutkan dengan kode seperti biasa


from streamlit_webrtc import webrtc_streamer
import streamlit.components.v1 as components
from face_classification.src.video_camera import EmotionProcessor


import os

# di file utama (misal app.py)
import threading
import json
print("Streamlit module path:", st.__file__)
print("Streamlit version:", st.__version__)

emotion_lock = threading.Lock()
last_emotion = "neutral"  # default
def set_emotion(emotion):
    global last_emotion
    with emotion_lock:
        last_emotion = emotion


@st.cache_resource
def get_emotion_lock():
    return emotion_lock, last_emotion

# Function to retrieve last emotion with a thread lock
def get_emotion():
    with emotion_lock:
        return last_emotion

# Page config

# Custom HTML & CSS Styling


# Webcam and emotion detection
def processor_factory():
    processor = EmotionProcessor()
    processor.set_emotion_callback = set_emotion
    return processor

st.markdown("""
    <div class="card">
        <h2>🎥 Kamera Deteksi Emosi</h2>
    </div>
""", unsafe_allow_html=True)
webrtc_streamer(
    key="emotion-detection",
    video_processor_factory=processor_factory,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Simpan emosi terbaru ke session_state
st.session_state["current_emotion"] = get_emotion()

# Tambahkan div tersembunyi yang bisa dibaca oleh JavaScript
st.markdown(f"""
    <div id="emotion-data" style="display:none;">{st.session_state["current_emotion"]}</div>
""", unsafe_allow_html=True)
