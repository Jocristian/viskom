import streamlit as st

st.set_page_config(page_title="Face Exercise & Emotion Detection", layout="wide")

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
st.markdown("""
    <style>
        body {
            font-family: 'Poppins', sans-serif;
        }
        .main {
            background-color: #f0f0f0;
            padding: 2rem;
        }
        .banner {
            background-image: url('https://plus.unsplash.com/premium_photo-1722728055718-20684f6bddbb?q=80&w=2070&auto=format&fit=crop');
            background-size: cover;
            background-position: center;
            padding: 4rem;
            border-radius: 12px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        .card {
            background-color: gray;
            padding: 1rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }
        .emotion-game {
            background-color: white;
            background-color: gray;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            text-align: center;
        }
        .emotion-game img {
            border-radius: 12px;
            width: 300px;
            height: auto;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .btn-restart {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            margin-top: 1rem;
            cursor: pointer;
        }
        .btn-restart:hover {
            background-color: #0056b3;
        }
    </style>
""", unsafe_allow_html=True)

# Banner
st.markdown("""
<div class="banner">
    <h1>Face Exercise & Real-Time Emotion Detection</h1>
    <p>Latihan wajah interaktif untuk meningkatkan kesehatan kulit dan ekspresi Anda!</p>
</div>
""", unsafe_allow_html=True)

# Description
st.markdown("""
<div class="card">
    <h2>🧠 Real-Time Emotion Detection</h2>
    <p>Aktifkan kamera Anda dan deteksi ekspresi wajah secara langsung menggunakan model klasifikasi emosi.</p>
</div>
""", unsafe_allow_html=True)

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




# Game section
components.html("""
<div class="emotion-game">
    <h2>🎮 Game Ekspresi Wajah</h2>
    <p>Tiru ekspresi yang muncul dalam waktu <span id="timer">5</span> detik!</p>
    <div>
        <strong>Target Ekspresi:</strong> <span id="target-emotion">...</span><br>
        <strong>Skor:</strong> <span id="score">0</span>
    </div>
    <br>
    <img id="emotion-img" src="" alt="Target Emosi" width="300"/>
    <br>
    <button class="btn-restart" onclick="startGame()">Mulai Ulang</button>
</div>
<!-- Ambil emosi dari backend (disisipkan oleh Streamlit) -->
<div id="emotion-data" style="display:none;">""" + st.session_state["current_emotion"] + """</div>

<script>
    let targetEmotion = "";
    let score = 0;
    let timeLeft = 5;
    let matchedThisRound = false;

    function getRandomEmotion() {
        const emotions = ['happy', 'sad', 'angry', 'surprise', 'neutral'];
        return emotions[Math.floor(Math.random() * emotions.length)];
    }

    function setTargetEmotion() {
        targetEmotion = getRandomEmotion();
        matchedThisRound = false;
        document.getElementById("target-emotion").textContent = targetEmotion;
        const emojiMap = {
            happy: "https://em-content.zobj.net/thumbs/240/apple/354/grinning-face_1f600.png",
            sad: "https://em-content.zobj.net/thumbs/240/apple/354/crying-face_1f622.png",
            angry: "https://em-content.zobj.net/thumbs/240/apple/354/pouting-face_1f621.png",
            surprise: "https://em-content.zobj.net/thumbs/240/apple/354/astonished-face_1f632.png",
            neutral: "https://em-content.zobj.net/thumbs/240/apple/354/neutral-face_1f610.png"
        };
        document.getElementById("emotion-img").src = emojiMap[targetEmotion];

    }

    function checkEmotion() {
        const currentEmotion = document.getElementById("emotion-data").textContent.trim();
        if (currentEmotion === targetEmotion && !matchedThisRound) {
            score++;
            matchedThisRound = true;
            document.getElementById("score").textContent = score;
        }
    }



    function countdown() {
        if (timeLeft > 0) {
            timeLeft--;
            document.getElementById("timer").textContent = timeLeft;
        } else {
            setTargetEmotion();
            timeLeft = 5;
        }
    }

    function startGame() {
        setTargetEmotion();
        score = 0;
        document.getElementById("score").textContent = score;
        timeLeft = 5;
    }

    startGame();
    setInterval(checkEmotion, 1000);
    setInterval(countdown, 1000);
</script>
<style>
    .emotion-game {
        background-color: #f7f7f7;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
        max-width: 500px;
        margin: 0 auto;
        font-family: 'Poppins', sans-serif;
    }

    .emotion-game h2 {
        font-size: 1.5rem;
        margin-bottom: 1rem;
        color: #333;
    }

    .emotion-game p {
        font-size: 1rem;
        margin-bottom: 1.5rem;
        color: #555;
    }

    .game-info {
        font-size: 1.2rem;
        margin-bottom: 1rem;
        color: #333;
    }

    .game-image-container {
        margin-bottom: 2rem;
    }

    .game-image-container img {
        border-radius: 8px;
        width: 250px;
        height: auto;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }

    .btn-restart {
        background-color: #007bff;
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        border: none;
        font-size: 1rem;
        cursor: pointer;
        transition: background-color 0.3s;
    }

    .btn-restart:hover {
        background-color: #0056b3;
    }

    .game-info span {
        font-weight: bold;
    }

    .timer {
        color: #ff5733;
        font-size: 1.5rem;
        font-weight: bold;
    }
</style>
""", height=600)
