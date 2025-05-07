from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
from keras.models import load_model
# Global variable
from streamlit.runtime.scriptrunner import get_script_run_ctx

# Ambil lock dan variabel dari app.py

from threading import Lock

last_emotion = "neutral"
emotion_lock = Lock()


import os
import sys

# Tambahkan path ke direktori 'src'
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from utils.datasets import get_labels
from utils.inference import detect_faces, draw_bounding_box, draw_text, apply_offsets, load_detection_model
from utils.preprocessor import preprocess_input


# Tambahkan path ke folder utils


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'trained_models'))

detection_model_path = os.path.join(MODEL_DIR, 'detection_models', 'haarcascade_frontalface_default.xml')
emotion_model_path = os.path.join(MODEL_DIR, 'emotion_models', 'fer2013_mini_XCEPTION.102-0.66.hdf5')


face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
emotion_labels = get_labels('fer2013')
emotion_target_size = emotion_classifier.input_shape[1:3]
emotion_offsets = (20, 40)

class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_window = 10
        self.emotion_window = []
        self.set_emotion_callback = None

    def update_emotion(self, emotion_mode):
        if self.set_emotion_callback:
            self.set_emotion_callback(emotion_mode)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        faces = detect_faces(face_detection, gray_image)

        for face_coordinates in faces:
            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]

            try:
                gray_face = cv2.resize(gray_face, emotion_target_size)
            except:
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)

            emotion_prediction = emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]
            self.emotion_window.append(emotion_text)

            if len(self.emotion_window) > self.frame_window:
                self.emotion_window.pop(0)

            try:
                from statistics import mode
                emotion_mode = mode(self.emotion_window)
            except:
                emotion_mode = emotion_text

            # ✅ Simpan emosi terakhir secara global
            self.update_emotion(emotion_mode)


            # Color based on emotion
            if emotion_mode == 'angry':
                color = emotion_probability * np.asarray((255, 0, 0))
            elif emotion_mode == 'sad':
                color = emotion_probability * np.asarray((0, 0, 255))
            elif emotion_mode == 'happy':
                color = emotion_probability * np.asarray((255, 255, 0))
            elif emotion_mode == 'surprise':
                color = emotion_probability * np.asarray((0, 255, 255))
            else:
                color = emotion_probability * np.asarray((0, 255, 0))

            color = color.astype(int).tolist()
            draw_bounding_box(face_coordinates, rgb_image, color)
            draw_text(face_coordinates, rgb_image, emotion_mode, color, 0, -45, 1, 1)

        output_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        return av.VideoFrame.from_ndarray(output_image, format="bgr24")
