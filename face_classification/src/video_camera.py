from statistics import mode
import cv2
from keras.models import load_model
import numpy as np
from utils.datasets import get_labels
from utils.inference import detect_faces, draw_text, draw_bounding_box, apply_offsets, load_detection_model
from utils.preprocessor import preprocess_input

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class VideoCamera:
    def __init__(self):
        # Model paths
        detection_model_path = os.path.join(BASE_DIR, 'face_classification', 'trained_models', 'detection_models', 'haarcascade_frontalface_default.xml')
        emotion_model_path = os.path.join(BASE_DIR, 'face_classification', 'trained_models', 'emotion_models', 'fer2013_mini_XCEPTION.102-0.66.hdf5')
        
        # Load models and config
        self.face_detection = load_detection_model(detection_model_path)
        self.emotion_classifier = load_model(emotion_model_path, compile=False)
        self.emotion_labels = get_labels('fer2013')
        self.emotion_target_size = self.emotion_classifier.input_shape[1:3]
        self.emotion_offsets = (20, 40)
        self.frame_window = 10
        self.emotion_window = []

        # OpenCV video capture
        self.video_capture = cv2.VideoCapture(0)
  # Use DirectShow backend

        self.last_emotion = "neutral"


    def __del__(self):
        if self.video_capture.isOpened():
            self.video_capture.release()


    def get_frame(self):
        success, bgr_image = self.video_capture.read()
        if not success:
            return None


        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        faces = detect_faces(self.face_detection, gray_image)

        for face_coordinates in faces:
            x1, x2, y1, y2 = apply_offsets(face_coordinates, self.emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]

            try:
                gray_face = cv2.resize(gray_face, self.emotion_target_size)
            except:
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)

            emotion_prediction = self.emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = self.emotion_labels[emotion_label_arg]
            self.emotion_window.append(emotion_text)

            if len(self.emotion_window) > self.frame_window:
                self.emotion_window.pop(0)

            try:
                emotion_mode = mode(self.emotion_window)
                self.last_emotion = emotion_mode
            except:
                emotion_mode = emotion_text

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

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        ret, jpeg = cv2.imencode('.jpg', bgr_image)
        return jpeg.tobytes()
    
    def get_last_detected_emotion(self):
        return self.last_emotion
    
